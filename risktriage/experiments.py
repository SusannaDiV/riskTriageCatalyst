"""Run the six RiskTriage experiments on OCx24 HER."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from risktriage.calibration import residual_intervals
from risktriage.data import HERTask, load_her_task
from risktriage.decisions import (
    coverage_of,
    evaluate_actions,
    fit_ensemble,
    policies_for_split,
    prediction_metrics,
    residual_mad,
    scaled_sigma,
)
from risktriage.models import make_model
from risktriage.plots import (
    plot_cost_sensitivity,
    plot_discovery,
    plot_frontier,
    plot_matched_risk,
    plot_ood_test_rate,
    plot_pred_scatter,
)
from risktriage.stats import bootstrap_mean_ci, min_risk_at_test_budget, min_test_at_risk, paired_bootstrap_delta
from risktriage.theory import (
    Costs,
    bayes_action_from_p,
    cost_aware_interval_action,
    gaussian_success_prob,
    interval_action,
    robust_action,
    voi_perfect_experiment,
)
from risktriage.replay import policy_order, ranking_metrics, topk_valuable
from risktriage.selection import propensity_weights
from risktriage.splits import grouped_split, loco_folds, ood_cluster_split




def _y_star(y_train: np.ndarray, q: float) -> float:
    return float(np.quantile(y_train, q))


def experiment1_prediction(task: HERTask, out: Path, smoke: bool = False) -> pd.DataFrame:
    X, y = task.X, task.y
    folds = loco_folds(task.loco_groups)
    models = ["linear", "ridge", "rf"] if smoke else ["linear", "ridge", "rf", "lgbm"]
    rows = []
    oof = {}
    for name in models:
        model = make_model(name, seed=0)
        pred = cross_val_predict(model, X, y, cv=folds)
        oof[name] = pred
        m = prediction_metrics(y, pred)
        m["model"] = name
        rows.append(m)
        plot_pred_scatter(y, pred, out / f"exp1_{name}_scatter.png", f"LOCO {name}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp1_prediction.csv", index=False)
    return df


def _one_split_fit(task: HERTask, split, n_members: int, seed: int, costs: Costs, q: float):
    tr, ca, te = split.train, split.cal, split.test
    y_star = _y_star(task.y[tr], q)
    from risktriage.models import LGBMEnsemble
    from risktriage.decisions import FittedSplit

    ens = LGBMEnsemble(n_members=n_members, seed=seed).fit(task.X[tr], task.y[tr])
    dtr, dca, dte = ens.predict_dist(task.X[tr]), ens.predict_dist(task.X[ca]), ens.predict_dist(task.X[te])
    pred_tr, std_tr = dtr.mean, np.maximum(dtr.std, 1e-4)
    pred_ca, std_ca = dca.mean, np.maximum(dca.std, 1e-4)
    pred_te, std_te = dte.mean, np.maximum(dte.std, 1e-4)
    rscale = residual_mad(pred_tr, task.y[tr])
    scale_ca = scaled_sigma(std_ca, rscale)
    fitted = FittedSplit(
        pred_mean=pred_te,
        pred_std=std_te,
        y=task.y[te],
        success=(task.y[te] >= y_star).astype(int),
        y_star=y_star,
        costs=costs,
        residual_scale=rscale,
        X=task.X[te],
        idx=te,
        sabatier=task.frame["sabatier_h"].to_numpy()[te],
        matched=task.frame["matched_int"].to_numpy()[te],
    )
    return fitted, pred_ca, task.y[ca], scale_ca, pred_tr, std_tr, pred_te, std_te, y_star, rscale


def experiment2_calibration(
    task: HERTask,
    out: Path,
    n_seeds: int,
    n_members: int,
    costs: Costs,
    alpha: float,
    r_star: float,
    q: float,
) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        split = grouped_split(task.groups, seed=seed)
        fitted, pred_ca, y_ca, scale_ca, *_ = _one_split_fit(task, split, n_members, seed, costs, q)
        pol = policies_for_split(fitted, pred_ca, y_ca, scale_ca, alpha, r_star)
        for name in ["uncalibrated", "point", "split_cp", "crc", "rac"]:
            ev = evaluate_actions(name, pol[name], fitted)
            ev["seed"] = seed
            iv = pol.get("_iv_cp") if name == "split_cp" else pol.get("_iv_crc") if name == "crc" else pol.get("_iv_rac")
            if iv is not None:
                ev["coverage"] = coverage_of(iv, fitted.y)
                ev["width"] = float(np.mean(iv.width))
            else:
                ev["coverage"] = np.nan
                ev["width"] = np.nan
            rows.append(ev)
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp2_calibration.csv", index=False)
    return df


def experiment3_frontier(
    task: HERTask,
    out: Path,
    n_seeds: int,
    n_members: int,
    costs: Costs,
    q: float,
    r_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        split = grouped_split(task.groups, seed=seed)
        fitted, pred_ca, y_ca, scale_ca, *rest = _one_split_fit(task, split, n_members, seed, costs, q)
        y_star = fitted.y_star
        scale_te = scaled_sigma(fitted.pred_std, fitted.residual_scale)
        p = gaussian_success_prob(fitted.pred_mean, scale_te, y_star)
        from risktriage.theory import bayes_action_from_p

        # confidence-threshold sweep via costs.c_e proxy: threshold on p
        for t in np.linspace(0.05, 0.45, 9):
            a = np.full(len(p), 1, dtype=int)
            a[p < t] = 0
            a[p > 1 - t] = 2
            ev = evaluate_actions("prob_threshold", a, fitted)
            ev.update({"seed": seed, "param": t})
            rows.append(ev)
        a_var = np.full(len(p), 0, dtype=int)
        # uncertainty ranking: TEST the top fraction by std; rest point classify
        order = np.argsort(-fitted.pred_std)
        for frac in np.linspace(0.1, 0.9, 9):
            k = max(1, int(frac * len(p)))
            a = np.where(fitted.pred_mean >= y_star, 2, 0)
            a[order[:k]] = 1
            ev = evaluate_actions("uncertainty", a, fitted)
            ev.update({"seed": seed, "param": frac})
            rows.append(ev)
        for alpha in alpha_grid:
            from risktriage.calibration import fit_split_conformal

            lam = fit_split_conformal(pred_ca, y_ca, scale_ca, alpha=float(alpha))
            iv = residual_intervals(fitted.pred_mean, scale_te, lam)
            a = interval_action(iv.lower, iv.upper, y_star)
            ev = evaluate_actions("split_cp", a, fitted)
            ev.update({"seed": seed, "param": float(alpha), "coverage": coverage_of(iv, fitted.y)})
            rows.append(ev)
        for r in r_grid:
            from risktriage.calibration import crc_lambda, rac_lambda

            lam = crc_lambda(pred_ca, y_ca, scale_ca, y_star, costs, float(r), False)
            iv = residual_intervals(fitted.pred_mean, scale_te, lam)
            a = interval_action(iv.lower, iv.upper, y_star)
            ev = evaluate_actions("crc", a, fitted)
            ev.update({"seed": seed, "param": float(r), "coverage": coverage_of(iv, fitted.y)})
            rows.append(ev)
            lam = rac_lambda(pred_ca, y_ca, scale_ca, y_star, costs, float(r))
            iv = residual_intervals(fitted.pred_mean, scale_te, lam)
            a = robust_action(iv.lower, iv.upper, y_star, costs)
            ev = evaluate_actions("rac", a, fitted)
            ev.update({"seed": seed, "param": float(r), "coverage": coverage_of(iv, fitted.y)})
            rows.append(ev)
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp3_frontier.csv", index=False)
    mean = df.groupby(["method", "param"], as_index=False)[["p_test", "triage_risk", "wrong_nontest"]].mean()
    plot_frontier(mean, out / "exp3_frontier.png")
    return df


def experiment4_replay(
    task: HERTask,
    out: Path,
    n_seeds: int,
    n_members: int,
    costs: Costs,
    alpha: float,
    r_star: float,
    q: float,
    budgets: list[int],
) -> pd.DataFrame:
    methods = ["random", "sabatier", "ml_rank", "variance", "ucb", "ei", "voi", "conformal_width", "risktriage"]
    curves = []
    summary = []
    for seed in range(n_seeds):
        split = grouped_split(task.groups, seed=seed)
        fitted, pred_ca, y_ca, scale_ca, *_ = _one_split_fit(task, split, n_members, seed, costs, q)
        pol = policies_for_split(fitted, pred_ca, y_ca, scale_ca, alpha, r_star)
        valuable = topk_valuable(fitted.y, 0.1)
        scale_te = scaled_sigma(fitted.pred_std, fitted.residual_scale)
        rng = np.random.default_rng(seed + 99)
        B = np.array(sorted({b for b in budgets if b >= 1}))
        B = B[B <= len(fitted.y)]
        if len(B) == 0:
            B = np.array([max(1, len(fitted.y) // 4)])
        for name in methods:
            order = policy_order(
                name,
                fitted.pred_mean,
                scale_te,
                fitted.sabatier,
                pol["_p"],
                pol["_voi"],
                fitted.y_star,
                rng,
                interval_width=pol["_iv_cp"].width,
                rac_test=(pol["rac"] == 1).astype(float),
                rac_action=pol["rac"],
            )
            met = ranking_metrics(fitted.y, order, B, valuable)
            for row in met["curve"]:
                curves.append({"seed": seed, "method": name, **row})
            summary.append({"seed": seed, "method": name, "audc": met["audc"], "ett80": met["ett80"]})
    cdf = pd.DataFrame(curves)
    sdf = pd.DataFrame(summary)
    cdf.to_csv(out / "exp4_replay_curves.csv", index=False)
    sdf.to_csv(out / "exp4_replay_summary.csv", index=False)
    mean_curve = cdf.groupby(["method", "B"], as_index=False)["recall"].mean()
    plot_discovery(mean_curve, out / "exp4_discovery.png")
    # bootstrap AUDC: RiskTriage vs ml_rank
    if "risktriage" in sdf.method.values and "ml_rank" in sdf.method.values:
        a = sdf.pivot_table(index="seed", columns="method", values="audc")
        if "risktriage" in a.columns and "ml_rank" in a.columns:
            boot = paired_bootstrap_delta(a["risktriage"].to_numpy(), a["ml_rank"].to_numpy(), n_boot=2000, seed=0)
            (out / "exp4_bootstrap_audc.json").write_text(json.dumps(boot, indent=2))
    return sdf


def experiment5_staged(
    task: HERTask,
    out: Path,
    n_seeds: int,
    n_members: int,
    costs: Costs,
    alpha: float,
    r_star: float,
    q: float,
) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        split = grouped_split(task.groups, seed=seed)
        tr, ca, te = split.train, split.cal, split.test
        y_star = _y_star(task.y[tr], q)
        from risktriage.models import LGBMEnsemble
        from risktriage.calibration import rac_lambda
        from risktriage.decisions import FittedSplit, evaluate_actions

        ens = LGBMEnsemble(n_members=n_members, seed=seed).fit(task.X[tr], task.y[tr])
        dtr, dca, dte = ens.predict_dist(task.X[tr]), ens.predict_dist(task.X[ca]), ens.predict_dist(task.X[te])
        pred_tr, pred_ca, pred_te = dtr.mean, dca.mean, dte.mean
        std_ca, std_te = np.maximum(dca.std, 1e-4), np.maximum(dte.std, 1e-4)
        rscale = residual_mad(pred_tr, task.y[tr])
        scale_ca = scaled_sigma(std_ca, rscale)
        scale_te = scaled_sigma(std_te, rscale)
        lam = rac_lambda(pred_ca, task.y[ca], scale_ca, y_star, costs, r_star)
        iv = residual_intervals(pred_te, scale_te, lam)
        a_direct = robust_action(iv.lower, iv.upper, y_star, costs)
        success = (task.y[te] >= y_star).astype(int)

        fitted = FittedSplit(pred_te, std_te, task.y[te], success, y_star, costs, rscale, task.X[te], te, task.frame["sabatier_h"].to_numpy()[te], task.frame["matched_int"].to_numpy()[te])
        ev = evaluate_actions("direct", a_direct, fitted)
        ev["seed"] = seed
        rows.append(ev)

        # staged: after XRF/XRD, use staged features; drop unmatched unless VOI still high
        ens_s = LGBMEnsemble(n_members=n_members, seed=seed + 1).fit(task.X_staged[tr], task.y[tr])
        dtr_s, dca_s, dte_s = ens_s.predict_dist(task.X_staged[tr]), ens_s.predict_dist(task.X_staged[ca]), ens_s.predict_dist(task.X_staged[te])
        pred_tr_s, pred_ca_s, pred_te_s = dtr_s.mean, dca_s.mean, dte_s.mean
        std_te_s = np.maximum(dte_s.std, 1e-4)
        rscale_s = residual_mad(pred_tr_s, task.y[tr])
        scale_ca_s = scaled_sigma(np.maximum(dca_s.std, 1e-4), rscale_s)
        scale_te_s = scaled_sigma(std_te_s, rscale_s)
        lam_s = rac_lambda(pred_ca_s, task.y[ca], scale_ca_s, y_star, costs, r_star)
        iv_s = residual_intervals(pred_te_s, scale_te_s, lam_s)
        a_st = robust_action(iv_s.lower, iv_s.upper, y_star, costs)
        matched = task.frame["matched_int"].to_numpy()[te] > 0.5
        # unmatched realizations: convert TEST -> DROP (stop before electrochemistry)
        a_filter = a_st.copy()
        a_filter[(~matched) & (a_st == 1)] = 0
        fitted_s = FittedSplit(pred_te_s, std_te_s, task.y[te], success, y_star, costs, rscale_s, task.X_staged[te], te, task.frame["sabatier_h"].to_numpy()[te], matched.astype(float))
        evs = evaluate_actions("staged_xrd_update", a_st, fitted_s)
        evs["seed"] = seed
        rows.append(evs)
        evf = evaluate_actions("staged_filter_unmatched", a_filter, fitted_s)
        evf["seed"] = seed
        # extra funnel stats
        evf["tests_avoided"] = float(((~matched) & (a_st == 1)).mean())
        evf["good_lost"] = float(((~matched) & (a_st == 1) & (success == 1)).mean())
        rows.append(evf)
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp5_staged.csv", index=False)
    return df


def experiment6_ood(
    task: HERTask,
    out: Path,
    n_seeds: int,
    n_members: int,
    costs: Costs,
    alpha: float,
    r_star: float,
    q: float,
) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        iid = grouped_split(task.groups, seed=seed)
        ood = ood_cluster_split(task.X, task.loco_groups, seed=seed)
        for tag, split in [("iid", iid), ("ood", ood)]:
            if len(split.test) < 5 or len(split.cal) < 5 or len(split.train) < 8:
                continue
            fitted, pred_ca, y_ca, scale_ca, *_ = _one_split_fit(task, split, n_members, seed, costs, q)
            pol = policies_for_split(fitted, pred_ca, y_ca, scale_ca, alpha, r_star)
            for name in ["split_cp", "crc", "rac"]:
                ev = evaluate_actions(name, pol[name], fitted)
                iv = pol["_iv_cp"] if name == "split_cp" else pol["_iv_crc"] if name == "crc" else pol["_iv_rac"]
                ev.update({"seed": seed, "split": tag, "coverage": coverage_of(iv, fitted.y), "width": float(np.mean(iv.width))})
                rows.append(ev)
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp6_ood.csv", index=False)
    if len(df):
        plot_ood_test_rate(df, out / "exp6_ood_test_rate.png")
    return df


def experiment_ablations(exp2: pd.DataFrame, exp4: pd.DataFrame, exp5: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    if len(exp2):
        g = exp2.groupby("method")[["wrong_nontest", "p_test", "triage_risk"]].mean().reset_index()
        g["source"] = "exp2"
        rows.append(g)
    if len(exp4):
        g = exp4.groupby("method")[["audc"]].mean().reset_index()
        g["source"] = "exp4"
        rows.append(g)
    if len(exp5):
        g = exp5.groupby("method")[["wrong_nontest", "p_test", "triage_risk"]].mean().reset_index()
        g["source"] = "exp5"
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "ablations.csv", index=False)
    return df


def experiment_selection(task: HERTask, out: Path) -> dict:
    """Appendix: IPW using XRD-matched as a proxy for electrochemical prioritization."""
    X, y = task.X, task.y
    tested = task.frame["matched_int"].to_numpy()
    w = propensity_weights(X, tested)
    mask = tested > 0.5
    if mask.sum() < 5:
        note = {"ok": False, "reason": "too few matched samples"}
        (out / "appendix_selection.json").write_text(json.dumps(note, indent=2))
        return note
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("m", Ridge())])
    pipe.fit(X[mask], y[mask], **{"m__sample_weight": w[mask]})
    pred = pipe.predict(X)
    naive = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("m", Ridge())])
    naive.fit(X[mask], y[mask])
    pred_n = naive.predict(X)
    outj = {
        "ok": True,
        "n_matched": int(mask.sum()),
        "ipw_mae_matched": float(np.mean(np.abs(pred[mask] - y[mask]))),
        "naive_mae_matched": float(np.mean(np.abs(pred_n[mask] - y[mask]))),
        "mean_weight_matched": float(w[mask].mean()),
        "note": "Matched XRD is a coarse proxy for OCx24 electrochemical prioritization, not a known logging policy.",
    }
    (out / "appendix_selection.json").write_text(json.dumps(outj, indent=2))
    return outj


def write_claims(out: Path, exp1, exp2, exp4, exp5, exp6):
    claims = {}
    if len(exp1):
        claims["claim1_loco_r2"] = {str(k): float(v) for k, v in exp1.set_index("model")["r2"].items()}
    if len(exp2):
        g = exp2.groupby("method")[["p_test", "triage_risk", "wrong_nontest"]].mean()
        claims["claim2_decision"] = {m: {c: float(g.loc[m, c]) for c in g.columns} for m in g.index}
    if len(exp4):
        claims["claim3_audc"] = {str(k): float(v) for k, v in exp4.groupby("method")["audc"].mean().items()}
    if len(exp5):
        g = exp5.groupby("method")[["p_test", "triage_risk"]].mean()
        claims["claim4_staged"] = {m: {c: float(g.loc[m, c]) for c in g.columns} for m in g.index}
    if len(exp6):
        g = exp6.groupby(["method", "split"])["p_test"].mean().reset_index()
        claims["ood_p_test"] = g.to_dict(orient="records")
    (out / "claims.json").write_text(json.dumps(claims, indent=2))


RISK_LEVELS = (0.03, 0.05, 0.075, 0.10, 0.125)
TEST_BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)
COST_GRID = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)


def _copy_fitted(fitted, costs: Costs):
    from risktriage.decisions import FittedSplit

    return FittedSplit(
        pred_mean=fitted.pred_mean,
        pred_std=fitted.pred_std,
        y=fitted.y,
        success=fitted.success,
        y_star=fitted.y_star,
        costs=costs,
        residual_scale=fitted.residual_scale,
        X=fitted.X,
        idx=fitted.idx,
        sabatier=fitted.sabatier,
        matched=fitted.matched,
    )


def _dense_frontier_points(fitted, pred_ca, y_ca, scale_ca, seed: int, costs: Costs, dense: bool) -> list[dict]:
    from risktriage.calibration import _lambda_grid, fit_split_conformal

    y_star = fitted.y_star
    scale_te = scaled_sigma(fitted.pred_std, fitted.residual_scale)
    rows = []
    n_lam = 48 if dense else 16
    lam_grid = _lambda_grid(pred_ca, y_ca, scale_ca, n=n_lam)
    for lam in lam_grid:
        iv = residual_intervals(fitted.pred_mean, scale_te, float(lam))
        for name, a in (
            ("crc", interval_action(iv.lower, iv.upper, y_star)),
            ("rac", robust_action(iv.lower, iv.upper, y_star, costs)),
            ("rac_interval_cost", cost_aware_interval_action(iv.lower, iv.upper, y_star, costs, fitted.pred_mean)),
        ):
            ev = evaluate_actions(name, a, fitted)
            ev.update({"seed": seed, "param": float(lam)})
            rows.append(ev)
        p = gaussian_success_prob(fitted.pred_mean, scale_te * max(float(lam), 1e-6), y_star)
        ev = evaluate_actions("risktriage", bayes_action_from_p(p, costs), fitted)
        ev.update({"seed": seed, "param": float(lam)})
        rows.append(ev)
    order = np.argsort(-fitted.pred_std)
    fracs = np.linspace(0.0, 1.0, 21 if dense else 9)
    for frac in fracs:
        k = int(round(float(frac) * len(order)))
        a = np.where(fitted.pred_mean >= y_star, 2, 0)
        if k > 0:
            a[order[:k]] = 1
        ev = evaluate_actions("uncertainty", a, fitted)
        ev.update({"seed": seed, "param": float(frac)})
        rows.append(ev)
    alphas = np.linspace(0.02, 0.50, 17 if dense else 6)
    for alpha in alphas:
        lam = fit_split_conformal(pred_ca, y_ca, scale_ca, alpha=float(alpha))
        iv = residual_intervals(fitted.pred_mean, scale_te, lam)
        ev = evaluate_actions("split_cp", interval_action(iv.lower, iv.upper, y_star), fitted)
        ev.update({"seed": seed, "param": float(alpha), "coverage": coverage_of(iv, fitted.y)})
        rows.append(ev)
    return rows


def experiment_cost_sensitivity(
    task: HERTask,
    out: Path,
    n_seeds: int,
    n_members: int,
    q: float,
    r_star: float = 0.15,
    smoke: bool = False,
) -> pd.DataFrame:
    from risktriage.calibration import crc_lambda, crc_lambda_bayes

    grid = COST_GRID if not smoke else (0.05, 0.2, 0.6)
    rows = []
    for seed in range(n_seeds):
        split = grouped_split(task.groups, seed=seed)
        fitted0, pred_ca, y_ca, scale_ca, *_ = _one_split_fit(task, split, n_members, seed, Costs(), q)
        scale_te = scaled_sigma(fitted0.pred_std, fitted0.residual_scale)
        p_base = gaussian_success_prob(fitted0.pred_mean, scale_te, fitted0.y_star)
        for c_e in grid:
            cc = Costs(c_fp=1.0, c_fn=1.0, c_e=float(c_e))
            fitted = _copy_fitted(fitted0, cc)
            # Proposition 1 on the uncalibrated Gaussian
            ev = evaluate_actions("bayes", bayes_action_from_p(p_base, cc), fitted)
            ev.update({"seed": seed, "c_e": float(c_e), "theta_drop": cc.p_drop, "testing_region": int(cc.c_e < cc.harmonic_cap)})
            rows.append(ev)
            # RAC λ from Bayes CRC, then Bayes action — c_E enters both λ and thresholds
            lam_b = crc_lambda_bayes(pred_ca, y_ca, scale_ca, fitted.y_star, cc, r_star)
            p_rac = gaussian_success_prob(fitted.pred_mean, scale_te * max(lam_b, 1e-6), fitted.y_star)
            ev = evaluate_actions("risktriage", bayes_action_from_p(p_rac, cc), fitted)
            ev.update({"seed": seed, "c_e": float(c_e), "lambda": lam_b, "theta_drop": cc.p_drop, "testing_region": int(cc.c_e < cc.harmonic_cap)})
            rows.append(ev)
            # Interval RAC (max-min): theoretically insensitive until c_E >= min(c_FP, c_FN)
            lam_i = crc_lambda(pred_ca, y_ca, scale_ca, fitted.y_star, cc, r_star, use_robust=True)
            iv = residual_intervals(fitted.pred_mean, scale_te, lam_i)
            ev = evaluate_actions("rac_interval", robust_action(iv.lower, iv.upper, fitted.y_star, cc), fitted)
            ev.update({"seed": seed, "c_e": float(c_e), "lambda": lam_i})
            rows.append(ev)
            ev = evaluate_actions(
                "rac_interval_cost",
                cost_aware_interval_action(iv.lower, iv.upper, fitted.y_star, cc, fitted.pred_mean),
                fitted,
            )
            ev.update({"seed": seed, "c_e": float(c_e), "lambda": lam_i})
            rows.append(ev)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cost_sensitivity.csv", index=False)
    mean = df.groupby(["method", "c_e"], as_index=False)[["p_test", "p_trust", "p_drop", "triage_risk", "full_risk"]].mean()
    mean.to_csv(out / "cost_sensitivity_mean.csv", index=False)
    plot_cost_sensitivity(mean, out / "cost_sensitivity.png")
    return df


def experiment_matched_efficiency(
    task: HERTask,
    out: Path,
    n_seeds: int,
    n_members: int,
    costs: Costs,
    q: float,
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dense = not smoke
    frontier_rows = []
    for seed in range(n_seeds):
        split = grouped_split(task.groups, seed=seed)
        fitted, pred_ca, y_ca, scale_ca, *_ = _one_split_fit(task, split, n_members, seed, costs, q)
        frontier_rows.extend(_dense_frontier_points(fitted, pred_ca, y_ca, scale_ca, seed, costs, dense))
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(out / "exp3_dense_frontier.csv", index=False)

    t_rows, r_rows = [], []
    methods = ["risktriage", "crc", "rac", "split_cp", "uncertainty"]
    for method in methods:
        for seed, g in frontier[frontier.method == method].groupby("seed"):
            pt, rk = g["p_test"].to_numpy(), g["triage_risk"].to_numpy()
            for r in RISK_LEVELS:
                t_rows.append({"method": method, "seed": int(seed), "r": r, "T": min_test_at_risk(pt, rk, r)})
            for t in TEST_BUDGETS:
                r_rows.append({"method": method, "seed": int(seed), "t": t, "R": min_risk_at_test_budget(pt, rk, t)})
    t_seed = pd.DataFrame(t_rows)
    r_seed = pd.DataFrame(r_rows)
    t_seed.to_csv(out / "matched_risk_by_seed.csv", index=False)
    r_seed.to_csv(out / "reverse_frontier_by_seed.csv", index=False)

    t_sum, r_sum = [], []
    for method in methods:
        for r in RISK_LEVELS:
            x = t_seed[(t_seed.method == method) & (t_seed.r == r)]["T"].to_numpy()
            ci = bootstrap_mean_ci(x, n_boot=2000, seed=0)
            t_sum.append({"method": method, "r": r, "T_mean": ci["mean"], "T_ci_lo": ci["ci_lo"], "T_ci_hi": ci["ci_hi"], "n": ci["n"]})
        for t in TEST_BUDGETS:
            x = r_seed[(r_seed.method == method) & (r_seed.t == t)]["R"].to_numpy()
            ci = bootstrap_mean_ci(x, n_boot=2000, seed=0)
            r_sum.append({"method": method, "t": t, "R_mean": ci["mean"], "R_ci_lo": ci["ci_lo"], "R_ci_hi": ci["ci_hi"], "n": ci["n"]})
    t_tab = pd.DataFrame(t_sum)
    r_tab = pd.DataFrame(r_sum)
    t_tab.to_csv(out / "matched_risk_table.csv", index=False)
    r_tab.to_csv(out / "reverse_frontier_table.csv", index=False)
    plot_matched_risk(t_tab, out / "matched_risk.png")

    # headline comparisons at r=0.10 and t=0.10
    def _cell(tab, method, key, val, mean_col, lo, hi):
        row = tab[(tab.method == method) & (tab[key] == val)]
        if not len(row):
            return None
        return {"mean": float(row[mean_col].iloc[0]), "ci_lo": float(row[lo].iloc[0]), "ci_hi": float(row[hi].iloc[0])}

    headline = {
        "T_at_r_0.10": {m: _cell(t_tab, m, "r", 0.10, "T_mean", "T_ci_lo", "T_ci_hi") for m in methods},
        "R_at_t_0.10": {m: _cell(r_tab, m, "t", 0.10, "R_mean", "R_ci_lo", "R_ci_hi") for m in methods},
    }
    (out / "matched_risk_headline.json").write_text(json.dumps(headline, indent=2))
    return t_tab, r_tab


def run_all(
    outdir: Path,
    smoke: bool = False,
    n_seeds: int | None = None,
    costs: Costs | None = None,
) -> None:
    import warnings

    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    costs = costs or Costs()
    n_seeds = n_seeds if n_seeds is not None else (3 if smoke else 20)
    n_members = 3 if smoke else 5
    alpha = 0.1
    r_star = 0.15 * max(costs.c_fp, costs.c_fn)
    q = 0.75
    task = load_her_task()
    (out / "task_meta.json").write_text(json.dumps(task.meta, indent=2))

    exp1 = experiment1_prediction(task, out, smoke=smoke)
    exp2 = experiment2_calibration(task, out, n_seeds, n_members, costs, alpha, r_star, q)
    r_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4]) * max(costs.c_fp, costs.c_fn)
    if smoke:
        r_grid = r_grid[::2]
    alpha_grid = np.array([0.05, 0.1, 0.2, 0.3] if not smoke else [0.1, 0.2])
    experiment3_frontier(task, out, n_seeds, n_members, costs, q, r_grid, alpha_grid)
    budgets = [5, 10, 15, 20, 25, 30, 40] if not smoke else [5, 10, 20]
    exp4 = experiment4_replay(task, out, n_seeds, n_members, costs, alpha, r_star, q, budgets)
    exp5 = experiment5_staged(task, out, n_seeds, n_members, costs, alpha, r_star, q)
    exp6 = experiment6_ood(task, out, n_seeds, n_members, costs, alpha, r_star, q)
    experiment_ablations(exp2, exp4, exp5, out)
    experiment_selection(task, out)
    experiment_cost_sensitivity(task, out, n_seeds, n_members, q, r_star, smoke=smoke)
    experiment_matched_efficiency(task, out, n_seeds, n_members, costs, q, smoke=smoke)
    write_claims(out, exp1, exp2, exp4, exp5, exp6)
    (out / "DONE.txt").write_text("risktriage complete\n")
