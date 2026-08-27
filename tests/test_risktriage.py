import numpy as np
import pytest

from risktriage.features import parse_composition, composition_features
from risktriage.splits import grouped_split
from risktriage.theory import (
    Costs,
    bayes_action_from_p,
    interval_action,
    has_testing_region,
    voi_perfect_experiment,
    DROP,
    TEST,
    TRUST,
)
from risktriage.calibration import fit_split_conformal, residual_intervals, crc_lambda


def test_parse_ocx_composition():
    d = parse_composition("Ag-0.417-Au-0.167-Zn-0.417")
    assert set(d) == {"Ag", "Au", "Zn"}
    assert abs(sum(d.values()) - 1.0) < 1e-8
    v = composition_features("Ag-0.5-Au-0.5")
    assert np.isfinite(v).all()


def test_proposition1_thresholds():
    c = Costs(c_fp=1.0, c_fn=1.0, c_e=0.2)
    assert has_testing_region(c)
    p = np.array([0.05, 0.5, 0.95])
    a = bayes_action_from_p(p, c)
    assert a[0] == DROP
    assert a[1] == TEST
    assert a[2] == TRUST


def test_proposition2_variance_vs_voi():
    # x1: large variance, never succeeds
    y1 = np.array([-100.0] * 99 + [-1.0])
    p1 = 0.0
    # x2: straddles threshold
    p2 = 0.5
    c = Costs(c_fp=1, c_fn=1, c_e=0.2)
    assert np.var(y1) > 1.0
    assert voi_perfect_experiment(np.array([p1]), c)[0] == 0.0
    assert voi_perfect_experiment(np.array([p2]), c)[0] == pytest.approx(0.5)


def test_theorem1_wrong_nontest_subset_of_uncovered():
    y_star = 0.0
    y = np.array([-1.0, 1.0, 0.5, -0.2])
    lo = np.array([0.1, 0.2, -2.0, -0.5])
    hi = np.array([0.2, 2.0, 2.0, -0.1])
    a = interval_action(lo, hi, y_star)
    uncovered = (y < lo) | (y > hi)
    wrong = ((a == TRUST) & (y < y_star)) | ((a == DROP) & (y >= y_star))
    assert np.all(wrong <= uncovered)


def test_grouped_split_no_group_leak():
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4)
    sp = grouped_split(groups, seed=0, frac_train=0.5, frac_cal=0.25)
    gtr, gca, gte = set(groups[sp.train]), set(groups[sp.cal]), set(groups[sp.test])
    assert gtr.isdisjoint(gca) and gtr.isdisjoint(gte) and gca.isdisjoint(gte)


def test_crc_larger_lambda_fewer_errors():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=80)
    y = pred + rng.normal(scale=0.4, size=80)
    scale = np.ones(80)
    c = Costs()
    y_star = 0.0
    lam_lo = crc_lambda(pred, y, scale, y_star, c, r_star=0.5)
    lam_hi = crc_lambda(pred, y, scale, y_star, c, r_star=0.05)
    assert lam_hi >= lam_lo


def test_split_conformal_quantile_finite_sample():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    from risktriage.calibration import split_conformal_quantile

    q = split_conformal_quantile(scores, 0.2)
    assert q >= 0.4


def test_bayes_test_rate_falls_with_c_e():
    p = np.linspace(0.02, 0.98, 50)
    rates = []
    for c_e in (0.05, 0.2, 0.45, 0.6):
        a = bayes_action_from_p(p, Costs(c_e=c_e))
        rates.append(float((a == TEST).mean()))
    assert rates[0] > rates[1] > rates[2]
    assert rates[3] == 0.0


def test_min_test_at_risk_and_reverse():
    from risktriage.stats import min_risk_at_test_budget, min_test_at_risk

    p_test = np.array([0.4, 0.2, 0.05, 0.0])
    risk = np.array([0.02, 0.06, 0.11, 0.15])
    assert min_test_at_risk(p_test, risk, 0.10) == pytest.approx(0.2)
    assert min_risk_at_test_budget(p_test, risk, 0.10) == pytest.approx(0.11)


def test_load_her_if_present():
    from risktriage.data import DATA_DIR, load_her_task

    if not (DATA_DIR / "processed_data" / "HER_40_70_all.csv").exists():
        pytest.skip("OCx24 data not downloaded")
    task = load_her_task()
    assert task.n >= 100
    assert "voltage_she" in task.frame.columns
    assert task.X.shape[0] == task.n
    assert np.isfinite(task.y).all()
