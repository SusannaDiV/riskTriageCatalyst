# RiskTriage: Risk-Averse Calibration for Simulation-to-Experiment Catalyst Decisions

**4-page workshop paper.** Compile `paper/risktriage.tex`. Numbers from `results/risktriage/` (20 grouped seeds).

---

**Abstract.** Computational screening produces uncertain catalyst recommendations, but physical validation is costly. We formulate translation as a risk-sensitive sequential decision: whether to synthesize, characterize, electrochemically test, or reject a computational candidate. On Open Catalyst Experiments 2024 (OCx24) HER, a LightGBM predictor of cell voltage at \(50\,\mathrm{mA\,cm^{-2}}\) attains leave-one-composition-out \(R^2=0.61\): useful signal with enough residual uncertainty that triage matters. Calibrating nested prediction sets to a catalyst-specific decision loss, rather than to generic coverage, yields \(T(r)=\min P(\mathrm{TEST})\) subject to decision risk \(R\le r\). At \(R\le 0.10\), RiskTriage requires **8.4%** [4.2, 13.4] physical tests versus **16.5%** [9.2, 24.4] for uncertainty sampling and **25.9%** [22.3, 29.6] for split conformal (bootstrap 95% CI). Under composition shift the same policy becomes more conservative (\(7\%\to 21\%\) tests). Interval max-min rules are insensitive to experimental cost \(c_E\) until the Bayes testing region empties; cost-aware Bayes thresholds shut off testing at the theoretical cap \(c_E\ge c_{\mathrm{FP}}c_{\mathrm{FN}}/(c_{\mathrm{FP}}+c_{\mathrm{FN}})\).

---

## 1. Introduction

The translational bottleneck in computational catalysis is not a missing predictor. It is a *decision*: given an uncertain computational score, should a laboratory spend synthesis, X-ray characterization, and electrolysis on this candidate? Ordinary conformal prediction answers what interval covers \(Y\) with probability \(1-\alpha\), and can force almost every candidate into the experimental queue. Risk-averse calibration (Kiyani et al., ICML 2025) and conformal risk control (Angelopoulos et al., ICLR 2024) attach uncertainty sets to a downstream loss.

We study hydrogen evolution on OCx24 (Abed et al., 2024), whose experimental campaign was a funnel—targeted synthesis, XRF/XRD filtering, prioritized electrolysis—not a uniform reveal of labels.

**Claim.** Decision-calibrated uncertainty beats prediction-calibrated uncertainty for experimentally relevant outcomes: at a fixed tolerance for wrong TRUST/DROP actions, fewer physical tests recover the same scientific risk budget.

We do not claim a new conformal theorem, a GNN, or that RiskTriage dominates ranking for discovery.

## 2. A three-action catalyst problem

Let \(Y\) be experimental HER voltage vs SHE at \(50\,\mathrm{mA\,cm^{-2}}\) (higher / less negative is better). Success is \(G=\mathbf{1}\{Y\ge y^\star\}\) with \(y^\star\) the training 75th percentile (\(\approx -1.27\,\mathrm{V}\); 25% positives). Actions \(a\in\{\mathrm{DROP},\mathrm{TEST},\mathrm{TRUST}\}\) incur

\[
L(\mathrm{TRUST},G)=c_{\mathrm{FP}}(1-G),\quad
L(\mathrm{DROP},G)=c_{\mathrm{FN}}G,\quad
L(\mathrm{TEST},G)=c_E.
\]

Triage risk excludes \(c_E\): TEST pays money, not a scientific misclassification. The operational program is

\[
\min_\lambda\; \mathbb{E}[\mathbf{1}\{a_\lambda=\mathrm{TEST}\}]
\quad\text{s.t.}\quad
R_{\mathrm{triage}}(\lambda)\le r^\star.
\]

**Proposition 1 (Bayes testing region).** If \(c_E < c_{\mathrm{FP}}c_{\mathrm{FN}}/(c_{\mathrm{FP}}+c_{\mathrm{FN}})\), then DROP for \(p\le c_E/c_{\mathrm{FN}}\), TRUST for \(p\ge 1-c_E/c_{\mathrm{FP}}\), and TEST in between. Otherwise TEST is never uniquely optimal.

Value of a perfect experiment is \(\mathrm{VOI}=\min\{pc_{\mathrm{FN}},(1-p)c_{\mathrm{FP}}\}\); TEST iff \(\mathrm{VOI}>c_E\).

**Proposition 2.** There exist \(x_1,x_2\) with \(\mathrm{Var}(Y\mid x_1)>\mathrm{Var}(Y\mid x_2)\) but \(\mathrm{VOI}(x_1)<\mathrm{VOI}(x_2)\). (A law on \(\{-100,-1\}\) can have arbitrary variance and \(\mathrm{VOI}=0\).)

Given \(C(x)=[L(x),U(x)]\): TRUST if \(L\ge y^\star\), DROP if \(U<y^\star\), else TEST.

**Theorem 1.** If \(P(Y\in C(X))\ge 1-\alpha\), then \(P(\text{wrong TRUST or DROP})\le\alpha\). A catalyst-specific corollary of conformal validity, not a new coverage theorem.

For default costs \((c_{\mathrm{FP}},c_{\mathrm{FN}},c_E)=(1,1,0.2)\), the RAC max-min action on an interval coincides with Theorem 1, so RAC and CRC select the same nested residual set. Cost \(c_E\) enters Bayes thresholds, not this max-min rule, until the testing region is empty.

With XRF/XRD features \(Z\), \(\mathbb{E}[V(X,Z)\mid X]\ge V(X)\) (tower property). The empirical question is how much electrochemistry \(Z\) avoids.

## 3. OCx24 HER protocol

Public OCx24 joined HER table: **179** targets at \(50\,\mathrm{mA\,cm^{-2}}\), 166 SIDs, 45 alloy families, 43 XRD-matched. Features: mean/Wulff/Boltzmann adsorption energies for H, OH, CO, C, CHO, COCOH plus composition descriptors. No GNN, no new DFT. LightGBM ensemble (\(K=5\)) supplies scale for \(C_\lambda(x)=[\hat y\pm\lambda\sigma(x)]\).

SID-grouped 65/17.5/17.5 splits; OOD holds out composition clusters; **20 seeds**. \(y^\star\) from train only.

## 4. Results

### Claim 1 — Predictions are useful and insufficient

LOCO \(R^2\): linear 0.52, ridge 0.56, RF 0.60, LightGBM **0.61** (MAE 0.079 V, Spearman 0.79). Matches OCx24’s HER signal (\(\approx 0.59\)–\(0.61\)). Residual error is why triage matters.

### Claim 2 — Coverage-optimal sets are experiment-expensive

Split conformal (\(\alpha=0.1\)): coverage 0.94, wrong-non-test 0.3%, but **75%** TEST. CRC/RAC at \(r^\star=0.15\): **7.4%** TEST, triage risk 0.114 (point predictor: 0% tests, risk 0.140).

### Claim 3 — Matched-risk efficiency (main table)

**Table 1.** \(T(r)=\min P(\mathrm{TEST})\) s.t. \(R\le r\), percent, mean [bootstrap 95% CI].

| \(r\) | RAC / CRC | Uncertainty | Split CP |
|------:|----------:|------------:|---------:|
| 0.05 | **25.2** [18.3, 32.6] | 47.0 [35.3, 58.0] | 32.0 [26.8, 37.4] |
| 0.075 | **14.7** [9.3, 20.5] | 26.3 [16.4, 36.5] | 26.3 [22.7, 29.8] |
| **0.10** | **8.4** [4.2, 13.4] | 16.5 [9.2, 24.4] | 25.9 [22.3, 29.6] |
| 0.125 | **6.9** [3.3, 11.2] | 15.2 [8.1, 22.9] | 25.1 [21.1, 28.9] |

At \(R\le 0.10\), RAC needs about half the tests of uncertainty sampling and about one third of split conformal.

Reverse: at a **10% testing budget**, RAC risk is **0.096** [0.074, 0.122] vs **0.113** [0.086, 0.142] for uncertainty. Split CP rarely attains \(P(\mathrm{TEST})\le 10\%\) (2/20 seeds); do not quote that cell.

Discovery AUDC (top-decile recall vs budget): RiskTriage 0.929, ML rank 0.927, UCB 0.925, random 0.55, Sabatier 0.27. Bootstrap \(\Delta\) vs rank: 0.002 [0.000, 0.006], \(P(\Delta>0)=0.63\) — **not a discovery win**. The gain is safe abstention.

### Claim 4 — Characterization is not free

XRF/XRD in the predictor: TEST \(7.4\%\to 5.1\%\), risk \(0.114\to 0.125\). Hard-drop unmatched before electrolysis: TEST \(1.1\%\), risk \(0.146\), ~2% good catalysts lost.

OOD: RAC TEST \(7.4\%\to 21\%\) (CP stays \(75\%\to 81\%\)).

**Cost.** Bayes \(P(\mathrm{TEST})\) falls \(5.9\%\to 0\%\) as \(c_E\) goes \(0.01\to 0.8\), and is exactly 0 for \(c_E\ge 1/2\) (Prop. 1). Interval max-min RAC stays at 7.4% for all \(c_E\le 0.8<1\). Treat interval RAC as risk-calibrated, not cost-calibrated, unless Bayes thresholds or the cap rule are used.

## 5. Limitations

\(n=179\); CIs on \(T(0.10)\) are wide. Labels are not a logging policy over 19k computational candidates. RAC \(\equiv\) CRC for this three-action utility. CO₂RR left to future work. OCx24 replicate noise \(\sim 0.043\,\mathrm{V}\); MAE \(0.079\,\mathrm{V}\) is above that floor.

## 6. Conclusion

Uncertainty for catalyst translation should be calibrated to the physical decision it controls. On OCx24 HER, nested-set risk control meets a scientific error tolerance with substantially fewer electrochemical tests than coverage conformalization or variance sampling, becomes more conservative under composition shift, and does not automatically improve top-decile discovery over predicted-performance ranking.

## References

1. Abed et al. Open Catalyst Experiments 2024 (OCx24). arXiv:2411.11783, 2024.
2. Angelopoulos, Bates, Fisch, Lei, Zrnic. Conformal risk control. ICLR, 2024.
3. Kiyani, Pappas, Roth, Hassani. Decision theoretic foundations for conformal prediction. ICML, 2025.
