# Credit Default Risk Modeling and Validation

An end-to-end credit-risk project comparing an interpretable logistic-regression baseline with a gradient-boosting challenger. The emphasis is not only predictive performance, but also leakage prevention, probability calibration, stability monitoring, and defensible model validation.

## Current status

**Milestone 3 — validation and decision policy:** in progress

- [x] Reproducible download from the official UCI repository
- [x] Data-contract and leakage audit
- [x] Locked, stratified development/test split
- [ ] Exploratory analysis and feature policy
- [x] Logistic-regression baseline
- [x] Gradient-boosting challenger
- [x] Calibration and cost-sensitive thresholding
- [x] PSI and drift stress tests
- [ ] Final model card and recruiter-ready summary

## Business question

Can we estimate the probability that a credit-card client will default next month, while producing probabilities that are useful for risk decisions rather than only optimizing classification accuracy?

## Data

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository. It contains 30,000 client records, 23 explanatory variables, and a binary next-month default outcome. The source workbook is downloaded by script and is not committed to GitHub.

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)
- License: CC BY 4.0

## Validation design

The source data includes six months of repayment history but **does not include a borrower observation date suitable for a true out-of-time split**. This project therefore does not label any split as out-of-time.

Instead, it uses:

1. a fixed, stratified development/test split;
2. cross-validation only inside the development sample;
3. a final test set that remains locked until model and threshold choices are complete;
4. calibration diagnostics and Brier score;
5. PSI and synthetic/segment drift stress tests, clearly labeled as stability analyses rather than temporal validation.

## Planned model comparison

| Model | Role | Main evaluation focus |
|---|---|---|
| Logistic regression | Interpretable baseline | coefficients, discrimination, calibration |
| Gradient boosting | Nonlinear challenger | incremental lift, calibration, stability |

Primary metrics will include ROC-AUC, PR-AUC, Brier score, precision, recall, and expected decision cost at a selected threshold.

## Baseline result

The first model is an unweighted, L2-regularized logistic regression. Numerical variables are standardized, documented categorical variables are one-hot encoded, and undocumented education/marital-status codes are consolidated into an explicit other/unknown category rather than silently deleted.

The values below are **out-of-fold predictions from five-fold stratified cross-validation on the 24,000-row development sample**. The 6,000-row final test sample has not been accessed.

| Metric | Development OOF result |
|---|---:|
| ROC-AUC | 0.724 |
| PR-AUC / Average precision | 0.505 |
| Brier score | 0.145 |
| Log loss | 0.465 |
| 10-bin expected calibration error | 0.048 |

![Logistic-regression baseline validation](reports/figures/logistic_baseline_validation.png)

The baseline provides meaningful risk ranking, but the calibration plot shows room for improvement.

## Fair challenger comparison

The XGBoost challenger uses the same 24,000 development rows, target, feature policy, five stratified folds, random seed, and evaluation metrics as the logistic baseline. Numeric scaling is omitted for XGBoost because tree splits are scale invariant; all other data boundaries remain identical. Hyperparameters were specified before the comparison, and neither model accessed the locked final test set.

| Model | ROC-AUC | PR-AUC | Brier score | Log loss | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.724 | 0.505 | 0.145 | 0.465 | 0.048 |
| XGBoost | **0.781** | **0.553** | **0.134** | **0.429** | **0.005** |

![Fair model comparison](reports/figures/model_comparison_validation.png)

A 500-repetition paired bootstrap on the same out-of-fold rows estimated:

- ROC-AUC gain: 0.057 (95% interval: 0.051 to 0.064)
- PR-AUC gain: 0.048 (95% interval: 0.041 to 0.056)
- Brier-score reduction: 0.010 (95% interval: 0.009 to 0.011)

These results support XGBoost as the current challenger winner on development data. They do not yet constitute final generalization evidence; stability tests and one final locked-test evaluation remain.

## Calibration decision and cost-sensitive policy

Calibration was evaluated with a second layer of five-fold cross-fitting on the development OOF scores. This prevents a calibration map from being fitted and assessed on the same observations.

| Calibration method | Brier score | Log loss | ECE (10 bins) |
|---|---:|---:|---:|
| Raw XGBoost | **0.134445** | **0.428913** | 0.004745 |
| Platt scaling | 0.134454 | 0.428933 | 0.005053 |
| Isotonic regression | 0.134680 | 0.433603 | **0.003824** |

The pre-specified selection rule prioritizes Brier score, with log loss as a tie-breaker. The raw XGBoost probabilities are therefore retained: adding a calibrator would marginally worsen overall probability accuracy.

![Calibration and illustrative decision-cost policy](reports/figures/calibration_and_cost_policy.png)

Decision thresholds are framed as **enhanced-review policies**, not automatic credit denials. Because the dataset contains no real bank cost ledger, the analysis reports transparent hypothetical scenarios with false-positive cost normalized to one:

| Assumed false-negative cost | Theoretical threshold | Flag rate | Precision | Recall | Specificity |
|---:|---:|---:|---:|---:|---:|
| 2x false positive | 0.333 | 19.1% | 58.0% | 50.1% | 89.7% |
| 5x false positive | 0.167 | 42.6% | 38.5% | 74.1% | 66.3% |
| 10x false positive | 0.091 | 75.9% | 27.1% | 93.0% | 29.0% |

The 5x scenario is the primary illustration, not a claim about an actual institution's economics. Its threshold follows directly from calibrated probability decision theory: `1 / (1 + 5) = 0.167`. The policy remains frozen before the final test is accessed.

## PSI stability stress tests

Because the dataset has no account-level observation date, this project does not fabricate temporal monitoring data. Instead, the fitted development model is subjected to transparent synthetic stresses, while a random-half control checks the PSI implementation against ordinary sampling variation.

| Scenario | Mean risk: reference | Mean risk: comparison | PSI | Descriptive band |
|---|---:|---:|---:|---|
| Same-distribution random-half control | 0.223 | 0.220 | 0.001 | Low shift |
| Payment amounts reduced by 30% | 0.221 | 0.232 | 0.024 | Low shift |
| Non-negative repayment-delay codes increased by one | 0.221 | 0.309 | 0.577 | High shift |
| Combined delay and payment stress | 0.221 | 0.322 | 0.713 | High shift |

![Development-only stability stress test](reports/figures/stability_stress_test.png)

The 0.10 and 0.25 PSI bands are descriptive monitoring heuristics, not regulatory pass/fail rules. The results show that the model's score distribution is relatively insensitive to a payment-amount reduction alone but highly sensitive to broad deterioration in repayment status. These scenarios are robustness probes, not claims about future portfolio performance.

## Repository structure

```text
credit-risk-model-validation/
├── data/
│   ├── raw/                 # downloaded locally; never committed
│   ├── interim/             # local intermediate data
│   └── processed/           # reproducible split IDs and safe outputs
├── docs/                    # data and validation documentation
├── models/                  # local serialized models
├── notebooks/               # analysis notebooks
├── reports/
│   ├── figures/
│   └── tables/
├── scripts/                 # executable project steps
├── src/credit_risk/         # reusable project code
└── tests/                   # automated data-contract tests
```

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_data_audit.py
pytest
```

## Responsible-use note

This is an educational model-validation project, not a production lending system. Demographic fields require careful policy, fairness, and legal review. They will be used for audit and subgroup diagnostics; their inclusion in any candidate model will be explicitly justified rather than assumed.
