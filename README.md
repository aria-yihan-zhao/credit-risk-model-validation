# Credit Default Risk Modeling and Validation

An end-to-end credit-risk project comparing an interpretable logistic-regression baseline with a gradient-boosting challenger. The emphasis is not only predictive performance, but also leakage prevention, probability calibration, stability monitoring, and defensible model validation.

## Current status

**Milestone 1 — data foundation:** in progress

- [x] Reproducible download from the official UCI repository
- [x] Data-contract and leakage audit
- [x] Locked, stratified development/test split
- [ ] Exploratory analysis and feature policy
- [ ] Logistic-regression baseline
- [ ] Gradient-boosting challenger
- [ ] Calibration and cost-sensitive thresholding
- [ ] PSI and drift stress tests
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
