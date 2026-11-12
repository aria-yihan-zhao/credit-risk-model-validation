# Model card: credit default risk challenger

## Model overview

- Model: regularized XGBoost binary classifier
- Outcome: default in the following month
- Intended decision: prioritize accounts for enhanced review
- Status: educational portfolio project; not approved for production use
- Calibration decision: retain raw XGBoost probability
- Primary illustrative threshold: 0.167

## Data

The model uses the UCI Default of Credit Card Clients dataset: 30,000 client records with a 22.12% default rate. A reproducible, stratified split assigns 24,000 rows to development and 6,000 rows to a locked final test.

The source data is historical and does not contain a borrower-level observation date suitable for genuine out-of-time validation. Raw data is downloaded from UCI by a checksum-verified script and is not committed to the repository.

## Features

Predictive inputs are limited to:

- credit limit;
- six repayment-status variables;
- six bill-amount variables;
- six prior-payment-amount variables.

`SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are excluded from model training. They are retained only for post-score subgroup diagnostics. Account `ID` is never a predictive feature.

## Development methodology

1. The test sample was locked before analysis.
2. Logistic regression and XGBoost used the same development rows and five stratified folds.
3. Model comparison used out-of-fold predictions only.
4. XGBoost improvements were checked with 500 paired bootstrap repetitions.
5. Raw, Platt-scaled, and isotonic probabilities were compared through a second cross-fitting layer.
6. Raw XGBoost was retained because it had the best cross-fitted Brier score and log loss.
7. The primary 0.167 threshold was fixed under an illustrative 5:1 false-negative to false-positive cost ratio.
8. Feature policy, model, calibration choice, and threshold were committed before the final test was opened.

## Final performance

The single locked 6,000-row test evaluation produced:

| Metric | Result | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.787 | 0.772 to 0.802 |
| PR-AUC | 0.571 | 0.542 to 0.598 |
| Brier score | 0.132 | 0.126 to 0.137 |
| Log loss | 0.422 | 0.408 to 0.436 |

At the frozen threshold:

- recall: 0.738;
- precision: 0.385;
- specificity: 0.666;
- flag rate: 0.424.

The development-to-test score PSI is 0.0029.

## Subgroup audit

Development OOF scores were audited by recorded sex, education, marital status, and age bands. Demographics were not model inputs. The audit reports AUC, Brier score, flag rate, precision, recall, and false-positive rate for groups with at least 100 rows.

The audit is descriptive and does not establish causal fairness, equalized odds, legal compliance, or production acceptability. Small categories—notably education other/unknown and age 60+—have wide sampling uncertainty and should not be overinterpreted.

## Stability analysis

Synthetic development-only stresses test lower payment amounts, worsening repayment status, and their combination. A same-distribution random-half control produces PSI near zero. Repayment-delay stresses create substantial score shifts in the expected direction.

These scenarios are not macroeconomic forecasts or temporal validation.

## Limitations

- The source data is dated and represents a specific historical portfolio.
- The dataset lacks a genuine temporal ordering for out-of-time validation.
- Cost ratios are illustrative; no bank loss ledger is available.
- The project does not estimate causal effects.
- PSI bands are monitoring heuristics, not regulatory rules.
- Subgroup results depend on encoded source categories and sample sizes.
- A production system would require current data, independent validation, policy review, monitoring ownership, security controls, and jurisdiction-specific legal review.

## Prohibited use

This model must not be used to automatically deny credit, reduce limits, price a product, or take an adverse action. The threshold represents enhanced review only.

## Governance record

- Pre-test frozen commit: `c72e35839b3758e24e82bd85812423085a58cd2d`
- Final test rows: 6,000
- Calibration method: raw probability
- Frozen threshold: 1/6
- Post-test retuning: none
- Repeat-test rule: prohibited by the evaluation script while the final result artifact exists
