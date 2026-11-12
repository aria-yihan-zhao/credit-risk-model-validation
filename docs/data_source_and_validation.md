# Data source and validation boundary

## Source

- Dataset: Default of Credit Card Clients
- Repository: UCI Machine Learning Repository
- DOI: 10.24432/C55S3H
- License: CC BY 4.0
- Original authors: I-Cheng Yeh and Che-hui Lien
- Source archive SHA-256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`

The download script verifies the archive checksum before extracting the workbook. Raw data is intentionally excluded from version control.

## Prediction target

The target is whether a client defaults in the following month. The predictor set contains credit limit, age, encoded demographic variables, six months of repayment status, six bill amounts, and six previous-payment amounts.

## Leakage controls

- `ID` is treated as an identifier, never a predictive feature.
- The target is separated before preprocessing.
- All learned transformations must be fit only on development folds.
- The final test sample is locked before exploratory model development.
- Threshold selection is performed without reference to the final test outcomes.

## Responsible feature policy

`SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are excluded from both predictive models. They are retained only for descriptive subgroup diagnostics. The candidate models use credit limit, repayment-status history, bill amounts, and prior payment amounts.

## Why this is not an out-of-time validation

The monthly columns describe each client's historical payment behavior, but the dataset does not supply an account-level observation timestamp or origination ordering. Therefore, a true temporal holdout cannot be reconstructed. A random stratified split is used transparently, with PSI and drift scenarios added as robustness tests—not presented as temporal evidence.
