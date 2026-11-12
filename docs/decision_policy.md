# Probability calibration and decision policy

## Calibration protocol

The base XGBoost predictions are already out-of-fold: each development row is scored by a model that did not train on that row. Calibration is assessed with an additional five-fold cross-fitting layer. In each calibration fold, the mapping is fitted on four groups of OOF scores and applied to the fifth.

Three alternatives are evaluated:

1. raw XGBoost probability;
2. Platt scaling on the log-odds of the XGBoost probability;
3. isotonic regression.

The selection rule is fixed before the final test: minimize cross-fitted Brier score, using log loss only as a tie-breaker. Raw XGBoost wins under that rule and is retained. The decision demonstrates that calibration is a validation question, not a mandatory cosmetic step.

## Threshold interpretation

The binary decision means **send the account for enhanced review**. It does not mean automatically decline, reduce credit, or take an adverse action.

No institutional cost data is available. The project therefore treats all cost ratios as explicit scenarios:

- false positive: a non-defaulting account is unnecessarily reviewed;
- false negative: a defaulting account is not flagged;
- false-positive cost is normalized to 1;
- false-negative costs of 2, 5, and 10 are examined.

For calibrated probability `p`, flagging is preferred when:

```text
p >= false_positive_cost / (false_positive_cost + false_negative_cost)
```

The primary 5x scenario therefore uses a threshold of 1/6, or 0.167. This is an illustrative policy, not an estimate of a bank's operational loss.

## Test-set governance

The calibration choice, cost scenarios, primary 5x policy, and threshold are frozen using development OOF predictions. They will be applied unchanged in the one-time final test evaluation.
