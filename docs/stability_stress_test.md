# Stability stress-test design

## Boundary

The UCI dataset contains historical repayment variables but no account-level observation timestamp. It cannot support a genuine out-of-time monitoring exercise. All scenarios below use development rows only and are explicitly labeled synthetic.

## PSI implementation

Population Stability Index bins are defined from reference-score deciles. Those same bin edges are then applied to the comparison scores. A small epsilon prevents division by zero. Bin-level contributions are saved so the aggregate PSI can be audited.

The familiar 0.10 and 0.25 bands are used only as descriptive heuristics:

- below 0.10: low shift;
- 0.10 to below 0.25: moderate shift;
- 0.25 or above: high shift.

They are not regulatory requirements and would need institution-specific governance in production.

## Scenarios

### Same-distribution control

The development population is randomly divided into equal halves. A PSI near zero confirms that ordinary sampling variation does not generate a false drift alarm.

### Payment-amount stress

All six prior payment-amount variables are multiplied by 0.70. This scenario asks whether lower payments alone materially shift predicted risk.

### Repayment-delay stress

For non-negative repayment-status codes, one unit is added and capped at eight. Negative special-status codes are left unchanged because incrementing them would not have a clean ordinal interpretation.

### Combined stress

The payment-amount and repayment-delay stresses are applied together.

## Interpretation limit

The stresses do not estimate macroeconomic causality or future default rates. They test whether the fitted score responds in an economically plausible direction and provide a reproducible example of model-monitoring logic.
