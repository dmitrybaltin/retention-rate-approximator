# AGENTS.ML

Rules for ML and analytics code in this repository.

## Typing
- Every new or changed ML-facing function must be fully type annotated.
- Do not add untyped parameters, untyped returns, or broad `Any` unless there is a hard external library boundary.
- Prefer precise unions and literals over loose types.
- Keep tensor/dataframe helper interfaces explicit about accepted shapes or required columns in docstrings or naming.

## Validation
- After changing modeling, training, confidence intervals, or plotting derived from model outputs, run:
  - static checks / type checks when applicable
  - unit tests
  - at least one sanity check on real-looking output values
- Sanity checks must verify that predicted retention stays in a plausible numeric range.
- If a change affects decomposition logic, also check that the result is still interpretable, not only that tests pass.

## Product constraints
- This project is not only about minimizing loss. The decomposition must remain interpretable:
  - main trend
  - patch jumps
  - weekly seasonality
- Avoid fixes that improve fit while making patch effects less meaningful.

## Confidence intervals
- Confidence bands are drawn around `predicted`, not `predicted_trend`, unless explicitly changed.
- Segment-level interval width must be allowed to differ between patch segments.
- New confidence logic must not silently alter the fitted mean curve itself.

## Change discipline
- Separate ML/model changes from UI changes into different commits whenever practical.
- When a feature depends on ML output, confirm first that the underlying fitted values are still sane before debugging the visualization.
