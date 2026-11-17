# Verification

- 12 standard-library unittest checks cover deterministic generation, forbidden columns, label maturity, chronological separation, training-only imputation, unseen categories, missing numeric values, cost threshold selection, exact explanation reconstruction, metric accounting, holdout-label isolation, single-class rejection, and input ranges. Some checks share a test method.
- The end-to-end CLI generates its report, prediction CSV, and two chart formats.
- Results in `examples/demo-report.json` and `examples/evaluation.svg` come from the 12,000-row seed-42 synthetic demo.
- GitHub Actions repeats the tests and demo; its current status is visible in the Actions tab.
- No real commercial data, intervention outcomes, or deployment performance was evaluated.
