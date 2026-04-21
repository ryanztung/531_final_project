# Fairness Analysis Extensions

This pipeline keeps the original tables/plots and adds modular fairness tests for stronger statistical evidence.

## Added tests and why they matter

- `disparate_impact_table`: Computes selection-rate gaps and impact ratios (4/5ths rule) to detect adverse impact in practical terms.
- `bootstrap_metric_difference` and `bootstrap_impact_ratio`: Adds nonparametric 95% confidence intervals for mean/selection differences and impact ratios to quantify uncertainty.
- `interaction_tests`: Tests demographic x wording interactions to evaluate whether wording changes outcomes differently by group.
- `regression_with_controls`: Estimates demographic effects while controlling for wording and qualification tier, separating potential confounding from group effects.
- `paired_variant_tests`: Uses within-base-resume pairing to isolate demographic cue effects from baseline resume quality differences.
- `variance_checks`: Uses Levene's tests to detect uneven score spread (instability) across groups.

## Interpretation guidance

- Statistical significance and practical significance are different: report both p-values and effect size magnitude.
- Non-significant findings do not prove fairness; they can reflect low power or limited sample size.
- Use matched/paired results as primary evidence when variants share the same base resume.
- Treat impact ratio thresholds (e.g., 0.80) as screening heuristics, not definitive legal conclusions.

## New output files

New tables are written to `outputs/tables/`:

- `disparate_impact_gender.csv`
- `disparate_impact_race.csv`
- `bootstrap_fairness_ci.csv`
- `interaction_tests_gender_wording.csv`
- `interaction_tests_race_wording.csv` (when sample size is adequate)
- `regression_controls_overall.csv`
- `regression_controls_selection.csv`
- `paired_variant_tests.csv`
- `variance_checks_gender.csv`
- `variance_checks_race.csv`
- `presentation_bullets.csv`
- `report_paragraphs.csv`

Existing outputs are preserved:

- `selection_rate_gap.csv`
- `confidence_intervals.csv`
- `statistical_tests.csv`
- `interaction_summary.csv`
- `race_summary.csv`
