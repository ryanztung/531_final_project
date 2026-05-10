# Analysis Pipeline — Fairness Audit of an LLM Resume Scorer

This folder contains the fairness analysis pipeline for the DSCI 531 final project on demographic invariance in multi-stage LLM resume screening. It consumes the per-resume scoring CSV produced by the upstream ATS simulator and produces tables, figures, and narrative outputs evaluating fairness across ground-truth gender and race.

## What's in this folder

```
Analysis Pipeline/
├── Analysis_Pipeline_updated.ipynb   # main notebook — run this end-to-end
├── fairness_extensions.py            # modular fairness functions imported by the notebook
├── similarity_scores_all.csv         # input: 840 model evaluations from the ATS simulator
├── outputs/
│   ├── tables/                       # CSV tables (created on run)
│   └── figures/                      # PNG figures (created on run)
└── README.md
```

## Prerequisites

- Python 3.10 or newer
- Jupyter (for executing the notebook)
- The Python packages listed below

## Setup

From this folder, install the required packages:

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn shap jupyter nbconvert ipykernel
```

If you prefer an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn shap jupyter nbconvert ipykernel
```

## Inputs

The notebook reads `similarity_scores_all.csv` from this folder. The file must contain at least these columns (the upstream ATS simulator produces all of them):

- Identifiers: `resume_id`, `variant_id`, `run_id`, `model_name`, `batch_id`, `jd_role`
- Experimental conditions: `name_condition`, `wording_condition`, `format_condition`, `qualification_tier`
- Ground-truth labels: `gender_true`, `race_true`
- Model perception (diagnostic only): `race_predicted`
- Scores: `overall_score`, `leadership_score`, `experience_score`, `skills_score`
- Outcome: `hire_decision` (text values `hire` / `reject` / `consider`, normalized to 0/1 in the notebook)


## How to run

### Option A — interactive (recommended for inspecting plots)

```bash
jupyter notebook Analysis_Pipeline_updated.ipynb
```

Then choose `Cell → Run All`.

The main analysis uses ground-truth columns (`gender_true`, `race_true`) throughout. The diagnostic at the end is the only place that reads `name_condition` and `race_predicted`, and it is clearly labeled as diagnostic.

## Outputs

### Tables (`outputs/tables/`)


- `selection_rate_gap.csv` — selection rates and male − female gap
- `confidence_intervals.csv` — bootstrap CIs for overall and leadership scores by `gender_true` and `race_true`
- `statistical_tests.csv` — t-test on overall_score by gender_true
- `interaction_summary.csv` — wording × gender_true means
- `race_summary.csv` — per-race counts, selection rate, and mean sub-scores

Extended fairness suite:

- `disparate_impact_gender.csv`, `disparate_impact_race.csv`
- `disparate_impact_gender_predicted.csv`, `disparate_impact_race_predicted.csv` (perception-side, for completeness)
- `bootstrap_fairness_ci.csv`
- `interaction_tests_gender_wording.csv`, `interaction_tests_race_wording.csv`
- `regression_controls_overall.csv`, `regression_controls_selection.csv`
- `paired_variant_tests.csv` (currently empty — see Limitations below)
- `variance_checks_gender.csv`, `variance_checks_race.csv`
- `presentation_bullets.csv`, `report_paragraphs.csv`

Diagnostic (predicted vs ground truth):

- `gender_cue_vs_truth_confusion.csv`
- `race_predicted_vs_truth_confusion.csv`
- `race_prediction_recall.csv`
- `selection_rate_true_vs_predicted_race.csv`



