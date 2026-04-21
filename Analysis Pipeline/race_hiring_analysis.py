import pandas as pd
from pathlib import Path
import numpy as np
from scipy import stats

# Load the data
csv_path = Path("/Users/kavinphabiani/Documents/USC/DSCI531/531_final_project/final/Analysis Pipeline/similarity_scores_all .csv")
df = pd.read_csv(csv_path)

# Ensure hire_decision is numeric
df['hire_decision'] = pd.to_numeric(df['hire_decision'], errors='coerce')

# Group by race and compute summary statistics
race_summary = (
    df.groupby("race", as_index=False)
    .agg(
        count=("resume_id", "count"),
        selection_rate=("hire_decision", "mean"),
        mean_overall_score=("overall_score", "mean"),
        mean_leadership_score=("leadership_score", "mean"),
        mean_experience_score=("experience_score", "mean"),
        mean_skills_score=("skills_score", "mean"),
    )
    .round(4)
)

# Save to CSV
output_path = Path("outputs/tables/race_hiring_impact.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
race_summary.to_csv(output_path, index=False)

print(f"Race impact analysis saved to: {output_path}")
print(race_summary)

# Optional: Statistical test for selection rates across races
races = df['race'].unique()
selection_rates = [df[df['race'] == race]['hire_decision'].values for race in races]

# ANOVA if more than 2 groups
if len(races) > 2:
    f_stat, p_val = stats.f_oneway(*selection_rates)
    print(f"\nANOVA for selection rates across races: F={f_stat:.4f}, p={p_val:.4f}")
else:
    # t-test for 2 groups
    t_stat, p_val = stats.ttest_ind(selection_rates[0], selection_rates[1])
    print(f"\nt-test for selection rates between {races[0]} and {races[1]}: t={t_stat:.4f}, p={p_val:.4f}")