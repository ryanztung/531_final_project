from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm


@dataclass
class ColumnMap:
    base_id: str
    variant_id: str
    gender: str
    race: str
    wording: str
    qualification: str
    selected: str
    overall: str
    leadership: str
    experience: str
    skills: str


def _first_existing(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def infer_column_map(df: pd.DataFrame) -> ColumnMap:
    return ColumnMap(
        base_id=_first_existing(df, ["resume_id", "base_resume_id", "original_resume_id", "base_id"]),
        variant_id=_first_existing(df, ["variant_id", "resume_variant_id"], required=False) or "variant_id",
        gender=_first_existing(df, ["name_condition", "gender_condition", "gender", "sex"]),
        race=_first_existing(df, ["race", "race_condition", "ethnicity"]),
        wording=_first_existing(df, ["wording_condition", "wording", "prompt_wording"]),
        qualification=_first_existing(df, ["qualification_tier", "qualification_level", "tier"]),
        selected=_first_existing(df, ["hire_decision", "selected", "callback", "outcome"]),
        overall=_first_existing(df, ["overall_score", "overall", "total_score"]),
        leadership=_first_existing(df, ["leadership_score", "leadership"]),
        experience=_first_existing(df, ["experience_score", "experience"]),
        skills=_first_existing(df, ["skills_score", "skills"]),
    )


def _safe_rate(s: pd.Series) -> float:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else np.nan


def disparate_impact_table(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    reference_group: Optional[str] = None,
    min_n: int = 5,
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, as_index=False)
        .agg(n=(outcome_col, "count"), selection_rate=(outcome_col, _safe_rate))
        .sort_values("selection_rate", ascending=False)
    )
    grouped["selection_rate"] = grouped["selection_rate"].astype(float)
    if reference_group is None:
        reference_group = grouped.iloc[0][group_col]
    ref_rate = grouped.loc[grouped[group_col] == reference_group, "selection_rate"].iloc[0]

    grouped["reference_group"] = reference_group
    grouped["absolute_gap_vs_reference"] = grouped["selection_rate"] - ref_rate
    grouped["impact_ratio_vs_reference"] = grouped["selection_rate"] / ref_rate if ref_rate else np.nan
    grouped["passes_4_5_rule"] = grouped["impact_ratio_vs_reference"] >= 0.8
    grouped["small_sample_warning"] = grouped["n"] < min_n
    return grouped.round(4)


def bootstrap_metric_difference(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    metric_col: str,
    n_boot: int = 3000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    a = pd.to_numeric(df.loc[df[group_col] == group_a, metric_col], errors="coerce").dropna().values
    b = pd.to_numeric(df.loc[df[group_col] == group_b, metric_col], errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return {"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_a": len(a), "n_b": len(b)}

    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True).mean()
        sb = rng.choice(b, size=len(b), replace=True).mean()
        diffs.append(sa - sb)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(a.mean() - b.mean()), "ci_lower": float(lo), "ci_upper": float(hi), "n_a": len(a), "n_b": len(b)}


def bootstrap_impact_ratio(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    protected_group: str,
    reference_group: str,
    n_boot: int = 3000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    p = pd.to_numeric(df.loc[df[group_col] == protected_group, outcome_col], errors="coerce").dropna().values
    r = pd.to_numeric(df.loc[df[group_col] == reference_group, outcome_col], errors="coerce").dropna().values
    if len(p) < 2 or len(r) < 2:
        return {"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_protected": len(p), "n_reference": len(r)}

    ratios = []
    for _ in range(n_boot):
        ps = rng.choice(p, size=len(p), replace=True).mean()
        rs = rng.choice(r, size=len(r), replace=True).mean()
        ratios.append(ps / rs if rs > 0 else np.nan)
    ratios = np.array(ratios, dtype=float)
    ratios = ratios[~np.isnan(ratios)]
    if len(ratios) == 0:
        return {"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_protected": len(p), "n_reference": len(r)}
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": float(p.mean() / r.mean()) if r.mean() > 0 else np.nan,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_protected": len(p),
        "n_reference": len(r),
    }


def interaction_tests(
    df: pd.DataFrame,
    dem_col: str,
    wording_col: str,
    score_col: str,
    outcome_col: str,
    max_rows_for_logit: int = 50000,
) -> pd.DataFrame:
    out: List[Dict[str, object]] = []

    ols_model = smf.ols(f"{score_col} ~ C({dem_col}) * C({wording_col})", data=df).fit()
    anova = anova_lm(ols_model, typ=2).reset_index().rename(columns={"index": "term"})
    for _, row in anova.iterrows():
        out.append(
            {
                "model_type": "ols_anova",
                "dependent_variable": score_col,
                "term": row["term"],
                "statistic": row.get("F", np.nan),
                "p_value": row.get("PR(>F)", np.nan),
                "effect_direction": "n/a",
            }
        )

    logit_df = df
    if len(df) > max_rows_for_logit:
        logit_df = df.sample(n=max_rows_for_logit, random_state=42)
    try:
        logit_model = smf.logit(f"{outcome_col} ~ C({dem_col}) * C({wording_col})", data=logit_df).fit(disp=False, maxiter=100, method="lbfgs")
        for term, coef in logit_model.params.items():
            if ":" in term:
                out.append(
                    {
                        "model_type": "logit",
                        "dependent_variable": outcome_col,
                        "term": term,
                        "statistic": coef,
                        "p_value": logit_model.pvalues.get(term, np.nan),
                        "effect_direction": "positive" if coef > 0 else "negative",
                    }
                )
    except Exception as exc:
        out.append(
            {
                "model_type": "logit",
                "dependent_variable": outcome_col,
                "term": "model_error",
                "statistic": np.nan,
                "p_value": np.nan,
                "effect_direction": f"warning: {exc}",
            }
        )
    return pd.DataFrame(out).round(6)


def regression_with_controls(
    df: pd.DataFrame,
    outcome_col: str,
    demographic_cols: List[str],
    control_cols: List[str],
    logistic: bool = False,
    max_rows_for_logit: int = 50000,
) -> pd.DataFrame:
    terms = [f"C({c})" for c in demographic_cols + control_cols]
    formula = f"{outcome_col} ~ " + " + ".join(terms)
    if logistic:
        fit_df = df.sample(n=max_rows_for_logit, random_state=42) if len(df) > max_rows_for_logit else df
        model = smf.logit(formula, data=fit_df).fit(disp=False, maxiter=100, method="lbfgs")
    else:
        model = smf.ols(formula, data=df).fit()
    result = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "p_value": model.pvalues.values,
            "conf_low": model.conf_int()[0].values,
            "conf_high": model.conf_int()[1].values,
            "model_type": "logit" if logistic else "ols",
            "dependent_variable": outcome_col,
        }
    )
    return result.round(6)


def paired_variant_tests(
    df: pd.DataFrame,
    base_id_col: str,
    condition_col: str,
    metric_col: str,
    first_group: str,
    second_group: str,
) -> Dict[str, float]:
    a = (
        df[df[condition_col] == first_group]
        .groupby(base_id_col, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: f"{metric_col}_{first_group}"})
    )
    b = (
        df[df[condition_col] == second_group]
        .groupby(base_id_col, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: f"{metric_col}_{second_group}"})
    )
    paired = a.merge(b, on=base_id_col, how="inner").dropna()
    if len(paired) < 3:
        return {
            "n_pairs": len(paired),
            "mean_diff_first_minus_second": np.nan,
            "paired_t_stat": np.nan,
            "paired_t_p": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan,
        }

    x = paired[f"{metric_col}_{first_group}"]
    y = paired[f"{metric_col}_{second_group}"]
    t_stat, t_p = stats.ttest_rel(x, y, nan_policy="omit")
    try:
        w_stat, w_p = stats.wilcoxon(x, y)
    except Exception:
        w_stat, w_p = np.nan, np.nan
    return {
        "n_pairs": int(len(paired)),
        "mean_diff_first_minus_second": float((x - y).mean()),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if pd.notna(w_stat) else np.nan,
        "wilcoxon_p": float(w_p) if pd.notna(w_p) else np.nan,
    }


def variance_checks(df: pd.DataFrame, group_col: str, score_cols: List[str], min_n: int = 3) -> pd.DataFrame:
    rows = []
    for score_col in score_cols:
        samples = []
        for _, g in df.groupby(group_col):
            vals = pd.to_numeric(g[score_col], errors="coerce").dropna()
            if len(vals) >= min_n:
                samples.append(vals.values)
        if len(samples) < 2:
            rows.append({"group_col": group_col, "score_col": score_col, "levene_stat": np.nan, "p_value": np.nan, "warning": "insufficient samples"})
            continue
        stat, p = stats.levene(*samples, center="median")
        rows.append({"group_col": group_col, "score_col": score_col, "levene_stat": stat, "p_value": p, "warning": ""})
    return pd.DataFrame(rows).round(6)


def presentation_bullets(disparate_df: pd.DataFrame, interaction_df: pd.DataFrame, paired_df: pd.DataFrame) -> List[str]:
    bullets: List[str] = []
    for _, row in disparate_df.iterrows():
        bullets.append(
            f"{row['group']} selection rate={row['selection_rate']:.3f}, impact ratio={row['impact_ratio']:.3f} vs {row['reference_group']} ({'passes' if row['passes_4_5_rule'] else 'fails'} 4/5ths rule)."
        )
    sig_interactions = interaction_df[(interaction_df["term"].astype(str).str.contains(":")) & (interaction_df["p_value"] < 0.05)]
    if len(sig_interactions):
        bullets.append("At least one demographic x wording interaction is statistically significant; wording likely shifts outcomes differently by group.")
    else:
        bullets.append("No statistically significant demographic x wording interaction detected at alpha=0.05; practical differences may still exist.")
    if not paired_df.empty:
        row = paired_df.iloc[0]
        n_pairs = int(row.get("n_pairs", 0))
        mean_diff = row.get("mean_diff_first_minus_second", np.nan)
        t_p = row.get("paired_t_p", np.nan)
        w_p = row.get("wilcoxon_p", np.nan)
        if pd.isna(mean_diff) or n_pairs < 3:
            bullets.append(
                f"Matched within-resume test had only {n_pairs} pairs; insufficient paired sample size for robust inference."
            )
        else:
            bullets.append(
                f"Matched within-resume test ({n_pairs} pairs) mean diff={mean_diff:.3f}, paired t p={t_p:.3f}, Wilcoxon p={w_p:.3f}."
            )
    return bullets


def report_paragraphs(
    bootstrap_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    variance_df: pd.DataFrame,
) -> List[str]:
    sections = []
    sections.append(
        "Bootstrap confidence intervals quantify uncertainty around group differences and impact ratios. Intervals that include zero (for differences) or one (for ratios) indicate limited evidence of robust disparity under this sample."
    )
    dem = reg_df[reg_df["term"].str.contains("name_condition|race", case=False, regex=True, na=False)].copy()
    if len(dem):
        sig = (dem["p_value"] < 0.05).sum()
        sections.append(
            f"Regression models with controls (wording and qualification tier) estimate whether demographic coefficients persist after adjustment. Here, {sig} demographic terms are significant at alpha=0.05; effect magnitude should be read alongside confidence intervals."
        )
    var_sig = (variance_df["p_value"] < 0.05).sum() if len(variance_df) else 0
    sections.append(
        f"Variance checks using Levene's test flag {int(var_sig)} score distributions with unequal spread across groups. Unequal variance can indicate instability even when mean gaps are small."
    )
    sections.append(
        "Interpretation is conservative: statistical non-significance does not prove fairness, and practical significance, sample size, and power constraints remain central to conclusions."
    )
    return sections


def run_extended_fairness_suite(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
    min_group_n: int = 5,
    min_pair_n: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full fairness extension suite and return all result tables.
    This orchestrates disparate impact, bootstrap CIs, interaction tests,
    controlled regressions, paired tests, variance checks, and narratives.
    """
    colmap = infer_column_map(df)
    results: Dict[str, pd.DataFrame] = {}

    # 1) Disparate impact tables.
    gender_ref = "male_coded" if "male_coded" in set(df[colmap.gender].dropna().unique()) else None
    impact_gender_df = disparate_impact_table(
        df,
        group_col=colmap.gender,
        outcome_col=colmap.selected,
        reference_group=gender_ref,
        min_n=min_group_n,
    )
    impact_gender_df["group"] = impact_gender_df[colmap.gender]
    impact_gender_df = impact_gender_df.rename(
        columns={
            "absolute_gap_vs_reference": "absolute_gap",
            "impact_ratio_vs_reference": "impact_ratio",
        }
    )
    results["disparate_impact_gender"] = impact_gender_df

    impact_race_df = disparate_impact_table(
        df,
        group_col=colmap.race,
        outcome_col=colmap.selected,
        reference_group=None,
        min_n=min_group_n,
    )
    impact_race_df["group"] = impact_race_df[colmap.race]
    impact_race_df = impact_race_df.rename(
        columns={
            "absolute_gap_vs_reference": "absolute_gap",
            "impact_ratio_vs_reference": "impact_ratio",
        }
    )
    results["disparate_impact_race"] = impact_race_df

    # 2) Bootstrap confidence intervals.
    boot_rows: List[Dict[str, object]] = []
    if {"male_coded", "female_coded"}.issubset(set(df[colmap.gender].dropna().unique())):
        score_boot = bootstrap_metric_difference(
            df,
            group_col=colmap.gender,
            group_a="female_coded",
            group_b="male_coded",
            metric_col=colmap.overall,
            n_boot=n_boot,
            seed=seed,
        )
        boot_rows.append(
            {
                "comparison": "female_minus_male",
                "metric": colmap.overall,
                "estimate": score_boot["estimate"],
                "ci_lower": score_boot["ci_lower"],
                "ci_upper": score_boot["ci_upper"],
                "n_a": score_boot["n_a"],
                "n_b": score_boot["n_b"],
            }
        )
        sr_boot = bootstrap_metric_difference(
            df,
            group_col=colmap.gender,
            group_a="female_coded",
            group_b="male_coded",
            metric_col=colmap.selected,
            n_boot=n_boot,
            seed=seed,
        )
        boot_rows.append(
            {
                "comparison": "female_minus_male",
                "metric": "selection_rate",
                "estimate": sr_boot["estimate"],
                "ci_lower": sr_boot["ci_lower"],
                "ci_upper": sr_boot["ci_upper"],
                "n_a": sr_boot["n_a"],
                "n_b": sr_boot["n_b"],
            }
        )
        ratio_boot = bootstrap_impact_ratio(
            df,
            group_col=colmap.gender,
            outcome_col=colmap.selected,
            protected_group="female_coded",
            reference_group="male_coded",
            n_boot=n_boot,
            seed=seed,
        )
        boot_rows.append(
            {
                "comparison": "female_vs_male",
                "metric": "impact_ratio",
                "estimate": ratio_boot["estimate"],
                "ci_lower": ratio_boot["ci_lower"],
                "ci_upper": ratio_boot["ci_upper"],
                "n_a": ratio_boot["n_protected"],
                "n_b": ratio_boot["n_reference"],
            }
        )
    results["bootstrap_fairness_ci"] = pd.DataFrame(boot_rows).round(6)

    # 3) Interaction effects.
    results["interaction_tests_gender_wording"] = interaction_tests(
        df,
        dem_col=colmap.gender,
        wording_col=colmap.wording,
        score_col=colmap.overall,
        outcome_col=colmap.selected,
    )
    if df[colmap.race].nunique() >= 2 and df.groupby(colmap.race).size().min() >= min_group_n:
        results["interaction_tests_race_wording"] = interaction_tests(
            df,
            dem_col=colmap.race,
            wording_col=colmap.wording,
            score_col=colmap.overall,
            outcome_col=colmap.selected,
        )
    else:
        results["interaction_tests_race_wording"] = pd.DataFrame(
            [{"warning": "Race interaction skipped due to small sample sizes."}]
        )

    # 4) Regression with controls.
    results["regression_controls_overall"] = regression_with_controls(
        df,
        outcome_col=colmap.overall,
        demographic_cols=[colmap.gender, colmap.race],
        control_cols=[colmap.wording, colmap.qualification],
        logistic=False,
    )
    results["regression_controls_selection"] = regression_with_controls(
        df,
        outcome_col=colmap.selected,
        demographic_cols=[colmap.gender, colmap.race],
        control_cols=[colmap.wording, colmap.qualification],
        logistic=True,
    )

    # 5) Paired tests by base resume.
    paired_rows: List[Dict[str, object]] = []
    if {"male_coded", "female_coded"}.issubset(set(df[colmap.gender].dropna().unique())):
        paired_row = paired_variant_tests(
            df,
            base_id_col=colmap.base_id,
            condition_col=colmap.gender,
            metric_col=colmap.overall,
            first_group="female_coded",
            second_group="male_coded",
        )
        if paired_row.get("n_pairs", 0) < min_pair_n:
            paired_row["warning"] = "Insufficient matched pairs for robust inference."
        paired_rows.append({"comparison": "female_vs_male_overall", **paired_row})
    results["paired_variant_tests"] = pd.DataFrame(paired_rows).round(6)

    # 6) Variance checks.
    score_cols = [colmap.overall, colmap.leadership, colmap.experience, colmap.skills]
    results["variance_checks_gender"] = variance_checks(df, group_col=colmap.gender, score_cols=score_cols)
    results["variance_checks_race"] = variance_checks(df, group_col=colmap.race, score_cols=score_cols)

    # Narrative outputs.
    results["presentation_bullets"] = pd.DataFrame(
        {
            "bullet": presentation_bullets(
                impact_gender_df[["group", "selection_rate", "impact_ratio", "reference_group", "passes_4_5_rule"]],
                results["interaction_tests_gender_wording"],
                results["paired_variant_tests"],
            )
        }
    )
    results["report_paragraphs"] = pd.DataFrame(
        {
            "paragraph": report_paragraphs(
                results["bootstrap_fairness_ci"],
                pd.concat(
                    [results["regression_controls_overall"], results["regression_controls_selection"]],
                    ignore_index=True,
                ),
                pd.concat([results["variance_checks_gender"], results["variance_checks_race"]], ignore_index=True),
            )
        }
    )
    return results
