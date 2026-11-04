"""
Track A of the analysis: descriptive statistics and paired tests against the ground truth.

The seeds are repeated measures: every k level runs on the same seeds as the
ground truth. Every test here pairs the values by seed (the rows of the wide
table of framework.analysis.tables). The differences are k level minus ground
truth, so a positive difference means that the k level scores higher.

  descriptive()   for each k level: n, mean, SD, median, min, max and the
                  confidence interval of the mean (one group, t distribution)
  paired_tests()  for each k level against the ground truth: the mean and SD
                  of the differences and their confidence interval; the paired
                  t-test (primary test) with the Bonferroni correction; Cohen's
                  d_z; the Shapiro-Wilk test of the differences (a check of the
                  t-test assumption only); the Wilcoxon signed-rank test with
                  the Bonferroni correction (sensitivity check)
  friedman()      the Friedman rank test over all k levels (sensitivity check
                  of the repeated-measures omnibus test)

Bonferroni: p_bonf = min(1, p x family). A test is significant when p_bonf is
below alpha. The confidence level is 1 - alpha.

A test that cannot give a value (for example all differences equal, so the
standard deviation is 0) gets an empty value and a text in the column note.
The other tests continue.
"""

import numpy as np
import pandas as pd
from scipy import stats

EQUAL_TOLERANCE = 1e-12  # differences closer than this count as equal

DESCRIPTIVE_COLUMNS = ("config", "n", "mean", "sd", "median", "min", "max", "ci_level", "ci_lower",
                       "ci_upper", "note")
PAIRED_COLUMNS = ("config", "n", "diff_mean", "diff_sd", "ci_level", "ci_lower", "ci_upper", "t_stat",
                  "p_value", "bonferroni_family", "p_bonf_variant", "significant_bonf_variant", "cohens_dz",
                  "shapiro_w", "shapiro_p", "wilcoxon_n_nonzero", "wilcoxon_stat", "wilcoxon_p",
                  "wilcoxon_p_bonf_variant", "note")
FRIEDMAN_COLUMNS = ("n_seeds", "n_configs", "chi2", "df", "p_value", "note")


def _all_equal(values):
    values = np.asarray(values, dtype=float)
    return values.size == 0 or float(np.max(values) - np.min(values)) <= EQUAL_TOLERANCE


def _bonferroni(p_value, family):
    return min(1.0, p_value * family) if np.isfinite(p_value) else np.nan


def mean_interval(values, level):
    """(lower, upper) of the t confidence interval of the mean; NaN with fewer than 2 values."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        return np.nan, np.nan
    half = stats.t.ppf(0.5 + level / 2, n - 1) * np.std(values, ddof=1) / np.sqrt(n)
    return float(np.mean(values) - half), float(np.mean(values) + half)


def descriptive(wide, configs, alpha):
    """One row for each k level (in the given order)."""
    level = 1 - alpha
    rows = []
    for name in configs:
        values = wide[name].to_numpy(dtype=float)
        lower, upper = mean_interval(values, level)
        rows.append({"config": name, "n": int(values.size), "mean": float(np.mean(values)),
                     "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                     "median": float(np.median(values)), "min": float(np.min(values)),
                     "max": float(np.max(values)), "ci_level": level, "ci_lower": lower, "ci_upper": upper,
                     "note": "" if values.size > 1 else "one value: no SD and no confidence interval"})
    return pd.DataFrame(rows, columns=DESCRIPTIVE_COLUMNS)


def _paired_row(name, diff, family, alpha):
    level = 1 - alpha
    n = int(diff.size)
    notes = []
    row = {"config": name, "n": n, "diff_mean": float(np.mean(diff)),
           "diff_sd": float(np.std(diff, ddof=1)) if n > 1 else np.nan, "ci_level": level,
           "ci_lower": np.nan, "ci_upper": np.nan, "t_stat": np.nan, "p_value": np.nan,
           "bonferroni_family": family, "p_bonf_variant": np.nan, "significant_bonf_variant": False,
           "cohens_dz": np.nan, "shapiro_w": np.nan, "shapiro_p": np.nan,
           "wilcoxon_n_nonzero": int(np.sum(np.abs(diff) > EQUAL_TOLERANCE)), "wilcoxon_stat": np.nan,
           "wilcoxon_p": np.nan, "wilcoxon_p_bonf_variant": np.nan}
    if n < 3:
        notes.append(f"{n} pair(s): no tests (3 or more are needed)")
    elif _all_equal(diff):
        notes.append("all differences equal: no t-test, no Cohen's d_z, no Shapiro-Wilk")
    else:
        row["ci_lower"], row["ci_upper"] = mean_interval(diff, level)
        t_stat, p_value = stats.ttest_rel(diff, np.zeros(n))
        row["t_stat"], row["p_value"] = float(t_stat), float(p_value)
        row["p_bonf_variant"] = _bonferroni(row["p_value"], family)
        row["significant_bonf_variant"] = bool(row["p_bonf_variant"] < alpha)
        row["cohens_dz"] = row["diff_mean"] / row["diff_sd"]
        w_stat, w_p = stats.shapiro(diff)
        row["shapiro_w"], row["shapiro_p"] = float(w_stat), float(w_p)
    if n >= 3:
        nonzero = row["wilcoxon_n_nonzero"]
        if nonzero == 0:
            notes.append("all differences 0: no Wilcoxon test")
        else:
            if nonzero < n:
                notes.append(f"{n - nonzero} zero difference(s) left out of the Wilcoxon test")
            kept = diff[np.abs(diff) > EQUAL_TOLERANCE]
            result = stats.wilcoxon(kept, zero_method="wilcox", alternative="two-sided")
            row["wilcoxon_stat"], row["wilcoxon_p"] = float(result.statistic), float(result.pvalue)
            row["wilcoxon_p_bonf_variant"] = _bonferroni(row["wilcoxon_p"], family)
    row["note"] = "; ".join(notes)
    return row


def paired_tests(wide, configs, family, alpha):
    """One row for each k level after the first (the ground truth), in the given order."""
    reference = wide[configs[0]].to_numpy(dtype=float)
    rows = [_paired_row(name, wide[name].to_numpy(dtype=float) - reference, family, alpha)
            for name in configs[1:]]
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def friedman(wide, configs):
    """One row: the Friedman test over all k levels (seeds are the blocks)."""
    n, k = len(wide), len(configs)
    row = {"n_seeds": n, "n_configs": k, "chi2": np.nan, "df": k - 1, "p_value": np.nan, "note": ""}
    values = wide[list(configs)].to_numpy(dtype=float)
    if k < 3 or n < 2:
        row["note"] = "the Friedman test needs 3 or more k levels and 2 or more seeds"
    elif all(_all_equal(line) for line in values):
        row["note"] = "every seed has equal values for all k levels: no Friedman test"
    else:
        chi2, p_value = stats.friedmanchisquare(*[values[:, j] for j in range(k)])
        row["chi2"], row["p_value"] = float(chi2), float(p_value)
    return pd.DataFrame([row], columns=FRIEDMAN_COLUMNS)
