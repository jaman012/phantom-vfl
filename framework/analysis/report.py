"""
Plain-text reports (Markdown) of the analysis.

variant_report() writes the report of one run config (report.md in its analysis
folder); summary_report() writes the report of a summary across variants. The
functions only format results that the other modules computed; they compute no
test. The seed-count part of the summary report appears only when the summary
asked for it, and it says that its numbers are advice only.
"""

from pathlib import Path

import numpy as np

from framework.analysis import select_round

NO_EQUIVALENCE = ("\"No separation found\" is not a proof of equal performance; no equivalence test is made.")
ADVICE_TEXT = (
    "The numbers in this part are advice only. They change no run and no result: the seeds of a run come only "
    "from the section training of its run config, and no part of the analysis reads this advice. The advice "
    "answers one question: with how many seeds would a true drop of delta in the main metric probably be "
    "detected?")


def _number(value, digits=4):
    """A short text for a number; an empty text for a missing value."""
    if value is None or value == "":
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def _flag(value):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return {"true": "yes", "false": "no"}.get(value.strip().lower(), value)
    return "yes" if bool(value) else "no"


def _cell(text):
    return str(text).replace("|", "/").replace("\n", " ")


def table(header, rows):
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def warning_lines(log_files):
    """The warning and error lines of the open run log so far, or None when no run log is open."""
    if log_files is None:
        return None
    path = Path(log_files.warnings_path)
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _warning_part(lines):
    if lines is None:
        return ["No run log was open; the warning lines are in the screen output only."]
    if not lines:
        return ["None."]
    return [f"- {_cell(line)}" for line in lines]


def variant_report(info, results, seeds, missing, created, run_log_path=None, warnings=None):
    """
    The text of report.md of one run config.

    info: the dict of settings.load_analysis (with r_versions when Track B is on);
    results: the main-metric results of the driver (paired, omnibus, ceiling);
    seeds: the complete seeds; missing: {seed: [k levels without a used run]};
    created: a date-time text; warnings: the warning lines of the run (None: no run log).
    """
    settings = info["settings"]
    metric = settings["metric"]
    lines = [f"# Analysis report: {info['results_tag']}, split {info['setup']['split_ratio']}", "",
             f"Created (UTC): {created}  ",
             f"Data folder: {info['setup']['gen']['data_dir']}  ",
             f"Main metric: {metric}. {select_round.rule_text(settings)}  ",
             f"Alpha {settings['alpha']}; Bonferroni family of Track A: {settings['bonferroni_family']} k levels.  "]
    if settings["track_b"]:
        versions = info.get("r_versions") or {}
        lines.append(f"Track B: on (R {versions.get('r', '')}, emmeans {versions.get('emmeans', '')}).")
    else:
        lines.append("Track B: off (analysis.track_b false).")
    lines += ["", "## Seeds", "",
              f"Planned: {', '.join(str(s) for s in info['seeds'])}.  ",
              f"Used (complete seeds): {', '.join(str(s) for s in seeds)} ({len(seeds)} seeds)."]
    if missing:
        lines.append("Left out of every table:")
        lines += [f"- seed {seed}: no used run for {', '.join(names)}" for seed, names in missing.items()]
    else:
        lines.append("Left out: none.")
    paired = results["paired"]
    lines += ["", f"## Track A: paired t-tests against the ground truth ({metric}, k level minus ground truth)", ""]
    rows = []
    for _, row in paired.iterrows():
        rows.append([row["config"], int(row["n"]), _number(row["diff_mean"]),
                     f"{_number(row['ci_lower'])} to {_number(row['ci_upper'])}" if np.isfinite(row["ci_lower"]) else "",
                     _number(row["t_stat"]), _number(row["p_value"]), _number(row["p_bonf_variant"]),
                     _flag(row["significant_bonf_variant"]) if np.isfinite(row["p_value"]) else "",
                     _number(row["cohens_dz"]), _number(row["shapiro_p"]), _number(row["wilcoxon_p_bonf_variant"]),
                     row["note"]])
    lines.append(table(["k level", "n", "mean difference", f"CI ({_number(1 - settings['alpha'])})", "t", "p",
                        "p (Bonferroni)", "significant", "d_z", "Shapiro-Wilk p", "Wilcoxon p (Bonferroni)",
                        "note"], rows))
    lines += ["", "## Track B: repeated-measures ANOVA", ""]
    omnibus = results.get("omnibus")
    if omnibus is None:
        lines.append("Track B is off; the ceiling below uses Track A only.")
    elif not np.isfinite(omnibus["f_stat"]):
        lines.append(f"No F test: {omnibus['note']}.")
    else:
        lines.append(f"F({_number(omnibus['df1'])}, {_number(omnibus['df2'])}) = {_number(omnibus['f_stat'])}, "
                     f"p = {_number(omnibus['p_uncorrected'])}; Greenhouse-Geisser epsilon "
                     f"{_number(omnibus['gg_epsilon'])}, p = {_number(omnibus['p_gg'])}; Huynh-Feldt p = "
                     f"{_number(omnibus['p_hf'])}; "
                     + (f"Mauchly p = {_number(omnibus['mauchly_p'])}.  " if _number(omnibus["mauchly_p"])
                        else "no Mauchly test.  "))
        lines.append(f"Significant after the Greenhouse-Geisser correction: {_flag(omnibus['omnibus_significant_gg'])}."
                     " The pairwise Tukey tests are in track_b_pairwise.csv; they are reported without this "
                     "condition.")
    lines += ["", "## Viable ceiling (the k level before the first separated k level)", ""]
    rows = [[row["track"], row["ceiling"] or "none", row["literal"], row["exceptions"], row["untested"],
             row["first_separated"], f"{row['kstar_folder']}: {row['kstar_separated']}" if row["kstar_folder"] else "",
             row["note"]] for _, row in results["ceiling"].iterrows()]
    lines.append(table(["track", "ceiling", "literal", "exceptions", "untested", "first separated",
                        "k* separated", "note"], rows))
    lines += ["", NO_EQUIVALENCE, "", "## Warning lines of this run", ""]
    lines += _warning_part(warnings)
    if run_log_path:
        lines += ["", f"Run log: {run_log_path}"]
    lines += ["", f"The CSV files of this folder hold every value. {select_round.second_text(settings)}", ""]
    return "\n".join(lines)


def summary_report(created, config_path, variants, track_a_all, ceilings_all, omnibus_all, family_all, alpha,
                   advice=None, warnings=None):
    """
    The text of report.md of a summary across variants.

    variants: one dict per variant (label, source, n_seeds, n_levels, in_target, note);
    track_a_all, ceilings_all, omnibus_all: the tables across the analysis folders (may be empty);
    advice: None, or a dict with settings, summary (the seed-count summary table) and beside (the same for
    the variants that do not enter the target).
    """
    lines = ["# Summary across variants", "", f"Created (UTC): {created}  ", f"Summary config: {config_path}", "",
             "## Variants", ""]
    rows = [[v["label"], v["source"], v["n_seeds"], v["n_levels"], "yes" if v["in_target"] else "no", v["note"]]
            for v in variants]
    lines.append(table(["variant", "source", "seeds", "k levels", "in the target", "note"], rows))
    lines += ["", "## Results of the analysis folders", ""]
    if track_a_all is None or len(track_a_all) == 0:
        lines.append("No analysis folder is in this summary (stored results are used for the advice only).")
    else:
        lines.append(f"Track A: number of significant k levels (alpha {alpha}) with the Bonferroni family of the "
                     f"variant and with the family of all analysis folders ({family_all} tests).")
        lines.append("")
        rows = []
        for label, part in track_a_all.groupby("variant", sort=False):
            tested = part[np.isfinite(part["p_value"].astype(float))]
            rows.append([label, len(part), int(tested["significant_bonf_variant"].astype(bool).sum()),
                         int(tested["significant_bonf_all"].astype(bool).sum())])
        lines.append(table(["variant", "k levels", "significant (variant family)", "significant (all)"], rows))
        lines += ["", "Viable ceilings:", ""]
        rows = [[r["variant"], r["track"], r["ceiling"] or "none", r["literal"], r["exceptions"], r["untested"]]
                for _, r in ceilings_all.iterrows()]
        lines.append(table(["variant", "track", "ceiling", "literal", "exceptions", "untested"], rows))
        if omnibus_all is not None and len(omnibus_all):
            lines += ["", "Track B omnibus tests:", ""]
            rows = [[r["variant"], _number(r["f_stat"]), _number(r["p_gg"]), _flag(r["omnibus_significant_gg"]),
                     r["note"]] for _, r in omnibus_all.iterrows()]
            lines.append(table(["variant", "F", "p (Greenhouse-Geisser)", "significant", "note"], rows))
        lines += ["", NO_EQUIVALENCE]
    if advice is not None:
        s = advice["settings"]
        lines += ["", "## Seed-count advice (advice only)", "", ADVICE_TEXT, "",
                  f"Settings: target power {s['target_power']}; the spread of the data at its upper "
                  f"{s['sd_upper_confidence']} confidence limit; {s['replicates']} simulated data sets for each "
                  f"number of seeds (Track B); random seed {s['random_seed']}; seeds from {s['n_min']} to "
                  f"{s['n_max']}. Reading \"variant\": Track A at alpha / (k levels of the variant), Track B at "
                  f"alpha. Reading \"all\": Track A at alpha / (all tests of the target variants), Track B at alpha "
                  f"/ (number of target variants).", "",
                  "Smallest number of seeds over all k levels and target variants (empty: not reached within "
                  f"{s['n_max']} seeds):", ""]
        lines.append(_advice_table(advice["summary"]))
        if advice.get("beside") is not None and len(advice["beside"]):
            lines += ["", "Shown beside, not part of the target (see the notes of the variants):", ""]
            lines.append(_advice_table(advice["beside"]))
        lines += ["", "The per-level values are in seed_count_advice.csv; the power curves of Track B are in "
                      "power_curves.csv."]
    lines += ["", "## Warning lines of this run", ""]
    lines += _warning_part(warnings)
    lines.append("")
    return "\n".join(lines)


def _advice_table(summary):
    rows = []
    for _, r in summary.iterrows():
        rows.append([_number(r["delta"]), r["reading"], r["track_a"], r["track_b_omnibus"], r["track_b_pair"],
                     r["largest"], r["not_reached"]])
    return table(["delta", "reading", "Track A", "Track B omnibus", "Track B pair", "largest", "not reached"], rows)
