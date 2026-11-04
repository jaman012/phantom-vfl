"""
The viable ceiling: the highest k level up to which no separation from the ground truth is found.

The k levels are read in the order of the plan (the ground truth first, then
for example k*, 5, 10, ... 40). For each track:

  ceiling          the k level just before the first separated k level; when no
                   k level is separated, the last k level; empty when the first
                   k level is separated
  literal          the highest k level that is not separated (it differs from
                   the ceiling when a k level above the first separated one is
                   not separated)
  exceptions       the k levels above the first separated one that are not separated
  untested         the k levels without a test result (for example all
                   differences equal); an untested k level counts as separated,
                   so the ceiling never rests on a missing test

Track B: a k level is separated when its Tukey-adjusted p-value against the
ground truth is below alpha. Track A: when its Bonferroni-corrected paired
t-test is significant. "No separation found" is not a proof of equal
performance; no equivalence test is made.
"""

import numpy as np
import pandas as pd

COLUMNS = ("track", "ceiling", "literal", "exceptions", "untested", "first_separated", "kstar_folder",
           "kstar_separated", "note")


def ceiling_of(levels, separated):
    """
    levels: the k levels after the ground truth, in plan order.
    separated: {level: True, False or None (untested)}.
    Return a dict with ceiling, literal, exceptions, untested, first_separated, note.
    """
    flags = [True if separated.get(level) is None else bool(separated[level]) for level in levels]
    untested = [level for level in levels if separated.get(level) is None]
    first = next((index for index, flag in enumerate(flags) if flag), None)
    not_separated = [level for level, flag in zip(levels, flags) if not flag]
    if first is None:
        ceiling, exceptions, note = levels[-1], [], "no k level is separated"
    elif first == 0:
        ceiling, exceptions = "", [level for level, flag in zip(levels[1:], flags[1:]) if not flag]
        note = "the first k level is separated: no viable k level"
    else:
        ceiling = levels[first - 1]
        exceptions = [level for level, flag in zip(levels[first + 1:], flags[first + 1:]) if not flag]
        note = ""
    if untested:
        note = "; ".join(filter(None, [note, "untested k levels count as separated"]))
    return {"ceiling": ceiling, "literal": not_separated[-1] if not_separated else "",
            "exceptions": ";".join(exceptions), "untested": ";".join(untested),
            "first_separated": levels[first] if first is not None else "", "note": note}


def _flag(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value)


def ceiling_table(levels, kstar_folder, track_a=None, track_b=None, ground_truth="ground_truth"):
    """
    One row for each track that has results.
    track_a: the paired table of Track A (columns config, p_value, significant_bonf_variant).
    track_b: the pairwise table of Track B (columns group1, group2, p_tukey, reject).
    """
    rows = []
    if track_b is not None:
        gt_rows = track_b[track_b["group1"] == ground_truth].set_index("group2")
        separated = {level: (_flag(gt_rows.loc[level, "reject"]) if np.isfinite(gt_rows.loc[level, "p_tukey"])
                             else None) for level in levels}
        rows.append(("track_b_tukey", separated))
    if track_a is not None:
        table = track_a.set_index("config")
        separated = {level: (_flag(table.loc[level, "significant_bonf_variant"])
                             if np.isfinite(table.loc[level, "p_value"]) else None) for level in levels}
        rows.append(("track_a_paired_t_bonferroni", separated))
    out = []
    for track, separated in rows:
        row = {"track": track, **ceiling_of(levels, separated), "kstar_folder": kstar_folder or ""}
        row["kstar_separated"] = ("" if kstar_folder not in levels
                                  else ("untested" if separated[kstar_folder] is None else separated[kstar_folder]))
        out.append(row)
    return pd.DataFrame(out, columns=COLUMNS)
