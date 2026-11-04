"""
Round selection of each training run.

Two rules choose a round:
  own round   each run (seed, k level) at its own round with the highest value
              of the selection column; equal values: the higher tie-break
              value, then the earlier round
  gt round    for each seed, the own round of the ground truth, used for every
              k level of that seed

analysis.round_source decides the columns and which rule gives the main tables:
  "validation"  selection column selection_metric, tie-break tie_break (both
                validation columns). Main tables: own round. Second tables
                (folder sensitivity_gt_round/): gt round. The test set never
                chooses the round.
  "test"        selection column metric (a test column), no tie-break (equal
                values: the earlier round). Main tables: gt round. Second tables
                (folder sensitivity_own_round/): own round. This is the rule of
                the published results, for a split without a validation set:
                the test set chooses the round.
The results of a run are the test columns of the chosen round.
"""

import pandas as pd

from framework.analysis import settings as analysis_settings

GT_ROUND_FOLDER = "sensitivity_gt_round"
OWN_ROUND_FOLDER = "sensitivity_own_round"


def rule_columns(settings):
    """(selection column, tie-break column or None) of analysis.round_source."""
    if settings["round_source"] == "test":
        return settings["metric"], None
    return settings["selection_metric"], settings["tie_break"]


def second_folder(settings):
    """The folder of the second tables: sensitivity_gt_round or sensitivity_own_round."""
    return OWN_ROUND_FOLDER if settings["round_source"] == "test" else GT_ROUND_FOLDER


def rule_text(settings):
    """The rule of the main tables in one sentence."""
    if settings["round_source"] == "test":
        return (f"Round of each seed: the round with the highest {settings['metric']} of the ground truth, then the "
                f"earlier round; every k level of the seed uses that round. The round is chosen on the test set "
                f"(analysis.round_source \"test\").")
    return (f"Round of each run: the highest {settings['selection_metric']}, then the highest "
            f"{settings['tie_break']}, then the earlier round.")


def second_text(settings):
    """What the second tables hold, in one sentence."""
    folder = second_folder(settings)
    if settings["round_source"] == "test":
        return f"The folder {folder}/ holds the same tables with each run at its own round."
    return f"The folder {folder}/ holds the same tables with the ground truth's round for every k level of a seed."


def select_row(frame, selection_metric, tie_break=None):
    """The row of the own round of one run: the highest selection value, then the higher tie-break, then the earlier round."""
    columns = [selection_metric] + ([tie_break] if tie_break else []) + ["round"]
    ordered = frame.sort_values(by=columns, ascending=[False] * (len(columns) - 1) + [True], kind="mergesort")
    return ordered.iloc[0]


def _record_row(record, row, settings):
    frame = record["frame"]
    selection, tie_break = rule_columns(settings)
    values = {
        "seed": record["seed"],
        "config": record["config"],
        "selected_round": int(row["round"]),
        "last_round": int(frame["round"].max()),
        selection: row[selection],
    }
    if tie_break:
        values[tie_break] = row[tie_break]
    for name in analysis_settings.TEST_COLUMNS:
        if name in frame.columns:
            values[name] = row[name]
    return values


def selection_table(records, settings):
    """One row for each used run at its own round, in the order of the records."""
    selection, tie_break = rule_columns(settings)
    rows = [_record_row(record, select_row(record["frame"], selection, tie_break), settings)
            for record in records]
    return pd.DataFrame(rows)


def gt_round_table(records, selection, settings, ground_truth):
    """
    One row for each used run of a seed with a used ground-truth run: the run at
    the ground truth's selected round of that seed. Raise ValueError when a run
    has no row for that round.
    """
    gt_rounds = {int(row.seed): int(row.selected_round)
                 for row in selection.itertuples() if row.config == ground_truth}
    rows = []
    for record in records:
        if record["seed"] not in gt_rounds:
            continue
        target = gt_rounds[record["seed"]]
        frame = record["frame"]
        match = frame[frame["round"] == target]
        if match.empty:
            raise ValueError(f"seed {record['seed']}, {record['config']}: no row for round {target} "
                             f"(the selected round of the ground truth)")
        rows.append(_record_row(record, match.iloc[0], settings))
    return pd.DataFrame(rows)


def lens_tables(records, own, settings, ground_truth):
    """
    (main table, second table or None) of analysis.round_source; own is the
    selection_table of the records. The second table is None when
    analysis.sensitivity_gt_round is false.
    """
    second = settings["sensitivity_gt_round"]
    if settings["round_source"] == "test":
        return gt_round_table(records, own, settings, ground_truth), (own if second else None)
    return own, (gt_round_table(records, own, settings, ground_truth) if second else None)
