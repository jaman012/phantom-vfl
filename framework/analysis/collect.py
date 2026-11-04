"""
Collect the training runs of one run config for the analysis.

For each seed and k level of the training plan, the analysis uses the run only
when all of these are true:
  - training_runs.csv (the state file of the training runner) has a row for it;
  - the row has the status "completed";
  - the test-metrics file exists and its MD5 checksum equals test_metrics_md5 of the row.
Otherwise the run is "left out", with the reason. A used test-metrics file must
have the column round, one row for each round, and the selection, tie-break and
test columns of the analysis, without empty values; else the analysis stops.
With analysis.round_source "validation", a file without validation columns
stops the analysis.
"""

import pandas as pd

from framework.analysis import select_round
from framework.analysis import settings as analysis_settings
from framework.training import run_training

USED = "used"
LEFT_OUT = "left out"


def required_columns(settings):
    """Columns that a used test-metrics file must have."""
    names = ["round", *[name for name in select_round.rule_columns(settings) if name]]
    names += analysis_settings.metric_columns(settings)
    return list(dict.fromkeys(names))


def read_metrics(path, columns):
    """Read one test-metrics file and check it. Raise ValueError on a fault."""
    frame = pd.read_csv(path)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        validation = [name for name in missing if name.startswith("val_")]
        hint = (" The split has no validation set. With analysis.round_source \"test\" (the default for a split "
                "without a validation set), the round is chosen on the test set."
                if validation else "")
        raise ValueError(f"{path}: missing columns {missing}.{hint}")
    if frame.empty:
        raise ValueError(f"{path}: no rows")
    if frame["round"].duplicated().any():
        raise ValueError(f"{path}: a round occurs more than once")
    if frame[columns].isna().any().any():
        empty = [name for name in columns if frame[name].isna().any()]
        raise ValueError(f"{path}: empty values in {empty}")
    return frame


def collect_runs(info):
    """
    One record for each seed and k level of the plan, in plan order (seed, then k level).

    Each record: seed, config, path, md5, status ("used" or "left out"), reason,
    frame (the test-metrics table of a used run, else None).
    """
    setup, settings = info["setup"], info["settings"]
    state = run_training.read_state_runs(info["state_folder"])
    columns = required_columns(settings)
    records = []
    for seed in info["seeds"]:
        for config in info["configs"]:
            path = run_training.test_metrics_path(setup, seed, config)
            row = state.get((seed, config))
            record = {"seed": seed, "config": config, "path": path, "md5": "",
                      "status": LEFT_OUT, "reason": "", "frame": None}
            if row is None:
                record["reason"] = "no row in training_runs.csv"
            elif row["status"] != "completed":
                record["reason"] = f"status '{row['status']}' in training_runs.csv"
            elif not path.exists():
                record["reason"] = "test-metrics file not found"
            else:
                record["md5"] = run_training.file_md5(path)
                if record["md5"] != row["test_metrics_md5"]:
                    record["reason"] = "MD5 of the test-metrics file differs from training_runs.csv"
                else:
                    record["frame"] = read_metrics(path, columns)
                    record["status"] = USED
            records.append(record)
    return records
