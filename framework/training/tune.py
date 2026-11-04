r"""
Tuning driver: random search of run-config values with Flower (James Mansueto's own code).

The driver reads the sections generation, data_preparation and tuning of
run_config.json. It draws the values of every trial from the search space
before the first run, starts one Flower run for each trial, seed and k level
with the launcher framework/training/run_flower.py (run_flower.main), and
ranks the trials on the validation metrics. From the test-metrics file of a
run it reads only the column round and the two val_* columns of the settings
metric and tie_break; it never reads a test_* column.

The tuning section (every key except study and space has a default):

  "tuning": {
    "study": "bm_focal_gt",                      required: letters, digits, _ and -, 1 to 64 characters
    "seeds": [1000, 1008, 1016],                 default: generation.seeds
    "k_levels": ["ground_truth"],                default: ["ground_truth"]
    "feature_form": "model",                     "model" (one-hot form) or "integer"; default "model"
    "trials": 60,                                1 to 999; default 20
    "sampler_seed": 2026,                        seed of the draws; default 0
    "space": {                                   required: one key or more
      "focal-alpha":   {"distribution": "uniform",     "low": 0.05,   "high": 0.95, "decimals": 3},
      "learning-rate": {"distribution": "log_uniform", "low": 0.0001, "high": 0.1,  "significant": 3}
    },
    "run_config": {"loss-type": "focal"},        fixed values of every trial
    "metric": "val_f1",                          default "val_f1"
    "tie_break": "val_pr_auc"                    default "val_pr_auc"
  }

space: keys of [tool.flwr.app.config] in flower_app/pyproject.toml that have a
decimal default. uniform: low + u x (high - low), rounded to 'decimals'
places (default 3). log_uniform: exp(log low + u x (log high - log low)),
rounded to 'significant' digits (default 3); low must be above 0. low and high
must not change when they are rounded, so a rounded value stays in the range.
run_config: as training.run_config of run_config.json, without the key
results-tag (the driver sets it) and without the keys of the space.
The split must have a validation set (data_preparation.validation_share above 0).

Files of a study (the study folder is <data-dir>/07_tuning/<study>/):

  study.json         the settings of the study (written before the first run)
  trials_plan.csv    trial, results_tag and the values of each trial (written before the first run)
  runs.csv           one row for each run (written again after each run): trial, seed, k,
                     results_tag, status, exit_code, reason, metric_peak, peak_round,
                     tie_break_at_peak, last_round, test_metrics_md5, run_log
  ranking.csv        one row for each trial (written again after each run)
  best_run_config.json  the fixed values and the values of the trial with rank 1

The Flower results of trial <n> go to <data-dir>/06_flower_vfl_results/tune_<study>_t<nnn>/
(framework.training.results); the run logs go to <data-dir>/logs/ (the launcher).

Rules:
- The driver checks the whole run_config.json before the first run and stops
  with a list of all faults. It reads each --run-config text back with
  Flower's reader and stops if a value differs from the plan.
- Order of the runs: trial 1 with each seed and each k level, then trial 2, and so on.
- Peak of a run: the highest value of 'metric' over the evaluated rounds (the
  earliest round if two rounds have the same value; rows without a number are
  not used). A completed run has a test-metrics file with the two columns and
  a last round equal to num-server-rounds.
- Rank: only trials with every run completed. Order: the mean of the peaks
  over the runs (highest first), then the mean of tie_break at the peak rounds
  (highest first), then the trial number.
- The driver does not change run_config.json. Copy the run_config of
  best_run_config.json into training.run_config yourself.
- Without --resume, an existing study folder stops the driver. With --resume,
  the driver stops if study.json or trials_plan.csv differs from the settings,
  keeps each run that runs.csv shows as completed when its test-metrics file
  still has the recorded MD5, and starts every other run. With --resume and no
  study folder, the driver starts a new study.
- After the plan, the driver waits for Enter (the confirmation gate); type q
  to stop. --yes runs without the gate.
- A failed run does not stop the driver. Exit code: 0 when every run
  completed, 1 when a run failed, 2 when the user stopped at the gate.
  Ctrl-C stops the driver; the run that was interrupted has no row in runs.csv.

Usage (with the Python of the environment that has flwr):

  python <repository root>/framework/training/tune.py --config run_config.json [--yes] [--resume]
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
if __package__ in (None, ""):  # started as a file: make the package 'framework' importable
    sys.path.insert(0, str(ROOT))

from framework import run_pipeline  # noqa: E402
from framework.training import results, run_flower, run_training  # noqa: E402

SECTION = "tuning"
TUNING_KEYS = ("study", "seeds", "k_levels", "feature_form", "trials", "sampler_seed",
               "space", "run_config", "metric", "tie_break")
DEFAULT_K_LEVELS = [run_training.GROUND_TRUTH]
DEFAULT_TRIALS = 20
MAX_TRIALS = 999
DEFAULT_SAMPLER_SEED = 0
DEFAULT_METRIC = "val_f1"
DEFAULT_TIE_BREAK = "val_pr_auc"
METRIC_PREFIX = "val_"
DISTRIBUTIONS = {"uniform": ("decimals", 3), "log_uniform": ("significant", 3)}  # rounding key and its default
SPACE_KEYS = ("distribution", "low", "high")
DRIVER_KEYS = ("results-tag",)  # the driver sets this key for each trial
STUDY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
TUNING_FOLDER = "07_tuning"
TAG_PREFIX = "tune"
STUDY_FILE = "study.json"
PLAN_FILE = "trials_plan.csv"
RUNS_FILE = "runs.csv"
RANKING_FILE = "ranking.csv"
BEST_FILE = "best_run_config.json"
RUN_COLUMNS = ("trial", "seed", "k", "results_tag", "status", "exit_code", "reason", "metric_peak",
               "peak_round", "tie_break_at_peak", "last_round", "test_metrics_md5", "run_log")
STUDY_SCHEMA = 1
ERROR_PREFIX = run_training.ERROR_PREFIX


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def round_value(value, distribution, digits):
    """The value rounded: 'digits' decimal places (uniform) or 'digits' significant digits (log_uniform)."""
    if distribution == "uniform":
        return round(value, digits)
    return float(f"{value:.{digits - 1}e}")


def check_space(space, defaults, fixed):
    """Return a list of error messages for tuning.space (empty if valid)."""
    if not isinstance(space, dict) or not space:
        return [f"{SECTION}.space must be an object with one key or more"]
    errors = []
    for key, spec in space.items():
        name = f"{SECTION}.space.{key}"
        if key in run_training.RESERVED_KEYS or key in DRIVER_KEYS:
            errors.append(f"{name}: the runner or the driver sets this key. Remove it")
            continue
        if key not in defaults:
            errors.append(f"{name} is not a key of flower_app/pyproject.toml")
            continue
        if not isinstance(defaults[key], float):
            errors.append(f"{name}: only a key with a decimal default can be tuned "
                          f"(default {defaults[key]!r})")
        if key in fixed:
            errors.append(f"{name} is also in {SECTION}.run_config. Remove it from one of them")
        if not isinstance(spec, dict):
            errors.append(f"{name} must be an object")
            continue
        distribution = spec.get("distribution")
        if distribution not in DISTRIBUTIONS:
            errors.append(f"{name}.distribution must be one of {list(DISTRIBUTIONS)}")
            continue
        digits_key, _ = DISTRIBUTIONS[distribution]
        unknown = set(spec) - set(SPACE_KEYS) - {digits_key}
        if unknown:
            errors.append(f"{name}: unknown keys {sorted(unknown)}")
        low, high = spec.get("low"), spec.get("high")
        if not (_is_number(low) and _is_number(high)):
            errors.append(f"{name}: low and high must be numbers")
            continue
        if not low < high:
            errors.append(f"{name}: low must be less than high")
        if distribution == "log_uniform" and low <= 0:
            errors.append(f"{name}: low must be above 0 for log_uniform")
        digits = spec.get(digits_key, DISTRIBUTIONS[distribution][1])
        if not (isinstance(digits, int) and not isinstance(digits, bool) and 1 <= digits <= 10):
            errors.append(f"{name}.{digits_key} must be a whole number from 1 to 10")
            continue
        for bound in (low, high):
            if bound > 0 or distribution == "uniform":
                if round_value(float(bound), distribution, digits) != bound:
                    errors.append(f"{name}: {bound!r} changes when it is rounded to {digits} "
                                  f"{'decimal places' if distribution == 'uniform' else 'significant digits'}")
    return errors


def check_tuning(tuning, gen, defaults):
    """Return a list of error messages for the tuning section (empty if valid)."""
    if not isinstance(tuning, dict):
        return [f"the {SECTION} section is required and must be an object"]
    errors = []
    unknown = set(tuning) - set(TUNING_KEYS)
    if unknown:
        errors.append(f"{SECTION}: unknown keys {sorted(unknown)}")
    study = tuning.get("study")
    if not (isinstance(study, str) and STUDY_NAME.match(study)):
        errors.append(f"{SECTION}.study is required: 1 to 64 letters, digits, _ or - (first: a letter or a digit)")
    seeds = tuning.get("seeds", gen["seeds"])
    if not isinstance(seeds, list) or not seeds or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds):
        errors.append(f"{SECTION}.seeds must be a non-empty list of whole numbers (0 or more)")
    elif len(set(seeds)) != len(seeds):
        errors.append(f"{SECTION}.seeds has a repeated seed")
    try:
        run_training.k_folders(tuning.get("k_levels", DEFAULT_K_LEVELS), gen["kstar_folder"])
    except ValueError as error:
        errors.append(f"{SECTION}.k_levels: {error}")
    if tuning.get("feature_form", run_training.DEFAULT_FEATURE_FORM) not in run_training.FEATURE_FORMS:
        errors.append(f"{SECTION}.feature_form must be one of {list(run_training.FEATURE_FORMS)}")
    trials = tuning.get("trials", DEFAULT_TRIALS)
    if not (isinstance(trials, int) and not isinstance(trials, bool) and 1 <= trials <= MAX_TRIALS):
        errors.append(f"{SECTION}.trials must be a whole number from 1 to {MAX_TRIALS}")
    sampler_seed = tuning.get("sampler_seed", DEFAULT_SAMPLER_SEED)
    if not (isinstance(sampler_seed, int) and not isinstance(sampler_seed, bool) and sampler_seed >= 0):
        errors.append(f"{SECTION}.sampler_seed must be a whole number (0 or more)")
    fixed = tuning.get("run_config", {})
    if not isinstance(fixed, dict):
        errors.append(f"{SECTION}.run_config must be an object")
        fixed = {}
    for key, value in fixed.items():
        if key in run_training.RESERVED_KEYS or key in DRIVER_KEYS:
            errors.append(f"{SECTION}.run_config.{key}: the runner or the driver sets this key. Remove it")
        elif key not in defaults:
            errors.append(f"{SECTION}.run_config.{key} is not a key of flower_app/pyproject.toml")
        elif not run_training._same_type(value, defaults[key]):
            errors.append(f"{SECTION}.run_config.{key} must have the type of its default "
                          f"{defaults[key]!r} ({type(defaults[key]).__name__})")
    errors.extend(check_space(tuning.get("space"), defaults, fixed))
    for key, default in (("metric", DEFAULT_METRIC), ("tie_break", DEFAULT_TIE_BREAK)):
        value = tuning.get(key, default)
        if not (isinstance(value, str) and value.startswith(METRIC_PREFIX) and len(value) > len(METRIC_PREFIX)):
            errors.append(f"{SECTION}.{key} must be the name of a {METRIC_PREFIX}* column (for example {default})")
    if not gen["data_preparation"]["validation_share"] > 0:
        errors.append("data_preparation.validation_share must be above 0: the tuning uses the validation set")
    if not Path(gen["data_dir"]).is_dir():
        errors.append(f"generation.data_dir does not exist: {gen['data_dir']}")
    return errors


def load_tuning(config_path):
    """
    Read and check run_config.json for a study. Stop with a list of all faults.

    Returns:
        dict: gen, study, seeds, k_folders, feature_form, split_ratio,
        feature_splits, trials, sampler_seed, space (with the rounding digits),
        run_config (the fixed values), metric, tie_break, defaults, study_dir
    """
    gen = run_pipeline.load_run_config(config_path)
    tuning = json.loads(Path(config_path).read_text(encoding="utf-8")).get(SECTION)
    defaults = run_training.app_defaults()
    errors = check_tuning(tuning, gen, defaults)
    if errors:
        raise run_pipeline.RunConfigError(f"Run config {config_path} is not valid:\n  - " + "\n  - ".join(errors))
    prep = gen["data_preparation"]
    form = tuning.get("feature_form", run_training.DEFAULT_FEATURE_FORM)
    split_folder = run_pipeline.data_io.split_folder_name(
        prep["test_share"], prep["validation_share"], model=(form == "model"))
    cfg = run_pipeline.load_dataset_config(gen["dataset_config"])
    fixed = {key: float(value) if isinstance(defaults[key], float) else value
             for key, value in tuning.get("run_config", {}).items()}
    space = {}
    for key in sorted(tuning["space"]):
        spec = tuning["space"][key]
        digits_key, digits_default = DISTRIBUTIONS[spec["distribution"]]
        space[key] = {"distribution": spec["distribution"], "low": float(spec["low"]),
                      "high": float(spec["high"]), digits_key: spec.get(digits_key, digits_default)}
    return {
        "gen": gen,
        "study": tuning["study"],
        "seeds": list(tuning.get("seeds", gen["seeds"])),
        "k_folders": run_training.k_folders(tuning.get("k_levels", DEFAULT_K_LEVELS), gen["kstar_folder"]),
        "feature_form": form,
        "split_ratio": split_folder[len("split_"):],
        "feature_splits": ",".join(str(len(client)) for client in cfg.clients),
        "trials": tuning.get("trials", DEFAULT_TRIALS),
        "sampler_seed": tuning.get("sampler_seed", DEFAULT_SAMPLER_SEED),
        "space": space,
        "run_config": fixed,
        "metric": tuning.get("metric", DEFAULT_METRIC),
        "tie_break": tuning.get("tie_break", DEFAULT_TIE_BREAK),
        "defaults": defaults,
        "study_dir": Path(gen["data_dir"]) / TUNING_FOLDER / tuning["study"],
    }


def draw_trials(space, trials, sampler_seed):
    """
    The values of each trial: one draw u in [0, 1) for each key, keys in name
    order, trial after trial, from random.Random(sampler_seed). Returns a list
    of dicts (trial 1 first).
    """
    rng = random.Random(sampler_seed)
    plan = []
    for _ in range(trials):
        values = {}
        for key in sorted(space):
            spec = space[key]
            u = rng.random()
            if spec["distribution"] == "uniform":
                value = spec["low"] + u * (spec["high"] - spec["low"])
                values[key] = round_value(value, "uniform", spec["decimals"])
            else:
                low, high = math.log(spec["low"]), math.log(spec["high"])
                values[key] = round_value(math.exp(low + u * (high - low)), "log_uniform", spec["significant"])
        plan.append(values)
    return plan


def trial_tag(study, trial):
    """Results tag of a trial: tune_<study>_t<nnn>."""
    return f"{TAG_PREFIX}_{study}_t{trial:03d}"


def study_record(setup):
    """The settings of the study as written to study.json (resume compares them)."""
    gen = setup["gen"]
    return {
        "schema": STUDY_SCHEMA, "study": setup["study"], "data_dir": gen["data_dir"],
        "dataset_config": gen["dataset_config"], "split_ratio": setup["split_ratio"],
        "feature_form": setup["feature_form"], "seeds": setup["seeds"], "k_folders": setup["k_folders"],
        "trials": setup["trials"], "sampler_seed": setup["sampler_seed"], "space": setup["space"],
        "run_config": setup["run_config"], "metric": setup["metric"], "tie_break": setup["tie_break"],
        "app_defaults": setup["defaults"],
    }


def plan_rows(setup, plan):
    """Rows of trials_plan.csv (values as text)."""
    return [{"trial": number, "results_tag": trial_tag(setup["study"], number),
             **{key: repr(value) for key, value in values.items()}}
            for number, values in enumerate(plan, start=1)]


def make_runs(setup, plan):
    """The runs: each trial with each seed and each k level, with the arguments of the launcher."""
    runs = []
    for number, trial_values in enumerate(plan, start=1):
        tag = trial_tag(setup["study"], number)
        for seed in setup["seeds"]:
            for k in setup["k_folders"]:
                values = run_training.run_values(setup, seed, k)
                values.update(trial_values)
                values["results-tag"] = tag
                text = run_training.run_config_text(values)
                run_training.check_text(text, values)
                runs.append({"trial": number, "seed": seed, "k": k, "tag": tag, "values": values,
                             "argv": ["--split-ratio", setup["split_ratio"], "--run-config", text]})
    return runs


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def metrics_path(setup, run):
    folder = results.results_dir(setup["gen"]["data_dir"], run["tag"], setup["split_ratio"], run["seed"], run["k"])
    return folder / results.test_metrics_file_name(run["k"], run["seed"])


def _number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def read_peak(path, metric, tie_break, last_round):
    """
    Peak of one run from its test-metrics file: (metric_peak, peak_round,
    tie_break_at_peak, last_round of the file). Reads only the column round
    and the columns metric and tie_break. Raises ValueError when the file has
    no such column, no row with a number, or a last round other than last_round.
    """
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [name for name in ("round", metric, tie_break) if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"columns missing in {path.name}: {missing}")
        peak = None
        file_last = None
        for row in reader:
            number = int(row["round"])
            file_last = number if file_last is None else max(file_last, number)
            value = _number(row[metric])
            if value is not None and (peak is None or value > peak[0]):
                peak = (value, number, _number(row[tie_break]))
    if peak is None:
        raise ValueError(f"no row of {path.name} has a number in {metric}")
    if file_last != last_round:
        raise ValueError(f"last round of {path.name} is {file_last}, expected {last_round}")
    return peak[0], peak[1], peak[2], file_last


def _text(value):
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def run_one(setup, run, launcher):
    """Start one run with the launcher and return its row of runs.csv."""
    settings = {**setup["defaults"], **run["values"]}
    start = time.time()
    reason = ""
    try:
        code = launcher(run["argv"])
    except KeyboardInterrupt:
        raise
    except BaseException as error:  # an error of one run does not stop the other runs
        code = 1
        reason = f"{type(error).__name__}: {error}"
        print(f"{ERROR_PREFIX} The run stopped with an error: {reason}", flush=True)
    row = {"trial": run["trial"], "seed": run["seed"], "k": run["k"], "results_tag": run["tag"],
           "status": "failed", "exit_code": code, "reason": reason, "metric_peak": "", "peak_round": "",
           "tie_break_at_peak": "", "last_round": "", "test_metrics_md5": "",
           "run_log": run_training.find_run_log(settings, start)}
    if code != 0:
        row["reason"] = reason or "see the run log"
        return row
    path = metrics_path(setup, run)
    try:
        peak, peak_round, tie, last = read_peak(path, setup["metric"], setup["tie_break"],
                                                settings["num-server-rounds"])
    except (OSError, ValueError) as error:
        row["reason"] = str(error)
        print(f"{ERROR_PREFIX} {error}", flush=True)
        return row
    row.update({"status": "completed", "metric_peak": _text(peak), "peak_round": peak_round,
                "tie_break_at_peak": _text(tie), "last_round": last, "test_metrics_md5": file_md5(path)})
    print(f"Trial {run['trial']}, seed {run['seed']}, {run['k']}: peak {setup['metric']} "
          f"{peak:.4f} at round {peak_round}", flush=True)
    return row


def rank_trials(setup, plan, rows):
    """Rows of ranking.csv: one row for each trial, rank 1 first, then the trials without a rank."""
    planned = len(setup["seeds"]) * len(setup["k_folders"])
    metric, tie = setup["metric"], setup["tie_break"]
    table = []
    for number, values in enumerate(plan, start=1):
        done = [row for row in rows if row["trial"] == number and row["status"] == "completed"]
        entry = {"rank": "", "trial": number, "results_tag": trial_tag(setup["study"], number),
                 **{key: repr(value) for key, value in values.items()},
                 "runs_completed": len(done), "runs_planned": planned,
                 f"mean_{metric}": "", f"sd_{metric}": "", f"mean_{tie}": "", "peak_rounds": ""}
        if len(done) == planned:
            peaks = [float(row["metric_peak"]) for row in done]
            ties = [_number(row["tie_break_at_peak"]) for row in done]
            entry[f"mean_{metric}"] = statistics.fmean(peaks)
            entry[f"sd_{metric}"] = statistics.stdev(peaks) if len(peaks) > 1 else 0.0
            entry[f"mean_{tie}"] = statistics.fmean(ties) if all(t is not None for t in ties) else ""
            entry["peak_rounds"] = ";".join(str(row["peak_round"]) for row in done)
        table.append(entry)

    def order(entry):
        tie_value = entry[f"mean_{tie}"]
        return (-entry[f"mean_{metric}"], -(tie_value if tie_value != "" else -math.inf), entry["trial"])

    ranked = sorted((entry for entry in table if entry[f"mean_{metric}"] != ""), key=order)
    for position, entry in enumerate(ranked, start=1):
        entry["rank"] = position
    return ranked + [entry for entry in table if entry["rank"] == ""]


def best_record(setup, plan, ranking):
    """Content of best_run_config.json, or None when no trial has a rank."""
    if not ranking or ranking[0]["rank"] != 1:
        return None
    best = ranking[0]
    metric = setup["metric"]
    return {
        "study": setup["study"], "trial": best["trial"], "results_tag": best["results_tag"],
        f"mean_{metric}": best[f"mean_{metric}"], f"sd_{metric}": best[f"sd_{metric}"],
        "trials_ranked": sum(1 for entry in ranking if entry["rank"] != ""), "trials_planned": len(plan),
        "run_config": {**setup["run_config"], **plan[best["trial"] - 1]},
        "note": "Copy run_config into training.run_config of run_config.json. "
                "The tuning driver does not change run_config.json.",
    }


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _text(row.get(key)) for key in columns} for row in rows)


def write_outputs(setup, plan, rows):
    """Write runs.csv, ranking.csv and best_run_config.json (when a trial has a rank)."""
    folder = setup["study_dir"]
    write_csv(folder / RUNS_FILE, RUN_COLUMNS, rows)
    ranking = rank_trials(setup, plan, rows)
    columns = (["rank", "trial", "results_tag"] + sorted(setup["space"]) +
               ["runs_completed", "runs_planned", f"mean_{setup['metric']}", f"sd_{setup['metric']}",
                f"mean_{setup['tie_break']}", "peak_rounds"])
    write_csv(folder / RANKING_FILE, columns, ranking)
    best = best_record(setup, plan, ranking)
    if best is not None:
        (folder / BEST_FILE).write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    return ranking


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def start_study(setup, plan, resume):
    """
    Make the study folder or check it for a resume. Return the rows of runs.csv
    that stay (completed runs whose test-metrics file has the recorded MD5).
    """
    folder = setup["study_dir"]
    record = json.loads(json.dumps(study_record(setup)))
    expected_plan = [{key: str(value) for key, value in row.items()} for row in plan_rows(setup, plan)]
    if folder.exists() and not resume:
        raise run_pipeline.RunConfigError(
            f"The study folder exists: {folder}. Use --resume to continue it, or choose a new study name.")
    if not folder.exists():
        folder.mkdir(parents=True)
        (folder / STUDY_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        columns = ["trial", "results_tag"] + sorted(setup["space"])
        write_csv(folder / PLAN_FILE, columns, plan_rows(setup, plan))
        if resume:
            print(f"--resume: no study folder; a new study starts in {folder}")
        return []
    saved = json.loads((folder / STUDY_FILE).read_text(encoding="utf-8"))
    if saved != record:
        changed = sorted(key for key in set(saved) | set(record) if saved.get(key) != record.get(key))
        raise run_pipeline.RunConfigError(
            f"--resume: the settings differ from {folder / STUDY_FILE} (keys {changed}). "
            "Use the settings of the study, or choose a new study name.")
    if _read_csv(folder / PLAN_FILE) != expected_plan:
        raise run_pipeline.RunConfigError(f"--resume: {folder / PLAN_FILE} differs from the plan of the settings.")
    kept = []
    runs_path = folder / RUNS_FILE
    old_rows = _read_csv(runs_path) if runs_path.exists() else []
    for row in old_rows:
        if row["status"] != "completed":
            continue
        run = {"tag": row["results_tag"], "seed": int(row["seed"]), "k": row["k"]}
        path = metrics_path(setup, run)
        if path.exists() and file_md5(path) == row["test_metrics_md5"]:
            row["trial"], row["seed"] = int(row["trial"]), int(row["seed"])
            kept.append(row)
        else:
            print(f"--resume: trial {row['trial']}, seed {row['seed']}, {row['k']}: the test-metrics file "
                  "is missing or changed; the run starts again")
    return kept


def show_plan(config_path, setup, runs, to_run, yes):
    gen = setup["gen"]
    print("=" * 80)
    print("PHANTOM VFL TUNING DRIVER")
    print("=" * 80)
    print(f"Run config: {config_path}")
    print(f"  study: {setup['study']} ({setup['study_dir']})")
    print(f"  data_dir: {gen['data_dir']}")
    print(f"  feature form: {setup['feature_form']} (SPLIT_RATIO={setup['split_ratio']})")
    print(f"  seeds: {setup['seeds']}")
    print(f"  k levels (folders): {setup['k_folders']}")
    print(f"  trials: {setup['trials']} (sampler_seed {setup['sampler_seed']})")
    for key, spec in setup["space"].items():
        print(f"  space {key}: {spec}")
    print(f"  {SECTION}.run_config: {setup['run_config'] or '(none: the defaults of flower_app/pyproject.toml)'}")
    print(f"  selection: peak {setup['metric']}, mean over the runs of a trial; tie-break {setup['tie_break']}")
    print(f"  runs: {len(runs)} planned, {len(runs) - len(to_run)} already completed, {len(to_run)} to start")
    print(f"  confirmation gate: {'off (--yes)' if yes else 'on'}")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Random search of run-config values with Flower (the tuning section of run_config.json).")
    parser.add_argument("--config", required=True, help="Path of run_config.json")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation gate")
    parser.add_argument("--resume", action="store_true", help="Continue the study in its existing folder")
    return parser.parse_args(argv)


def main(argv=None, launcher=None):
    """
    Run the study of the tuning section. Return 0 when every run completed, 1
    when a run failed, 2 when the user stopped at the confirmation gate.
    launcher replaces run_flower.main (for tests only).
    """
    args = parse_arguments(argv)
    launcher = launcher or run_flower.main
    setup = load_tuning(args.config)
    plan = draw_trials(setup["space"], setup["trials"], setup["sampler_seed"])
    runs = make_runs(setup, plan)
    folder_existed = setup["study_dir"].exists()
    if folder_existed and not args.resume:
        raise run_pipeline.RunConfigError(
            f"The study folder exists: {setup['study_dir']}. Use --resume to continue it, or choose a new study name.")
    kept = start_study(setup, plan, args.resume) if folder_existed else []
    done = {(row["trial"], row["seed"], row["k"]) for row in kept}
    to_run = [run for run in runs if (run["trial"], run["seed"], run["k"]) not in done]
    show_plan(args.config, setup, runs, to_run, args.yes)
    if not args.yes and not run_pipeline.confirm(f"{len(to_run)} Flower run(s) of the study {setup['study']}"):
        print("\nStopped by the user before the first run.")
        return 2
    if not folder_existed:
        start_study(setup, plan, args.resume)
    rows = {(row["trial"], row["seed"], row["k"]): row for row in kept}
    order = [(run["trial"], run["seed"], run["k"]) for run in runs]
    failed = []
    for number, run in enumerate(to_run, start=1):
        print("\n" + "#" * 80)
        print(f"# Run {number} of {len(to_run)}: trial {run['trial']}, seed {run['seed']}, {run['k']}")
        print("#" * 80, flush=True)
        row = run_one(setup, run, launcher)
        rows[(run["trial"], run["seed"], run["k"])] = row
        if row["status"] != "completed":
            failed.append(row)
        write_outputs(setup, plan, [rows[key] for key in order if key in rows])
    ranking = write_outputs(setup, plan, [rows[key] for key in order if key in rows])
    print("\n" + "=" * 80)
    print(f"TUNING RUNS: {len(to_run) - len(failed)} completed, {len(failed)} failed "
          f"({len(runs) - len(to_run)} kept from before)")
    for row in failed:
        print(f"{ERROR_PREFIX} Trial {row['trial']}, seed {row['seed']}, {row['k']} failed: {row['reason']}")
    if ranking and ranking[0]["rank"] == 1:
        best = ranking[0]
        print(f"Rank 1: trial {best['trial']}, mean {setup['metric']} {best['mean_' + setup['metric']]:.4f}; "
              f"values in {setup['study_dir'] / BEST_FILE}")
    else:
        print("No trial has a rank (no trial has every run completed).")
    print(f"Study folder: {setup['study_dir']}")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except (run_pipeline.RunConfigError, run_pipeline.DatasetConfigError, ValueError, FileNotFoundError) as error:
        sys.exit(f"{ERROR_PREFIX} {error}")
    sys.exit(exit_code)
