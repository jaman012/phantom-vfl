r"""
Training runner: the Flower runs of one run_config.json (James Mansueto's own code).

The runner reads the sections generation, data_preparation and training of
run_config.json, makes the plan (each seed with each k level, in this order)
and starts each run with the launcher framework/training/run_flower.py
(run_flower.main). Each run has its own run log in <data-dir>/logs/ (the
launcher writes it). The runner writes a summary file of the runs, again after
each run:

  <data-dir>/logs/training_<date>-<time>_summary.csv
      run, seed, k, results_tag, exit_code, status, reason, run_log
      (status: completed, failed, or kept for a run that --resume keeps)

and keeps the state of the runs of one results tag and split folder:

  <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/training_settings.json
      the settings that every run of the folder shares (no seeds, no k levels)
  <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/training_runs.csv
      seed, k, status, exit_code, reason, last_round, test_metrics_md5, run_log
      (one row for each seed and k level; written again after each run)

The training section is optional, and every key in it is optional:

  "training": {
    "seeds": [42],                               default: generation.seeds
    "k_levels": ["ground_truth", "kstar", 20],   default: "ground_truth", then generation.k_levels
    "feature_form": "model",                     "model" (one-hot form) or "integer"; default "model"
    "run_config": {"loss-type": "focal", "num-server-rounds": 250}
  }

k_levels: "ground_truth", "kstar" (the folder generation.kstar_folder) or a
whole percentage from 1 to 99 (the folder k<p>).
run_config: keys of [tool.flwr.app.config] in flower_app/pyproject.toml, with
the same names and the same value types (a whole number is accepted for a
decimal value). A key that is not given keeps its default in pyproject.toml.
The runner sets five keys itself, so run_config must not have them: data-dir
and dataset-config (from generation), phantom-dataset and random-seed (from
the plan), and feature-splits (the number of features of each client in
dataset.json). The split folder (SPLIT_RATIO) comes from
data_preparation.test_share and data_preparation.validation_share, with
_onehot for the model form.

Rules:
- The runner checks the whole run_config.json before the first run and stops
  with a list of all faults. It reads each --run-config text back with
  Flower's reader and stops if a value differs from the plan.
- After the plan, the runner waits for Enter (the confirmation gate); type q to
  stop. --yes runs without the gate.
- A run is completed when the launcher returns 0 and its test-metrics file has
  a last round equal to num-server-rounds. A failed run does not stop the
  runner. At the end the runner lists the failed runs. Exit code: 0 when every
  started run completed, 1 when a run failed, 2 when the user stopped at the
  gate. Ctrl-C stops the runner; the run that was interrupted has no row in
  training_runs.csv.
- The first start for a results tag and split folder writes
  training_settings.json (after the gate). When this file exists, a start
  without --resume stops, so that a new start does not overwrite the results.
- With --resume, the runner stops when the settings differ from
  training_settings.json (it lists the keys), keeps each run of the plan that
  training_runs.csv shows as completed when its test-metrics file still has
  the recorded MD5, and starts every other run of the plan. Rows of seeds and
  k levels that are not in the plan stay in training_runs.csv, so the seeds
  can run in batches. With --resume and no state file, every run starts.
- A run that starts overwrites the result files of an earlier run with the
  same results tag, split, seed and k level.

Usage (with the Python of the environment that has flwr):

  python <repository root>/framework/training/run_training.py --config run_config.json [--yes] [--resume]
"""

import argparse
import csv
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
if __package__ in (None, ""):  # started as a file: make the package 'framework' importable
    sys.path.insert(0, str(ROOT))

from framework import run_pipeline  # noqa: E402
from framework.training import results, run_flower  # noqa: E402

SECTION = "training"
TRAINING_KEYS = ("seeds", "k_levels", "feature_form", "run_config")
FEATURE_FORMS = ("model", "integer")
DEFAULT_FEATURE_FORM = "model"
GROUND_TRUTH = run_pipeline.data_io.GROUND_TRUTH_FOLDER
RESERVED_KEYS = ("data-dir", "dataset-config", "phantom-dataset", "random-seed", "feature-splits")
SUMMARY_NAME = "training"
SUMMARY_COLUMNS = ("run", "seed", "k", "results_tag", "exit_code", "status", "reason", "run_log")
STATE_SETTINGS_FILE = "training_settings.json"
STATE_RUNS_FILE = "training_runs.csv"
STATE_COLUMNS = ("seed", "k", "status", "exit_code", "reason", "last_round", "test_metrics_md5", "run_log")
STATE_SCHEMA = 1
ERROR_PREFIX = run_flower.run_log.ERROR_PREFIX


def app_defaults(app_dir=run_flower.APP_DIR):
    """[tool.flwr.app.config] of flower_app/pyproject.toml: the keys of the app and their default values."""
    with open(Path(app_dir) / "pyproject.toml", "rb") as file:
        return dict(run_flower.tomllib.load(file)["tool"]["flwr"]["app"]["config"])


def _same_type(value, default):
    """True when the value has the type of the default (a whole number is accepted for a decimal default)."""
    if isinstance(value, bool) or isinstance(default, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float))
    return type(value) is type(default)


def k_folders(k_levels, kstar_folder):
    """
    Folder names of the training k levels, in the given order: 'ground_truth',
    the k* folder for 'kstar', k<p> for a whole percentage p. Stop on a wrong
    value or on two levels with the same folder.
    """
    if not isinstance(k_levels, list) or not k_levels:
        raise ValueError("must be a non-empty list")
    folders = []
    for level in k_levels:
        if isinstance(level, str) and level.strip().lower() == GROUND_TRUTH:
            folders.append(GROUND_TRUTH)
        else:
            folders += [folder for _, folder, _ in
                        run_pipeline.phantom_stage.k_levels_with_folders([level], kstar_folder)]
    duplicates = sorted({folder for folder in folders if folders.count(folder) > 1})
    if duplicates:
        raise ValueError(f"two k levels use the same folder: {duplicates}")
    return folders


def check_training(training, gen, defaults):
    """Return a list of error messages for the training section (empty if valid)."""
    if not isinstance(training, dict):
        return [f"{SECTION} must be an object"]
    errors = []
    unknown = set(training) - set(TRAINING_KEYS)
    if unknown:
        errors.append(f"{SECTION}: unknown keys {sorted(unknown)}")
    seeds = training.get("seeds", gen["seeds"])
    if not isinstance(seeds, list) or not seeds or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds):
        errors.append(f"{SECTION}.seeds must be a non-empty list of whole numbers (0 or more)")
    elif len(set(seeds)) != len(seeds):
        errors.append(f"{SECTION}.seeds has a repeated seed")
    if "k_levels" in training:
        try:
            k_folders(training["k_levels"], gen["kstar_folder"])
        except ValueError as error:
            errors.append(f"{SECTION}.k_levels: {error}")
    if training.get("feature_form", DEFAULT_FEATURE_FORM) not in FEATURE_FORMS:
        errors.append(f"{SECTION}.feature_form must be one of {list(FEATURE_FORMS)}")
    flower = training.get("run_config", {})
    if not isinstance(flower, dict):
        errors.append(f"{SECTION}.run_config must be an object")
        flower = {}
    for key, value in flower.items():
        if key in RESERVED_KEYS:
            errors.append(f"{SECTION}.run_config.{key}: the runner sets this key. Remove it")
        elif key not in defaults:
            errors.append(f"{SECTION}.run_config.{key} is not a key of flower_app/pyproject.toml "
                          f"(keys: {sorted(set(defaults) - set(RESERVED_KEYS))})")
        elif not _same_type(value, defaults[key]):
            errors.append(f"{SECTION}.run_config.{key} must have the type of its default "
                          f"{defaults[key]!r} ({type(defaults[key]).__name__})")
    if not Path(gen["data_dir"]).is_dir():
        errors.append(f"generation.data_dir does not exist: {gen['data_dir']}")
    return errors


def load_training(config_path):
    """
    Read and check run_config.json for the training runs. Stop with a list of all faults.

    Returns:
        dict: gen (the generation settings of run_pipeline.load_run_config), seeds,
        k_folders, feature_form, split_ratio, feature_splits, run_config (the
        values of the training section) and defaults (of pyproject.toml)
    """
    gen = run_pipeline.load_run_config(config_path)
    training = json.loads(Path(config_path).read_text(encoding="utf-8")).get(SECTION, {})
    defaults = app_defaults()
    errors = check_training(training, gen, defaults)
    if errors:
        raise run_pipeline.RunConfigError(f"Run config {config_path} is not valid:\n  - " + "\n  - ".join(errors))
    prep = gen["data_preparation"]
    form = training.get("feature_form", DEFAULT_FEATURE_FORM)
    split_folder = run_pipeline.data_io.split_folder_name(
        prep["test_share"], prep["validation_share"], model=(form == "model"))
    cfg = run_pipeline.load_dataset_config(gen["dataset_config"])
    flower = {key: float(value) if isinstance(defaults[key], float) else value
              for key, value in training.get("run_config", {}).items()}
    return {
        "gen": gen,
        "seeds": list(training.get("seeds", gen["seeds"])),
        "k_folders": k_folders(training.get("k_levels", [GROUND_TRUTH] + list(gen["k_levels"])),
                               gen["kstar_folder"]),
        "feature_form": form,
        "split_ratio": split_folder[len("split_"):],
        "feature_splits": ",".join(str(len(client)) for client in cfg.clients),
        "run_config": flower,
        "defaults": defaults,
    }


def toml_text(value):
    """A value as TOML text for --run-config (a text in quotes)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value and "\\" not in value:
        return f'"{value}"'
    raise ValueError(f"The text {value!r} cannot go into --run-config (it has a quote and another quote or a backslash)")


def run_values(setup, seed, k):
    """The run-config values of one run: the five keys of the runner, then the training section."""
    gen = setup["gen"]
    return {"data-dir": gen["data_dir"], "dataset-config": gen["dataset_config"], "phantom-dataset": k,
            "random-seed": seed, "feature-splits": setup["feature_splits"], **setup["run_config"]}


def run_config_text(values):
    """The --run-config text of the values ('key=value' pairs, separated by spaces)."""
    return " ".join(f"{key}={toml_text(value)}" for key, value in values.items())


def check_text(text, values):
    """Read the text back with Flower's reader; stop when it does not give the values."""
    try:
        from flwr.common.config import parse_config_args
    except ImportError as error:
        raise SystemExit(
            f"{ERROR_PREFIX} flwr.common.config.parse_config_args was not found ({error}). "
            "Start the runner with the Python of the environment that has flwr."
        ) from error
    read = parse_config_args([text])
    if read != values:
        raise run_pipeline.RunConfigError(
            f"The --run-config text does not give the planned values.\n  text: {text}\n"
            f"  read: {read}\n  planned: {values}")


def make_plan(setup):
    """The runs: each seed with each k level, with the arguments of the launcher."""
    runs = []
    for seed in setup["seeds"]:
        for k in setup["k_folders"]:
            values = run_values(setup, seed, k)
            text = run_config_text(values)
            check_text(text, values)
            runs.append({"seed": seed, "k": k, "values": values,
                         "argv": ["--split-ratio", setup["split_ratio"], "--run-config", text]})
    return runs


def show_plan(config_path, setup, runs, yes, kept=0):
    gen = setup["gen"]
    print("=" * 80)
    print("PHANTOM VFL TRAINING RUNNER")
    print("=" * 80)
    print(f"Run config: {config_path}")
    print(f"  data_dir: {gen['data_dir']}")
    print(f"  dataset_config: {gen['dataset_config']}")
    print(f"  seeds: {setup['seeds']}")
    print(f"  k levels (folders): {setup['k_folders']}")
    print(f"  feature form: {setup['feature_form']} (SPLIT_RATIO={setup['split_ratio']})")
    print(f"  feature-splits: {setup['feature_splits']}")
    print(f"  {SECTION}.run_config: {setup['run_config'] or '(none: the defaults of flower_app/pyproject.toml)'}")
    print(f"  runs: {len(runs)} planned, {kept} kept from before, {len(runs) - kept} to start")
    print(f"  confirmation gate: {'off (--yes)' if yes else 'on'}")
    if runs:
        print(f"  first run: --split-ratio {setup['split_ratio']} --run-config \"{runs[0]['argv'][-1]}\"")


def file_md5(path):
    """MD5 of a file."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def run_results_tag(setup):
    """The results tag of the runs: results-tag, or the loss type when it is empty (framework.training.results)."""
    settings = {**setup["defaults"], **setup["run_config"]}
    return results.results_tag(settings.get("results-tag", ""), settings["loss-type"])


def state_folder(setup):
    """The folder of the state files: <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/."""
    return (Path(setup["gen"]["data_dir"]) / results.RESULTS_FOLDER / run_results_tag(setup)
            / f"split_{setup['split_ratio']}")


def test_metrics_path(setup, seed, k):
    """The test-metrics file of one run."""
    folder = results.results_dir(setup["gen"]["data_dir"], run_results_tag(setup), setup["split_ratio"], seed, k)
    return folder / results.test_metrics_file_name(k, seed)


def last_round(path):
    """The highest value in the column round of a test-metrics file. Raises ValueError without the column or rows."""
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if "round" not in (reader.fieldnames or []):
            raise ValueError(f"no column round in {Path(path).name}")
        rounds = [int(row["round"]) for row in reader]
    if not rounds:
        raise ValueError(f"no row in {Path(path).name}")
    return max(rounds)


def settings_record(setup):
    """The settings that every run of the state folder shares (training_settings.json; no seeds, no k levels)."""
    record = {
        "schema": STATE_SCHEMA,
        "results_tag": run_results_tag(setup),
        "split_ratio": setup["split_ratio"],
        "feature_form": setup["feature_form"],
        "feature_splits": setup["feature_splits"],
        "dataset_config_md5": file_md5(setup["gen"]["dataset_config"]),
        "run_config": setup["run_config"],
        "app_defaults": setup["defaults"],
    }
    return json.loads(json.dumps(record))


def read_state_runs(folder):
    """The rows of training_runs.csv as {(seed, k): row}, in the order of the file; empty without the file."""
    path = Path(folder) / STATE_RUNS_FILE
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as file:
        return {(int(row["seed"]), row["k"]): row for row in csv.DictReader(file)}


def write_state_runs(folder, rows):
    """Write training_runs.csv again (all rows, in their order)."""
    with open(Path(folder) / STATE_RUNS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())


def start_state(setup, runs, resume):
    """
    Read the state of the state folder before the gate. Without --resume, stop
    when training_settings.json exists. With --resume, stop when the settings
    differ, and keep each run of the plan that training_runs.csv shows as
    completed when its test-metrics file still has the recorded MD5.

    Returns:
        dict: folder, record (the settings), existed (training_settings.json
        exists), rows (the rows of training_runs.csv) and kept ({(seed, k): row})
    """
    folder = state_folder(setup)
    record = settings_record(setup)
    settings_path = folder / STATE_SETTINGS_FILE
    state = {"folder": folder, "record": record, "existed": settings_path.exists(), "rows": {}, "kept": {}}
    if not state["existed"]:
        if resume:
            print(f"--resume: no state file in {folder}; every run starts")
        return state
    if not resume:
        raise run_pipeline.RunConfigError(
            f"The state file exists: {settings_path}. Use --resume to continue these runs, "
            "or choose another results-tag.")
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    if saved != record:
        changed = sorted(key for key in set(saved) | set(record) if saved.get(key) != record.get(key))
        raise run_pipeline.RunConfigError(
            f"--resume: the settings differ from {settings_path} (keys {changed}). "
            "Use the settings of these runs, or choose another results-tag.")
    state["rows"] = read_state_runs(folder)
    planned = {(run["seed"], run["k"]) for run in runs}
    for key, row in state["rows"].items():
        if key not in planned or row["status"] != "completed":
            continue
        path = test_metrics_path(setup, *key)
        if path.exists() and file_md5(path) == row["test_metrics_md5"]:
            state["kept"][key] = row
        else:
            print(f"--resume: seed {key[0]}, {key[1]}: the test-metrics file is missing or changed; "
                  "the run starts again")
    return state


def write_settings(state):
    """Write training_settings.json at the first start (after the gate)."""
    if not state["existed"]:
        state["folder"].mkdir(parents=True, exist_ok=True)
        (state["folder"] / STATE_SETTINGS_FILE).write_text(
            json.dumps(state["record"], indent=2) + "\n", encoding="utf-8")
        state["existed"] = True


def new_summary_path(data_dir):
    """A new summary file name in <data-dir>/logs/ (a number is added if the name exists)."""
    logs_dir = Path(data_dir) / run_flower.run_log.LOGS_FOLDER
    logs_dir.mkdir(exist_ok=True)
    base = f"{SUMMARY_NAME}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    candidate, number = base, 1
    while (logs_dir / f"{candidate}_summary.csv").exists():
        number += 1
        candidate = f"{base}_{number}"
    return logs_dir / f"{candidate}_summary.csv"


def write_summary(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def find_run_log(settings, start):
    """The run log that the launcher wrote for the run (the newest one since start); '' if none."""
    logs_dir = Path(settings["data-dir"]) / run_flower.run_log.LOGS_FOLDER
    found = [path for path in logs_dir.glob(f"{run_flower.log_name(settings)}_*.log")
             if path.stat().st_mtime >= start - 1]
    return str(max(found, key=lambda path: path.stat().st_mtime)) if found else ""


def run_all(runs, setup, launcher, summary_path, state=None):
    """
    Start each run with the launcher; write the summary after each run. Return
    the summary rows. state (from start_state) holds the rows of
    training_runs.csv and the kept runs: a kept run does not start and gets the
    status 'kept'; after each started run, training_runs.csv is written again.
    Without state, no state file is written.
    """
    state = state or {"folder": None, "rows": {}, "kept": {}}
    to_start = sum(1 for run in runs if (run["seed"], run["k"]) not in state["kept"])
    rows = []
    started = 0
    for number, run in enumerate(runs, start=1):
        settings = {**setup["defaults"], **run["values"]}
        tag = results.results_tag(settings.get("results-tag", ""), settings["loss-type"])
        key = (run["seed"], run["k"])
        if key in state["kept"]:
            rows.append({"run": number, "seed": run["seed"], "k": run["k"], "results_tag": tag, "exit_code": "",
                         "status": "kept", "reason": "", "run_log": state["kept"][key].get("run_log", "")})
            write_summary(summary_path, rows)
            continue
        started += 1
        print("\n" + "#" * 80)
        print(f"# Run {started} of {to_start}: seed {run['seed']}, {run['k']}")
        print("#" * 80, flush=True)
        start = time.time()
        reason = ""
        try:
            code = launcher(run["argv"])
        except KeyboardInterrupt:
            raise
        except BaseException as error:  # an error of one run does not stop the other runs
            code = 1
            reason = f"{type(error).__name__}: {error}"
            print(f"{ERROR_PREFIX} Run {number} stopped with an error: {reason}", flush=True)
        last = ""
        path = test_metrics_path(setup, run["seed"], run["k"])
        if code == 0:
            try:
                last = last_round(path)
                if last != settings["num-server-rounds"]:
                    reason = f"last round of {path.name} is {last}, expected {settings['num-server-rounds']}"
            except (OSError, ValueError) as error:
                reason = f"test-metrics file: {error}"
            if reason:
                print(f"{ERROR_PREFIX} Run {number}: {reason}", flush=True)
        status = "completed" if code == 0 and not reason else "failed"
        if status == "failed" and not reason:
            reason = "see the run log"
        run_log = find_run_log(settings, start)
        rows.append({"run": number, "seed": run["seed"], "k": run["k"], "results_tag": tag, "exit_code": code,
                     "status": status, "reason": reason, "run_log": run_log})
        write_summary(summary_path, rows)
        if state["folder"] is not None:
            state["rows"][key] = {"seed": run["seed"], "k": run["k"], "status": status, "exit_code": code,
                                  "reason": reason, "last_round": last,
                                  "test_metrics_md5": file_md5(path) if status == "completed" else "",
                                  "run_log": run_log}
            write_state_runs(state["folder"], state["rows"])
    return rows


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Start the Flower runs of run_config.json (each seed with each k level).")
    parser.add_argument("--config", required=True, help="Path of run_config.json")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation gate")
    parser.add_argument("--resume", action="store_true",
                        help="Keep the completed runs of training_runs.csv and start the other runs of the plan")
    return parser.parse_args(argv)


def main(argv=None, launcher=None):
    """
    Start the Flower runs of run_config.json. Return 0 when every started run
    completed, 1 when a run failed, 2 when the user stopped at the confirmation
    gate. launcher replaces run_flower.main (for tests only).
    """
    args = parse_arguments(argv)
    launcher = launcher or run_flower.main
    setup = load_training(args.config)
    runs = make_plan(setup)
    state = start_state(setup, runs, args.resume)
    kept = len(state["kept"])
    show_plan(args.config, setup, runs, args.yes, kept=kept)
    if not args.yes and not run_pipeline.confirm(f"{len(runs) - kept} Flower run(s)"):
        print("\nStopped by the user before the first run.")
        return 2
    write_settings(state)
    summary_path = new_summary_path(setup["gen"]["data_dir"])
    rows = run_all(runs, setup, launcher, summary_path, state)
    failed = [row for row in rows if row["status"] == "failed"]
    print("\n" + "=" * 80)
    print(f"TRAINING RUNS: {len(rows) - len(failed) - kept} completed, {len(failed)} failed "
          f"({kept} kept from before)")
    for row in failed:
        print(f"{ERROR_PREFIX} Run {row['run']} (seed {row['seed']}, {row['k']}) failed: {row['reason']}")
    print(f"State: {state['folder']}")
    print(f"Summary: {summary_path}")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except (run_pipeline.RunConfigError, run_pipeline.DatasetConfigError, ValueError, FileNotFoundError) as error:
        sys.exit(f"{ERROR_PREFIX} {error}")
    sys.exit(exit_code)
