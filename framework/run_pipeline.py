"""
PHANTOM VFL RUNNER: RUN THE GENERATION STAGES FROM ONE run_config.json

run_config.json is the one place for the settings of a run. The runner checks
it, then runs these stages for each seed, in this order:
  1. split_train_test.py               train-test split, integer maps, test set
  2. generate_ground_truth_dataset.py  ground truth (integer form and model form)
  3. generate_phantom_datasets.py      phantom datasets (k* and the imposed k levels)

After each stage it shows the summary of the stage and waits for Enter (the
confirmation gate). Type q to stop. --yes runs without the gates (for a campaign
of many seeds). --resume passes through to stage 3.

The screen output of the run also goes to a log file and a warnings file in
<data folder>/logs/ (run_log.py).

Relative paths in run_config.json are relative to the folder of run_config.json.

run_config.json (schema_version 1):
  {
    "schema_version": 1,
    "generation": {
      "dataset_config": "datasets/<name>/dataset.json",         required
      "data_dir": "data/<name>",                                 required
      "seeds": [42],                                             required
      "label_sharing": true,                                     required (no default)
      "k_levels": ["kstar", 5, 10, 15, 20, 25, 30, 35, 40],      optional
      "resume": false,                                           optional
      "kstar_folder": "k0",                                      optional
      "cramers_v_yates": false,                                  optional
      "phase2_trials": 1                                         optional
    },
    "data_preparation": {                                        optional section
      "test_share": 0.2,                                         optional
      "validation_share": 0.0,                                   optional
      "min_level_share": 0.01,                                   optional
      "scaler_columns": "continuous_ordinal"                     optional
    }
  }
scaler_columns gives the columns that the scalers of stage 2 change:
"continuous_ordinal" (the continuous and ordinal columns; binary and one-hot
columns pass through) or "all" (every column). The data files do not depend
on it; only the scaler files and the scaling text of config.json do.
cramers_v_yates: true gives the Yates correction of a 2 x 2 table in the
Cramer's V check of Phase 2, as in the scripts of the published results;
false (default) gives no correction.
phase2_trials: the number of random removals that Phase 2 tests for each
budget, in the marginal and the bivariate search (an odd number; majority
vote). 1 gives the files of the scripts of the published results.
The shares are whole percentages; the training share is the rest. The split
folder names come from the shares: split_80_20 and split_80_20_onehot with the
defaults, split_60_20_20 and split_60_20_20_onehot with a validation share of
0.2. Before stage 1, the runner checks every seed for k folders made with the
other label-sharing setting and stops before it writes a file.
The training section is for framework/training/run_training.py and the tuning
section for framework/training/tune.py; this runner accepts them and does not
read them. Later work adds its own sections (analysis).

Usage:
    python run_pipeline.py --config run_config.json [--yes] [--resume]
"""

import argparse
import json
import sys
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parent
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

import data_io  # noqa: E402
import generate_ground_truth_dataset as ground_truth_stage  # noqa: E402
import generate_phantom_datasets as phantom_stage  # noqa: E402
import split_train_test as split_stage  # noqa: E402
from dataset_config import DatasetConfigError, load_dataset_config  # noqa: E402
from run_log import run_log  # noqa: E402

RUN_CONFIG_SCHEMA_VERSION = 1
SECTIONS = ("generation", "data_preparation", "training", "tuning", "analysis")  # training: read and checked by framework/training/run_training.py; tuning: by framework/training/tune.py; analysis: by framework/analysis/settings.py
GENERATION_REQUIRED = ("dataset_config", "data_dir", "seeds", "label_sharing")
GENERATION_DEFAULTS = {
    "k_levels": list(phantom_stage.DEFAULT_K_LEVELS),
    "resume": False,
    "kstar_folder": phantom_stage.DEFAULT_KSTAR_FOLDER,
    "cramers_v_yates": False,
    "phase2_trials": phantom_stage.DEFAULT_PHASE2_TRIALS,
}
# Keys that are not settings: the split folder names come from the shares
GENERATION_REMOVED = {
    "split_folder": "the split folder name comes from data_preparation.test_share and "
                    "data_preparation.validation_share (for example split_80_20)",
    "model_split_folder": "the model-form split folder name comes from data_preparation.test_share and "
                          "data_preparation.validation_share (for example split_80_20_onehot)",
}
DATA_PREPARATION_DEFAULTS = {
    "test_share": data_io.DEFAULT_TEST_SHARE,
    "validation_share": data_io.DEFAULT_VALIDATION_SHARE,
    "min_level_share": split_stage.DEFAULT_MIN_LEVEL_SHARE,
    "scaler_columns": ground_truth_stage.DEFAULT_SCALER_COLUMNS,
}
STOP_ANSWERS = ("q", "quit", "stop")


class RunConfigError(ValueError):
    """run_config.json has a missing field or a wrong value."""


def _is_folder_name(value):
    return isinstance(value, str) and value and "/" not in value and "\\" not in value


def check_generation(gen):
    """Return a list of error messages for the generation section (empty if valid)."""
    if not isinstance(gen, dict):
        return ["generation must be an object"]
    errors = []
    for key in GENERATION_REQUIRED:
        if key not in gen:
            errors.append(f"generation.{key} is required")
    for key in sorted(set(gen) & set(GENERATION_REMOVED)):
        errors.append(f"generation.{key} is not a setting: {GENERATION_REMOVED[key]}. Remove this key")
    unknown = set(gen) - set(GENERATION_REQUIRED) - set(GENERATION_DEFAULTS) - set(GENERATION_REMOVED)
    if unknown:
        errors.append(f"generation: unknown keys {sorted(unknown)}")
    for key in ("dataset_config", "data_dir"):
        if key in gen and (not isinstance(gen[key], str) or not gen[key]):
            errors.append(f"generation.{key} must be a non-empty text")
    seeds = gen.get("seeds")
    if "seeds" in gen and (
        not isinstance(seeds, list) or not seeds
        or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)
    ):
        errors.append("generation.seeds must be a non-empty list of whole numbers (0 or more)")
    elif isinstance(seeds, list) and len(set(seeds)) != len(seeds):
        errors.append("generation.seeds has a repeated seed")
    for key in ("label_sharing", "resume", "cramers_v_yates"):
        if key in gen and not isinstance(gen[key], bool):
            errors.append(f"generation.{key} must be true or false")
    if "kstar_folder" in gen and not _is_folder_name(gen["kstar_folder"]):
        errors.append("generation.kstar_folder must be a folder name")
    if "phase2_trials" in gen:
        try:
            phantom_stage.check_phase2_trials(gen["phase2_trials"])
        except ValueError as error:
            errors.append(f"generation.{error}")
    settings = {**GENERATION_DEFAULTS, **gen}
    if not isinstance(settings["k_levels"], list):
        errors.append("generation.k_levels must be a list")
    elif _is_folder_name(settings["kstar_folder"]):
        try:
            phantom_stage.k_levels_with_folders(settings["k_levels"], settings["kstar_folder"])
        except ValueError as error:
            errors.append(f"generation.k_levels: {error}")
    return errors


def check_data_preparation(prep):
    """Return a list of error messages for the data_preparation section (empty if valid)."""
    if not isinstance(prep, dict):
        return ["data_preparation must be an object"]
    errors = []
    unknown = set(prep) - set(DATA_PREPARATION_DEFAULTS)
    if unknown:
        errors.append(f"data_preparation: unknown keys {sorted(unknown)}")
    settings = {**DATA_PREPARATION_DEFAULTS, **prep}
    try:
        data_io.split_percentages(settings["test_share"], settings["validation_share"])
    except ValueError as error:
        errors.append(f"data_preparation: {error}")
    try:
        split_stage.check_min_level_share(settings["min_level_share"])
    except ValueError as error:
        errors.append(f"data_preparation: {error}")
    try:
        ground_truth_stage.check_scaler_columns(settings["scaler_columns"])
    except ValueError as error:
        errors.append(f"data_preparation: {error}")
    return errors


def load_run_config(path):
    """
    Read and check run_config.json. Stop with a list of all faults.

    Returns:
        the generation settings with defaults (dataset_config and data_dir
        resolved against the folder of run_config.json), and under the key
        data_preparation the data_preparation settings with defaults
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RunConfigError(f"Run config {path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise RunConfigError(f"Run config {path} must be a JSON object")
    errors = []
    if data.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RUN_CONFIG_SCHEMA_VERSION}")
    unknown = set(data) - {"schema_version", *SECTIONS}
    if unknown:
        errors.append(f"unknown sections {sorted(unknown)} (known: {list(SECTIONS)})")
    if "generation" not in data:
        errors.append("the generation section is required")
    else:
        errors.extend(check_generation(data["generation"]))
    errors.extend(check_data_preparation(data.get("data_preparation", {})))
    if errors:
        raise RunConfigError(f"Run config {path} is not valid:\n  - " + "\n  - ".join(errors))

    gen = {**GENERATION_DEFAULTS, **data["generation"]}
    gen["data_preparation"] = {**DATA_PREPARATION_DEFAULTS, **data.get("data_preparation", {})}
    base = path.resolve().parent
    for key in ("dataset_config", "data_dir"):
        gen[key] = str(base / gen[key])
    load_dataset_config(gen["dataset_config"])  # stops early on a wrong dataset config
    return gen


def stages(gen, seed):
    """[(title, run function, settings)] of the stages for one seed, in order."""
    prep = gen["data_preparation"]
    shares = {"test_share": prep["test_share"], "validation_share": prep["validation_share"]}
    return [
        ("Stage 1: train-test split", split_stage.run, split_stage.SplitSettings(
            dataset_config=gen["dataset_config"], data_dir=gen["data_dir"], seed=seed,
            min_level_share=prep["min_level_share"], **shares)),
        ("Stage 2: ground truth", ground_truth_stage.run, ground_truth_stage.GroundTruthSettings(
            dataset_config=gen["dataset_config"], data_dir=gen["data_dir"], seed=seed,
            scaler_columns=prep["scaler_columns"], **shares)),
        ("Stage 3: phantom datasets", phantom_stage.run, phantom_stage.GenerationSettings(
            dataset_config=gen["dataset_config"], data_dir=gen["data_dir"], seed=seed,
            label_sharing=gen["label_sharing"], k_levels=list(gen["k_levels"]),
            kstar_folder=gen["kstar_folder"], resume=gen["resume"],
            cramers_v_yates=gen["cramers_v_yates"], phase2_trials=gen["phase2_trials"], **shares)),
    ]


def show_summary(title, summary):
    print("\n" + "#" * 80)
    print(f"# SUMMARY - {title}")
    print("#" * 80)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("#" * 80)


def confirm(next_title):
    """Confirmation gate. Returns False if the user stops the run."""
    try:
        answer = input(f"\nPress Enter to continue with '{next_title}', or type q to stop: ")
    except EOFError:
        print("\nNo input is available. Use --yes to run without the confirmation gates.")
        return False
    return answer.strip().lower() not in STOP_ANSWERS


def run_pipeline(config_path, yes=False, resume=False):
    """
    Run the generation stages for every seed of run_config.json.

    Returns:
        {"completed": bool, "summaries": [(seed, title, summary)], "stopped_before": title or None}
    """
    gen = load_run_config(config_path)
    if resume:
        gen["resume"] = True
    with run_log(gen["data_dir"], "run"):
        return _run_stages(gen, config_path, yes)


def _run_stages(gen, config_path, yes):
    """Run the stages of every seed (called by run_pipeline inside the run log)."""
    print("=" * 80)
    print("PHANTOM VFL RUNNER")
    print("=" * 80)
    print(f"Run config: {config_path}")
    for key in ("dataset_config", "data_dir", "seeds", "label_sharing", "k_levels", "resume", "kstar_folder",
                "cramers_v_yates", "phase2_trials"):
        print(f"  {key}: {gen[key]}")
    prep = gen["data_preparation"]
    for key in ("test_share", "validation_share", "min_level_share", "scaler_columns"):
        print(f"  data_preparation.{key}: {prep[key]}")
    split_folder = data_io.split_folder_name(prep["test_share"], prep["validation_share"])
    print(f"  split folders: {split_folder}, "
          f"{data_io.split_folder_name(prep['test_share'], prep['validation_share'], model=True)}")
    print(f"  confirmation gates: {'off (--yes)' if yes else 'on'}")

    # One data folder for each label-sharing setting: check
    # every seed before stage 1, so that a wrong data folder stops the run
    # before any file is written (stage 3 checks again and warns)
    for seed in gen["seeds"]:
        phantom_stage.check_label_sharing_of_k_folders(
            data_io.federated_seed_dir(gen["data_dir"], seed, split_folder),
            gen["label_sharing"], gen["kstar_folder"], warn=False)

    plan = [(seed, title, function, settings)
            for seed in gen["seeds"] for title, function, settings in stages(gen, seed)]
    summaries = []
    for position, (seed, title, function, settings) in enumerate(plan):
        summary = function(settings)
        summaries.append((seed, title, summary))
        show_summary(f"{title}, seed {seed}", summary)
        if position + 1 < len(plan) and not yes:
            next_seed, next_title = plan[position + 1][0], plan[position + 1][1]
            if not confirm(f"{next_title}, seed {next_seed}"):
                print(f"\nStopped by the user before '{next_title}, seed {next_seed}'.")
                return {"completed": False, "summaries": summaries,
                        "stopped_before": f"{next_title}, seed {next_seed}"}

    print("\n" + "=" * 80)
    print(f"ALL STAGES COMPLETED ({len(gen['seeds'])} seed(s))")
    print("=" * 80)
    return {"completed": True, "summaries": summaries, "stopped_before": None}


def main(argv=None):
    """Command line of the runner."""
    parser = argparse.ArgumentParser(description="Run the Phantom VFL generation stages from run_config.json")
    parser.add_argument("--config", required=True, help="Path of run_config.json")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation gates")
    parser.add_argument("--resume", action="store_true",
                        help="Stage 3 skips each k level whose completion marker matches this run")
    args = parser.parse_args(argv)
    return run_pipeline(args.config, yes=args.yes, resume=args.resume)


if __name__ == "__main__":
    try:
        result = main()
    except (RunConfigError, DatasetConfigError, ValueError, FileNotFoundError) as error:
        sys.exit(f"ERROR: {error}")
    sys.exit(0 if result["completed"] else 2)
