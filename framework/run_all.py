r"""
Orchestrator: the whole chain of one or more run configs with one command.

Two modes:

  python framework/run_all.py --new
      asks questions, writes a run_config.json and can start it

  python framework/run_all.py --config <run_config.json> [--config <...> ...]
                              [--summary <summary_config.json>] [--steps <list>]
                              [--yes] [--resume] [--overwrite]
      runs the chain of each run config in the given order:
        prepare     the example prepare script of the dataset, only when the
                    cleaned CSV is missing (datasets/<name>/prepare.py)
        generation  framework/run_pipeline.py (stages 1 to 3)
        training    framework/training/run_training.py (the Flower runs)
        analysis    framework/analysis/run_analysis.py
      and then, with --summary, framework/analysis/summarise.py.

Rules:
- The orchestrator reads and checks every run config before it starts a
  step, and stops with a list of all faults.
- It shows the plan of all steps once and waits for Enter (the confirmation
  gate); type q to stop. --yes runs without the gate. Each step runs with
  --yes, so the steps have no gates of their own.
- Each step runs as its own process with the same Python, from the code
  root, and writes its own run log in <data folder>/logs/. The orchestrator
  writes a run log of its own (name run_all) in the logs folder of the data
  folder of the first run config.
- The chain stops at the first step that fails. The orchestrator then prints
  the step, its exit code and the command to continue.
- --resume passes to the generation and the training. --overwrite passes to
  the analysis and the summary.
- The generation runs once for each data folder and generation settings: a
  later run config with the same data folder and the same settings of the
  sections generation and data_preparation skips it.
- The prepare step runs once for each data folder. A missing cleaned CSV
  with no prepare script beside dataset.json is a fault.
- The checks of the framework need an existing data folder. Before the
  checks, the orchestrator makes each missing data folder (an empty folder)
  and names it in the plan.
- The analysis step is skipped when the training plan has fewer seeds than
  analysis.min_complete_seeds. --summary then is a fault.
- Exit code: 0 when every step completed, 1 after a fault or a failed step,
  2 when the user stopped at the confirmation gate.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = ("prepare", "generation", "training", "analysis")
SCRIPTS = {
    "generation": Path("framework") / "run_pipeline.py",
    "training": Path("framework") / "training" / "run_training.py",
    "analysis": Path("framework") / "analysis" / "run_analysis.py",
    "summary": Path("framework") / "analysis" / "summarise.py",
}
PREPARE_SCRIPT = "prepare.py"
EXAMPLES_FOLDER = "examples"
LOG_NAME = "run_all"
STOP_ANSWERS = ("q", "quit", "stop")
WARNING_PREFIX = "[!]"
ERROR_PREFIX = "[ERROR]"


class OrchestratorError(ValueError):
    """A run config, a summary config or a command-line option has a fault."""


# ============================================================================
# READING A RUN CONFIG
# ============================================================================

def inspect_config(config_path, with_analysis):
    """
    Read and check one run config with the checks of the framework. Stop with
    the list of all faults of that run config.

    Args:
        config_path: path of run_config.json
        with_analysis: True to check the section analysis as well (the
            training k levels must then include the ground truth)

    Returns:
        dict: data_dir, dataset_config, dataset_name, cleaned_csv, prepare_script
        (or None), generation_key, label_sharing, seeds, k_folders,
        split_folder, results_tag, training_state_file, analysis_settings_file
        (or None), min_complete_seeds (or None)
    """
    from framework import run_pipeline
    from framework.training import run_training

    if with_analysis:
        from framework.analysis import run_analysis, settings as analysis_settings
        info = analysis_settings.load_analysis(config_path)
        setup = info["setup"]
        settings_file = Path(info["output_folder"]) / run_analysis.SETTINGS_FILE
        min_seeds = info["settings"]["min_complete_seeds"]
    else:
        setup = run_training.load_training(config_path)
        settings_file, min_seeds = None, None
    gen = setup["gen"]
    dataset_config = Path(gen["dataset_config"]).resolve()
    cfg = run_pipeline.load_dataset_config(str(dataset_config))
    prepare_script = dataset_config.parent / PREPARE_SCRIPT
    key_settings = {name: value for name, value in gen.items() if name not in ("resume", "data_dir")}
    return {
        "data_dir": Path(gen["data_dir"]).resolve(),
        "dataset_config": dataset_config,
        "dataset_name": dataset_config.parent.name,
        "cleaned_csv": Path(run_pipeline.data_io.cleaned_csv_path(gen["data_dir"], cfg)).resolve(),
        "prepare_script": prepare_script if prepare_script.is_file() else None,
        "generation_key": json.dumps({"data_dir": str(Path(gen["data_dir"]).resolve()), **key_settings},
                                     sort_keys=True, default=str),
        "label_sharing": gen["label_sharing"],
        "seeds": [int(seed) for seed in setup["seeds"]],
        "k_folders": list(setup["k_folders"]),
        "split_folder": f"split_{setup['split_ratio']}",
        "results_tag": run_training.run_results_tag(setup),
        "training_state_file": run_training.state_folder(setup) / run_training.STATE_SETTINGS_FILE,
        "analysis_settings_file": settings_file,
        "min_complete_seeds": min_seeds,
    }


def make_missing_data_folder(config_path):
    """
    Make the data folder (generation.data_dir) of a run config when it does
    not exist, so that the checks of the framework can run. Return the new
    folder, or None. A run config that cannot be read is left to the checks.
    """
    try:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        text = data["generation"]["data_dir"]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if not isinstance(text, str) or not text:
        return None
    folder = Path(text) if Path(text).is_absolute() else Path(config_path).resolve().parent / text
    if folder.exists():
        return None
    folder.mkdir(parents=True)
    return folder.resolve()


def summary_inputs(summary_path):
    """Check that the summary config exists and is a JSON object. Return its output folder text."""
    path = Path(summary_path)
    if not path.is_file():
        raise OrchestratorError(f"summary config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OrchestratorError(f"summary config {path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise OrchestratorError(f"summary config {path} must be a JSON object")
    return data.get("output_dir", "")


# ============================================================================
# THE PLAN
# ============================================================================

def parse_steps(text):
    """The steps of --steps (a comma-separated list), in the order of the chain."""
    names = [name.strip().lower() for name in text.split(",") if name.strip()]
    unknown = sorted(set(names) - set(STEPS))
    if unknown or not names:
        raise OrchestratorError(f"--steps must be a comma-separated list of {list(STEPS)}; got {text!r}")
    return [name for name in STEPS if name in names]


def _command(script, *arguments):
    return [sys.executable, str(script), *[str(argument) for argument in arguments]]


def make_plan(config_paths, steps, summary, resume, overwrite, inspect=None):
    """
    Read every run config and make the plan. Raise OrchestratorError with
    the list of all faults.

    Returns:
        dict: chains (one for each run config: config, info, actions) and
        summary (the action of the summary, or None). An action is a dict:
        step, run (True or False), command (or None), note.
    """
    inspect = inspect or inspect_config
    errors, chains = [], []
    prepared, generated = {}, {}
    for number, text in enumerate(config_paths, start=1):
        config = Path(text).resolve()
        if not config.is_file():
            errors.append(f"run config {number} not found: {config}")
            continue
        created = make_missing_data_folder(config)
        try:
            info = inspect(config, "analysis" in steps)
        except Exception as error:  # the checks of the framework raise several error types
            errors.append(f"run config {number} ({config}): {error}")
            continue
        actions = []
        for step in steps:
            action = plan_step(step, number, config, info, resume, overwrite, prepared, generated, errors)
            actions.append(action)
        chains.append({"config": config, "info": info, "actions": actions, "created": created})
    summary_action = None
    if summary:
        summary_path = Path(summary).resolve()
        try:
            summary_inputs(summary_path)
        except OrchestratorError as error:
            errors.append(str(error))
        skipped = [chain["config"] for chain in chains for action in chain["actions"]
                   if action["step"] == "analysis" and not action["run"]]
        if skipped:
            errors.append("--summary needs the analysis of every run config; the analysis is skipped for "
                          + ", ".join(str(path) for path in skipped))
        arguments = ["--config", summary_path, "--yes"] + (["--overwrite"] if overwrite else [])
        summary_action = {"step": "summary", "run": True,
                          "command": _command(ROOT / SCRIPTS["summary"], *arguments), "note": str(summary_path)}
    if errors:
        raise OrchestratorError("the plan has faults:\n  - " + "\n  - ".join(errors))
    return {"chains": chains, "summary": summary_action}


def plan_step(step, number, config, info, resume, overwrite, prepared, generated, errors):
    """The action of one step of one run config; faults go to errors."""
    data_dir = str(info["data_dir"])
    if step == "prepare":
        if data_dir in prepared:
            return {"step": step, "run": False, "command": None,
                    "note": f"made by the prepare step of run config {prepared[data_dir]}"}
        if info["cleaned_csv"].is_file():
            return {"step": step, "run": False, "command": None,
                    "note": f"the cleaned CSV exists ({info['cleaned_csv']})"}
        if info["prepare_script"] is None:
            errors.append(f"run config {number}: the cleaned CSV {info['cleaned_csv']} is missing, and the "
                          f"dataset has no {PREPARE_SCRIPT} beside {info['dataset_config'].name}. "
                          f"Make the cleaned CSV first.")
            return {"step": step, "run": False, "command": None, "note": "fault"}
        prepared[data_dir] = number
        return {"step": step, "run": True, "note": f"{info['prepare_script']}",
                "command": _command(info["prepare_script"], "--data-dir", data_dir)}
    if step == "generation":
        key = info["generation_key"]
        if key in generated:
            return {"step": step, "run": False, "command": None,
                    "note": f"same data folder and generation settings as run config {generated[key]}"}
        generated[key] = number
        arguments = ["--config", config, "--yes"] + (["--resume"] if resume else [])
        return {"step": step, "run": True, "command": _command(ROOT / SCRIPTS["generation"], *arguments),
                "note": f"stages 1 to 3, {len(info['seeds'])} seed(s)" + (", --resume" if resume else "")}
    if step == "training":
        if info["training_state_file"].is_file() and not resume:
            errors.append(f"run config {number}: training results exist ({info['training_state_file']}). "
                          f"Use --resume to keep them and start the other runs.")
        runs = len(info["seeds"]) * len(info["k_folders"])
        arguments = ["--config", config, "--yes"] + (["--resume"] if resume else [])
        return {"step": step, "run": True, "command": _command(ROOT / SCRIPTS["training"], *arguments),
                "note": f"{runs} Flower run(s) in the plan" + (", --resume" if resume else ""), "runs": runs}
    # analysis
    seeds, minimum = len(info["seeds"]), info["min_complete_seeds"]
    if seeds < minimum:
        return {"step": step, "run": False, "command": None,
                "note": f"{seeds} seed(s); the analysis needs at least {minimum} (analysis.min_complete_seeds)"}
    if info["analysis_settings_file"].is_file() and not overwrite:
        errors.append(f"run config {number}: an analysis exists ({info['analysis_settings_file']}). "
                      f"Use --overwrite to write it again.")
    arguments = ["--config", config, "--yes"] + (["--overwrite"] if overwrite else [])
    return {"step": step, "run": True, "command": _command(ROOT / SCRIPTS["analysis"], *arguments),
            "note": "Track A and, if on, Track B" + (", --overwrite" if overwrite else "")}


def show_plan(plan, yes, steps):
    print("=" * 80)
    print("PHANTOM VFL ORCHESTRATOR")
    print("=" * 80)
    print(f"Code root: {ROOT}")
    print(f"Steps: {', '.join(steps)}")
    total = 0
    count = len(plan["chains"])
    for number, chain in enumerate(plan["chains"], start=1):
        info = chain["info"]
        print(f"\nRun config {number} of {count}: {chain['config']}")
        print(f"  dataset: {info['dataset_name']} ({info['dataset_config']})")
        print(f"  data folder: {info['data_dir']}"
              + (" (made now: it did not exist)" if chain.get("created") else ""))
        print(f"  label sharing: {'on' if info['label_sharing'] else 'off'}")
        print(f"  seeds: {info['seeds']}")
        print(f"  k levels (folders): {info['k_folders']}")
        print(f"  split folder: {info['split_folder']}; results tag: {info['results_tag']}")
        for action in chain["actions"]:
            print(f"  {action['step']:<11} {'run ' if action['run'] else 'skip'}  {action['note']}")
            if action["run"] and action["step"] == "training":
                total += action["runs"]
    if plan["summary"]:
        print(f"\nSummary: {plan['summary']['note']}")
    print(f"\nFlower runs in the plan: {total}")
    print(f"Confirmation gate: {'off (--yes)' if yes else 'on'}")


def confirm():
    """Confirmation gate. Returns False if the user stops the run."""
    try:
        answer = input("\nPress Enter to start, or type q to stop: ")
    except EOFError:
        print("\nNo input is available. Use --yes to run without the confirmation gate.")
        return False
    return answer.strip().lower() not in STOP_ANSWERS


# ============================================================================
# RUNNING THE PLAN
# ============================================================================

def continue_command(argv):
    """The command line to continue after a failed step: the same options with --resume."""
    words = ["python", "framework/run_all.py", *argv]
    if "--resume" not in argv:
        words.append("--resume")
    return " ".join(shlex.quote(word) for word in words)


def run_plan(plan, argv, runner=None):
    """
    Run the actions in order. Stop at the first failed action. Return the
    exit code (0 or 1).
    """
    runner = runner or (lambda command: subprocess.run(command, cwd=ROOT).returncode)
    actions = [(chain["config"], action) for chain in plan["chains"] for action in chain["actions"]]
    if plan["summary"]:
        actions.append((None, plan["summary"]))
    planned = sum(1 for _, action in actions if action["run"])
    done = 0
    for config, action in actions:
        if not action["run"]:
            continue
        done += 1
        where = f" of {config}" if config else ""
        print("\n" + "#" * 80)
        print(f"# STEP {done} of {planned}: {action['step']}{where}")
        print("#" * 80, flush=True)
        code = runner(action["command"])
        if code != 0:
            print(f"\n{ERROR_PREFIX} The step {action['step']}{where} stopped with exit code {code}.")
            print("Read its run log in the logs folder of its data folder (the step printed the path).")
            if action["step"] in ("generation", "training"):
                print(f"Correct the cause, then continue with:\n  {continue_command(argv)}")
            else:
                print("Correct the cause, then start the same command again with the steps that are left "
                      "(--steps) or with --overwrite.")
            return 1
        print(f"\nStep {done} of {planned} completed: {action['step']}{where}")
    print("\n" + "=" * 80)
    print(f"ALL STEPS COMPLETED ({planned} step(s) run)")
    print("=" * 80)
    return 0


def run(config_paths, summary=None, steps=STEPS, yes=False, resume=False, overwrite=False,
        argv=(), inspect=None, runner=None):
    """Make the plan, show it, wait for the gate and run it. Return the exit code (0, 1 or 2)."""
    from framework.run_log import run_log

    plan = make_plan(config_paths, list(steps), summary, resume, overwrite, inspect=inspect)
    log_dir = plan["chains"][0]["info"]["data_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    with run_log(log_dir, LOG_NAME):
        show_plan(plan, yes, list(steps))
        if not yes and not confirm():
            print("\nStopped by the user before the first step.")
            return 2
        return run_plan(plan, list(argv), runner=runner)


# ============================================================================
# QUESTIONS MODE (--new)
# ============================================================================

LOSS_KEYS = {
    "bce": (),
    "weighted_bce": (),
    "focal": ("focal-alpha", "focal-gamma"),
    "class_balanced": ("cb-beta",),
}
DEFAULT_K_LEVELS = "kstar,5,10,15,20,25,30,35,40"


def ask(question, default, parse=str, answer=None):
    """
    Ask one question and return the parsed answer. Enter gives the default.
    A wrong answer asks again. No input stops the questions.
    """
    answer = answer or input
    while True:
        try:
            text = answer(f"{question} [{default}]: ").strip()
        except EOFError:
            raise OrchestratorError("no input is available for the questions") from None
        try:
            return parse(text if text else str(default))
        except ValueError as error:
            print(f"  {error}")


def _yes_no(text):
    value = text.strip().lower()
    if value in ("y", "yes"):
        return True
    if value in ("n", "no"):
        return False
    raise ValueError("answer y or n")


def _seeds(text):
    try:
        seeds = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError:
        raise ValueError("give whole numbers separated by commas, for example 42,50,58") from None
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("give one or more different whole numbers of 0 or more")
    return seeds


def _k_levels(text):
    levels = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if part.lower() == "kstar":
            levels.append("kstar")
            continue
        try:
            value = int(part)
        except ValueError:
            raise ValueError("give kstar and whole percentages from 1 to 99, separated by commas") from None
        if not 1 <= value <= 99:
            raise ValueError("a k level must be kstar or a whole percentage from 1 to 99")
        levels.append(value)
    if not levels or len(set(map(str, levels))) != len(levels):
        raise ValueError("give one or more different k levels")
    return levels


def _share(text):
    value = float(text)
    if not 0 <= value < 1:
        raise ValueError("give a share from 0 to less than 1, for example 0.2")
    return value


def _positive_number(text):
    value = float(text)
    if value <= 0:
        raise ValueError("give a number above 0")
    return value


def _whole(text):
    value = int(text)
    if value < 1:
        raise ValueError("give a whole number of 1 or more")
    return value


def _value_text(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def find_dataset_configs():
    """The dataset configs of the repository: datasets/<name>/dataset.json."""
    return sorted((ROOT / "datasets").glob("*/dataset.json"))


def find_examples():
    """The example run configs: examples/*.json that have a section generation."""
    found = []
    for path in sorted((ROOT / EXAMPLES_FOLDER).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "generation" in data:
            found.append((path, data))
    return found


def app_defaults():
    """The run-config keys of the Flower app and their defaults (flower_app/pyproject.toml)."""
    from framework.training import run_training
    return run_training.app_defaults()


def _choose(question, options, default_number, answer):
    for number, text in enumerate(options, start=1):
        print(f"  {number}. {text}")

    def parse(text):
        if text.isdigit() and 1 <= int(text) <= len(options):
            return int(text)
        raise ValueError(f"give a number from 1 to {len(options)}")
    return ask(question, default_number, parse, answer)


def questions(answer=None, defaults=None, rscript_found=None):
    """
    Ask the questions of a new run config. Return (path of the new run config, data).
    answer replaces input, defaults replaces app_defaults() and rscript_found
    replaces the search for Rscript (for tests only).
    """
    answer = answer or input
    defaults = defaults if defaults is not None else app_defaults()
    if rscript_found is None:
        rscript_found = shutil.which("Rscript") is not None
    print("=" * 80)
    print("NEW RUN CONFIG")
    print("=" * 80)
    print("Press Enter to take the value in brackets.\n")

    start = {}
    examples = find_examples()
    if examples:
        print("Start from an example run config (its values become the defaults):")
        names = ["no example"] + [path.name for path, _ in examples]
        choice = _choose("Example", names, 1, answer)
        if choice > 1:
            path, start = examples[choice - 2]
            start = dict(start)
            dataset = (path.parent / start["generation"]["dataset_config"]).resolve()
            start["generation"] = {**start["generation"], "dataset_config": str(dataset)}

    found = find_dataset_configs()
    start_dataset = start.get("generation", {}).get("dataset_config")
    print("\nDataset config:")
    options = [str(path.relative_to(ROOT)) for path in found] + ["another dataset.json (give its path)"]
    default = 1
    if start_dataset and Path(start_dataset) in found:
        default = found.index(Path(start_dataset)) + 1
    choice = _choose("Dataset", options, default, answer)
    if choice <= len(found):
        dataset_config = found[choice - 1]
    else:
        def existing_file(text):
            path = Path(text).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"file not found: {path}")
            return path
        dataset_config = ask("Path of dataset.json", "", existing_file, answer)

    gen_start = start.get("generation", {})
    label_sharing = ask("Label sharing (y/n)", "y" if gen_start.get("label_sharing", True) else "n",
                        _yes_no, answer)
    folder_name = f"{dataset_config.parent.name}_{'ls' if label_sharing else 'nls'}"
    data_dir = ask("Data folder", ROOT / "data" / folder_name,
                   lambda text: Path(text).expanduser().resolve(), answer)
    seeds = ask("Seeds (comma-separated)", ",".join(map(str, gen_start.get("seeds", [42]))), _seeds, answer)
    k_levels = ask("Phantom k levels (kstar and percentages)",
                   ",".join(map(str, gen_start.get("k_levels", DEFAULT_K_LEVELS.split(",")))), _k_levels, answer)

    prep_start = start.get("data_preparation", {})
    print("\nSplit: the test share, and the validation share (0 means no validation set).")
    test_share = ask("Test share", _value_text(prep_start.get("test_share", 0.2)), _share, answer)
    validation_share = ask("Validation share", _value_text(prep_start.get("validation_share", 0.0)),
                           _share, answer)

    run_start = start.get("training", {}).get("run_config", {})
    losses = list(LOSS_KEYS)
    print("\nLoss of the head model:")
    loss_default = losses.index(run_start["loss-type"]) + 1 if run_start.get("loss-type") in losses else 1
    loss = losses[_choose("Loss", losses, loss_default, answer) - 1]
    run_config = {"loss-type": loss}
    for key in LOSS_KEYS[loss] + ("learning-rate",):
        run_config[key] = ask(key, _value_text(run_start.get(key, defaults[key])), _positive_number, answer)
    run_config["num-server-rounds"] = ask("num-server-rounds",
                                          run_start.get("num-server-rounds", defaults["num-server-rounds"]),
                                          _whole, answer)

    print("\nTrack B of the analysis needs R with the R package emmeans."
          + ("" if rscript_found else " Rscript is not on the PATH."))
    track_b = ask("Track B (y/n)", "y" if rscript_found else "n", _yes_no, answer)
    analysis = {"track_b": track_b}
    if track_b and not rscript_found:
        analysis["rscript"] = ask("Rscript command or full path", "Rscript", str, answer)

    data = {
        "schema_version": 1,
        "generation": {
            "dataset_config": str(dataset_config),
            "data_dir": str(data_dir),
            "seeds": seeds,
            "label_sharing": label_sharing,
            "k_levels": k_levels,
        },
        "data_preparation": {"test_share": test_share, "validation_share": validation_share},
        "training": {"feature_form": "model", "run_config": run_config},
        "analysis": analysis,
    }
    config_path = ask("Write the run config to", data_dir / f"config_{loss}.json",
                      lambda text: Path(text).expanduser().resolve(), answer)
    if config_path.exists() and not ask(f"{config_path} exists. Replace it (y/n)", "n", _yes_no, answer):
        raise OrchestratorError(f"the run config {config_path} exists; nothing is written")
    return config_path, data


def new_config(answer=None, defaults=None, rscript_found=None, inspect=None, start_run=None):
    """
    Questions mode: ask, write the run config, check it and offer to start it.
    Return the exit code.
    """
    answer = answer or input
    config_path, data = questions(answer, defaults, rscript_found)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nRun config written: {config_path}")
    inspect = inspect or inspect_config
    try:
        inspect(config_path, True)
    except Exception as error:  # the checks of the framework raise several error types
        print(f"{ERROR_PREFIX} The run config has faults. Correct the file, then start it:\n{error}")
        return 1
    command = f"python framework/run_all.py --config {shlex.quote(str(config_path))}"
    print(f"The run config passed the checks. Start it from the code root with:\n  {command}")
    if ask("Start it now (y/n)", "n", _yes_no, answer):
        start_run = start_run or (lambda path: run([path], argv=["--config", str(path)]))
        return start_run(config_path)
    return 0


# ============================================================================
# COMMAND LINE
# ============================================================================

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Run the whole chain (prepare, generation, training, analysis) of one or more run "
                    "configs, or write a new run config with --new.")
    parser.add_argument("--new", action="store_true", help="Ask questions and write a new run config")
    parser.add_argument("--config", action="append", default=[],
                        help="Path of a run config; give it once for each run config, in the order of the runs")
    parser.add_argument("--summary", help="Path of a summary config: run the summary after the last chain")
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"Comma-separated steps of each chain (default: {','.join(STEPS)})")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation gate")
    parser.add_argument("--resume", action="store_true", help="Pass --resume to the generation and the training")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to the analysis and the summary")
    args = parser.parse_args(argv)
    if args.new and (args.config or args.summary):
        parser.error("--new takes no --config and no --summary")
    if not args.new and not args.config:
        parser.error("give --new or at least one --config")
    return args


def main(argv=None):
    """Command line of the orchestrator. Return the exit code (0, 1 or 2)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(argv)
    if args.new:
        return new_config()
    steps = parse_steps(args.steps)
    return run(args.config, summary=args.summary, steps=steps, yes=args.yes, resume=args.resume,
               overwrite=args.overwrite, argv=argv)


if __name__ == "__main__":
    try:
        exit_code = main()
    except OrchestratorError as error:
        sys.exit(f"{ERROR_PREFIX} {error}")
    except KeyboardInterrupt:
        sys.exit(f"\n{ERROR_PREFIX} Stopped by the user (Ctrl-C). Start the same command again with --resume.")
    sys.exit(exit_code)
