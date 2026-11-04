r"""
Launcher of one Flower run (James Mansueto's own code).

The launcher starts 'flwr run' for the app in flower_app/ and copies every line
of its screen output (server, clients and Ray) to a run log (framework.run_log):

  <data-dir>/logs/flower_<results-tag>_<k>_rs<seed>_<date>-<time>.log            all output
  <data-dir>/logs/flower_<results-tag>_<k>_rs<seed>_<date>-<time>_warnings.txt   warnings and errors

Usage (with the Python of the environment that has flwr):

  python <repository root>/framework/training/run_flower.py --split-ratio 80_20_onehot \
      --run-config "data-dir='<data folder>' dataset-config='<path of dataset.json>' phantom-dataset='k0' random-seed=42 loss-type='bce'"

Rules:
- --run-config has the same text as for 'flwr run', and 'flwr run' gets it
  unchanged. The launcher reads it with Flower's own reader
  (flwr.common.config.parse_config_args) and takes the missing keys from
  flower_app/pyproject.toml, only to name the log files.
- --split-ratio is required: 80_20 (integer form) or 80_20_onehot (model form).
  The launcher sets the environment variable SPLIT_RATIO of the run; without it
  the server app reads the integer form.
- The environment of 'flwr run' also gets PYTHONPATH (the repository root first,
  so that the app finds framework.training) and PYTHONUNBUFFERED=1 (the lines
  arrive in order).
- The terminal colour codes of the Flower output are removed.
- A Flower line with the level WARNING, or a Python warning line, gets the prefix
  "[!]"; a Flower line with the level ERROR, the first line of a traceback, or a
  line that reports a non-zero exit status gets the prefix "[ERROR]".
  framework.run_log copies these lines also to the warnings file. All other lines
  stay unchanged.
- The run counts as failed when 'flwr run' stops with an exit code other than 0,
  when the output has at least one "[ERROR]" line, or when one of the three
  result files of the run (test metrics, training loss and class-balance record,
  framework.training.results) is missing or older than the start of the run.
  'flwr run' of flwr 1.23.0 returns 0 also when the simulation fails, so the
  exit code alone is not enough. A failed run prints "[ERROR] The run failed:
  <reasons>", and the launcher stops with exit code 1 (or the exit code of
  'flwr run' when it is not 0). A run that completes prints the paths of its
  three result files, and the launcher stops with 0.
- When the key data-dir is empty, the run has no run log and no check of the
  result files.
- The launcher writes no results file; the server app writes them
  (framework.training.results).
- The runner framework/training/run_training.py calls main() for each run of
  run_config.json.
"""

import argparse
import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
if __package__ in (None, ""):  # started as a file: make the package 'framework' importable
    sys.path.insert(0, str(ROOT))

from framework import run_log  # noqa: E402
from framework.training import results  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

APP_DIR = ROOT / "flower_app"
LOG_NAME_PREFIX = "flower"
_COLOUR = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # terminal colour codes
_LEVEL = re.compile(r"^(?:\(\S+ pid=\d+\)\s*)?(WARNING|ERROR)\s*:")  # Flower level, also after a Ray worker name
_PYTHON_WARNING = re.compile(r"\b[A-Z]\w*Warning: ")  # for example "UserWarning: "
_TRACEBACK = "Traceback (most recent call last)"
_EXIT_STATUS = re.compile(r"returned non-zero exit status \d+")  # for example the end line of a failed simulation


def strip_colours(line):
    """The line without terminal colour codes."""
    return _COLOUR.sub("", line)


def mark_line(line):
    """
    The line with the prefix of framework.run_log when it is a warning or an error
    line (see the rules above); other lines unchanged.
    """
    stripped = line.lstrip()
    if stripped.startswith((run_log.WARNING_PREFIX, run_log.ERROR_PREFIX)):
        return line
    level = _LEVEL.match(stripped)
    if (level and level.group(1) == "ERROR") or stripped.startswith(_TRACEBACK) or _EXIT_STATUS.search(stripped):
        return f"{run_log.ERROR_PREFIX} {line}"
    if level or _PYTHON_WARNING.search(stripped):
        return f"{run_log.WARNING_PREFIX} {line}"
    return line


def read_settings(run_config_texts, app_dir=APP_DIR):
    """
    The run-config values of the run: the defaults of <app_dir>/pyproject.toml,
    replaced by the values of the run-config texts (Flower's own reader).
    """
    try:
        from flwr.common.config import parse_config_args
    except ImportError as error:
        raise SystemExit(
            f"{run_log.ERROR_PREFIX} flwr.common.config.parse_config_args was not found ({error}). "
            "Start the launcher with the Python of the environment that has flwr."
        ) from error
    with open(Path(app_dir) / "pyproject.toml", "rb") as file:
        settings = dict(tomllib.load(file)["tool"]["flwr"]["app"]["config"])
    settings.update(parse_config_args(list(run_config_texts)))
    return settings


def run_tag(settings):
    """The results tag of the run (framework.training.results)."""
    return results.results_tag(settings.get("results-tag", ""), settings["loss-type"])


def log_name(settings):
    """First part of the log file names: flower_<results-tag>_<k>_rs<seed>."""
    return f"{LOG_NAME_PREFIX}_{run_tag(settings)}_{settings['phantom-dataset']}_rs{settings['random-seed']}"


def flwr_program():
    """The program flwr next to the running Python, else the one on PATH."""
    beside = Path(sys.executable).parent / "flwr"
    if beside.exists():
        return str(beside)
    found = shutil.which("flwr")
    if found is None:
        raise SystemExit(f"{run_log.ERROR_PREFIX} The program flwr was not found (next to {sys.executable} or on PATH).")
    return found


def child_environment(split_ratio):
    """Environment of 'flwr run': SPLIT_RATIO, PYTHONPATH with the repository root first, PYTHONUNBUFFERED."""
    env = dict(os.environ)
    env["SPLIT_RATIO"] = split_ratio
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_child(command, env, cwd):
    """
    Run the command, print each line of its output (colour codes removed,
    warning and error lines marked) and return its exit code and the number of
    lines that start with "[ERROR]". Ctrl-C also stops the command.
    """
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    errors = 0
    try:
        for raw in process.stdout:
            line = mark_line(strip_colours(raw.rstrip("\r\n")))
            if line.lstrip().startswith(run_log.ERROR_PREFIX):
                errors += 1
            print(line, flush=True)
        return process.wait(), errors
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        raise


def result_files(settings, split_ratio):
    """The three result files of the run: test metrics, training loss and class-balance record (framework.training.results)."""
    k, seed = settings["phantom-dataset"], settings["random-seed"]
    folder = results.results_dir(settings["data-dir"], run_tag(settings), split_ratio, seed, k)
    return [folder / results.test_metrics_file_name(k, seed), folder / results.train_loss_file_name(k, seed),
            folder / results.class_balance_file_name(k, seed)]


def failure_reasons(code, errors, files, start):
    """The reasons why the run failed; an empty list for a run that completed."""
    reasons = []
    if code != 0:
        reasons.append(f"flwr run stopped with exit code {code}")
    if errors:
        reasons.append(f"{errors} error line(s) in the output")
    for path in files:
        if not path.exists():
            reasons.append(f"result file missing: {path}")
        elif path.stat().st_mtime < start:
            reasons.append(f"result file older than the start of the run: {path}")
    return reasons


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Start one Flower run and copy its screen output to <data-dir>/logs/.")
    parser.add_argument("--split-ratio", required=True,
                        help="split folder: 80_20 (integer form) or 80_20_onehot (model form)")
    parser.add_argument("--run-config", action="append", default=[],
                        help="run-config text as for 'flwr run' (can be given more than once)")
    return parser.parse_args(argv)


def main(argv=None, child_command=None):
    """
    Start one Flower run with a run log. Return 0 for a run that completed, and
    1 (or the exit code of 'flwr run' when it is not 0) for a failed run.
    child_command replaces [flwr, 'run', '.'] (for tests only).
    """
    args = parse_arguments(argv)
    settings = read_settings(args.run_config)
    command = list(child_command) if child_command else [flwr_program(), "run", "."]
    for text in args.run_config:
        command += ["--run-config", text]
    data_dir = settings.get("data-dir", "")
    if data_dir:
        log = run_log.run_log(data_dir, log_name(settings))
    else:
        print("No run log: the run-config key data-dir is empty.")
        log = contextlib.nullcontext()
    with log:
        run_log.set_context(f"{run_tag(settings)}, {settings['phantom-dataset']}, seed {settings['random-seed']}")
        print(f"Flower run: k={settings['phantom-dataset']}; seed={settings['random-seed']}; "
              f"loss-type={settings['loss-type']}; results tag={run_tag(settings)}; SPLIT_RATIO={args.split_ratio}")
        print(f"App folder: {APP_DIR}")
        print(f"Command: {shlex.join(command)}")
        start = time.time()  # result files older than this time come from an earlier run
        code, errors = run_child(command, child_environment(args.split_ratio), APP_DIR)
        files = result_files(settings, args.split_ratio) if data_dir else []
        reasons = failure_reasons(code, errors, files, start)
        if reasons:
            print(f"{run_log.ERROR_PREFIX} The run failed: {'; '.join(reasons)}")
            return code if code != 0 else 1
        print("Run completed. Result files:")
        for path in files:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
