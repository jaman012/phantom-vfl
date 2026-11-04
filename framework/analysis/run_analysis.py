"""
Analysis of the training runs of one run config.

The driver reads run_config.json (the sections generation, data_preparation,
training and the optional section analysis), collects the completed training
runs, selects the round of each run (framework.analysis.select_round: on the
validation set, or on the test set with analysis.round_source "test") and
writes:

  <data_dir>/<output_folder>/<results tag>/split_<ratio>/
    selection.csv           one row for each used run: the selected round and the test values of that round
    wide_<metric>.csv       complete seeds: one row for each seed, one column for each k level
    long_<metric>.csv       the same values as seed, config, value
    descriptive_<metric>.csv  each k level: n, mean, SD, median, min, max, confidence interval of the mean
    track_a_<metric>.csv    each k level against the ground truth (k level minus ground truth): paired
                            t-test with Bonferroni, Cohen's d_z, Shapiro-Wilk of the differences,
                            Wilcoxon signed-rank with Bonferroni (framework.analysis.track_a)
    friedman_<metric>.csv   the Friedman rank test over all k levels
    track_b_omnibus.csv     main metric: repeated-measures ANOVA, Mauchly's test, Greenhouse-Geisser
                            and Huynh-Feldt corrections (framework/analysis/track_b.R, in R)
    track_b_pairwise.csv    main metric: all pairs of k levels, Tukey-adjusted (R package emmeans)
    r_session.txt           the versions of R and emmeans
    ceiling.csv             main metric: the viable ceiling of each track (framework.analysis.ceiling)
    report.md               the main numbers in plain text, with the warning lines of the run
    sensitivity_gt_round/   round_source "validation": the same files with the ground truth's round for
                            every k level of a seed
    sensitivity_own_round/  round_source "test": the same files with each run at its own round
    analysis_settings.json  the settings, the seeds used and left out, the MD5 checksums of the inputs

analysis_settings.json is written first with "complete": false and last with
"complete": true. An output folder that holds analysis_settings.json is not
written again without --overwrite. The driver copies its screen output to a
run log in <data_dir>/logs/. With analysis.track_b true (the default) the
driver checks R and the R package emmeans before the confirmation gate and
stops when one of them is missing; with false it writes Track A only and a
ceiling of Track A only.

Usage (from the code root):
    python framework/analysis/run_analysis.py --config <run_config.json> [--yes] [--overwrite]

Exit codes: 0 done; 1 stopped by a fault in the results (for example fewer
complete seeds than analysis.min_complete_seeds); 2 stopped by the user at the
confirmation gate.
"""

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
if __package__ in (None, ""):  # started as a file: make the package 'framework' importable
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from framework import run_log, run_pipeline  # noqa: E402
from framework.analysis import ceiling, collect, report, run_r, select_round, tables, track_a  # noqa: E402
from framework.analysis import settings as analysis_settings  # noqa: E402
from framework.training import run_training  # noqa: E402

SETTINGS_FILE = "analysis_settings.json"
SELECTION_FILE = "selection.csv"
CEILING_FILE = "ceiling.csv"
REPORT_FILE = "report.md"
SENSITIVITY_FOLDER = "sensitivity_gt_round"
LOG_NAME = "analysis"
SCHEMA = 1
WARNING_PREFIX = run_log.WARNING_PREFIX
ERROR_PREFIX = run_log.ERROR_PREFIX


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Select the round of each training run and write the analysis tables.")
    parser.add_argument("--config", required=True, help="Path of run_config.json")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation gate")
    parser.add_argument("--overwrite", action="store_true",
                        help="Write again into an output folder that holds an analysis")
    return parser.parse_args(argv)


def check_output_folder(folder, overwrite):
    """Raise ValueError when the folder holds an analysis and overwrite is off."""
    if (Path(folder) / SETTINGS_FILE).exists() and not overwrite:
        raise ValueError(f"the output folder {folder} holds an analysis ({SETTINGS_FILE}). "
                         f"Use --overwrite to write it again.")


def show_plan(config_path, info, yes, overwrite):
    settings, setup = info["settings"], info["setup"]
    print("=" * 80)
    print("PHANTOM VFL ANALYSIS")
    print("=" * 80)
    print(f"Run config: {config_path}")
    print(f"  data_dir: {setup['gen']['data_dir']}")
    print(f"  results tag: {info['results_tag']} (SPLIT_RATIO={setup['split_ratio']})")
    print(f"  seeds: {info['seeds']}")
    print(f"  k levels (folders): {info['configs']}")
    print(f"  {select_round.rule_text(settings)}")
    if settings["round_source"] == "test":
        print(f"{WARNING_PREFIX} the round is chosen on the test set (analysis.round_source \"test\"), as in the "
              f"published results; the test values of the chosen round are not independent of that choice")
    print(f"  metrics: {analysis_settings.metric_columns(settings)}")
    if settings["track_b"]:
        versions = info["r_versions"]
        print(f"  Track B: on ({settings['metric']}; {versions['rscript']}, R {versions['r']}, "
              f"emmeans {versions['emmeans']})")
    else:
        print("  Track B: off (analysis.track_b false; Track A only)")
    print(f"  output: {info['output_folder']}{' (overwrite)' if overwrite else ''}")
    print(f"  confirmation gate: {'off (--yes)' if yes else 'on'}")


def _relative(path, base):
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


def _md5_or_empty(path):
    return run_training.file_md5(path) if Path(path).exists() else ""


def settings_record(config_path, info, records, seeds, missing, complete, files=()):
    """The content of analysis_settings.json."""
    data_dir = Path(info["setup"]["gen"]["data_dir"])
    state = Path(info["state_folder"])
    return {
        "schema": SCHEMA,
        "complete": complete,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_config": str(Path(config_path).resolve()),
        "run_config_md5": run_training.file_md5(config_path),
        "results_tag": info["results_tag"],
        "split_ratio": info["setup"]["split_ratio"],
        "configs": info["configs"],
        "seeds_planned": info["seeds"],
        "seeds_used": seeds,
        "seeds_left_out": {str(seed): names for seed, names in missing.items()},
        "analysis": info["settings"],
        "ci_level": 1 - info["settings"]["alpha"],
        "training_settings_md5": _md5_or_empty(state / run_training.STATE_SETTINGS_FILE),
        "training_runs_md5": _md5_or_empty(state / run_training.STATE_RUNS_FILE),
        "inputs": [{"seed": record["seed"], "config": record["config"],
                    "file": _relative(record["path"], data_dir), "md5": record["md5"],
                    "status": record["status"], "reason": record["reason"]} for record in records],
        "files": list(files),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "r": info.get("r_versions", "not used")},
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_track_a(folder, wide, metric, info, label):
    """Write the Track A files of one metric; print a warning line for each note. Return (file names, paired table)."""
    settings, configs = info["settings"], info["configs"]
    described = track_a.descriptive(wide, configs, settings["alpha"])
    paired = track_a.paired_tests(wide, configs, settings["bonferroni_family"], settings["alpha"])
    ranked = track_a.friedman(wide, configs)
    names = [f"descriptive_{metric}.csv", f"track_a_{metric}.csv", f"friedman_{metric}.csv"]
    for table, name in zip((described, paired, ranked), names):
        table.to_csv(folder / name, index=False)
    notes = [(row["config"], row["note"]) for table in (described, paired) for _, row in table.iterrows()]
    notes += [("all k levels", note) for note in ranked["note"]]
    for config, note in notes:
        if note:
            print(f"{WARNING_PREFIX} ({label}, {metric}, {config}) {note}")
    return names, paired


def write_track_b(folder, info, label, paired):
    """
    Run Track B on the long table of the main metric (when analysis.track_b is
    true) and write the ceiling file. Return (file names, results), where results
    has the keys paired, omnibus, pairwise and ceiling.
    """
    settings = info["settings"]
    metric = settings["metric"]
    names, omnibus, pairwise = [], None, None
    if settings["track_b"]:
        omnibus, pairwise = run_r.run_track_b(info["r_versions"]["rscript"], folder / f"long_{metric}.csv", folder,
                                              settings["alpha"], info["configs"])
        names += [run_r.OMNIBUS_FILE, run_r.PAIRWISE_FILE, run_r.SESSION_FILE]
        if omnibus["note"]:
            print(f"{WARNING_PREFIX} ({label}, {metric}, Track B) {omnibus['note']}")
        if not omnibus["omnibus_significant_gg"]:
            print(f"{WARNING_PREFIX} ({label}, {metric}, Track B) the repeated-measures ANOVA is not significant "
                  f"after the Greenhouse-Geisser correction (p = {omnibus['p_gg']:.4g}); the pairwise Tukey "
                  f"results are reported without this condition")
    levels = info["configs"][1:]
    table = ceiling.ceiling_table(levels, info["setup"]["gen"]["kstar_folder"], track_a=paired, track_b=pairwise,
                                  ground_truth=info["configs"][0])
    table.to_csv(folder / CEILING_FILE, index=False)
    names.append(CEILING_FILE)
    return names, {"paired": paired, "omnibus": omnibus, "pairwise": pairwise, "ceiling": table}


def write_tables(folder, selection, info, seeds, label):
    """
    Write the selection file; for each metric, the wide and long tables and the
    Track A files; for the main metric, the Track B files and the ceiling file.
    Return (file names, results of the main metric).
    """
    folder.mkdir(parents=True, exist_ok=True)
    table = selection.copy()
    table.insert(2, "complete_seed", table["seed"].isin(seeds))
    table.to_csv(folder / SELECTION_FILE, index=False)
    names = [SELECTION_FILE]
    main = None
    for metric in analysis_settings.metric_columns(info["settings"]):
        wide = tables.wide_table(selection, metric, info["configs"], seeds)
        wide.to_csv(folder / f"wide_{metric}.csv", index=False)
        tables.long_table(wide, info["configs"]).to_csv(folder / f"long_{metric}.csv", index=False)
        names += [f"wide_{metric}.csv", f"long_{metric}.csv"]
        track_names, paired = write_track_a(folder, wide, metric, info, label)
        names += track_names
        if main is None:
            main = paired
    track_b_names, results = write_track_b(folder, info, label, main)
    names += track_b_names
    return names, results


def show_track_a(paired, settings):
    """Print the paired tests of the main metric."""
    print(f"\nTrack A ({settings['metric']}, k level minus ground truth; Bonferroni family "
          f"{settings['bonferroni_family']}, alpha {settings['alpha']}):")
    for _, row in paired.iterrows():
        if np.isfinite(row["p_value"]):
            verdict = "significant" if row["significant_bonf_variant"] else "not significant"
            print(f"  {row['config']:<14} diff {row['diff_mean']:+.4f}  p {row['p_value']:.4g}  "
                  f"p_bonf {row['p_bonf_variant']:.4g}  {verdict}")
        else:
            print(f"  {row['config']:<14} diff {row['diff_mean']:+.4f}  no t-test ({row['note']})")


def show_track_b(results, settings):
    """Print the Track B omnibus test, the ground-truth pairs and the ceilings of the main metric."""
    omnibus, pairwise = results["omnibus"], results["pairwise"]
    if omnibus is not None:
        print(f"\nTrack B ({settings['metric']}, repeated-measures ANOVA in R):")
        if np.isfinite(omnibus["f_stat"]):
            print(f"  F({omnibus['df1']:g}, {omnibus['df2']:g}) = {omnibus['f_stat']:.4f}, p = "
                  f"{omnibus['p_uncorrected']:.4g}; Greenhouse-Geisser epsilon {omnibus['gg_epsilon']:.4f}, "
                  f"p = {omnibus['p_gg']:.4g}; Huynh-Feldt p = {omnibus['p_hf']:.4g}; Mauchly p = "
                  f"{omnibus['mauchly_p']:.4g}")
        else:
            print(f"  no F test ({omnibus['note']})")
        ground = pairwise[pairwise["group1"] == pairwise["group1"].iloc[0]]
        for _, row in ground.iterrows():
            if np.isfinite(row["p_tukey"]):
                verdict = "separated" if row["reject"] else "not separated"
                print(f"  {row['group2']:<14} diff {row['diff']:+.4f}  p_tukey {row['p_tukey']:.4g}  {verdict}")
            else:
                print(f"  {row['group2']:<14} diff {row['diff']:+.4f}  no Tukey test")
    print("\nViable ceiling (the k level before the first separated k level):")
    for _, row in results["ceiling"].iterrows():
        text = row["ceiling"] if row["ceiling"] else "none"
        extra = [f"exceptions {row['exceptions']}" if row["exceptions"] else "",
                 f"untested {row['untested']}" if row["untested"] else "",
                 f"k* ({row['kstar_folder']}) separated: {row['kstar_separated']}" if row["kstar_folder"] else ""]
        print(f"  {row['track']:<28} {text}  " + "; ".join(filter(None, extra)))


def analyse(config_path, info):
    """Collect, select and write. Return the exit code."""
    settings, folder = info["settings"], Path(info["output_folder"])
    records = collect.collect_runs(info)
    used = [record for record in records if record["status"] == collect.USED]
    for record in records:
        if record["status"] != collect.USED:
            print(f"{WARNING_PREFIX} (seed {record['seed']}, {record['config']}) run left out: {record['reason']}")
    if not used:
        print(f"{ERROR_PREFIX} No run can be used. Nothing is written.")
        return 1
    selection = select_round.selection_table(used, settings)
    seeds, missing = tables.complete_seeds(selection, info["seeds"], info["configs"])
    for seed, names in missing.items():
        print(f"{WARNING_PREFIX} (seed {seed}) seed left out of every table: no used run for {names}")
    print(f"Runs: {len(records)} planned, {len(used)} used, {len(records) - len(used)} left out. "
          f"Complete seeds: {len(seeds)} of {len(info['seeds'])}.")
    if len(seeds) < settings["min_complete_seeds"]:
        print(f"{ERROR_PREFIX} {len(seeds)} complete seed(s); the analysis needs at least "
              f"{settings['min_complete_seeds']} (analysis.min_complete_seeds). Nothing is written.")
        return 1
    main_table, second_table = select_round.lens_tables(used, selection, settings, analysis_settings.GROUND_TRUTH)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / SETTINGS_FILE, settings_record(config_path, info, records, seeds, missing, False))
    files, main_results = write_tables(folder, main_table, info, seeds, "main")
    if second_table is not None:
        second = select_round.second_folder(settings)
        second_files, _ = write_tables(folder / second, second_table, info, seeds, second)
        files += [f"{second}/{name}" for name in second_files]
    log_files = info.get("run_log")
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    text = report.variant_report(info, main_results, seeds, missing, created,
                                 log_files.log_path if log_files else None, report.warning_lines(log_files))
    (folder / REPORT_FILE).write_text(text, encoding="utf-8")
    files.append(REPORT_FILE)
    write_json(folder / SETTINGS_FILE, settings_record(config_path, info, records, seeds, missing, True, files))
    complete = main_table[main_table["seed"].isin(seeds)]
    print(f"\nSelected rounds and mean {settings['metric']} over the {len(seeds)} complete seeds:")
    for name in info["configs"]:
        part = complete[complete["config"] == name]
        print(f"  {name:<14} rounds {int(part['selected_round'].min())}-{int(part['selected_round'].max())}  "
              f"mean {part[settings['metric']].mean():.4f}")
    show_track_a(main_results["paired"], settings)
    show_track_b(main_results, settings)
    print(f"\nOutput: {folder}")
    return 0


def main(argv=None):
    """Run the analysis of one run config. Return the exit code (0, 1 or 2)."""
    args = parse_arguments(argv)
    info = analysis_settings.load_analysis(args.config)
    with run_log.run_log(info["setup"]["gen"]["data_dir"], LOG_NAME) as log_files:
        info["run_log"] = log_files
        check_output_folder(info["output_folder"], args.overwrite)
        if info["settings"]["track_b"]:
            info["r_versions"] = run_r.check_r(info["settings"]["rscript"])
        show_plan(args.config, info, args.yes, args.overwrite)
        if not args.yes and not run_pipeline.confirm("analysis"):
            print("\nStopped by the user before the analysis.")
            return 2
        return analyse(args.config, info)


if __name__ == "__main__":
    try:
        exit_code = main()
    except (run_pipeline.RunConfigError, run_pipeline.DatasetConfigError, ValueError, FileNotFoundError) as error:
        sys.exit(f"{ERROR_PREFIX} {error}")
    sys.exit(exit_code)
