"""
Summary across variants, with an optional seed-count advice.

The command reads a small summary config (JSON):

    {
      "schema_version": 1,
      "output_dir": "<folder>",
      "alpha": 0.05,
      "variants": [
        {"label": "ai_ls", "config": "<run_config.json>"},
        {"label": "old_ai_ls", "long_csv": "<file>", "value_column": "f1", "note": "..."},
        {"label": "old_le_ai_ls", "wide_csv": "<file>", "configs": ["ground_truth", "k0", ...],
         "column_suffix": "_f1", "in_target": false, "power_tracks": ["track_a"]}
      ],
      "power": {"deltas": [0.01, 0.02], "target_power": 0.8, "sd_upper_confidence": 0.8,
                "replicates": 2000, "random_seed": 20261103, "n_min": 3, "n_max": 100}
    }

A variant is one of:
  config    a run config whose analysis (framework/analysis/run_analysis.py) is complete;
            its tables enter the tables across variants
  long_csv  stored results in long form (seed, config, value column)
  wide_csv  stored results in wide form (seed and one column for each k level)
Stored results are an input of the advice only; they enter no table of results.

Without a "power" block the command writes only the tables across the analysis
folders and the report. With a "power" block it also writes the seed-count
advice. The advice is optional and changes nothing: the seeds of a run come only
from the section training of its run config.

Output files in output_dir:
    summary_settings.json        the config with defaults, the inputs with MD5 checksums, the versions
    track_a_all.csv              Track A of every analysis folder, with the Bonferroni correction over all
                                 of them (p_bonf_all)
    ceilings_all.csv             the ceiling rows of every analysis folder
    track_b_omnibus_all.csv      the Track B omnibus row of every analysis folder (when Track B ran)
    seed_count_advice.csv        (power only) one row for each variant, k level, test, delta and reading
    seed_count_advice_summary.csv (power only) the largest number of seeds for each delta and reading
    power_curves.csv             (power only) the Track B power for each number of seeds
    report.md                    the main numbers in plain text

Usage (from the code root):
    python framework/analysis/summarise.py --config <summary_config.json> [--yes] [--overwrite]

Exit codes: 0 done; 1 stopped by a fault (for example an analysis folder that is
not complete); 2 stopped by the user at the confirmation gate.
"""

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
if __package__ in (None, ""):  # started as a file: make the package 'framework' importable
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from framework import run_log, run_pipeline  # noqa: E402
from framework.analysis import power, report  # noqa: E402
from framework.analysis import settings as analysis_settings  # noqa: E402
from framework.training import run_training  # noqa: E402

SCHEMA = 1
SETTINGS_FILE = "summary_settings.json"
REPORT_FILE = "report.md"
LOG_NAME = "summary"
WARNING_PREFIX = run_log.WARNING_PREFIX
ERROR_PREFIX = run_log.ERROR_PREFIX
SOURCES = ("config", "long_csv", "wide_csv")
VARIANT_KEYS = {"label", "note", "in_target", "power_tracks", "reference", "value_column", "configs",
                "column_suffix"} | set(SOURCES)
TOP_KEYS = {"schema_version", "output_dir", "alpha", "variants", "power"}
TRACKS = ("track_a", "track_b")
POWER_DEFAULTS = {"deltas": [0.01, 0.02], "target_power": power.DEFAULT_TARGET,
                  "sd_upper_confidence": power.DEFAULT_CONFIDENCE, "replicates": 2000, "random_seed": 20261103,
                  "n_min": power.MIN_SEEDS, "n_max": 100}


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Tables across variants and, when the summary config asks for it, a seed-count advice.")
    parser.add_argument("--config", required=True, help="Path of the summary config (JSON)")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation gate")
    parser.add_argument("--overwrite", action="store_true", help="Write again into an output folder with a summary")
    return parser.parse_args(argv)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _whole(value, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _fraction(value):
    return _number(value) and 0 < value < 1


def _path(base, text):
    path = Path(text)
    return path if path.is_absolute() else (base / path)


def check_power(section, errors):
    """The power block with defaults, or None; faults go to errors."""
    if section is None:
        return None
    if not isinstance(section, dict):
        errors.append("power must be an object")
        return None
    unknown = sorted(set(section) - set(POWER_DEFAULTS))
    if unknown:
        errors.append(f"power has unknown keys {unknown}")
    values = {**POWER_DEFAULTS, **section}
    deltas = values["deltas"]
    if not (isinstance(deltas, list) and deltas and all(_number(d) and d > 0 for d in deltas)
            and len(set(deltas)) == len(deltas)):
        errors.append("power.deltas must be a list of different positive numbers")
    for key in ("target_power", "sd_upper_confidence"):
        if not _fraction(values[key]):
            errors.append(f"power.{key} must be a number between 0 and 1")
    if not _whole(values["replicates"], 1):
        errors.append("power.replicates must be a whole number of at least 1")
    if not _whole(values["random_seed"], 0):
        errors.append("power.random_seed must be a whole number of at least 0")
    if not _whole(values["n_min"], power.MIN_SEEDS):
        errors.append(f"power.n_min must be a whole number of at least {power.MIN_SEEDS}")
    elif not _whole(values["n_max"], values["n_min"]):
        errors.append("power.n_max must be a whole number of at least power.n_min")
    return values


def load_summary(config_path):
    """Read and check the summary config. Stop with a list of all faults. Return the checked settings."""
    config_path = Path(config_path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    errors = []
    if not isinstance(data, dict):
        raise run_pipeline.RunConfigError(f"Summary config {config_path} must be a JSON object")
    unknown = sorted(set(data) - TOP_KEYS)
    if unknown:
        errors.append(f"unknown keys {unknown}")
    if data.get("schema_version") != SCHEMA:
        errors.append(f"schema_version must be {SCHEMA}")
    if not isinstance(data.get("output_dir"), str) or not data.get("output_dir"):
        errors.append("output_dir must be a folder path")
    alpha = data.get("alpha", 0.05)
    if not _fraction(alpha):
        errors.append("alpha must be a number between 0 and 1")
    variants = data.get("variants")
    checked = []
    if not isinstance(variants, list) or not variants:
        errors.append("variants must be a non-empty list")
        variants = []
    labels = set()
    for index, variant in enumerate(variants):
        name = f"variants[{index}]"
        if not isinstance(variant, dict):
            errors.append(f"{name} must be an object")
            continue
        label = variant.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"{name}.label must be a non-empty text")
        elif label in labels:
            errors.append(f"{name}.label '{label}' is used twice")
        else:
            labels.add(label)
            name = f"variant '{label}'"
        unknown = sorted(set(variant) - VARIANT_KEYS)
        if unknown:
            errors.append(f"{name} has unknown keys {unknown}")
        sources = [key for key in SOURCES if key in variant]
        if len(sources) != 1:
            errors.append(f"{name} must have exactly one of {list(SOURCES)}")
            continue
        source = sources[0]
        if not isinstance(variant[source], str) or not variant[source]:
            errors.append(f"{name}.{source} must be a file path")
            continue
        tracks = variant.get("power_tracks", list(TRACKS))
        if not (isinstance(tracks, list) and tracks and set(tracks) <= set(TRACKS) and len(set(tracks)) == len(tracks)):
            errors.append(f"{name}.power_tracks must be a list from {list(TRACKS)}")
        if not isinstance(variant.get("in_target", True), bool):
            errors.append(f"{name}.in_target must be true or false")
        if not isinstance(variant.get("note", ""), str):
            errors.append(f"{name}.note must be a text")
        for key in ("reference", "value_column", "column_suffix"):
            if key in variant and not isinstance(variant[key], str):
                errors.append(f"{name}.{key} must be a text")
        if source == "config" and set(variant) & {"reference", "value_column", "configs", "column_suffix"}:
            errors.append(f"{name}: reference, value_column, configs and column_suffix are for stored results only")
        if source == "long_csv" and set(variant) & {"configs", "column_suffix"}:
            errors.append(f"{name}: configs and column_suffix are for wide_csv only")
        if source == "wide_csv":
            if "value_column" in variant:
                errors.append(f"{name}: value_column is for long_csv only")
            configs = variant.get("configs")
            if not (isinstance(configs, list) and len(configs) >= 2 and all(isinstance(c, str) and c for c in configs)
                    and len(set(configs)) == len(configs)):
                errors.append(f"{name}.configs must be a list of at least 2 different column names")
        checked.append({
            "label": label, "source": source, "path": _path(base, variant[source]),
            "note": variant.get("note", ""), "in_target": variant.get("in_target", True),
            "power_tracks": tracks, "reference": variant.get("reference", analysis_settings.GROUND_TRUTH),
            "value_column": variant.get("value_column", "value"), "configs": variant.get("configs"),
            "column_suffix": variant.get("column_suffix", "")})
    power_settings = check_power(data.get("power"), errors)
    if power_settings is not None and not any(v["in_target"] for v in checked):
        errors.append("the seed-count advice needs at least one variant with in_target true")
    if errors:
        raise run_pipeline.RunConfigError(f"Summary config {config_path} is not valid:\n  - " + "\n  - ".join(errors))
    return {"config_path": config_path, "output_dir": _path(base, data["output_dir"]), "alpha": alpha,
            "variants": checked, "power": power_settings}


def _ordered(configs, reference, label):
    if reference not in configs:
        raise ValueError(f"variant '{label}': the reference '{reference}' is not among the k levels {configs}")
    return [reference] + [c for c in configs if c != reference]


def _complete_rows(wide, label):
    """Keep the seeds with a value in every column; print a warning for each seed left out."""
    bad = wide[wide.isna().any(axis=1)]
    for seed in bad.index:
        print(f"{WARNING_PREFIX} (variant {label}) seed {seed} left out: a value is missing")
    return wide.dropna()


def load_variant(variant, alpha):
    """
    Read one variant. Return a dict with label, source, configs (reference first),
    data (seeds x configs, a numpy array), seeds, inputs ({file: md5}) and, for an
    analysis folder, the tables track_a, ceiling and omnibus.
    """
    label, path = variant["label"], variant["path"]
    item = {"label": label, "source": variant["source"], "note": variant["note"], "in_target": variant["in_target"],
            "power_tracks": variant["power_tracks"], "inputs": {}}
    if variant["source"] == "config":
        info = analysis_settings.load_analysis(path)
        folder = Path(info["output_folder"])
        settings_path = folder / "analysis_settings.json"
        if not settings_path.exists():
            raise ValueError(f"variant '{label}': no analysis in {folder}; run framework/analysis/run_analysis.py first")
        record = json.loads(settings_path.read_text(encoding="utf-8"))
        if not record.get("complete"):
            raise ValueError(f"variant '{label}': the analysis in {folder} is not complete")
        if not np.isclose(record["analysis"]["alpha"], alpha):
            raise ValueError(f"variant '{label}': its analysis used alpha {record['analysis']['alpha']}, "
                             f"the summary uses {alpha}")
        metric = record["analysis"]["metric"]
        configs = list(record["configs"])
        files = {"settings": settings_path, "wide": folder / f"wide_{metric}.csv",
                 "track_a": folder / f"track_a_{metric}.csv", "ceiling": folder / "ceiling.csv"}
        for name in ("wide", "track_a", "ceiling"):
            if not files[name].exists():
                raise ValueError(f"variant '{label}': missing file {files[name]}")
        wide = pd.read_csv(files["wide"]).set_index("seed")[configs]
        item["track_a"] = pd.read_csv(files["track_a"], keep_default_na=False, na_values=[""])
        item["ceiling"] = pd.read_csv(files["ceiling"], keep_default_na=False)
        omnibus_path = folder / "track_b_omnibus.csv"
        if omnibus_path.exists():
            files["omnibus"] = omnibus_path
            item["omnibus"] = pd.read_csv(omnibus_path, keep_default_na=False, na_values=["NA"]).iloc[0].to_dict()
        item["inputs"] = {str(p): run_training.file_md5(p) for p in files.values()}
        item["metric"] = metric
        item["family"] = int(record["analysis"]["bonferroni_family"])
    else:
        if not path.exists():
            raise ValueError(f"variant '{label}': file not found: {path}")
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            raise ValueError(f"variant '{label}': {path} has no column 'seed'")
        if variant["source"] == "long_csv":
            column = variant["value_column"]
            missing = [c for c in ("config", column) if c not in frame.columns]
            if missing:
                raise ValueError(f"variant '{label}': {path} has no column(s) {missing}")
            if frame.duplicated(["seed", "config"]).any():
                raise ValueError(f"variant '{label}': {path} has more than one row for a seed and k level")
            configs = _ordered(list(dict.fromkeys(frame["config"])), variant["reference"], label)
            wide = frame.pivot(index="seed", columns="config", values=column)[configs]
        else:
            columns = [c + variant["column_suffix"] for c in variant["configs"]]
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise ValueError(f"variant '{label}': {path} has no column(s) {missing}")
            if frame["seed"].duplicated().any():
                raise ValueError(f"variant '{label}': {path} has more than one row for a seed")
            wide = frame.set_index("seed")[columns]
            wide.columns = list(variant["configs"])
            configs = _ordered(list(variant["configs"]), variant["reference"], label)
            wide = wide[configs]
        item["inputs"] = {str(path): run_training.file_md5(path)}
        item["metric"] = variant["value_column"] if variant["source"] == "long_csv" else "values"
        item["family"] = len(configs) - 1
    wide = _complete_rows(wide.astype(float), label)
    item["configs"] = configs
    item["seeds"] = [int(s) for s in wide.index]
    item["data"] = wide.to_numpy(dtype=float)
    if len(item["seeds"]) < 2:
        raise ValueError(f"variant '{label}': fewer than 2 complete seeds")
    return item


def tables_across(items, alpha):
    """Track A, ceiling and omnibus rows of the analysis folders. family_all = all their Track A tests."""
    folders = [item for item in items if item["source"] == "config"]
    family_all = sum(len(item["track_a"]) for item in folders)
    track_a_parts, ceiling_parts, omnibus_rows = [], [], []
    for item in folders:
        part = item["track_a"].copy()
        part.insert(0, "variant", item["label"])
        p = part["p_value"].astype(float)
        part["bonferroni_family_all"] = family_all
        part["p_bonf_all"] = np.where(np.isfinite(p), np.minimum(1.0, p * family_all), np.nan)
        part["significant_bonf_all"] = np.where(np.isfinite(p), part["p_bonf_all"] < alpha, False)
        track_a_parts.append(part)
        ceiling = item["ceiling"].copy()
        ceiling.insert(0, "variant", item["label"])
        ceiling_parts.append(ceiling)
        if "omnibus" in item:
            omnibus_rows.append({"variant": item["label"], **item["omnibus"]})
    track_a_all = pd.concat(track_a_parts, ignore_index=True) if track_a_parts else pd.DataFrame()
    ceilings_all = pd.concat(ceiling_parts, ignore_index=True) if ceiling_parts else pd.DataFrame()
    return track_a_all, ceilings_all, pd.DataFrame(omnibus_rows), family_all


ADVICE_COLUMNS = ("variant", "in_target", "level", "test", "delta", "reading", "alpha", "n_data", "sd_diff",
                  "sd_upper", "variance_factor", "n_needed", "power_at_n", "note")
CURVE_COLUMNS = ("variant", "level", "delta", "n", "alpha", "power_omnibus", "power_pair", "mc_se_omnibus",
                 "mc_se_pair", "n_no_residual")


def advice(items, alpha, settings):
    """
    The seed-count advice. Returns (advice rows, curve rows).

    Track A: for each k level the SD of the paired differences, raised to its
    upper confidence limit, gives the smallest number of seeds for each delta at
    alpha / (k levels of the variant) and at alpha / (all tests of the target
    variants). Track B: the covariance of the k levels, raised by the same
    factor, gives the Monte Carlo power curve of the omnibus test and of the
    Tukey test of the ground truth against the k level with the drop, at alpha
    and at alpha / (number of target variants).
    """
    target = [item for item in items if item["in_target"]]
    family_all = sum(item["family"] for item in target)
    alpha_b_all = alpha / len(target)
    n_grid = range(settings["n_min"], settings["n_max"] + 1)
    rows, curves = [], []
    for item in items:
        data, n = item["data"], len(item["seeds"])
        factor = power.variance_factor(n, settings["sd_upper_confidence"])
        cov = np.cov(data, rowvar=False) * factor
        for j, level in enumerate(item["configs"][1:], start=1):
            diff = data[:, j] - data[:, 0]
            s = float(np.std(diff, ddof=1))
            s_up = power.sd_upper(s, n, settings["sd_upper_confidence"])
            base = {"variant": item["label"], "in_target": item["in_target"], "level": level, "n_data": n,
                    "sd_diff": s, "sd_upper": s_up, "variance_factor": factor}
            for delta in settings["deltas"]:
                if "track_a" in item["power_tracks"]:
                    for reading, a in (("variant", alpha / item["family"]), ("all", alpha / family_all)):
                        needed, value, note = power.track_a_n(delta, s_up, a, settings["target_power"],
                                                              settings["n_min"], settings["n_max"])
                        rows.append({**base, "test": "track_a", "delta": delta, "reading": reading, "alpha": a,
                                     "n_needed": needed, "power_at_n": value, "note": note})
                if "track_b" in item["power_tracks"]:
                    start = time.time()
                    curve = power.track_b_power_curve(cov, delta, j, n_grid, [alpha, alpha_b_all],
                                                      settings["replicates"], settings["random_seed"])
                    for row in curve:
                        curves.append({"variant": item["label"], "level": level, "delta": delta, **row})
                    for reading, a in (("variant", alpha), ("all", alpha_b_all)):
                        for test, column in (("track_b_omnibus", "power_omnibus"), ("track_b_pair", "power_pair")):
                            needed, note = power.smallest_n(curve, column, a, settings["target_power"])
                            at_n = next((r[column] for r in curve if r["n"] == needed and np.isclose(r["alpha"], a)),
                                        np.nan)
                            rows.append({**base, "sd_diff": np.nan, "sd_upper": np.nan, "test": test, "delta": delta,
                                         "reading": reading, "alpha": a, "n_needed": needed, "power_at_n": at_n,
                                         "note": note})
                    print(f"  {item['label']} {level} delta {delta}: Track B curve in {time.time() - start:.1f} s")
    return rows, curves


def advice_summary(advice_table, in_target):
    """The largest number of seeds over the k levels and variants, for each delta and reading."""
    part = advice_table[advice_table["in_target"] == in_target]
    rows = []
    for (delta, reading), group in part.groupby(["delta", "reading"], sort=True):
        row = {"delta": delta, "reading": reading}
        largest, not_reached = [], []
        for test in ("track_a", "track_b_omnibus", "track_b_pair"):
            tests = group[group["test"] == test]
            if not len(tests):
                row[test] = ""
                continue
            missing = tests[tests["n_needed"].isna()]
            found = tests["n_needed"].dropna()
            row[test] = "" if len(missing) else int(found.max())
            if len(missing):
                not_reached += [f"{test} {v} {lv}" for v, lv in zip(missing["variant"], missing["level"])]
            else:
                largest.append(int(found.max()))
        row["largest"] = "" if not_reached or not largest else max(largest)
        row["not_reached"] = "; ".join(not_reached)
        rows.append(row)
    return pd.DataFrame(rows, columns=["delta", "reading", "track_a", "track_b_omnibus", "track_b_pair", "largest",
                                       "not_reached"])


def show_plan(config_path, summary, yes, overwrite):
    print("=" * 80)
    print("PHANTOM VFL SUMMARY ACROSS VARIANTS")
    print("=" * 80)
    print(f"Summary config: {config_path}")
    for v in summary["variants"]:
        print(f"  {v['label']:<16} {v['source']:<9} {v['path']}{'' if v['in_target'] else '  (not in the target)'}")
    s = summary["power"]
    if s is None:
        print("  seed-count advice: off (no power block)")
    else:
        curves = sum(1 for v in summary["variants"] if "track_b" in v["power_tracks"])
        print(f"  seed-count advice: on (advice only); deltas {s['deltas']}; target power {s['target_power']}; "
              f"seeds {s['n_min']} to {s['n_max']}; {s['replicates']} replicates; random seed {s['random_seed']}; "
              f"Track B for {curves} variant(s)")
    print(f"  output: {summary['output_dir']}{' (overwrite)' if overwrite else ''}")
    print(f"  confirmation gate: {'off (--yes)' if yes else 'on'}")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def settings_record(summary, items, complete, files=()):
    return {
        "schema": SCHEMA,
        "complete": complete,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_config": str(summary["config_path"]),
        "summary_config_md5": run_training.file_md5(summary["config_path"]),
        "alpha": summary["alpha"],
        "power": summary["power"],
        "variants": [{"label": item["label"], "source": item["source"], "in_target": item["in_target"],
                      "power_tracks": item["power_tracks"], "note": item["note"], "configs": item["configs"],
                      "seeds": item["seeds"], "inputs": item["inputs"]} for item in items],
        "files": list(files),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__},
    }


def summarise(summary, log_files=None):
    """Read the variants and write the summary files. Return the exit code."""
    out, alpha = Path(summary["output_dir"]), summary["alpha"]
    items = [load_variant(variant, alpha) for variant in summary["variants"]]
    for item in items:
        print(f"Variant {item['label']}: {item['source']}, {len(item['seeds'])} seeds, {len(item['configs'])} "
              f"configs ({item['configs'][0]} first)")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / SETTINGS_FILE, settings_record(summary, items, False))
    track_a_all, ceilings_all, omnibus_all, family_all = tables_across(items, alpha)
    files = []
    for name, table in (("track_a_all.csv", track_a_all), ("ceilings_all.csv", ceilings_all),
                        ("track_b_omnibus_all.csv", omnibus_all)):
        if len(table):
            table.to_csv(out / name, index=False)
            files.append(name)
    advice_part = None
    if summary["power"] is not None:
        print("\nSeed-count advice (advice only):")
        rows, curves = advice(items, alpha, summary["power"])
        advice_table = pd.DataFrame(rows, columns=ADVICE_COLUMNS)
        advice_table.to_csv(out / "seed_count_advice.csv", index=False)
        target = advice_summary(advice_table, True)
        beside = advice_summary(advice_table, False)
        summary_table = pd.concat([target.assign(group="target"), beside.assign(group="beside")], ignore_index=True)
        summary_table.to_csv(out / "seed_count_advice_summary.csv", index=False)
        files += ["seed_count_advice.csv", "seed_count_advice_summary.csv"]
        if curves:
            pd.DataFrame(curves, columns=CURVE_COLUMNS).to_csv(out / "power_curves.csv", index=False)
            files.append("power_curves.csv")
        advice_part = {"settings": summary["power"], "summary": target, "beside": beside}
        print("\nLargest number of seeds (target variants; empty: not reached):")
        for _, row in target.iterrows():
            print(f"  delta {row['delta']:<8g} {row['reading']:<8} Track A {row['track_a']!s:<4} Track B omnibus "
                  f"{row['track_b_omnibus']!s:<4} Track B pair {row['track_b_pair']!s:<4} largest {row['largest']}")
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    variants = [{"label": i["label"], "source": i["source"], "n_seeds": len(i["seeds"]),
                 "n_levels": len(i["configs"]) - 1, "in_target": i["in_target"], "note": i["note"]} for i in items]
    text = report.summary_report(created, summary["config_path"], variants, track_a_all, ceilings_all, omnibus_all,
                                 family_all, alpha, advice_part, report.warning_lines(log_files))
    (out / REPORT_FILE).write_text(text, encoding="utf-8")
    files.append(REPORT_FILE)
    write_json(out / SETTINGS_FILE, settings_record(summary, items, True, files))
    print(f"\nOutput: {out}")
    return 0


def main(argv=None):
    """Run the summary. Return the exit code (0, 1 or 2)."""
    args = parse_arguments(argv)
    summary = load_summary(args.config)
    out = Path(summary["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with run_log.run_log(out, LOG_NAME) as log_files:
        if (out / SETTINGS_FILE).exists() and not args.overwrite:
            raise ValueError(f"the output folder {out} holds a summary ({SETTINGS_FILE}). "
                             f"Use --overwrite to write it again.")
        show_plan(args.config, summary, args.yes, args.overwrite)
        if not args.yes and not run_pipeline.confirm("summary"):
            print("\nStopped by the user before the summary.")
            return 2
        return summarise(summary, log_files)


if __name__ == "__main__":
    try:
        exit_code = main()
    except (run_pipeline.RunConfigError, run_pipeline.DatasetConfigError, ValueError, FileNotFoundError) as error:
        sys.exit(f"{ERROR_PREFIX} {error}")
    sys.exit(exit_code)
