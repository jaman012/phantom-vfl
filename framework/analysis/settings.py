"""
Settings of the analysis: the optional section "analysis" of run_config.json.

The analysis reads the training plan of the same run config with
framework.training.run_training.load_training: the data folder, the seeds, the
k levels, the split folder and the results tag. The section "analysis" adds the
settings below. Every key is optional; a missing key takes its default.

  round_source         where the round of each run is chosen: "validation" (the validation
                       columns) or "test" (the test columns, as in the published results);
                       null: "validation" when the split has a validation set, else "test"
  selection_metric     validation column that selects the round of each run (higher is better;
                       round_source "validation" only)
  tie_break            validation column for two rounds with the same selection_metric (higher is
                       better; round_source "validation" only)
  metric               test column of the statistical tests
  extra_metrics        further test columns (tables and paired tests only)
  alpha                significance level of each test family
  bonferroni_family    number of tests of the Bonferroni correction; null: the number of k levels
  min_complete_seeds   the analysis stops with fewer seeds that have every k level
  sensitivity_gt_round also write the tables of the second round rule: with round_source
                       "validation", the ground truth's round for every k level; with "test",
                       each run at its own round (framework.analysis.select_round)
  track_b              run Track B in R (repeated-measures ANOVA and the Tukey test);
                       false: Track A only, without R
  rscript              the Rscript command
  output_folder        folder in the data folder for the analysis results
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
if __package__ in (None, ""):  # started as a file: make the package 'framework' importable
    sys.path.insert(0, str(ROOT))

from framework import run_pipeline  # noqa: E402
from framework.training import run_training  # noqa: E402

SECTION = "analysis"
GROUND_TRUTH = run_training.GROUND_TRUTH
METRICS = ("f1", "auc_roc", "pr_auc", "precision", "recall", "specificity", "accuracy",
           "loss", "tp", "tn", "fp", "fn")
HIGHER_IS_BETTER = ("f1", "auc_roc", "pr_auc", "precision", "recall", "specificity", "accuracy")
TEST_COLUMNS = tuple(f"test_{name}" for name in METRICS)
SELECTION_COLUMNS = tuple(f"val_{name}" for name in HIGHER_IS_BETTER)
ROUND_SOURCES = ("validation", "test")
DEFAULTS = {
    "round_source": None,
    "selection_metric": "val_f1",
    "tie_break": "val_pr_auc",
    "metric": "test_f1",
    "extra_metrics": ["test_pr_auc", "test_auc_roc"],
    "alpha": 0.05,
    "bonferroni_family": None,
    "min_complete_seeds": 3,
    "sensitivity_gt_round": True,
    "track_b": True,
    "rscript": "Rscript",
    "output_folder": "08_results_analysis",
}


def _whole_number(value, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def check_analysis(section, configs):
    """Return a list of error messages for the section "analysis" (empty if valid)."""
    if not isinstance(section, dict):
        return [f"{SECTION} must be an object"]
    errors = []
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        errors.append(f"{SECTION}: unknown keys {sorted(unknown)} (known: {sorted(DEFAULTS)})")
    values = {**DEFAULTS, **section}
    if values["round_source"] is not None and values["round_source"] not in ROUND_SOURCES:
        errors.append(f"{SECTION}.round_source must be null or one of {list(ROUND_SOURCES)}")
    for key in ("selection_metric", "tie_break"):
        if values[key] not in SELECTION_COLUMNS:
            errors.append(f"{SECTION}.{key} must be one of {list(SELECTION_COLUMNS)}")
    if values["tie_break"] == values["selection_metric"]:
        errors.append(f"{SECTION}.tie_break must differ from {SECTION}.selection_metric")
    if values["metric"] not in TEST_COLUMNS:
        errors.append(f"{SECTION}.metric must be one of {list(TEST_COLUMNS)}")
    extra = values["extra_metrics"]
    if not isinstance(extra, list) or not all(name in TEST_COLUMNS for name in extra):
        errors.append(f"{SECTION}.extra_metrics must be a list of names from {list(TEST_COLUMNS)}")
    elif len(set(extra)) != len(extra) or values["metric"] in extra:
        errors.append(f"{SECTION}.extra_metrics must not repeat a name or the metric")
    alpha = values["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        errors.append(f"{SECTION}.alpha must be a number above 0 and below 1")
    family = values["bonferroni_family"]
    if family is not None and not _whole_number(family, 1):
        errors.append(f"{SECTION}.bonferroni_family must be null or a whole number (1 or more)")
    if not _whole_number(values["min_complete_seeds"], 3):
        errors.append(f"{SECTION}.min_complete_seeds must be a whole number (3 or more)")
    if not isinstance(values["sensitivity_gt_round"], bool):
        errors.append(f"{SECTION}.sensitivity_gt_round must be true or false")
    if not isinstance(values["track_b"], bool):
        errors.append(f"{SECTION}.track_b must be true or false")
    if not isinstance(values["rscript"], str) or not values["rscript"].strip():
        errors.append(f"{SECTION}.rscript must be a non-empty text")
    folder = values["output_folder"]
    if not run_pipeline._is_folder_name(folder) or folder in (".", ".."):
        errors.append(f"{SECTION}.output_folder must be a folder name without '/' or '\\'")
    if GROUND_TRUTH not in configs:
        errors.append(f"the training k levels must include {GROUND_TRUTH} (the analysis compares with it)")
    elif len(configs) < 2:
        errors.append("the training k levels must include at least one level besides the ground truth")
    return errors


def load_analysis(config_path):
    """
    Read and check run_config.json for the analysis. Stop with a list of all faults.

    Returns:
        dict: setup (of run_training.load_training), settings (the section with
        defaults), configs (the k folders, ground truth first), seeds, state_folder
        (the folder of the training state files) and output_folder
    """
    setup = run_training.load_training(config_path)
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    section = data.get(SECTION, {})
    configs = list(setup["k_folders"])
    errors = check_analysis(section, configs)
    if errors:
        raise run_pipeline.RunConfigError(f"Run config {config_path} is not valid:\n  - " + "\n  - ".join(errors))
    settings = {**DEFAULTS, **section}
    settings["extra_metrics"] = list(settings["extra_metrics"])
    if settings["round_source"] is None:
        validation_share = setup["gen"]["data_preparation"]["validation_share"]
        settings["round_source"] = "validation" if validation_share > 0 else "test"
    configs = [GROUND_TRUTH] + [name for name in configs if name != GROUND_TRUTH]
    if settings["bonferroni_family"] is None:
        settings["bonferroni_family"] = len(configs) - 1
    tag = run_training.run_results_tag(setup)
    output = (Path(setup["gen"]["data_dir"]) / settings["output_folder"] / tag
              / f"split_{setup['split_ratio']}")
    return {
        "setup": setup,
        "settings": settings,
        "configs": configs,
        "seeds": [int(seed) for seed in setup["seeds"]],
        "results_tag": tag,
        "state_folder": run_training.state_folder(setup),
        "output_folder": output,
    }


def metric_columns(settings):
    """The test columns of the tables: metric first, then extra_metrics."""
    return [settings["metric"]] + list(settings["extra_metrics"])
