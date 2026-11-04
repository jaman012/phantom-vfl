"""
Data input and output of the generation stages.

The folder paths of one seed; the loader of the integer-form training data
(the ground-truth stage, and the row-set check of the phantom generation
stage); and the loader of the integer-form ground truth (the phantom
generation stage), which checks the ground truth against the training data.

Folder layout under the data folder (--data-dir):
  <cleaned CSV of the dataset config>                                (output of the dataset adapter)
  03_train_test_splits/<split folder>/rs<seed>/split_indices/       (train and test row indices)
  03_train_test_splits/<split folder>/rs<seed>/train/<train CSV of the dataset config>
  03_train_test_splits/<split folder>/rs<seed>/test/                (integer-form test set)
  03_train_test_splits/<split folder>/rs<seed>/validation/          (integer-form validation set, if any)
  03_train_test_splits/<split folder>/rs<seed>/encoding_maps.json   (integer maps of the split)
  04_federated_datasets/<split folder>/rs<seed>/                    (integer-form datasets)
  03_train_test_splits/<model split folder>/rs<seed>/test/          (model-form test set)
  03_train_test_splits/<model split folder>/rs<seed>/validation/    (model-form validation set, if any)
  04_federated_datasets/<model split folder>/rs<seed>/              (model-form datasets)

The split folder names come from the shares of the split (split_folder_name):
split_<train>_<test> (for example split_80_20), or split_<train>_<validation>_<test>
with a validation set (for example split_60_20_20). The model-form folder adds
_onehot. The shares are whole percentages; the training share is the rest.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from encoding import apply_integer_maps, load_integer_maps

SPLITS_FOLDER = "03_train_test_splits"
FEDERATED_FOLDER = "04_federated_datasets"
DEFAULT_TEST_SHARE = 0.2
DEFAULT_VALIDATION_SHARE = 0.0
MODEL_FOLDER_SUFFIX = "_onehot"
MAPS_FILE = "encoding_maps.json"
SPLIT_INDICES_FOLDER = "split_indices"
TEST_FOLDER = "test"
VALIDATION_FOLDER = "validation"


# ============================================================================
# SPLIT SHARES AND FOLDER NAMES
# ============================================================================

def share_percent(value, name):
    """The share as a whole percentage (0.2 gives 20). Stop if it is not a number or not a whole percentage."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    percent = round(value * 100)
    if abs(value * 100 - percent) > 1e-9:
        raise ValueError(f"{name} must be a whole percentage (for example 0.2 or 0.15), got {value}")
    return int(percent)


def split_percentages(test_share=DEFAULT_TEST_SHARE, validation_share=DEFAULT_VALIDATION_SHARE):
    """
    (training, validation, test) as whole percentages; the training share is
    the rest. Stop if a share is wrong: the test share must be 0.01 or more,
    the validation share 0 or more, and the training share 0.01 or more.
    """
    test = share_percent(test_share, "test_share")
    validation = share_percent(validation_share, "validation_share")
    if test < 1:
        raise ValueError(f"test_share must be 0.01 or more, got {test_share}")
    if validation < 0:
        raise ValueError(f"validation_share must be 0 or more, got {validation_share}")
    train = 100 - test - validation
    if train < 1:
        raise ValueError(f"test_share + validation_share must be 0.99 or less, got {test_share} + {validation_share}")
    return train, validation, test


def split_folder_name(test_share=DEFAULT_TEST_SHARE, validation_share=DEFAULT_VALIDATION_SHARE, model=False):
    """
    Split folder name from the shares: split_<train>_<test>, or
    split_<train>_<validation>_<test> with a validation set. model=True adds
    _onehot (the model-form folder).
    """
    train, validation, test = split_percentages(test_share, validation_share)
    name = f"split_{train}_{test}" if validation == 0 else f"split_{train}_{validation}_{test}"
    return name + (MODEL_FOLDER_SUFFIX if model else "")


DEFAULT_SPLIT_FOLDER = split_folder_name()
DEFAULT_MODEL_SPLIT_FOLDER = split_folder_name(model=True)


# ============================================================================
# PATHS
# ============================================================================

def split_seed_dir(data_dir, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Folder of one seed in the train-test splits."""
    return Path(data_dir) / SPLITS_FOLDER / split_folder / f"rs{int(seed)}"


def train_csv_path(data_dir, cfg, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Training CSV of one seed (file name from the dataset config)."""
    return split_seed_dir(data_dir, seed, split_folder) / "train" / cfg.train_csv


def split_maps_path(data_dir, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Integer maps that the split saved for one seed."""
    return split_seed_dir(data_dir, seed, split_folder) / MAPS_FILE


def federated_seed_dir(data_dir, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Folder of one seed in the generated datasets."""
    return Path(data_dir) / FEDERATED_FOLDER / split_folder / f"rs{int(seed)}"


def cleaned_csv_path(data_dir, cfg):
    """Cleaned CSV of the dataset (output of the dataset adapter; path from the dataset config)."""
    return Path(data_dir) / cfg.cleaned_csv


def split_indices_dir(data_dir, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Folder with the train and test row indices of one seed."""
    return split_seed_dir(data_dir, seed, split_folder) / SPLIT_INDICES_FOLDER


def split_test_dir(data_dir, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Folder with the test set of one seed."""
    return split_seed_dir(data_dir, seed, split_folder) / TEST_FOLDER


def split_validation_dir(data_dir, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Folder with the validation set of one seed (only with a validation share)."""
    return split_seed_dir(data_dir, seed, split_folder) / VALIDATION_FOLDER


# ============================================================================
# INTEGER-FORM TRAINING DATA
# ============================================================================

def label_codes(labels, cfg):
    """
    Return the label column as codes.

    'fit_on_train' dataset: each label text gets its code from the label map of
    the dataset config. A 'pre_encoded' dataset keeps its stored codes. A value
    that is not in the label map stops the run.
    """
    if cfg.integer_encoding == "pre_encoded":
        unknown = sorted(set(labels.unique()) - set(cfg.label_map.values()), key=str)
        if unknown:
            raise ValueError(f"Label column '{cfg.label_column}' has codes that are not in the "
                             f"label map of the dataset config: {unknown}")
        return labels.copy()
    text = labels.astype(str)
    unknown = sorted(set(text.unique()) - set(cfg.label_map))
    if unknown:
        raise ValueError(f"Label column '{cfg.label_column}' has values that are not in the "
                         f"label map of the dataset config: {unknown}")
    return text.map(cfg.label_map)


def load_integer_training_data(data_dir, cfg, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """
    Read the training split of one seed and return it in the integer form.

    A 'fit_on_train' dataset uses the integer maps that the split saved
    (encoding_maps.json); the maps are not built again. A 'pre_encoded'
    dataset has no maps.

    Args:
        data_dir: data folder (holds 03_train_test_splits/)
        cfg: DatasetConfig
        seed: seed of the split
        split_folder: name of the split folder

    Returns:
        (df_int, maps, path): the training rows in the integer form (features
        and label codes, row index as in the CSV), the maps ({} for a
        'pre_encoded' dataset), and the path of the training CSV
    """
    path = train_csv_path(data_dir, cfg, seed, split_folder)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}\n"
                                f"Run split_train_test.py with the same seed first.")
    raw = pd.read_csv(path)
    missing = [c for c in list(cfg.features) + [cfg.label_column] if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns of the dataset config missing in {path}: {missing}")

    if cfg.integer_encoding == "pre_encoded":
        maps = {}
    else:
        maps = load_integer_maps(split_maps_path(data_dir, seed, split_folder))
        needed = [f for f, spec in cfg.features.items() if spec["type"] != "continuous"]
        missing_maps = [c for c in needed + [cfg.label_column] if c not in maps]
        if missing_maps:
            raise ValueError(f"Integer maps of the split have no entry for: {missing_maps}")

    df_int = apply_integer_maps(raw, maps, cfg)
    df_int[cfg.label_column] = label_codes(raw[cfg.label_column], cfg)
    return df_int, maps, path


# ============================================================================
# INTEGER-FORM GROUND TRUTH (INPUT OF THE PHANTOM GENERATION STAGE)
# ============================================================================

GROUND_TRUTH_FOLDER = "ground_truth"
GROUND_TRUTH_HINT = "Run the ground-truth stage (generate_ground_truth_dataset.py) with the same seed first."


def ground_truth_dir(data_dir, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """Integer-form ground truth of one seed (written by the ground-truth stage)."""
    return federated_seed_dir(data_dir, seed, split_folder) / GROUND_TRUTH_FOLDER


def load_integer_ground_truth(data_dir, cfg, seed, split_folder=DEFAULT_SPLIT_FOLDER):
    """
    Read the integer-form ground truth of one seed and check it (row-set check).

    The ground truth is the full PSI result with no removal: every training
    row of the seed, all REAL. The run stops at the first fault:
    1. a file of the ground truth is missing;
    2. the row counts of labels.csv, the client files and config.json differ;
    3. the columns of a client file are not the client's features in the
       dataset config;
    4. ground_truth_analysis.csv does not list person_0 to person_<n-1>, all
       REAL, with the labels of labels.csv;
    5. the training split (read again with load_integer_training_data)
       differs from the ground truth in its row count, a feature value or a
       label.

    Returns:
        (df_int, maps, train_path, files): the ground-truth rows in the
        integer form (the features of each client, then the label column;
        row index 0 to n-1), the integer maps of the split, the path of the
        training CSV, and {name: path} of labels.csv and of the client files
        (names relative to the ground-truth folder, for the completion marker)
    """
    gt_dir = ground_truth_dir(data_dir, seed, split_folder)
    files = {"labels.csv": gt_dir / "labels.csv"}
    for client_index in range(cfg.n_clients):
        files[f"client_{client_index}/features.csv"] = gt_dir / f"client_{client_index}" / "features.csv"
    other_files = [gt_dir / "config.json", gt_dir / "ground_truth_analysis.csv"]
    missing = [str(path) for path in list(files.values()) + other_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Integer-form ground truth incomplete for seed {seed}: {missing}\n"
                                f"{GROUND_TRUTH_HINT}")

    def fault(text):
        return ValueError(f"Ground truth {gt_dir}: {text}. {GROUND_TRUTH_HINT}")

    # Checks 2 and 3: row counts and client columns
    labels = pd.read_csv(files["labels.csv"])
    if list(labels.columns) != ["label"]:
        raise fault(f"labels.csv has the columns {list(labels.columns)}, expected ['label']")
    n_rows = len(labels)
    with open(gt_dir / "config.json") as f:
        stated_rows = json.load(f).get("num_samples")
    if stated_rows != n_rows:
        raise fault(f"config.json states {stated_rows} rows, labels.csv has {n_rows}")
    parts = []
    for client_index, features in enumerate(cfg.clients):
        client = pd.read_csv(files[f"client_{client_index}/features.csv"])
        if list(client.columns) != list(features):
            raise fault(f"client_{client_index}/features.csv has the columns {list(client.columns)}, "
                        f"the dataset config gives {list(features)}")
        if len(client) != n_rows:
            raise fault(f"client_{client_index}/features.csv has {len(client)} rows, labels.csv has {n_rows}")
        parts.append(client)

    # Check 4: every row is a REAL individual with the label of labels.csv
    analysis = pd.read_csv(gt_dir / "ground_truth_analysis.csv")
    ids = [f"person_{index}" for index in range(n_rows)]
    if (not {"id", "status", "label"} <= set(analysis.columns) or len(analysis) != n_rows
            or analysis["id"].astype(str).tolist() != ids or not (analysis["status"] == "REAL").all()
            or not np.array_equal(analysis["label"].to_numpy(dtype=float), labels["label"].to_numpy(dtype=float))):
        raise fault("ground_truth_analysis.csv does not list person_0 to person_<n-1>, all REAL, "
                    "with the labels of labels.csv")

    df_int = pd.concat(parts, axis=1)
    df_int[cfg.label_column] = labels["label"].to_numpy()

    # Check 5: the ground truth equals the training split, row by row
    train, maps, train_path = load_integer_training_data(data_dir, cfg, seed, split_folder)
    if len(train) != n_rows or not train.index.equals(pd.RangeIndex(n_rows)):
        raise fault(f"the training split has {len(train)} rows, the ground truth has {n_rows}")
    for column in [f for features in cfg.clients for f in features] + [cfg.label_column]:
        expected = train[column].to_numpy(dtype=float)
        found = df_int[column].to_numpy(dtype=float)
        same = (expected == found) | (np.isnan(expected) & np.isnan(found))
        if not same.all():
            raise fault(f"column '{column}' differs from the training split "
                        f"(first at row {int(np.flatnonzero(~same)[0])})")
    return df_int, maps, train_path, files
