"""
Data access of the Flower app (James Mansueto's own code).

The Flower app reads the federated datasets and the test set that the framework
writes. The server app puts the run settings into environment variables, and
the client app and the server app call the functions of this module:

  PHANTOM_DATA_DIR        data folder of the dataset (run-config key data-dir)
  PHANTOM_DATASET_CONFIG  path of dataset.json (run-config key dataset-config)
  SPLIT_RATIO             split folder name without "split_", for example
                          80_20 (integer form) or 80_20_onehot (model form)
  RANDOM_SEED             seed of the split, for example 42
  PHANTOM_DATASET         k folder, for example k0 (k*), k20 or ground_truth

Files that this module reads (under the data folder):
  04_federated_datasets/split_<ratio>/rs<seed>/<k>/client_<i>/features.csv
  04_federated_datasets/split_<ratio>/rs<seed>/<k>/labels.csv
  04_federated_datasets/split_<ratio>/rs<seed>/scalers/scaler_client_<i>.pkl
  03_train_test_splits/split_<ratio>/rs<seed>/test/client_<i>_test_features.csv
  03_train_test_splits/split_<ratio>/rs<seed>/test/test_labels.csv
  04_federated_datasets/split_<ratio>/rs<seed>/ground_truth/labels.csv     (class-balance record)
  03_train_test_splits/split_<ratio>/rs<seed>/validation/client_<i>_validation_features.csv
  03_train_test_splits/split_<ratio>/rs<seed>/validation/validation_labels.csv
                                         (validation metrics and class-balance record;
                                          only with a validation share)

The feature names, the client split and the label map come from dataset.json.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

import data_io  # noqa: E402
from dataset_config import load_dataset_config  # noqa: E402

ENV_DATA_DIR = "PHANTOM_DATA_DIR"
ENV_DATASET_CONFIG = "PHANTOM_DATASET_CONFIG"
ENV_SPLIT_RATIO = "SPLIT_RATIO"
ENV_RANDOM_SEED = "RANDOM_SEED"
ENV_PHANTOM_DATASET = "PHANTOM_DATASET"

LABELS_FILE = "labels.csv"
LABELS_COLUMN = "label"
SCALERS_FOLDER = "scalers"

_config_cache = {}


def _required_env(name):
    """Value of an environment variable that the server app sets; stop if it is missing."""
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. The server app sets it from the run config "
            f"(keys data-dir and dataset-config); give both keys to 'flwr run'."
        )
    return value


def data_dir():
    """Data folder of the run (PHANTOM_DATA_DIR)."""
    return Path(_required_env(ENV_DATA_DIR))


def dataset_config():
    """DatasetConfig of the run (PHANTOM_DATASET_CONFIG), read once for each path."""
    path = _required_env(ENV_DATASET_CONFIG)
    if path not in _config_cache:
        _config_cache[path] = load_dataset_config(path)
    return _config_cache[path]


def split_ratio():
    """Split folder name without "split_" (SPLIT_RATIO)."""
    return os.getenv(ENV_SPLIT_RATIO, "80_20")


def random_seed():
    """Seed of the split (RANDOM_SEED), as text."""
    return os.getenv(ENV_RANDOM_SEED, "42")


def phantom_dataset():
    """k folder of the run (PHANTOM_DATASET)."""
    return os.getenv(ENV_PHANTOM_DATASET, "k0")


def feature_columns():
    """All feature names, client by client, in the order of 'clients' in dataset.json."""
    return [feature for client in dataset_config().clients for feature in client]


def client_splits():
    """{"Client_<i>": [feature names]} from 'clients' in dataset.json."""
    return {f"Client_{index}": list(client) for index, client in enumerate(dataset_config().clients)}


def federated_seed_dir():
    """04_federated_datasets/split_<ratio>/rs<seed>/ of the run."""
    return data_dir() / data_io.FEDERATED_FOLDER / f"split_{split_ratio()}" / f"rs{random_seed()}"


def test_dir():
    """03_train_test_splits/split_<ratio>/rs<seed>/test/ of the run."""
    return data_dir() / data_io.SPLITS_FOLDER / f"split_{split_ratio()}" / f"rs{random_seed()}" / "test"


def phantom_dataset_dir():
    """Folder of the k level of the run."""
    return federated_seed_dir() / phantom_dataset()


def _check_feature_splits(feature_splits):
    """The run-config value feature-splits must add up to the number of features in dataset.json."""
    n_features = len(feature_columns())
    if sum(feature_splits) != n_features:
        raise ValueError(f"Feature splits sum ({sum(feature_splits)}) must equal total features ({n_features})")


def _client_id(partition_id, feature_splits):
    """Client number of a node: the partition ID modulo the number of clients."""
    return partition_id % len(feature_splits)


def _scale(df, client_id):
    """Apply the pickled scaler of the client to all columns of df; return float32 values."""
    scaler_path = federated_seed_dir() / SCALERS_FOLDER / f"scaler_client_{client_id}.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\nRun the ground-truth stage of the framework first to create scalers"
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    normalized_data = scaler.transform(df.values)
    print(f"[Client {client_id}] Normalized with the scaler of the ground-truth stage")
    return normalized_data.astype(np.float32)


def load_client_features(partition_id, feature_splits):
    """
    Training features of one client at the k level of the run, scaled.

    Args:
        partition_id: partition ID of the node (node config 'partition-id')
        feature_splits: list of the client widths from feature-splits, for example [5, 6]

    Returns:
        float32 array (rows of the k level, features of the client)
    """
    _check_feature_splits(feature_splits)
    client_id = _client_id(partition_id, feature_splits)
    base_path = phantom_dataset_dir()
    features_path = base_path / f"client_{client_id}" / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Training data not found: {features_path}\nMake sure phantom dataset exists at: {base_path}")
    df = pd.read_csv(features_path)
    print(f"[Client {client_id}] Loaded training data: {features_path}")
    print(f"[Client {client_id}] Shape: {df.shape}")
    return _scale(df, client_id)


def load_client_test_features(partition_id, feature_splits):
    """
    Test features of one client (the test set has no phantoms), scaled with the
    same scaler as the training features.

    Args and return value as load_client_features.
    """
    _check_feature_splits(feature_splits)
    client_id = _client_id(partition_id, feature_splits)
    base_path = test_dir()
    features_path = base_path / f"client_{client_id}_test_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Test data not found: {features_path}\nMake sure clean test set exists at: {base_path}")
    df = pd.read_csv(features_path)
    print(f"[Client {client_id}] Loaded test data: {features_path}")
    print(f"[Client {client_id}] Test shape: {df.shape}")
    return _scale(df, client_id)


def load_client_validation_features(partition_id, feature_splits):
    """
    Validation features of one client (the validation set has no phantoms),
    scaled with the same scaler as the training features. Only with a
    validation share (validation_dir() exists).

    Args and return value as load_client_features.
    """
    _check_feature_splits(feature_splits)
    client_id = _client_id(partition_id, feature_splits)
    base_path = validation_dir()
    features_path = base_path / f"client_{client_id}_validation_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Validation data not found: {features_path}\nMake sure the validation set exists at: {base_path}")
    df = pd.read_csv(features_path)
    print(f"[Client {client_id}] Loaded validation data: {features_path}")
    print(f"[Client {client_id}] Validation shape: {df.shape}")
    return _scale(df, client_id)


def load_train_labels():
    """Training labels of the k level of the run (column 'label' of labels.csv), float32, shape (N, 1)."""
    labels_path = phantom_dataset_dir() / LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"Training labels not found: {labels_path}")
    df = pd.read_csv(labels_path)
    labels = df[LABELS_COLUMN].values.astype(np.float32)
    print(f"Loaded training labels: {labels_path}")
    print(f"Training labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")
    return labels.reshape(-1, 1)


def load_test_labels():
    """
    Test labels of the run, float32, shape (N, 1).

    The label column and the label map come from dataset.json. data_io.label_codes()
    maps text labels to codes and keeps stored codes; a value that is not in the
    label map stops the run.
    """
    cfg = dataset_config()
    labels_path = test_dir() / "test_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Test labels not found: {labels_path}")
    df = pd.read_csv(labels_path)
    labels = data_io.label_codes(df[cfg.label_column], cfg).astype(np.float32).values
    print(f"Loaded test labels: {labels_path}")
    print(f"Test labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")
    return labels.reshape(-1, 1)


def validation_dir():
    """03_train_test_splits/split_<ratio>/rs<seed>/validation/ of the run (only with a validation share)."""
    return data_dir() / data_io.SPLITS_FOLDER / f"split_{split_ratio()}" / f"rs{random_seed()}" / data_io.VALIDATION_FOLDER


def load_ground_truth_labels():
    """
    Training labels of the ground truth of the run (no phantoms; column 'label' of
    ground_truth/labels.csv), float32, shape (N, 1).
    """
    labels_path = federated_seed_dir() / data_io.GROUND_TRUTH_FOLDER / LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"Ground-truth labels not found: {labels_path}")
    labels = pd.read_csv(labels_path)[LABELS_COLUMN].values.astype(np.float32)
    print(f"Loaded ground-truth labels: {labels_path}")
    return labels.reshape(-1, 1)


def load_validation_labels():
    """
    Validation labels of the run, mapped as the test labels, float32, shape (N, 1);
    None when the split has no validation folder (validation share 0).
    """
    folder = validation_dir()
    if not folder.exists():
        return None
    cfg = dataset_config()
    labels_path = folder / "validation_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Validation labels not found: {labels_path}")
    df = pd.read_csv(labels_path)
    labels = data_io.label_codes(df[cfg.label_column], cfg).astype(np.float32).values
    print(f"Loaded validation labels: {labels_path}")
    return labels.reshape(-1, 1)
