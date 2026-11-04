"""
PHANTOM VFL STAGE 2: GENERATE THE GROUND-TRUTH DATASET

The ground truth has no phantom protocol: no PSI removal and no substitution.
Every training row of the seed is a real individual (0 % phantoms).

For one dataset and one seed, this script:
1. Loads the training rows in the integer form (data_io). Stage 3 reads the
   ground truth that this stage saves, and checks it against the same rows.
2. Saves the integer-form ground truth: the labels, the features of each
   client, the tracking files and config.json. It fits one StandardScaler for
   each client on these features and saves it (the setting scaler_columns
   gives the columns that the scaler changes; encoding.py).
3. Builds the model form (fit_model_form): the one-hot columns of the nominal
   features are learned from the training rows and applied to the test rows
   (and to the validation rows, if the split has them). It saves the
   model-form ground truth, the model-form test set (and validation set), one
   StandardScaler for each client, and config.json.

The features are saved unscaled; the training code applies the scalers.

Usage:
    python generate_ground_truth_dataset.py --dataset-config <dataset.json> \
        --data-dir <data folder> --seed 42 [--test-share 0.2] [--validation-share 0.0] \
        [--scaler-columns continuous_ordinal]

The shares give the split folder names; they must be the shares of stage 1.

The runner calls run() with the same settings (GroundTruthSettings).
When the script runs alone from its command line, its screen output also goes
to a log file and a warnings file in <data folder>/logs/ (run_log.py).
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FRAMEWORK_DIR = Path(__file__).resolve().parent
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

import data_io  # noqa: E402
from dataset_config import DatasetConfigError, load_dataset_config  # noqa: E402
from run_log import run_log, set_context  # noqa: E402
from encoding import (DEFAULT_SCALER_COLUMNS, SCALER_COLUMNS, check_scaler_columns,  # noqa: E402
                      client_model_columns, encode_model_from_raw, fit_scaler,
                      low_variance_columns, save_scaler, scaled_columns)

GROUND_TRUTH_FOLDER = "ground_truth"
SCALERS_FOLDER = "scalers"
DESCRIPTION = "Baseline VFL dataset without phantom protocol"
# Texts of config.json for each value of scaler_columns; 'all' gives the texts
# of the scripts of the published results (byte-identical files)
NORMALIZATION_TEXT = {
    "all": "RAW features saved, scalers fitted for ISLP compliance",
    "continuous_ordinal": ("RAW features saved; scalers fitted on the training rows; they scale the "
                           "continuous and ordinal columns, the other columns pass through (mean 0, scale 1)"),
}
MODEL_SCALING_TEXT = {
    "all": ("StandardScaler fit on full per-client matrix (continuous + all encoded columns) — "
            "mirrors baseline pipeline's uniform scaling so only encoding differs"),
    "continuous_ordinal": ("StandardScaler fit on the training rows of each client; it scales the continuous "
                           "and ordinal columns; binary and one-hot columns pass through (mean 0, scale 1)"),
}


@dataclass
class GroundTruthSettings:
    """Settings of one run of this stage (the runner fills them from run_config.json)."""
    dataset_config: str
    data_dir: str
    seed: int
    test_share: float = data_io.DEFAULT_TEST_SHARE
    validation_share: float = data_io.DEFAULT_VALIDATION_SHARE
    scaler_columns: str = DEFAULT_SCALER_COLUMNS


# ============================================================================
# INTEGER FORM
# ============================================================================

def write_tracking_files(labels, cfg, out_dir):
    """
    Write the files that document 0 % phantoms in the ground truth:
    ground_truth_analysis.csv (every ID is REAL) and phantom_rate_summary.json.
    """
    analysis = pd.DataFrame({
        "id": [f"person_{index}" for index in range(len(labels))],
        "is_phantom": False,
        "status": "REAL",
        "removed_by_count": 0,
        "removed_by_clients": "NONE",
        "label": labels.to_numpy(),
    })
    analysis.to_csv(out_dir / "ground_truth_analysis.csv", index=False)
    clients = [f"Client_{index}" for index in range(cfg.n_clients)]
    summary = {
        "dataset_type": "ground_truth",
        "total_ids_after_psi": len(labels),
        "total_ids_after_removal": len(labels),
        "client_removal_counts": {name: 0 for name in clients},
        "expected_phantom_rate_per_client": {name: 0.0 for name in clients},
        "overall_expected_phantom_rate": 0.0,
        "actual_phantom_count": 0,
        "actual_phantom_rate": 0.0,
        "removed_by_single_client": 0,
        "removed_by_multiple_clients": 0,
        "removed_by_all_clients": 0,
    }
    with open(out_dir / "phantom_rate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def save_integer_ground_truth(df_int, cfg, out_dir, scaler_dir, source_file,
                              scaler_columns=DEFAULT_SCALER_COLUMNS):
    """
    Save the integer-form ground truth and fit its scalers.

    source_file: path of the training CSV relative to the data folder, with
    '/' (written to config.json, so the file holds no machine-specific path).
    scaler_columns: the columns that each scaler changes (encoding.scaled_columns).

    Returns:
        the float labels (column 'label')
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = df_int[cfg.label_column].astype(float)
    pd.DataFrame({"label": labels}).to_csv(out_dir / "labels.csv", index=False)

    for client_index, features in enumerate(cfg.clients):
        client_dir = out_dir / f"client_{client_index}"
        client_dir.mkdir(parents=True, exist_ok=True)
        client_features = df_int[features]
        client_features.to_csv(client_dir / "features.csv", index=False)
        scaled = scaled_columns(features, cfg, scaler_columns)
        save_scaler(fit_scaler(client_features, scaled), scaler_dir / f"scaler_client_{client_index}.pkl")
        print(f"✓ client_{client_index}: {len(features)} features, {client_features.shape[0]:,} rows; "
              f"scaler fitted (scales {len(scaled)} of {len(features)} columns)")

    write_tracking_files(labels, cfg, out_dir)

    config = {
        "dataset_type": "ground_truth",
        "description": DESCRIPTION,
        "phantom_protocol_applied": False,
        "num_samples": len(df_int),
        "num_features": sum(len(features) for features in cfg.clients),
        "num_clients": cfg.n_clients,
        "feature_split": "+".join(str(len(features)) for features in cfg.clients),
        "vertical_split": {f"client_{index}": list(features) for index, features in enumerate(cfg.clients)},
        "label_distribution": {
            "class_0": int((labels == 0).sum()),
            "class_1": int((labels == 1).sum()),
        },
        "preprocessing": cfg.output_text["ground_truth_preprocessing"],
        "normalization": NORMALIZATION_TEXT[scaler_columns],
        "source_file": source_file,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return labels


# ============================================================================
# MODEL FORM
# ============================================================================

def fit_model_form(train_raw, test_raw, maps, cfg):
    """
    Learn the model form from the training rows and apply it to the test rows.

    The one-hot columns come from the training rows only (encode_model_from_raw).

    Returns:
        (train_model, test_model, client_columns): the model-form rows and the
        model columns of each client
    """
    train_model, test_model = encode_model_from_raw(train_raw, test_raw, maps, cfg)
    return train_model, test_model, client_model_columns(list(train_model.columns), cfg)


def model_test_labels(test_raw, cfg):
    """Test labels in the format that the training code reads (test_label_format)."""
    if cfg.test_label_format == "text":
        return test_raw[cfg.label_column].astype(str).str.strip().to_frame(cfg.label_column)
    return test_raw[cfg.label_column].astype(float).to_frame("label")


def save_model_form(train_model, test_model, client_columns, train_labels, test_raw, cfg,
                    out_dir, test_dir, scaler_dir, seed, split_folder,
                    scaler_columns=DEFAULT_SCALER_COLUMNS):
    """
    Save the model-form ground truth, test set and scalers, and config.json.

    scaler_columns: the columns that each scaler changes (encoding.scaled_columns).
    The near-zero-variance warning checks these columns only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": train_labels.to_numpy()}).to_csv(out_dir / "labels.csv", index=False)
    model_test_labels(test_raw, cfg).to_csv(test_dir / "test_labels.csv", index=False)

    for client_index, columns in enumerate(client_columns):
        client_train = train_model[columns]
        scaled = scaled_columns(columns, cfg, scaler_columns)
        low_variance = low_variance_columns(client_train[scaled])
        if low_variance:
            print(f"  [!] client_{client_index}: near-zero variance column(s) before scaling: "
                  f"{low_variance} (usually a rare category)")
        client_dir = out_dir / f"client_{client_index}"
        client_dir.mkdir(parents=True, exist_ok=True)
        client_train.to_csv(client_dir / "features.csv", index=False)
        test_model[columns].to_csv(test_dir / f"client_{client_index}_test_features.csv", index=False)
        save_scaler(fit_scaler(client_train, scaled), scaler_dir / f"scaler_client_{client_index}.pkl")
        print(f"✓ client_{client_index}: {len(columns)} model columns; scaler fitted "
              f"(scales {len(scaled)} of {len(columns)} columns)")

    config = {
        "dataset_type": "ground_truth_onehot",
        "purpose": cfg.output_text["model_purpose"],
        "onehot_features": cfg.nominal_in_model_order(),
        "continuous_features": [f for f in cfg.model_feature_order if cfg.feature_type(f) == "continuous"],
        "binary_passthrough": [f for f in cfg.model_feature_order if cfg.feature_type(f) == "binary"],
    }
    ordinal = [f for f in cfg.model_feature_order if cfg.feature_type(f) == "ordinal"]
    if ordinal:
        config["ordinal_features"] = ordinal
    config["scaling"] = MODEL_SCALING_TEXT[scaler_columns]
    config["seed"] = seed
    config["source_split"] = cfg.output_text["model_source_split"].format(split_folder=split_folder, seed=seed)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_model_validation(train_raw, validation_raw, maps, cfg, client_columns, validation_dir):
    """
    Encode the validation rows with the one-hot columns of the training rows
    (encode_model_from_raw, as for the test rows) and save the model-form
    validation set: the features of each client and the labels in the
    test-label format.
    """
    _, validation_model = encode_model_from_raw(train_raw, validation_raw, maps, cfg)
    validation_dir.mkdir(parents=True, exist_ok=True)
    model_test_labels(validation_raw, cfg).to_csv(validation_dir / "validation_labels.csv", index=False)
    for client_index, columns in enumerate(client_columns):
        validation_model[columns].to_csv(validation_dir / f"client_{client_index}_validation_features.csv",
                                         index=False)


# ============================================================================
# RUN
# ============================================================================

def run(settings):
    """
    Generate the ground truth of one seed (integer form and model form).

    Returns:
        summary for the runner gate: output folders, rows, label counts,
        features per client and model columns per client
    """
    seed = int(settings.seed)
    if seed < 0:
        raise ValueError(f"seed must be 0 or more, got {seed}")
    set_context(f"seed {seed}, stage 2")
    scaler_columns = check_scaler_columns(settings.scaler_columns)
    _, validation_pct, _ = data_io.split_percentages(settings.test_share, settings.validation_share)
    split_folder = data_io.split_folder_name(settings.test_share, settings.validation_share)
    model_split_folder = data_io.split_folder_name(settings.test_share, settings.validation_share, model=True)
    cfg = load_dataset_config(settings.dataset_config)
    data_dir = Path(settings.data_dir)

    print("\n" + "=" * 80)
    print(f"PHANTOM VFL STAGE 2: GROUND TRUTH - {cfg.name}")
    print("=" * 80)
    print(f"Seed: {seed}")
    print(f"Scaler columns: {scaler_columns}")

    df_int, maps, data_path = data_io.load_integer_training_data(data_dir, cfg, seed, split_folder)
    print(f"Training data: {data_path} ({len(df_int):,} rows)")

    # Integer form
    seed_dir = data_io.federated_seed_dir(data_dir, seed, split_folder)
    print("\nInteger form:")
    labels = save_integer_ground_truth(df_int, cfg, seed_dir / GROUND_TRUTH_FOLDER,
                                       seed_dir / SCALERS_FOLDER, data_path.relative_to(data_dir).as_posix(),
                                       scaler_columns)

    # Model form: raw training rows (the training CSV) and raw test rows
    # (the cleaned CSV, selected with the saved test indices)
    cleaned_path = data_io.cleaned_csv_path(data_dir, cfg)
    indices_dir = data_io.split_indices_dir(data_dir, seed, split_folder)
    test_indices_path = indices_dir / "test_indices.csv"
    validation_indices_path = indices_dir / "validation_indices.csv"
    needed = [cleaned_path, test_indices_path] + ([validation_indices_path] if validation_pct > 0 else [])
    for path in needed:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run split_train_test.py with the same seed "
                                    f"and shares first.")
    train_raw = pd.read_csv(data_path)
    cleaned = pd.read_csv(cleaned_path)
    test_indices = pd.read_csv(test_indices_path)["index"].tolist()
    test_raw = cleaned.loc[test_indices].reset_index(drop=True)

    train_model, test_model, client_columns = fit_model_form(train_raw, test_raw, maps, cfg)
    model_seed_dir = data_io.federated_seed_dir(data_dir, seed, model_split_folder)
    model_test_dir = data_io.split_test_dir(data_dir, seed, model_split_folder)
    print("\nModel form:")
    save_model_form(train_model, test_model, client_columns, labels, test_raw, cfg,
                    model_seed_dir / GROUND_TRUTH_FOLDER, model_test_dir, model_seed_dir / SCALERS_FOLDER,
                    seed, split_folder, scaler_columns)
    validation_rows = 0
    if validation_pct > 0:
        validation_indices = pd.read_csv(validation_indices_path)["index"].tolist()
        validation_raw = cleaned.loc[validation_indices].reset_index(drop=True)
        model_validation_dir = data_io.split_validation_dir(data_dir, seed, model_split_folder)
        save_model_validation(train_raw, validation_raw, maps, cfg, client_columns, model_validation_dir)
        validation_rows = len(validation_raw)
        print(f"✓ Saved model-form validation set ({validation_rows:,} rows): {model_validation_dir}")

    summary = {
        "dataset": cfg.name,
        "seed": seed,
        "output_dir": str(seed_dir / GROUND_TRUTH_FOLDER),
        "model_output_dir": str(model_seed_dir / GROUND_TRUTH_FOLDER),
        "rows": len(df_int),
        "label_counts": {"class_0": int((labels == 0).sum()), "class_1": int((labels == 1).sum())},
        "features_per_client": [len(features) for features in cfg.clients],
        "model_columns_per_client": [len(columns) for columns in client_columns],
        "test_rows": len(test_raw),
        "validation_rows": validation_rows,
        "scaler_columns": scaler_columns,
    }
    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETED")
    print("=" * 80 + "\n")
    return summary


def main(argv=None):
    """Command line of this stage."""
    parser = argparse.ArgumentParser(description="Stage 2: generate the ground truth of one seed")
    parser.add_argument("--dataset-config", required=True, help="Path of the dataset config (dataset.json)")
    parser.add_argument("--data-dir", default=".",
                        help="Data folder that holds the cleaned CSV and 03_train_test_splits/ (default: current folder)")
    parser.add_argument("--seed", type=int, required=True, help="Seed of the split")
    parser.add_argument("--test-share", type=float, default=data_io.DEFAULT_TEST_SHARE,
                        help="Test share of stage 1, a whole percentage; with the validation share it "
                             "gives the split folder names (default: %(default)s)")
    parser.add_argument("--validation-share", type=float, default=data_io.DEFAULT_VALIDATION_SHARE,
                        help="Validation share of stage 1, a whole percentage; 0: no validation set "
                             "(default: %(default)s)")
    parser.add_argument("--scaler-columns", choices=SCALER_COLUMNS, default=DEFAULT_SCALER_COLUMNS,
                        help="Columns that the scalers change: continuous_ordinal (the continuous and "
                             "ordinal columns; the other columns pass through) or all (default: %(default)s)")
    args = parser.parse_args(argv)
    settings = GroundTruthSettings(
        dataset_config=args.dataset_config,
        data_dir=args.data_dir,
        seed=args.seed,
        test_share=args.test_share,
        validation_share=args.validation_share,
        scaler_columns=args.scaler_columns,
    )
    with run_log(settings.data_dir, "stage2"):
        return run(settings)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, DatasetConfigError) as error:
        sys.exit(f"ERROR: {error}")
