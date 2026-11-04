"""
PHANTOM VFL STAGE 1: SPLIT THE CLEANED DATA INTO TRAINING AND TEST ROWS

For one dataset and one seed, this script:
1. Reads the cleaned CSV of the dataset (the output of the dataset adapter).
2. Makes a stratified split in two steps. Step 1 splits off
   the test rows (test_share; the strata are the label codes; random_state
   is the seed). Step 2 runs only with a validation share: it splits off the
   validation rows from the other rows in the same way (share
   validation_share / (1 - test_share), the same seed). So the test rows do
   not depend on the validation share. It saves the row indices of each set.
3. Builds the integer maps from the training rows only and saves them as
   encoding_maps.json ('fit_on_train' datasets; a 'pre_encoded' dataset has
   no maps and no file).
4. Saves the test set and, with a validation share, the validation set: the
   features of each client in the integer form, and the labels as they are
   in the cleaned CSV.
5. Saves the training rows unchanged. Stages 2 and 3 read them.
6. Level check of the non-continuous features, on
   the integer form. A warning ([!]) for each level that has less than
   min_level_share of the training rows, for each level that is in the
   training rows but not in the test (or validation) rows, and for each
   value of a 'pre_encoded' dataset that is in the test (or validation) rows
   but not in the training rows. The run does not stop. (In a 'fit_on_train'
   dataset, such a value stops the run when the integer maps are applied.)

The split folder name comes from the shares (data_io.split_folder_name), for
example split_80_20, or split_60_20_20 with a validation share of 0.2.

Usage:
    python split_train_test.py --dataset-config <dataset.json> \
        --data-dir <data folder> --seed 42 [--test-share 0.2] \
        [--validation-share 0.0] [--min-level-share 0.01]

The runner calls run() with the same settings (SplitSettings).
When the script runs alone from its command line, its screen output also goes
to a log file and a warnings file in <data folder>/logs/ (run_log.py).
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FRAMEWORK_DIR = Path(__file__).resolve().parent
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

import data_io  # noqa: E402
from dataset_config import DatasetConfigError, load_dataset_config  # noqa: E402
from run_log import run_log, set_context  # noqa: E402
from encoding import apply_integer_maps, build_integer_maps, save_integer_maps  # noqa: E402

DEFAULT_MIN_LEVEL_SHARE = 0.01  # level check: a smaller share of the training rows gives a warning


@dataclass
class SplitSettings:
    """Settings of one run of this stage (the runner fills them from run_config.json)."""
    dataset_config: str
    data_dir: str
    seed: int
    test_share: float = data_io.DEFAULT_TEST_SHARE
    validation_share: float = data_io.DEFAULT_VALIDATION_SHARE
    min_level_share: float = DEFAULT_MIN_LEVEL_SHARE


def check_min_level_share(value):
    """Stop if min_level_share is not a number from 0 (no share check) to less than 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError(f"min_level_share must be a number from 0 to less than 1, got {value!r}")
    return float(value)


def label_counts(codes):
    """{code: (count, percent)} of a series of label codes."""
    return {int(code): (int((codes == code).sum()), round(float((codes == code).mean()) * 100, 1))
            for code in sorted(codes.unique())}


def _level_text(code, names):
    """A level for the warning text: its name from the integer map, or the code of a pre-encoded feature."""
    if code in names:
        return repr(names[code])
    value = float(code)
    return str(int(value)) if value.is_integer() else str(value)


def level_warnings(train_int, other_sets, maps, cfg, min_level_share):
    """
    Level check of the non-continuous features, on the
    integer form.

    Args:
        train_int: training rows in the integer form
        other_sets: {name: rows in the integer form}, for example {"test": ...}
        maps: integer maps of the split ({} for a 'pre_encoded' dataset)
        cfg: DatasetConfig
        min_level_share: smallest share of the training rows with no warning
            (0: no share check)

    Returns:
        the warning texts, without the [!] prefix
    """
    texts = []
    n_train = len(train_int)
    for feature, spec in cfg.features.items():
        if spec["type"] == "continuous":
            continue
        names = {code: level for level, code in maps.get(feature, {}).items()}
        counts = train_int[feature].value_counts()
        for code in sorted(set(counts.index) | set(names)):
            count = int(counts.get(code, 0))
            if n_train and count / n_train < min_level_share:
                texts.append(f"'{feature}' level {_level_text(code, names)}: {count:,} of {n_train:,} training "
                             f"rows ({100 * count / n_train:.2f} %), less than min_level_share "
                             f"({100 * min_level_share:g} %)")
        for name, rows in other_sets.items():
            other_counts = rows[feature].value_counts()
            for code in sorted(counts.index):
                if code not in other_counts.index:
                    texts.append(f"'{feature}' level {_level_text(code, names)}: in the training rows, "
                                 f"not in the {name} rows")
            for code in sorted(set(other_counts.index) - set(counts.index)):
                note = " (no one-hot column in the model form)" if spec["type"] == "nominal" else ""
                texts.append(f"'{feature}' value {_level_text(code, names)}: in {int(other_counts[code]):,} "
                             f"{name} row(s), not in the training rows{note}")
    return texts


def run(settings):
    """
    Split the cleaned data of one seed and save the training and test sets.

    Returns:
        summary for the runner gate: output folder, row counts, label shares
        of both sets, and whether integer maps were saved
    """
    seed = int(settings.seed)
    if seed < 0:
        raise ValueError(f"seed must be 0 or more, got {seed}")
    set_context(f"seed {seed}, stage 1")
    train_pct, validation_pct, test_pct = data_io.split_percentages(settings.test_share, settings.validation_share)
    min_level_share = check_min_level_share(settings.min_level_share)
    split_folder = data_io.split_folder_name(settings.test_share, settings.validation_share)
    cfg = load_dataset_config(settings.dataset_config)
    data_dir = Path(settings.data_dir)

    print("\n" + "=" * 80)
    print(f"PHANTOM VFL STAGE 1: TRAIN-TEST SPLIT - {cfg.name}")
    print("=" * 80)
    print(f"Seed: {seed}")
    print(f"Shares: training {train_pct} %, validation {validation_pct} %, test {test_pct} % "
          f"(split folder {split_folder})")

    cleaned_path = data_io.cleaned_csv_path(data_dir, cfg)
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {cleaned_path}\n"
                                f"Run the import and clean scripts of the dataset first.")
    df = pd.read_csv(cleaned_path)
    missing = [c for c in list(cfg.features) + [cfg.label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns of the dataset config missing in {cleaned_path}: {missing}")
    print(f"Cleaned data: {cleaned_path} ({len(df):,} rows)")

    # Stratified split on the label codes. Step 1: the test rows (test_pct / 100
    # is the test share, for example 20 / 100 == 0.2).
    codes = data_io.label_codes(df[cfg.label_column], cfg)
    train_indices, test_indices = train_test_split(
        df.index.tolist(),
        test_size=test_pct / 100,
        stratify=codes,
        random_state=seed,
    )
    # Step 2 (validation share only): the validation rows from the other rows
    validation_indices = []
    if validation_pct > 0:
        train_indices, validation_indices = train_test_split(
            train_indices,
            test_size=validation_pct / (train_pct + validation_pct),
            stratify=codes.iloc[train_indices],
            random_state=seed,
        )
    train_codes = codes.iloc[train_indices]
    test_codes = codes.iloc[test_indices]
    validation_codes = codes.iloc[validation_indices]
    ratio_difference = abs(float((train_codes == 1).mean()) - float((test_codes == 1).mean()))
    print(f"Train: {len(train_indices):,} rows; test: {len(test_indices):,} rows; "
          f"difference of the class 1 share: {ratio_difference:.4f}")
    if validation_pct > 0:
        validation_difference = abs(float((train_codes == 1).mean()) - float((validation_codes == 1).mean()))
        print(f"Validation: {len(validation_indices):,} rows; "
              f"difference of the class 1 share (train and validation): {validation_difference:.4f}")

    seed_dir = data_io.split_seed_dir(data_dir, seed, split_folder)
    indices_dir = data_io.split_indices_dir(data_dir, seed, split_folder)
    indices_dir.mkdir(parents=True, exist_ok=True)
    sets = [("train", train_indices, train_codes), ("test", test_indices, test_codes)]
    if validation_pct > 0:
        sets.append(("validation", validation_indices, validation_codes))
    for name, indices, labels in sets:
        pd.DataFrame({"index": indices, "label": labels.values, "split": name}).to_csv(
            indices_dir / f"{name}_indices.csv", index=False)
    print(f"✓ Saved split indices: {indices_dir}")

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    validation_df = df.iloc[validation_indices].reset_index(drop=True)

    # Integer maps from the training rows only
    maps = build_integer_maps(train_df, cfg)
    if cfg.integer_encoding == "pre_encoded":
        print("✓ Integer maps: none (the dataset is pre-encoded)")
    else:
        save_integer_maps(maps, data_io.split_maps_path(data_dir, seed, split_folder))
        print(f"✓ Saved integer maps (training rows only): {data_io.split_maps_path(data_dir, seed, split_folder)}")

    # Test set (and validation set): integer-form features of each client,
    # labels as in the cleaned CSV
    held_out = [("test", test_df, data_io.split_test_dir(data_dir, seed, split_folder))]
    if validation_pct > 0:
        held_out.append(("validation", validation_df, data_io.split_validation_dir(data_dir, seed, split_folder)))
    held_out_int = {}
    for name, rows, out_dir in held_out:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows_int = apply_integer_maps(rows, maps, cfg)
        rows[[cfg.label_column]].to_csv(out_dir / f"{name}_labels.csv", index=False)
        for client_index, features in enumerate(cfg.clients):
            rows_int[features].to_csv(out_dir / f"client_{client_index}_{name}_features.csv", index=False)
        held_out_int[name] = rows_int
        print(f"✓ Saved {name} set ({cfg.n_clients} clients): {out_dir}")

    # Level check of the non-continuous features (warnings only)
    found = level_warnings(apply_integer_maps(train_df, maps, cfg), held_out_int, maps, cfg, min_level_share)
    print(f"Level check (min_level_share {100 * min_level_share:g} %): {len(found)} warning(s)")
    for text in found:
        print(f"  [!] {text}")

    # Training rows, unchanged
    train_path = data_io.train_csv_path(data_dir, cfg, seed, split_folder)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    print(f"✓ Saved training rows: {train_path}")

    summary = {
        "dataset": cfg.name,
        "seed": seed,
        "output_dir": str(seed_dir),
        "rows_total": len(df),
        "split_folder": split_folder,
        "rows_train": len(train_df),
        "rows_validation": len(validation_df),
        "rows_test": len(test_df),
        "train_labels": label_counts(train_codes),
        "validation_labels": label_counts(validation_codes),
        "test_labels": label_counts(test_codes),
        "integer_maps_saved": cfg.integer_encoding != "pre_encoded",
        "level_warnings": found,
    }
    print("\n" + "=" * 80)
    print("STAGE 1 COMPLETED")
    print("=" * 80 + "\n")
    return summary


def main(argv=None):
    """Command line of this stage."""
    parser = argparse.ArgumentParser(description="Stage 1: stratified train-test split of one seed")
    parser.add_argument("--dataset-config", required=True, help="Path of the dataset config (dataset.json)")
    parser.add_argument("--data-dir", default=".",
                        help="Data folder that holds the cleaned CSV and 03_train_test_splits/ (default: current folder)")
    parser.add_argument("--seed", type=int, required=True, help="Seed of the split")
    parser.add_argument("--test-share", type=float, default=data_io.DEFAULT_TEST_SHARE,
                        help="Share of the rows in the test set, a whole percentage (default: %(default)s)")
    parser.add_argument("--validation-share", type=float, default=data_io.DEFAULT_VALIDATION_SHARE,
                        help="Share of the rows in the validation set, a whole percentage; 0: no validation "
                             "set (default: %(default)s)")
    parser.add_argument("--min-level-share", type=float, default=DEFAULT_MIN_LEVEL_SHARE,
                        help="Level check: a level with a smaller share of the training rows gives a "
                             "warning; 0: no share check (default: %(default)s)")
    args = parser.parse_args(argv)
    settings = SplitSettings(
        dataset_config=args.dataset_config,
        data_dir=args.data_dir,
        seed=args.seed,
        test_share=args.test_share,
        validation_share=args.validation_share,
        min_level_share=args.min_level_share,
    )
    with run_log(settings.data_dir, "stage1"):
        return run(settings)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, DatasetConfigError) as error:
        sys.exit(f"ERROR: {error}")
