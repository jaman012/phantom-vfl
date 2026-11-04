"""
Example stage 0 for the Bank Marketing dataset: download the raw data and
write the cleaned CSV that datasets/bank_marketing/dataset.json describes.

The framework starts from a cleaned CSV (RUNNING.md, section 7). Cleaning
rules are specific to each dataset, so they are not part of the framework.
This script is a worked example: for a new dataset, copy it, change the
source constants and the function clean(), and write a dataset.json for the
cleaned CSV.

Source: the copy of the UCI Bank Marketing data (bank-full, 45,211 rows) on
the Hugging Face Hub, repository cestwc/bank-marketing, at the fixed revision
HF_REVISION. This copy stores the categorical columns as integer codes in the
order of its dataset card: job in alphabetical order with 11 = unknown;
education 0 = primary, 1 = secondary, 2 = tertiary, 3 = unknown; month
0 = January to 11 = December; y 0 = no, 1 = yes. It replaces pdays -1
(never contacted) with 999.

Cleaning (function clean()):
1. Drop default, contact, day, duration and poutcome.
2. Group job into 4 categories. Fill job 'unknown' with a random draw from
   the grouped values that are present (seed 42). The mapping makes missing
   values first, so job is a float column in the cleaned CSV (for example
   "0.0").
3. Fill education 'unknown' with a random draw from the codes 0 to 2
   (seed 42).
4. Change month to the calendar quarter (0 to 3).
5. Change campaign to 3 categories (1, 2, 3 or more contacts).
6. Change pdays to binary (0: 1 to 365 days; 1: any other value, which
   includes 999, never contacted).
7. Change previous to 3 categories (0, 1 to 2, 3 or more).
8. Rename y to label.

Output, in the data folder:
  01_bank_marketing_raw/bank_marketing_raw.csv     the download, unchanged
  <files.cleaned_csv of dataset.json>              the cleaned CSV
  logs/prepare_bank_marketing_<date>-<time>.log    the screen output

At the end, the script compares the MD5 of the cleaned CSV with
EXPECTED_CLEANED_MD5, the MD5 of the cleaned CSV of the published results.
A different MD5 gives a warning, not a stop: another pandas version can write
the same values with other bytes.

Usage (from the code root):
  python datasets/bank_marketing/prepare.py --data-dir <data folder> [--overwrite]

Needs pandas, numpy, the Hugging Face library 'datasets' and network access
to the Hugging Face Hub.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from framework.run_log import ERROR_PREFIX, WARNING_PREFIX, run_log  # noqa: E402

DATASET_NAME = "bank_marketing"
HF_REPO = "cestwc/bank-marketing"
HF_REVISION = "79f9e4af31ff6ab9868cc57a0bec513a9d84b26a"
HF_SPLIT = "train"
RAW_CSV = Path("01_bank_marketing_raw") / "bank_marketing_raw.csv"
EXPECTED_CLEANED_MD5 = "6154d28d55b06e2d3c6a2a058001c2b0"

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def md5_of(path):
    """MD5 of a file, as a hex string."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cleaned_csv_path(data_dir):
    """Path of the cleaned CSV: files.cleaned_csv of the dataset.json beside this script, in the data folder."""
    config_path = Path(__file__).resolve().parent / "dataset.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return Path(data_dir) / config["files"]["cleaned_csv"]


def download(raw_path):
    """Download the raw data at the fixed revision and write it unchanged to raw_path."""
    from datasets import load_dataset

    print(f"Downloading {HF_REPO} (revision {HF_REVISION}) from the Hugging Face Hub...")
    ds = load_dataset(HF_REPO, revision=HF_REVISION)
    train_df = ds[HF_SPLIT].to_pandas()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(raw_path, index=False)
    print(f"Saved to: {raw_path}")
    print(f"Shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")


def clean(df):
    """Clean the raw Bank Marketing data (steps 2 to 10 of the screen output)."""
    # ============================================================================
    # DROP FEATURES
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 2: DROPPING FEATURES")
    print(f"{'='*80}")

    features_to_drop = ['default', 'contact', 'day', 'duration', 'poutcome']
    print(f"\nDropping {len(features_to_drop)} features:")
    for feat in features_to_drop:
        print(f"  - {feat}")

    df = df.drop(columns=features_to_drop)
    print(f"\n✓ New shape: {df.shape}")

    # ============================================================================
    # TRANSFORM JOB FEATURE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 3: TRANSFORMING JOB FEATURE")
    print(f"{'='*80}")

    print("\nOriginal job distribution:")
    print(df['job'].value_counts().sort_index())

    # Define job category mapping
    job_mapping = {
        # White collar (0)
        0: 0,  # admin
        4: 0,  # management
        9: 0,  # technician
        # Blue collar (1)
        1: 1,  # blue-collar
        7: 1,  # services
        # Self-directed (2)
        2: 2,  # entrepreneur
        6: 2,  # self-employed
        # Non-working (3)
        5: 3,  # retired
        8: 3,  # student
        10: 3, # unemployed
        3: 3,  # housemaid
        # Unknown (11) - will be imputed
    }

    # Apply mapping
    df['job'] = df['job'].map(job_mapping)

    # Count unknowns (NaN after mapping)
    n_unknown = df['job'].isna().sum()
    print(f"\n✓ Job categories merged")
    print(f"Unknown values to impute: {n_unknown}")

    # Impute unknowns by random sampling from existing categories
    if n_unknown > 0:
        unknown_mask = df['job'].isna()
        random_jobs = df['job'].dropna().sample(n=n_unknown, replace=True, random_state=RANDOM_SEED).values
        df.loc[unknown_mask, 'job'] = random_jobs
        print(f"✓ Imputed {n_unknown} unknown values")

    print("\nFinal job distribution:")
    print(df['job'].value_counts().sort_index())
    print("\nJob category meanings:")
    print("  0 = White collar (admin, management, technician)")
    print("  1 = Blue collar (blue-collar, services)")
    print("  2 = Self-directed (entrepreneur, self-employed)")
    print("  3 = Non-working (retired, student, unemployed, housemaid)")

    # ============================================================================
    # TRANSFORM EDUCATION FEATURE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 4: TRANSFORMING EDUCATION FEATURE")
    print(f"{'='*80}")

    print("\nOriginal education distribution:")
    print(df['education'].value_counts().sort_index())

    # Count unknowns (category 3)
    n_unknown_edu = (df['education'] == 3).sum()
    print(f"\nUnknown values to impute: {n_unknown_edu}")

    # Impute unknowns by random sampling from categories 0, 1, 2
    if n_unknown_edu > 0:
        unknown_mask = df['education'] == 3
        valid_education = df.loc[df['education'].isin([0, 1, 2]), 'education']
        random_edu = valid_education.sample(n=n_unknown_edu, replace=True, random_state=RANDOM_SEED).values
        df.loc[unknown_mask, 'education'] = random_edu
        print(f"✓ Imputed {n_unknown_edu} unknown values")

    print("\nFinal education distribution:")
    print(df['education'].value_counts().sort_index())
    print("\nEducation category meanings:")
    print("  0 = Primary")
    print("  1 = Secondary")
    print("  2 = Tertiary")

    # ============================================================================
    # TRANSFORM MONTH TO QUARTERS
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 5: TRANSFORMING MONTH TO QUARTERS")
    print(f"{'='*80}")

    print("\nOriginal month distribution:")
    print(df['month'].value_counts().sort_index())

    # Define quarter mapping
    quarter_mapping = {
        0: 0, 1: 0, 2: 0,      # Q1
        3: 1, 4: 1, 5: 1,      # Q2
        6: 2, 7: 2, 8: 2,      # Q3
        9: 3, 10: 3, 11: 3     # Q4
    }

    df['month'] = df['month'].map(quarter_mapping)

    print("\n✓ Months converted to quarters")
    print("\nFinal quarter distribution:")
    print(df['month'].value_counts().sort_index())
    print("\nQuarter meanings:")
    print("  0 = Q1 (Jan-Mar)")
    print("  1 = Q2 (Apr-Jun)")
    print("  2 = Q3 (Jul-Sep)")
    print("  3 = Q4 (Oct-Dec)")

    # ============================================================================
    # TRANSFORM CAMPAIGN TO 3 CATEGORIES
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 6: TRANSFORMING CAMPAIGN FEATURE")
    print(f"{'='*80}")

    print("\nOriginal campaign distribution (first 10 values):")
    print(df['campaign'].value_counts().sort_index().head(10))

    # Convert to categories: 1→0, 2→1, 3+→2
    df['campaign'] = df['campaign'].apply(lambda x: 0 if x == 1 else (1 if x == 2 else 2))

    print("\n✓ Campaign converted to 3 categories")
    print("\nFinal campaign distribution:")
    print(df['campaign'].value_counts().sort_index())
    print("\nCampaign category meanings:")
    print("  0 = 1 contact")
    print("  1 = 2 contacts")
    print("  2 = 3+ contacts")

    # ============================================================================
    # TRANSFORM PDAYS TO BINARY
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 7: TRANSFORMING PDAYS TO BINARY")
    print(f"{'='*80}")

    print("\nOriginal pdays statistics:")
    print(f"  Min: {df['pdays'].min()}")
    print(f"  Max: {df['pdays'].max()}")
    print(f"  999 values (never contacted): {(df['pdays'] == 999).sum()}")
    print(f"  1-365 values: {((df['pdays'] >= 1) & (df['pdays'] <= 365)).sum()}")
    print(f"  366+ values: {((df['pdays'] >= 366) & (df['pdays'] < 999)).sum()}")

    # Convert to binary: 0 = within 1 year (1-365), 1 = not recent/never (366+/999)
    df['pdays'] = df['pdays'].apply(lambda x: 0 if 1 <= x <= 365 else 1)

    print("\n✓ Pdays converted to binary")
    print("\nFinal pdays distribution:")
    print(df['pdays'].value_counts().sort_index())
    print("\nPdays category meanings:")
    print("  0 = Contacted within 1 year (1-365 days)")
    print("  1 = Not contacted recently or never (366+ days or 999)")

    # ============================================================================
    # TRANSFORM PREVIOUS TO 3 CATEGORIES
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 8: TRANSFORMING PREVIOUS FEATURE")
    print(f"{'='*80}")

    print("\nOriginal previous distribution (first 10 values):")
    print(df['previous'].value_counts().sort_index().head(10))

    # Convert to categories: 0→0, 1-2→1, 3+→2
    df['previous'] = df['previous'].apply(lambda x: 0 if x == 0 else (1 if x <= 2 else 2))

    print("\n✓ Previous converted to 3 categories")
    print("\nFinal previous distribution:")
    print(df['previous'].value_counts().sort_index())
    print("\nPrevious category meanings:")
    print("  0 = Never contacted (0 prior contacts)")
    print("  1 = Contacted 1-2 times")
    print("  2 = Contacted 3+ times")

    # ============================================================================
    # ENCODE TARGET VARIABLE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 9: ENCODING TARGET VARIABLE")
    print(f"{'='*80}")

    print("\nOriginal target (y) distribution:")
    print(df['y'].value_counts())

    # Encode target: 1=yes (subscribed), 0=no (not subscribed)
    # Note: Already encoded as 0/1 in raw data, but rename for consistency
    df = df.rename(columns={'y': 'label'})

    print(f"\n✓ Renamed target variable to 'label'")
    print("\nFinal label distribution:")
    print(df['label'].value_counts())
    print("\nLabel meanings:")
    print("  0 = No (client did not subscribe)")
    print("  1 = Yes (client subscribed)")

    # ============================================================================
    # VERIFY FINAL DATASET
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 10: FINAL VERIFICATION")
    print(f"{'='*80}")

    print(f"\nFinal shape: {df.shape}")
    print(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠ WARNING: Missing values detected:")
        print(missing[missing > 0])
    else:
        print(f"\n✓ No missing values")

    # Show data types
    print(f"\nData types:")
    print(df.dtypes)

    # Show sample rows
    print(f"\nSample rows (first 5):")
    print(df.head())

    # Target variable distribution
    print(f"\nTarget variable (label) distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print(f"\nClass balance:")
    for lbl, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {lbl}: {count:,} ({pct:.1f}%)")
    return df


def prepare(data_dir, overwrite):
    """Download, clean and write the cleaned CSV. Returns the exit code (0 or 1)."""
    raw_path = data_dir / RAW_CSV
    output_path = cleaned_csv_path(data_dir)
    existing = [path for path in (raw_path, output_path) if path.exists()]
    if existing and not overwrite:
        for path in existing:
            print(f"{ERROR_PREFIX} The file exists: {path}. Use --overwrite to write it again.")
        return 1

    print("=" * 80)
    print("BANK MARKETING DATASET PREPARATION")
    print("=" * 80)
    print(f"\nSource: {HF_REPO}, revision {HF_REVISION}")
    print(f"Raw CSV: {raw_path}")
    print(f"Output: {output_path}")
    print(f"Random seed: {RANDOM_SEED}")

    print(f"\n{'=' * 80}")
    print("STEP 0: DOWNLOADING RAW DATA")
    print(f"{'=' * 80}")
    download(raw_path)
    print(f"MD5 of the raw CSV: {md5_of(raw_path)}")

    print(f"\n{'=' * 80}")
    print("STEP 1: LOADING RAW DATA")
    print(f"{'=' * 80}")
    df = pd.read_csv(raw_path)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Original columns ({len(df.columns)}): {df.columns.tolist()}")

    df = clean(df)

    print(f"\n{'=' * 80}")
    print("STEP 11: SAVING CLEANED DATASET")
    print(f"{'=' * 80}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"Shape: {df.shape}")

    cleaned_md5 = md5_of(output_path)
    print(f"MD5 of the cleaned CSV: {cleaned_md5}")
    if cleaned_md5 == EXPECTED_CLEANED_MD5:
        print("MATCH: the cleaned CSV is identical to the cleaned CSV of the published results.")
    else:
        print(f"{WARNING_PREFIX} DIFFERENT: the cleaned CSV of the published results has the MD5 "
              f"{EXPECTED_CLEANED_MD5}. The file is kept. Compare the values if you need the same data.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the bank_marketing data and write the cleaned CSV of its dataset.json.")
    parser.add_argument("--data-dir", required=True, help="data folder of the run")
    parser.add_argument("--overwrite", action="store_true",
                        help="write the raw CSV and the cleaned CSV again if they exist")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with run_log(data_dir, f"prepare_{DATASET_NAME}"):
        code = prepare(data_dir, args.overwrite)
    return code


if __name__ == "__main__":
    sys.exit(main())
