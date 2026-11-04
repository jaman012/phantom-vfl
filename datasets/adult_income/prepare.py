"""
Example stage 0 for the Adult Income dataset: download the raw data and write
the cleaned CSV that datasets/adult_income/dataset.json describes.

The framework starts from a cleaned CSV (RUNNING.md, section 7). Cleaning
rules are specific to each dataset, so they are not part of the framework.
This script is a worked example: for a new dataset, copy it, change the
source constants and the function clean(), and write a dataset.json for the
cleaned CSV.

Source: the copy of the UCI Adult (Census Income) data on the Hugging Face
Hub, repository scikit-learn/adult-census-income, at the fixed revision
HF_REVISION (32,561 rows).

Cleaning (function clean()):
1. Replace '?' with a missing value. Fill each missing categorical value with
   a random draw from the values that are present in its column (seed 42).
2. Merge capital.gain and capital.loss into capital.net (gain minus loss).
3. Group workclass, education, marital.status, occupation and race into
   fewer categories.
4. Binarise native.country (United-States or Other).
5. Drop fnlwgt (census sampling weight, not a personal characteristic) and
   education.num (the same information as education).

Output, in the data folder:
  01_adult_census_raw/adult_census_raw.csv       the download, unchanged
  <files.cleaned_csv of dataset.json>            the cleaned CSV
  logs/prepare_adult_income_<date>-<time>.log    the screen output

At the end, the script compares the MD5 of the cleaned CSV with
EXPECTED_CLEANED_MD5, the MD5 of the cleaned CSV of the published results.
A different MD5 gives a warning, not a stop: another pandas version can write
the same values with other bytes.

Usage (from the code root):
  python datasets/adult_income/prepare.py --data-dir <data folder> [--overwrite]

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

DATASET_NAME = "adult_income"
HF_REPO = "scikit-learn/adult-census-income"
HF_REVISION = "2a7ff8e6a6cd74fa596ed6622937081c52255fcb"
HF_SPLIT = "train"
RAW_CSV = Path("01_adult_census_raw") / "adult_census_raw.csv"
EXPECTED_CLEANED_MD5 = "5beb9fd66abdf0931ebe4121ff9c6d14"

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
    """Clean the raw Adult Income data (steps 2 to 11 of the screen output)."""
    # ============================================================================
    # HANDLE MISSING VALUES
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 2: HANDLING MISSING VALUES")
    print(f"{'='*80}")

    # Replace '?' and its variations with NaN
    df = df.replace('?', np.nan)
    df = df.replace(' ?', np.nan)
    df = df.replace('? ', np.nan)

    print("\nReplaced '?' with NaN")

    # Count missing values
    missing_counts = df.isnull().sum()
    missing_summary = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if len(missing_summary) > 0:
        print(f"\nFeatures with missing values:")
        for col, count in missing_summary.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")

        # Impute missing categorical values by random sampling
        for col in missing_summary.index:
            if df[col].dtype == 'object':
                missing_mask = df[col].isnull()
                n_missing = missing_mask.sum()

                # Randomly sample from non-missing values
                random_fill = df[col].dropna().sample(n=n_missing, replace=True, random_state=RANDOM_SEED).values
                df.loc[missing_mask, col] = random_fill

                print(f"\n✓ Filled {n_missing} missing values in '{col}'")

        # Verify no missing values remain
        remaining_missing = df.isnull().sum().sum()
        print(f"\n✓ Remaining missing values: {remaining_missing}")
    else:
        print("\n✓ No missing values found")

    # ============================================================================
    # MERGE CAPITAL FEATURES
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 3: MERGING CAPITAL FEATURES")
    print(f"{'='*80}")

    print(f"\ncapital.gain - Min: {df['capital.gain'].min()}, Max: {df['capital.gain'].max()}")
    print(f"capital.loss - Min: {df['capital.loss'].min()}, Max: {df['capital.loss'].max()}")

    # Create capital.net = capital.gain - capital.loss
    df['capital.net'] = df['capital.gain'] - df['capital.loss']

    print(f"\n✓ Created 'capital.net' = capital.gain - capital.loss")
    print(f"capital.net - Min: {df['capital.net'].min()}, Max: {df['capital.net'].max()}")

    # Drop original columns
    df = df.drop(columns=['capital.gain', 'capital.loss'])
    print(f"✓ Dropped 'capital.gain' and 'capital.loss'")

    # ============================================================================
    # GROUP WORKCLASS FEATURE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 4: GROUPING WORKCLASS FEATURE")
    print(f"{'='*80}")

    print("\nOriginal workclass distribution:")
    print(df['workclass'].value_counts().sort_index())

    # Strip whitespace first
    df['workclass'] = df['workclass'].str.strip()

    # Group Never-worked and Without-pay as Unemployed
    workclass_mapping = {
        'Never-worked': 'Unemployed',
        'Without-pay': 'Unemployed'
    }

    df['workclass'] = df['workclass'].replace(workclass_mapping)

    print(f"\n✓ Merged 'Never-worked' and 'Without-pay' into 'Unemployed'")
    print("\nFinal workclass distribution:")
    print(df['workclass'].value_counts().sort_index())

    # ============================================================================
    # GROUP EDUCATION FEATURE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 5: GROUPING EDUCATION FEATURE")
    print(f"{'='*80}")

    print("\nOriginal education distribution:")
    print(df['education'].value_counts().sort_index())

    # Strip whitespace
    df['education'] = df['education'].str.strip()

    # Define education grouping (16 categories → 8 categories)
    education_mapping = {
        # Dropout (Preschool through 12th grade incomplete)
        'Preschool': 'Dropout',
        '1st-4th': 'Dropout',
        '5th-6th': 'Dropout',
        '7th-8th': 'Dropout',
        '9th': 'Dropout',
        '10th': 'Dropout',
        '11th': 'Dropout',
        '12th': 'Dropout',
        # HS graduate
        'HS-grad': 'HS-grad',
        # Some college
        'Some-college': 'Some-college',
        # Associate degree
        'Assoc-acdm': 'Assoc',
        'Assoc-voc': 'Assoc',
        # Bachelor's
        'Bachelors': 'Bachelors',
        # Master's
        'Masters': 'Masters',
        # Professional school
        'Prof-school': 'Prof-school',
        # Doctorate
        'Doctorate': 'Doctorate'
    }

    df['education'] = df['education'].replace(education_mapping)

    print(f"\n✓ Grouped education into 8 categories")
    print("\nFinal education distribution:")
    print(df['education'].value_counts().sort_index())
    print("\nEducation category meanings:")
    print("  Dropout: Preschool through 12th grade incomplete")
    print("  HS-grad: High school graduate")
    print("  Some-college: Some college")
    print("  Assoc: Associate degree (academic or vocational)")
    print("  Bachelors: Bachelor's degree")
    print("  Masters: Master's degree")
    print("  Prof-school: Professional school (law, medicine)")
    print("  Doctorate: Doctorate degree")

    # ============================================================================
    # GROUP MARITAL STATUS FEATURE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 6: GROUPING MARITAL STATUS FEATURE")
    print(f"{'='*80}")

    print("\nOriginal marital.status distribution:")
    print(df['marital.status'].value_counts().sort_index())

    # Strip whitespace
    df['marital.status'] = df['marital.status'].str.strip()

    # Group all married categories together
    marital_mapping = {
        'Married-civ-spouse': 'Married',
        'Married-AF-spouse': 'Married',  # Armed Forces spouse
        'Married-spouse-absent': 'Married',
        'Never-married': 'Never-married',
        'Divorced': 'Divorced',
        'Separated': 'Separated',
        'Widowed': 'Widowed'
    }

    df['marital.status'] = df['marital.status'].replace(marital_mapping)

    print(f"\n✓ Grouped all married categories into 'Married'")
    print("\nFinal marital.status distribution:")
    print(df['marital.status'].value_counts().sort_index())
    print("\nMarital status meanings:")
    print("  Married: All married categories (civilian, armed forces, spouse absent)")
    print("  Never-married: Never married")
    print("  Divorced: Divorced")
    print("  Separated: Separated")
    print("  Widowed: Widowed")

    # ============================================================================
    # GROUP OCCUPATION FEATURE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 7: GROUPING OCCUPATION FEATURE")
    print(f"{'='*80}")

    print("\nOriginal occupation distribution:")
    print(df['occupation'].value_counts().sort_index())

    # Strip whitespace
    df['occupation'] = df['occupation'].str.strip()

    # Define occupation grouping (14 categories → 6 categories)
    occupation_mapping = {
        # White-collar
        'Exec-managerial': 'White-collar',
        'Prof-specialty': 'White-collar',
        'Tech-support': 'White-collar',
        # Sales-Admin
        'Sales': 'Sales-Admin',
        'Adm-clerical': 'Sales-Admin',
        # Blue-collar
        'Craft-repair': 'Blue-collar',
        'Machine-op-inspct': 'Blue-collar',
        'Transport-moving': 'Blue-collar',
        'Handlers-cleaners': 'Blue-collar',
        # Service
        'Other-service': 'Service',
        'Priv-house-serv': 'Service',
        'Protective-serv': 'Service',
        # Farming
        'Farming-fishing': 'Farming-fishing',
        # Armed Forces
        'Armed-Forces': 'Armed-Forces'
    }

    df['occupation'] = df['occupation'].replace(occupation_mapping)

    print(f"\n✓ Grouped occupation into 6 categories")
    print("\nFinal occupation distribution:")
    print(df['occupation'].value_counts().sort_index())
    print("\nOccupation category meanings:")
    print("  White-collar: Executive, professional, technical")
    print("  Sales-Admin: Sales and administrative support")
    print("  Blue-collar: Craft, machine operation, transport, handlers")
    print("  Service: Service workers, private household, protective service")
    print("  Farming-fishing: Farming and fishing")
    print("  Armed-Forces: Armed Forces")

    # ============================================================================
    # GROUP RACE FEATURE
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 8: GROUPING RACE FEATURE")
    print(f"{'='*80}")

    print("\nOriginal race distribution:")
    print(df['race'].value_counts().sort_index())

    # Strip whitespace
    df['race'] = df['race'].str.strip()

    # Group into White, Black, Other
    race_mapping = {
        'White': 'White',
        'Black': 'Black',
        'Amer-Indian-Eskimo': 'Other',
        'Asian-Pac-Islander': 'Other',
        'Other': 'Other'
    }

    df['race'] = df['race'].replace(race_mapping)

    print(f"\n✓ Grouped race into 3 categories")
    print("\nFinal race distribution:")
    print(df['race'].value_counts().sort_index())
    print("\nRace category meanings:")
    print("  White: White")
    print("  Black: Black")
    print("  Other: American Indian, Asian/Pacific Islander, Other")

    # ============================================================================
    # BINARIZE NATIVE.COUNTRY
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 9: BINARIZING NATIVE.COUNTRY")
    print(f"{'='*80}")

    print(f"\nOriginal native.country unique values: {df['native.country'].nunique()}")

    # Strip whitespace
    df['native.country'] = df['native.country'].str.strip()

    # Binarize: United-States vs Other
    df['native.country'] = df['native.country'].apply(
        lambda x: 'United-States' if x == 'United-States' else 'Other'
    )

    print(f"\n✓ Binarized native.country (United-States vs Other)")
    print("\nFinal native.country distribution:")
    print(df['native.country'].value_counts().sort_index())

    # ============================================================================
    # DROP FEATURES
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 10: DROPPING FEATURES")
    print(f"{'='*80}")

    features_to_drop = ['fnlwgt', 'education.num']
    print(f"\nDropping {len(features_to_drop)} features:")
    for feat in features_to_drop:
        if feat in df.columns:
            print(f"  - {feat}")

    df = df.drop(columns=[f for f in features_to_drop if f in df.columns])
    print(f"\n✓ New shape: {df.shape}")

    print("\nWhy dropped:")
    print("  fnlwgt: Census sampling weight (not a personal characteristic)")
    print("  education.num: Redundant with education categorical")

    # ============================================================================
    # VERIFY FINAL DATASET
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 11: FINAL VERIFICATION")
    print(f"{'='*80}")

    print(f"\nFinal shape: {df.shape}")
    print(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠  WARNING: Missing values detected:")
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
    print(f"\nTarget variable (income) distribution:")
    income_counts = df['income'].value_counts()
    print(income_counts)
    print(f"\nClass balance:")
    for label, count in income_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    # Feature summary
    print(f"\n{'='*80}")
    print("FINAL FEATURE SUMMARY")
    print(f"{'='*80}")

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'income' in numerical_features:
        numerical_features.remove('income')

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'income' in categorical_features:
        categorical_features.remove('income')

    print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Target: income")
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
    print("ADULT CENSUS INCOME DATASET PREPARATION")
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
    print("STEP 12: SAVING CLEANED DATASET")
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
        description="Download the adult_income data and write the cleaned CSV of its dataset.json.")
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
