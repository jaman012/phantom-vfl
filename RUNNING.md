# Running the Phantom VFL framework

Version: 4 Nov 2026. Applies to `framework/`, `flower_app/`, `datasets/`, `examples/` and `docs/` of this repository.

## 1. Status and scope

- This guide describes the Phantom VFL framework: one script for each stage of the data generation, one protocol module, one Flower app for the training, a training runner, a tuning driver, an analysis driver with a summary command, an orchestrator that runs these steps with one command (section 23), one dataset config for each dataset, and one settings file (`run_config.json`).
- The framework makes the datasets: the train-test split (with an optional validation set), the ground truth and the phantom datasets (k\* and the imposed k levels), in integer form and in model form (one-hot).
- The framework trains a vertical federated model with Flower on each dataset of a seed (the ground truth, k\* and each k level). Each run writes the test metrics and, with a validation set, the validation metrics of each evaluated round (sections 16 to 19).
- The tuning driver searches values of the training settings and ranks them on the validation set (section 18).
- Each run writes a run log and a warnings file (sections 10 and 16).
- The analysis tests the results of a training plan: each k level against the ground truth of the same seeds, one pair at a time (Track A, in Python) and all k levels together (Track B, in R). It gives the viable ceiling of each track (sections 20 to 22). A summary command puts the results of several run configs into one set of tables. It can also give an optional advice on the number of seeds (section 21).
- Two example scripts download and clean the two example datasets (section 7). They are not part of the framework, because cleaning rules are specific to each dataset.
- The folder `examples/` holds run configs that run as they are: a first test, a quick start and the six combinations of dataset, label sharing and loss of the published experiments (section 24).
- **Relation to the published results.** The published results came from earlier scripts that are not in this repository. This code has fixes that came after those results: the phantom generation is seeded, the substitution draws only from retained records that the server does not request, and the test set uses the encoding of the training set. The training code seeds the start weights of the models. So this code does not make the same phantom datasets or the same training results as the earlier scripts. The input is the same: the example prepare scripts make the cleaned CSVs of the published results byte for byte (section 7).
- **Tests of the consolidation.** Copies of the earlier scripts with the same fixes gave a reference. With the settings that give the earlier files (section 12: `scaler_columns` "all", `cramers_v_yates` true, `phase2_trials` 1), the generation stages give the same data files as this reference: on fake data for all six combinations of dataset, label sharing and loss, and on the real Adult Income data (seed 42, label sharing on and off). One difference is intended: `source_file` in the integer-form ground-truth `config.json` is a relative path. With the default settings, the data files stay the same on the tested data; the scaler files and their texts differ (section 12). The Flower app gives the same numbers as the earlier training code with the same data, the same start weights and the same seed (a test with all messages in one process; it does not cover the Flower runtime).

## 2. What the pipeline does

For each seed, the pipeline makes a set of vertically partitioned datasets for two or more clients.

- **Ground truth.** No phantom protocol, no removal. All individuals are real (100 % PSI match).
- **k\*.** The phantom rate that the protocol derives from the data with statistical checks (the natural budget). The datasets are in the folder `k0`. `k0` is k\*, not "no phantoms".
- **k = 5 % to 40 %.** Imposed phantom rates (folders `k5` … `k40`).
- Each seed gives one train-test split. The ground truth, k\* and the k levels of one seed use the same split, so you can compare them in pairs.
- The protocol works on the integer form. The model form (one-hot for nominal features) is made from the same rows.
- **Training.** For each seed, the training runner trains one model with Flower on the ground truth, on k\* and on each k level, and evaluates each model on the same test set (and validation set). Compare a k level with the ground truth of the same seed.
- **Analysis.** For each run, the analysis chooses one round and takes the test values of that round. With a validation set, the validation set chooses the round. Without a validation set (the default split 80-20), the test set chooses it, as in the published results (section 20). Then the analysis compares each k level with the ground truth over the seeds (sections 20 to 22).

## 3. Requirements

- Python 3.10. `requirements.txt` in the code root pins the tested versions of the Python packages: numpy 2.2.5, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.5.0, torch 2.6.0, flwr 1.23.0 (with the extra `simulation`), Ray 2.31.0, datasets 3.1.0, and tomli 2.2.1 (Python 3.10 only; the training code reads `pyproject.toml` with it). The tested environment had Python 3.10.19.
- Install the packages in a new environment, for example with conda:
  ```bash
  conda create -n phantom-vfl python=3.10
  conda activate phantom-vfl
  python -m pip install -r requirements.txt
  ```
- Other versions can give different files or numbers. Compare with a test run first (section 25).
- The training runs on the CPU. Tested computer: Apple M4 Pro, 14 cores, 24 GB memory.
- Start every command from the root of the code (the folder that holds `framework/`), with the Python of this environment. The examples write `python`.
- **R, for Track B of the analysis only** (sections 20 and 21): R with the R package emmeans. Tested: R 4.6.1 with emmeans 2.0.4. Install emmeans in R with `install.packages("emmeans")`. The analysis uses the `Rscript` command on the PATH; the setting `analysis.rscript` gives another command or a full path. With `analysis.track_b` false, the analysis runs without R (Track A only).
- The analysis uses numpy, pandas and scipy. It needs no other Python package.
- The example prepare scripts (section 7) need the Hugging Face library `datasets` (in `requirements.txt`) and network access to the Hugging Face Hub. Tested with huggingface_hub 1.0.1 and pyarrow 22.0.0 (installed with `datasets`).

## 4. Code layout

```
<code root>/
├── framework/
│   ├── run_all.py                       the orchestrator: all steps of one or more run configs (section 23)
│   ├── run_pipeline.py                  the generation runner (stages 1 to 3 from run_config.json)
│   ├── split_train_test.py              stage 1
│   ├── generate_ground_truth_dataset.py stage 2
│   ├── generate_phantom_datasets.py     stage 3
│   ├── phantom_protocol.py              the protocol (imported by stage 3)
│   ├── run_log.py                       run log and warnings file
│   ├── data_io.py                       split shares, folder names and paths; loaders of the training data and of the ground truth
│   ├── encoding.py                      integer maps, model encoding and the scaler rule
│   ├── dataset_config.py                loads and checks a dataset config
│   ├── analysis/
│   │   ├── run_analysis.py              the analysis driver: one run config (section 20)
│   │   ├── summarise.py                 the summary across run configs (section 21)
│   │   ├── settings.py                  reads and checks the section analysis
│   │   ├── collect.py                   finds the completed training runs and checks them
│   │   ├── select_round.py              the round of each run (on the validation set or the test set)
│   │   ├── tables.py                    the wide and long tables of the complete seeds
│   │   ├── track_a.py                   Track A: paired tests against the ground truth
│   │   ├── track_b.R                    Track B: repeated-measures ANOVA and Tukey (R)
│   │   ├── run_r.py                     finds Rscript and runs track_b.R
│   │   ├── ceiling.py                   the viable ceiling of each track
│   │   ├── power.py                     the power of both tracks (optional advice)
│   │   └── report.py                    report.md of an analysis and of a summary
│   └── training/
│       ├── run_flower.py                the launcher: one Flower run with a run log (section 16)
│       ├── run_training.py              the training runner (section 17)
│       ├── tune.py                      the tuning driver (section 18)
│       ├── flower_data.py               reads the data of a run for the Flower app
│       ├── losses.py                    the losses bce, weighted_bce, focal and class_balanced
│       ├── metrics.py                   the test and validation metrics
│       ├── class_balance.py             the class-balance record of a run
│       ├── seeding.py                   the seeded start weights of the models
│       └── results.py                   the names of the result folders and files
├── flower_app/
│   ├── pyproject.toml                   the Flower app and the defaults of its run-config keys
│   ├── vertical_fl/                     client_app.py, server_app.py, task.py
│   ├── LICENSE                          Apache License 2.0 of the Flower example
│   ├── ATTRIBUTION.md                   the source of the app
│   └── MODIFICATIONS.md                 the changes to the Flower example
├── datasets/
│   ├── adult_income/                    dataset.json; prepare.py (example: download and clean, section 7)
│   └── bank_marketing/                  dataset.json; prepare.py (example: download and clean, section 7)
├── docs/
│   └── flower_reference.md              source, changes and known limits of the Flower app
├── examples/
│   ├── README.md                        the examples, their run times and how to cut them
│   ├── first_test.json                  the first test: the quick start with 20 rounds (section 24)
│   ├── quickstart_adult_income_ls.json  the quick start (section 24)
│   └── full_design/                     six run configs and a summary config (section 24)
├── README.md                            the entry page of the repository
├── requirements.txt                     the Python packages, pinned (section 3)
└── RUNNING.md                           this guide
```

- The framework files contain no column names of a dataset. They read them from `dataset.json`.
- The Flower app is a modified version of the Flower example `examples/vertical-fl` (Apache License 2.0). `docs/flower_reference.md` gives the source and every change. Flower itself is a dependency.
- The data and the results are not part of the code. `data_dir` in `run_config.json` (or `--data-dir`) points to them. The example configs write to `runs/<name>/` in the code root (section 24).

## 5. Data folder layout

The data folder must hold the cleaned CSV at the path that `files.cleaned_csv` of the dataset config gives. The stages, the training runner and the tuning driver write the rest:

```
<data_dir>/
├── 01_<dataset>_raw/<raw CSV>                  the download of an example prepare script (section 7)
├── 02_<dataset>_cleaned/<cleaned CSV>          input (section 7)
├── 03_train_test_splits/
│   ├── <split folder>/rs<seed>/                stage 1: split_indices/, test/, train/, encoding_maps.json,
│   │                                           and validation/ (only with a validation share)
│   └── <model split folder>/rs<seed>/          stage 2: test/ (model form), and validation/ (model form,
│                                               only with a validation share)
├── 04_federated_datasets/
│   ├── <split folder>/rs<seed>/                stage 2: ground_truth/, scalers/
│   │                                           stage 3: encoding_maps.json, k0 (k*), k5 … k40
│   └── <model split folder>/rs<seed>/          stage 2: ground_truth/, scalers/
│                                               stage 3: k0, k5 … k40 (model form)
├── 06_flower_vfl_results/
│   └── <results tag>/split_<ratio>/            training runner: training_settings.json, training_runs.csv
│       └── rs<seed>/<k folder>/                one Flower run: test metrics, training loss,
│                                               class-balance record (section 19)
├── 07_tuning/<study>/                          tuning driver: study.json, trials_plan.csv, runs.csv,
│                                               ranking.csv, best_run_config.json (section 18)
├── 08_results_analysis/
│   └── <results tag>/split_<ratio>/            analysis driver: tables, tests, ceilings, report.md
│                                               and the second tables (sections 20 and 22)
└── logs/                                       the log and warnings files of each run (section 10),
                                                the Flower run logs (section 16) and the summary files
                                                of the training runner (section 17)
```

- **Folder names.** The shares make the split folder name: `split_<train>_<test>` (for example `split_80_20`), or `split_<train>_<validation>_<test>` with a validation share (for example `split_60_20_20`). The model-form folder adds `_onehot`. The training share is the rest. The names are not settings (section 8). Every stage makes them from the same two shares, so a stage that runs alone needs the same `--test-share` and `--validation-share` as stage 1.
- **Result folders.** `<ratio>` is the name of the split folder that the run reads, without `split_`: for example `80_20_onehot` or `60_20_20_onehot` (model form), or `80_20` (integer form). `<results tag>` is the run-config key `results-tag`, or the loss type when it is empty (section 16). A tuning trial writes its runs under the tag `tune_<study>_t<nnn>`. `<k folder>` is `ground_truth`, the k\* folder (`k0`) or `k<p>`.
- `k0` is the default name of the k\* folder. It is a setting (`kstar_folder`). The training runner and the tuning driver read the setting.
- **Validation set.** With a validation share above 0, the folder `validation/` holds `validation_labels.csv` and, for each client, `client_<i>_validation_features.csv`. They have the same format as the test files. Stage 1 writes the integer form. Stage 2 writes the model form (the one-hot columns of the training rows). The validation rows are not in the ground truth and not in the phantom generation.
- `encoding_maps.json` in `03_…` is written only for a dataset with `integer_encoding: fit_on_train`.
- Each integer-form k folder holds, for each client, `features.csv`, `metadata.json` and `phantom_mapping.json`, and also `labels.csv`, `config.json`, `ground_truth_analysis.csv`, `training_order.json`, `budget_computation_history.json`, `phantom_rate_summary.json` and `_complete.json` (section 14).

## 6. The dataset config

`datasets/<name>/dataset.json` holds every fact about one dataset. `dataset_config.py` checks it. A fault stops the run with a list of all faults.

| Field | Meaning |
|---|---|
| `schema_version` | 1 |
| `name` | short name, for example `adult_income` |
| `output_name` | dataset name written into the `config.json` files |
| `files.cleaned_csv` | path of the cleaned CSV, relative to the data folder |
| `files.train_csv` | file name of the training rows that stage 1 writes |
| `label.column`, `label.map` | label column; map of the two raw label values to 0 and 1 |
| `label.test_label_format` | `text` or `numeric`: the form of `test_labels.csv` that the stages write and the training code reads |
| `integer_encoding` | `fit_on_train` (stage 1 learns the integer codes from the training rows) or `pre_encoded` (the adapter gives codes already) |
| `features` | one entry for each feature, with its `type` (below) |
| `clients` | a list of feature lists, one for each client (2 or more); each feature in exactly one client |
| `model_feature_order` | optional; the order of the model-form columns (default: continuous, binary, ordinal, nominal) |
| `output_text` | optional; dataset-specific sentences for the `config.json` files of stage 2 |

Feature types:
- `continuous`: a number; used as it is.
- `binary`: two values. With a rule (`positive_value`, `positive_name`, `negative_name`), a text value equal to `positive_value` (after strip and upper case) gets 1, and every other value gets 0. Without a rule, the two training values get 0 and 1 in sorted order.
- `nominal`: categories with no order; one-hot in the model form.
- `ordinal`: categories with an order; `order` lists the levels from low to high; the codes follow it. The type is optional (section 12).

To add a dataset: (1) write a script that makes a cleaned CSV (no missing values; the label column and the feature columns; `datasets/<name>/prepare.py` of the example datasets shows one way, section 7); (2) write `datasets/<name>/dataset.json`; (3) as a first test, run the runner with one seed and `"k_levels": ["kstar"]`; (4) read the stage summaries.

## 7. Stage 0: the cleaned CSV

The framework starts from one cleaned CSV for each dataset. `files.cleaned_csv` of the dataset config gives its path in the data folder.

- The CSV has the label column (`label.column`) and every feature column of `features`, with the names of `dataset.json`.
- It has no missing values.
- The label column holds the two raw values of `label.map`.
- For a dataset with `integer_encoding: pre_encoded`, the categorical columns already hold integer codes. For `fit_on_train`, stage 1 learns the codes from the training rows.
- The two example datasets come from the UCI Machine Learning Repository: Adult (Census Income) and Bank Marketing. For each, the example script `datasets/<name>/prepare.py` downloads the copy on the Hugging Face Hub at a fixed revision and writes the cleaned CSV:

```
python datasets/adult_income/prepare.py --data-dir <data folder>
python datasets/bank_marketing/prepare.py --data-dir <data folder>
```

| Dataset | Hugging Face repository | Revision | Rows | MD5 of the cleaned CSV |
|---|---|---|---|---|
| Adult Income | `scikit-learn/adult-census-income` | `2a7ff8e6a6cd74fa596ed6622937081c52255fcb` | 32,561 | `5beb9fd66abdf0931ebe4121ff9c6d14` |
| Bank Marketing | `cestwc/bank-marketing` | `79f9e4af31ff6ab9868cc57a0bec513a9d84b26a` | 45,211 | `6154d28d55b06e2d3c6a2a058001c2b0` |

- The scripts need the Hugging Face library `datasets` (in `requirements.txt`) and network access (section 3). Without `datasets`, the run stops with the error "No module named 'datasets'".
- Each script writes the download unchanged (`01_<dataset>_raw/`), the cleaned CSV at `files.cleaned_csv` and a run log in `<data folder>/logs/`. It stops if one of the two CSVs exists; `--overwrite` writes them again.
- At the end, the script compares the MD5 of the cleaned CSV with the table. MATCH means that the file is the same as the cleaned CSV of the published results. DIFFERENT is a warning, not a stop: another pandas version can write the same values with other bytes (tested: pandas 2.2.3).
- The cleaning rules (groupings, imputation with seed 42, dropped columns) are in the docstring and in the function `clean()` of each script. For a new dataset, copy one script, change the source constants and `clean()`, and write a `dataset.json` for the cleaned CSV.

## 8. `run_config.json`

`run_config.json` is the one place for the settings of a run.

```json
{
  "schema_version": 1,
  "generation": {
    "dataset_config": "datasets/adult_income/dataset.json",
    "data_dir": "data/adult_income",
    "seeds": [42],
    "label_sharing": true,
    "k_levels": ["kstar", 5, 10, 15, 20, 25, 30, 35, 40],
    "resume": false,
    "kstar_folder": "k0",
    "cramers_v_yates": false,
    "phase2_trials": 1
  },
  "data_preparation": {
    "test_share": 0.2,
    "validation_share": 0.0,
    "min_level_share": 0.01,
    "scaler_columns": "continuous_ordinal"
  }
}
```

**`generation`**
- Required: `dataset_config`, `data_dir`, `seeds`, `label_sharing` (no default).
- Optional, with the defaults shown: `k_levels` (`"kstar"` and whole percentages), `resume`, `kstar_folder`, `cramers_v_yates` (`false`) and `phase2_trials` (1). The last two change Phase 2 (section 12).

**`data_preparation`** (the whole section is optional; each key is optional, with the defaults shown)
- `test_share`: share of the rows in the test set.
- `validation_share`: share of the rows in the validation set. 0 means no validation set.
- `min_level_share`: smallest share of the training rows that a level of a categorical feature can have with no warning (section 10). 0 means no share check.
- `scaler_columns`: the columns that the scalers of stage 2 change: `"continuous_ordinal"` (the continuous and ordinal columns; default) or `"all"` (every column). The data files do not depend on it (section 12).
- A share must be a whole percentage (0.2 or 0.15, not 0.125). `test_share` must be 0.01 or more. `test_share` plus `validation_share` must be 0.99 or less. `min_level_share` must be 0 or more and less than 1. The training share is the rest.
- Without the section, the split is 80-20 with no validation set.
- The shares make the split folder names (section 5). The test rows do not depend on the validation share: the test rows are split off first, with the same call and the same seed.

**`training` and `tuning`** (both optional): the settings of the Flower runs (section 17) and of a tuning study (section 18). The training runner and the tuning driver also read `generation` and `data_preparation`: the data folder, the dataset config, the seeds, the k levels and the split shares. Section 23 has a complete example.

**`analysis`** (optional): the settings of the analysis driver (section 20). Without the section, the defaults apply.

The orchestrator (section 23) runs the generation, the training and the analysis of a run config with one command. The folder `examples/` holds complete run configs (section 24).

**All sections**
- `split_folder` and `model_split_folder` are not settings any more. A `run_config.json` with one of them stops the run, and the message names the replacement. Remove these keys.
- Relative paths are relative to the folder of `run_config.json`.
- An unknown key or section stops the run. The known sections are `generation`, `data_preparation`, `training` (section 17), `tuning` (section 18) and `analysis` (section 20). The generation runner reads only `generation` and `data_preparation`. It accepts the other three sections and does not read them.

## 9. The generation runner

```bash
python framework/run_pipeline.py --config run_config.json
```

- The runner checks `run_config.json` and the dataset config. Then it checks every seed for k folders that were made with the other label-sharing setting (section 13). Then it runs stages 1, 2 and 3 for each seed, in the order of `seeds`.
- The runner prints its settings, including `data_preparation` and the split folder names. It copies the screen output of the run to `<data_dir>/logs/` (section 10).
- **Confirmation gate.** After each stage, the runner shows the summary of the stage and waits. Press Enter to continue. Type `q` to stop.
- `--yes` runs without the gates (for many seeds). `--resume` passes to stage 3 (section 14).
- Several seeds: put them in `seeds`, for example `[42, 50, 58, 66, 74, 82, 90, 98, 106, 114]` (the seeds of the published results), and use `--yes`.
- Run time (real Adult Income data, one seed, all 9 k levels): about 16 seconds on a 2-core machine. Phase 2 runs once for each seed.
- The end of a good run: "ALL STAGES COMPLETED (<n> seed(s))". After this line, the runner prints the number of warnings and errors and the paths of the warnings file and the log. Read the warnings file (section 10).

## 10. Run log and warnings

Terminal keeps only the newest lines of a long run. So the framework copies the screen output of each run to two text files in `<data_dir>/logs/`:

| File | Content |
|---|---|
| `<name>_<date>-<time>.log` | all screen output of the run. It also holds the traceback of an error. |
| `<name>_<date>-<time>_warnings.txt` | only the warning lines and the error lines |

- `<name>` is `run` for the generation runner. It is `stage1`, `stage2` or `stage3` for a stage that runs alone (section 11), and `flower_<results tag>_<k folder>_rs<seed>` for a Flower run (section 16). It is `analysis` for the analysis driver (section 20) and `summary` for the summary command (section 21). The summary command writes its logs to `<output_dir>/logs/`, not to `<data_dir>/logs/`. `<date>-<time>` looks like `20261031-110200`.
- A new run never overwrites an old log. If a name exists, the framework adds a number.
- The runner opens one log for the whole run. A stage that the runner starts does not open its own log.
- The first line of a run shows the path of the log. The last lines show the number of warnings and errors and the paths of both files. If the data folder does not exist, there is no log, and the run prints "No run log: …".
- The `logs/` folder holds time stamps. Skip it when you compare two data folders.
- Read the warnings file after each run.

**Warning lines and error lines**
- A warning line starts with `[!]`. No other line uses this prefix. A Python warning (for example from pandas) is also shown as a `[!]` line.
- A line that starts with `Note:` gives information. It is not a warning, and it is not in the warnings file.
- An error that stops the run starts with `[ERROR]`. It goes to both files. The traceback goes to the log only.
- The warnings file writes the context after the prefix. The context is the seed and the stage (`stage 1`, `stage 2`, `stage 3`), the k folder (`k0`, `k20`) or `model form`. Example: `[!] (seed 42, k20) Client_A: label mismatches: 46 (pot constraints)`.

**The warnings of the framework**

| Line (shortened) | Stage | Meaning |
|---|---|---|
| `'<feature>' level …: <n> of <N> training rows (…), less than min_level_share` | 1 | A level has few training rows (level check, below). |
| `'<feature>' level …: in the training rows, not in the test rows` | 1 | A level is in the training rows, but not in the test (or validation) rows. |
| `'<feature>' value …: in <n> test row(s), not in the training rows` | 1 | A value is in the test (or validation) rows, but not in the training rows. Only for a `pre_encoded` dataset. For a `fit_on_train` dataset, such a value stops the run. |
| `client_<i>: near-zero variance column(s) before scaling` | 2 | A column that the scaler changes has almost no variation in the training rows. Scaling can give very large values. The check looks only at the columns that the scaler changes (section 12). |
| `'<feature>' has value(s) … that the ground truth does not have` | 2, 3 | A model-form row has a value with no one-hot column. The row gets 0 in every one-hot column of that feature. |
| `k folder <name>: no completion marker with the label-sharing setting` | 3 | The k folder has no usable `_complete.json`, so its label-sharing setting is unknown. A requested k level overwrites it (section 13). |
| `<client>: label mismatches: <n> (pot constraints)` | 3 | Label mismatch (below). |
| `Python warning: …` | any | A warning of a Python package. |

**The level check (stage 1)**
- The check looks at every feature that is not continuous, on the integer form. It never stops the run.
- A warning means: look at this level. A level with very few training rows gives a one-hot column with almost no variation. A level that is not in the test rows is not tested.
- `min_level_share` sets the limit (default 0.01, which is 1 %). The stage option is `--min-level-share`. With 0, the share check is off. The other two checks stay on.
- Example: `[!] 'occupation' level 'Armed-Forces': 7 of 26,048 training rows (0.03 %), less than min_level_share (1 %)`.
- Real data, seed 42, 80-20 split: Adult Income gives two warnings, `workclass` Unemployed (18 rows, 0.07 %) and `occupation` Armed-Forces (7 rows, 0.03 %). Bank Marketing gives none. The framework does not change these two levels.

**Label mismatch (stage 3, label sharing on)**
- With label sharing on, the substitution takes a record with the same label. If the pot has no such record, it takes a record with the other label. This is a label mismatch.
- For a k level, each client with at least one mismatch prints `[!] <client>: label mismatches: <n> (pot constraints)`.
- On the real data (seed 42) the counts are small. The highest are 74 of 4,873 phantom positions (1.5 %, Adult Income, k25) and 57 of 8,669 (0.7 %, Bank Marketing, k40), for each client.
- The counts are only in the log and in the warnings file. They are not in `metadata.json` or in another data file. Keep the warnings file if you need them.
- With label sharing off, there is no such line. The substitution does not use labels.

## 11. Stages 1 to 3

Each stage can also run alone. Each needs the stage before it for the same seed. A stage that runs alone writes its own log and warnings file (section 10): `stage1_…`, `stage2_…`, `stage3_…`. The shares must be the same in all three stages, because they make the folder names.

**Stage 1: train-test split.**
```bash
python framework/split_train_test.py \
  --dataset-config datasets/adult_income/dataset.json \
  --data-dir <data_dir> --seed 42 \
  [--test-share 0.2] [--validation-share 0.0] [--min-level-share 0.01]
```
- Stratified split on the label, in two steps. Step 1 splits off the test rows (`random_state` = seed). With a validation share, step 2 splits the validation rows off the other rows, with the same seed. The test rows do not depend on the validation share.
- Writes `split_indices/` (a file for each set), `test/` and, with a validation share, `validation/` (integer form), `train/` and, for `fit_on_train`, `encoding_maps.json`. The integer maps come from the training rows only.
- Runs the level check (section 10). The warnings do not stop the stage.
- Success: "STAGE 1 COMPLETED"; the line "Level check (min_level_share 1 %): <n> warning(s)"; for two clients, 2 files in `split_indices/` (3 with a validation share), 3 in `test/`, 1 in `train/` (and 3 in `validation/`).

**Stage 2: ground truth.**
```bash
python framework/generate_ground_truth_dataset.py \
  --dataset-config datasets/adult_income/dataset.json \
  --data-dir <data_dir> --seed 42 \
  [--test-share 0.2] [--validation-share 0.0] \
  [--scaler-columns continuous_ordinal]
```
- Integer form: all training rows, all real; one `features.csv` for each client, `labels.csv`, `ground_truth_analysis.csv`, `config.json`, scalers. Model form: one-hot for nominal features; the model-form ground truth, test set and scalers. With a validation share, also the model-form validation set. The encoder and the scalers are learned from the ground truth (the training rows only). `--scaler-columns` gives the columns that each scaler changes: `continuous_ordinal` (default) or `all` (section 12).
- Success: "STAGE 2 COMPLETED"; for each client, a line for the integer form (features and rows) and a line for the model form (model columns); with a validation share, a line "Saved model-form validation set (<n> rows)".

**Stage 3: phantom datasets.**
```bash
python framework/generate_phantom_datasets.py \
  --dataset-config datasets/adult_income/dataset.json \
  --data-dir <data_dir> --seed 42 --label-sharing on \
  [--test-share 0.2] [--validation-share 0.0] \
  [--cramers-v-yates] [--phase2-trials 1]
```
- First it checks the k folders of the seed (section 13). A k folder that was made with the other label-sharing setting stops the stage before it writes a data file.
- Reads the integer-form ground truth of stage 2 and checks it against the training rows (row-set check). A difference stops the stage before the protocol, with the first different column and row.
- Runs Phases 0 to 2 once (k\*), then each k level. Writes the integer form and then the model form of each k level.
- Options: `--k-levels` (default `kstar,5,10,15,20,25,30,35,40`), `--kstar-folder`, `--test-share`, `--validation-share`, `--resume`, `--cramers-v-yates` (Phase 2: Yates correction on) and `--phase2-trials` (Phase 2: an odd number of trials, default 1; section 12). The options `--split-folder` and `--model-split-folder` do not exist any more: the shares make the folder names.
- Success: "Natural budget k\*: <n>"; for each k level and each client, a line "Reviewer check Client_<x>: … duplicates 0 … passed" (18 lines for 9 levels and 2 clients); "Reviewer check passed" true for every k level; "Model-form k folders written" lists every k level.

## 12. Data preparation

Data preparation is the work between the cleaned CSV and the phantom datasets: the split, the feature types, the encoding, the scaler and the checks of Phase 2. This section names each step, its setting and its default value. Every setting has a value that gives the same data files as before the setting existed.

**Steps and settings**

| Step | Stage | Setting (section of `run_config.json`) | Default | Stage option |
|---|---|---|---|---|
| Test share and validation share | 1 | `test_share`, `validation_share` (`data_preparation`) | 0.2 and 0.0 | `--test-share`, `--validation-share` |
| Level check | 1 | `min_level_share` (`data_preparation`) | 0.01 | `--min-level-share` |
| Scaler rule | 2 | `scaler_columns` (`data_preparation`) | `"continuous_ordinal"` | `--scaler-columns` |
| Yates correction in Cramér's V | 3 (Phase 2) | `cramers_v_yates` (`generation`) | `false` | `--cramers-v-yates` (no value) |
| Trials of Phase 2 | 3 (Phase 2) | `phase2_trials` (`generation`) | 1 | `--phase2-trials` |

The split and the validation set are in sections 5 and 8. The level check is in section 10.

**Feature types**

Each feature in `dataset.json` has one of four types (section 6). The type decides the encoding, the scaler rule and the checks of Phase 2.

| Type | Integer form | Model form | Scaled by the default rule | Type in Phase 2 |
|---|---|---|---|---|
| `continuous` | the number | the number | yes | `continuous` |
| `binary` | 0 or 1 | 0 or 1 | no | `categorical` |
| `nominal` | an integer code | one-hot columns | no | `categorical` |
| `ordinal` | the position of the level in `order` | the integer code | yes | `ordinal` |

- **The `ordinal` type is optional.** Declare a feature `ordinal` only if its levels have a true order. A user can declare every categorical feature `nominal`. The two example datasets declare no ordinal feature.
- `order` lists the levels from low to high. Stage 1 gives the codes 0, 1, 2 … in this order (for `fit_on_train`; for `pre_encoded`, `order` lists the numeric values of the cleaned data). A training value that is not in `order` stops the run. A level in `order` that the training rows do not have is allowed.
- Example (a feature of a new dataset):
  ```json
  "education_level": {"type": "ordinal", "order": ["primary", "secondary", "tertiary"]}
  ```
- Phase 2 gives an ordinal feature the checks of a categorical feature, and one more check (Spearman) for a pair with a continuous or an ordinal feature (below).

**The scaler rule (`scaler_columns`)**

- Stage 2 fits one `StandardScaler` for each client on the training rows (the ground truth) and saves it. The data files are saved unscaled, so this setting changes no data file.
- `"continuous_ordinal"` (default): the scaler changes the columns of continuous and ordinal features. Binary columns, one-hot columns and the integer codes of nominal features pass through. For these columns the saved scaler holds mean 0, variance 1 and scale 1, so `transform` returns them unchanged.
- `"all"`: the scaler changes every column. This is the rule of the earlier framework.
- The scaler keeps its class, its file name, the number of its columns and their order for both values. A training script that loads the scaler needs no change.
- Reason for the default: a one-hot column of a rare level has a very small standard deviation. In Adult Income, `occupation_Armed-Forces` (7 of 26,048 training rows) reaches 61.0 after scaling with `"all"` and stays 0 or 1 with the default rule.
- The text of the ground-truth `config.json` files follows the setting (keys `normalization` and `scaling`, section 15).
- The warning "near-zero variance column(s) before scaling" (section 10) looks only at the columns that the scaler changes.
- **Note for training.** With the default rule, a model that trains on the integer form gets the integer codes of nominal features unscaled. `"all"` gives the scalers of the earlier framework.

**Phase 2: marginal checks and bivariate checks**

Phase 2 runs for each client. It finds the largest number of individuals that the client can remove while its data stay statistically close to the data before the removal. The natural budget k\* comes from these results (section 2). Phase 2 has two parts:

- **Marginal checks:** one feature at a time. A removal passes only if every feature passes.
- **Bivariate checks:** every pair of features of the same client. The result cannot be higher than the result of the marginal checks. At most 25 % of the individuals can be removed (`max_removal_fraction`).

Phase 2 uses random removals. The limits below are fixed in the class `PhantomVFLConfig` of `phantom_protocol.py`. `run_config.json` does not set them.

| Marginal checks for | Both checks must pass | Limit |
|---|---|---|
| a continuous feature | Kolmogorov-Smirnov statistic | less than 0.05 (`epsilon_ks`) |
| | relative difference of the mean, the variance, the skewness and the kurtosis | each less than 0.10 (`epsilon_mom`) |
| a categorical feature (`binary`, `nominal`, `ordinal`) | chi-square test on the level counts | p-value above 0.05 (`alpha`) |
| | total variation distance of the level shares | less than 0.05 (`epsilon_tv`) |

- A moment difference of exactly 0.10 fails. The Kolmogorov-Smirnov p-value is stored and not used.
- The chi-square test uses only the levels with an expected count of 5 or more (`min_category_count`). It rescales the observed counts to the expected total. With fewer than 2 such levels, the check fails. The total variation distance uses all levels.

| Pair of features | Check | Limit |
|---|---|---|
| continuous and continuous | Pearson correlation | change of at most 0.05 (`delta_r_threshold`) |
| continuous and categorical | ANOVA F-statistic | relative change of at most 0.20 (`delta_f_threshold`) |
| categorical and categorical | Cramér's V | change of at most 0.05 (`delta_v_threshold`) |
| ordinal and ordinal, or ordinal and continuous | the check above for the pair (an ordinal feature counts as categorical) **and** the Spearman rank correlation | Spearman: change of at most 0.05 (`delta_rho_threshold`); both must pass |

- The Spearman check ranks an ordinal feature by `order` and gives tied values the average rank. A column with one value gives no correlation, and the check fails.
- A pair of an ordinal and a nominal or binary feature uses Cramér's V only.

**Cramér's V and the Yates correction (`cramers_v_yates`)**

- For a 2 × 2 table, the chi-square function of scipy applies the Yates correction by default. The formula of the protocol has no correction.
- `false` (default): no correction. `true`: the correction, as in the earlier scripts. The correction acts on a 2 × 2 table only.
- On the example data the setting changes no file. Real data, seed 42: Bank Marketing gives k\* 2,240 and the same 169 files with both values (the Cramér's V of `loan` and `pdays` is 0.027898 without and 0.027797 with the correction). Adult Income has no 2 × 2 pair inside a client. A dataset with binary features in the same client can differ.
- The completion marker records the setting. A change makes the next `--resume` run the k levels again (section 14).

**`phase2_trials` and the stability of k\***

- Phase 2 tests each candidate number of removals with a random removal. `phase2_trials` is the number of random removals for each candidate, in the marginal search and in the bivariate search.
- A candidate passes if most of its trials pass. A trial of the bivariate search passes only if all pairs pass. The value must be an odd whole number (1, 3, 5 …), so that there is no tie. Another value stops the run.
- With 1 (default), Phase 2 works as in the earlier scripts. The k-level `config.json` holds the key `phase2_trials` only if the value is not 1. Every completion marker holds the value (section 15).
- k\* is repeatable: the same seed gives the same k\*. It is not stable: another random stream or another seed gives another k\*. Measured values (label sharing on; 10 runs in each cell, so the standard deviations are rough):

| Dataset | `phase2_trials` | One split (seed 42), 10 random streams: mean, SD (smallest to largest) | The 10 seeds: mean, SD |
|---|---|---|---|
| Adult Income | 1 | 2,569, 514 (1,619 to 3,317) | 2,631, 540 |
| Adult Income | 3 | 2,675, 497 | 2,607, 495 |
| Adult Income | 5 | 2,821, 234 (2,449 to 3,103) | not measured |
| Bank Marketing | 1 | 4,325, 1,483 (1,976 to 6,338) | 3,542, 1,019 |
| Bank Marketing | 3 | 3,857, 995 | 3,642, 917 |
| Bank Marketing | 5 | 3,802, 985 (1,834 to 4,923) | not measured |

- Most of the spread comes from the random search, not from the split. 3 trials change the spread little. 5 trials halve the spread for Adult Income only.
- A value above 1 changes k\* itself. On seed 42, 3 trials give 2,080 for Adult Income (2,790 with 1 trial) and 3,413 for Bank Marketing (2,240 with 1 trial).
- Use the same value in all runs that you compare. Each seed computes its own k\* (section 26).

**What a setting changes**

| Setting (not the default) | Files that change |
|---|---|
| `scaler_columns: "all"` | the 4 scaler `.pkl` files and the scaling texts of the 2 ground-truth `config.json` files. No data file. |
| `cramers_v_yates: true` | the marker value. On the example data, no other file. |
| `phase2_trials: 3` (or more) | the folder `k0` (k\*), `natural_phantom_budget` in the `config.json` files, the key `phase2_trials` in the k-level `config.json`, and the marker value. |
| `validation_share` above 0 | the split folder names, and the folders `validation/` (section 5). |

## 13. Label sharing

- `label_sharing: true` (`--label-sharing on`): the clients get coded labels. Removal is stratified by label, and substitution prefers a substitute with the same label.
- `label_sharing: false` (`off`): no labels go to the clients. Removal is a simple random sample, and substitution takes the shuffled pot in position order.
- The setting has no default. The run stops if it is missing.
- **One data folder for each setting.** An on run and an off run must not share a data folder, because they write into the same k folders. Use two data folders, for example `data/adult_income_ls` and `data/adult_income_nls` (the names are examples).
- **The check.** Each complete k folder holds `_complete.json`, which records the setting. Before stage 1, the runner reads the k folders of every seed (`k<number>` and the k\* folder). Stage 3 reads them again. If one k folder has the other setting, the run stops before it writes any data file. The error is in the log files. The message names the k folders.
- A k folder with no usable `_complete.json` (an interrupted run, or a folder from older code) has an unknown setting. It gives a `[!]` warning. A requested k level overwrites it.
- `--resume` does not change this check (section 14).

## 14. `--resume` and the completion marker

- This section is about the generation (stage 3). The training runner and the tuning driver have their own `--resume` (sections 17 and 18).
- After each complete k folder, stage 3 writes `_complete.json` as the last file. It records the k level, the seed, the label-sharing setting, the Phase 2 settings (`cramers_v_yates` and `phase2_trials`), the SHA-256 checksums of the stage and protocol scripts, of the framework modules, of the dataset config, of the training CSV and of the integer-form ground truth (`ground_truth_sha256`), and the checksum of every file in the folder (all keys: section 15).
- With `--resume`, stage 3 skips a k folder only if its marker matches the current run in all of these. Otherwise the k level runs again, and the stage prints the reason.
- A change of `cramers_v_yates` or `phase2_trials` makes the marker differ, so the k levels run again.
- Phase 2 runs only if at least one k level must run.
- The model form of every requested k level is written again on each run, also for skipped k folders.
- Use `--resume` after a stopped run. Do not use it to keep files of older code: the checksums differ, so those k levels run again.

## 15. Keys of the generation files

This section defines the keys of five files that the generation writes. Section 19 has the files of the training and the tuning. The examples are from the real Adult Income run (seed 42, 80-20 split, label sharing on). In a path, `<split folder>` is the integer-form folder (for example `split_80_20`) and `<model split folder>` is the model-form folder (for example `split_80_20_onehot`).

**1. k-level `config.json`**

Path: `04_federated_datasets/<split folder>/rs<seed>/<k folder>/config.json`. Stage 3 writes one for each integer-form k folder. The model-form k folders hold only `client_<i>/features.csv` and `labels.csv`.

| Key | Type | Meaning |
|---|---|---|
| `dataset` | text | `output_name` of the dataset config (for example `adult_census_income`) |
| `phantom_proportion` | number | the k level as a fraction (0.2 for k20); 0.0 in the k\* folder |
| `num_clients` | integer | the number of clients |
| `num_training_samples` | integer | the number of positions in the training order of this k level. k\*: the training rows minus `natural_phantom_budget` (26,048 − 2,790 = 23,258). Imposed k level: about (1 − k) times the training rows (20,839 at k20) |
| `psi_common_ids` | integer | the number of IDs that all clients have (the PSI intersection). It equals the number of training rows |
| `natural_phantom_budget` | integer | k\* in individuals: the number that Phase 2 gives for this run (2,790 for the real run). Every k folder of the run holds the same value |
| `generation_seed` | integer | the seed of the run |
| `vertical_splits` | object | for each client name (`Client_A`, `Client_B` …), its list of features |
| `phase2_trials` | integer | the number of Phase 2 trials (section 12). The key is present only if the value is not 1 |

**2. Client `metadata.json`**

Path: `…/<k folder>/client_<i>/metadata.json`. Stage 3 writes one for each client and each integer-form k folder.

| Key | Type | Meaning |
|---|---|---|
| `client_id`, `party_name` | text | the client name (`Client_A` …); both keys hold the same value |
| `num_features` | integer | the number of features of the client |
| `feature_names` | list | the feature names, in the column order of `features.csv` |
| `feature_types` | object | the Phase 2 type of each feature: `continuous`, `categorical` or `ordinal`. A `binary` and a `nominal` feature both show as `categorical` |
| `num_training_samples` | integer | the same value as in the k-level `config.json` |
| `num_phantom_positions` | integer | the number of positions where this client has no real record and uses a phantom (a substitute) |
| `phantom_rate` | number | `num_phantom_positions` divided by `num_training_samples` (2,507 / 23,258 = 0.108 at k\*) |

The label-mismatch counts of section 10 are not in this file.

**3. Completion marker `_complete.json`**

Path: `…/<k folder>/_complete.json`. Stage 3 writes it as the last file of a complete k folder. `--resume` and the label-sharing check read it (sections 13 and 14).

| Key | Type | Meaning |
|---|---|---|
| `k_value` | number | the k level as a fraction (0.0 for k\*) |
| `generation_seed` | integer | the seed of the run |
| `protocol_script_sha256` | text | SHA-256 of `phantom_protocol.py` |
| `generation_script_sha256` | text | SHA-256 of `generate_phantom_datasets.py` |
| `training_data_sha256` | text | SHA-256 of the training CSV |
| `ground_truth_sha256` | object | SHA-256 of each file of the integer-form ground truth (`labels.csv` and `client_<i>/features.csv`) |
| `label_sharing` | true or false | the label-sharing setting of the run (section 13) |
| `phase2_settings` | object | the Phase 2 settings of the run: `cramers_v_yates` (true or false) and `phase2_trials` (integer) |
| `dataset_config_sha256` | text | SHA-256 of the dataset config |
| `framework_modules_sha256` | object | SHA-256 of `data_io.py`, `encoding.py` and `dataset_config.py` |
| `files` | object | the SHA-256 of every file in the k folder (relative path as key). The marker and hidden files are not in the list |

**4. Ground-truth `config.json`, integer form**

Path: `04_federated_datasets/<split folder>/rs<seed>/ground_truth/config.json`. Stage 2 writes it.

| Key | Type | Meaning |
|---|---|---|
| `dataset_type` | text | `ground_truth` |
| `description` | text | `Baseline VFL dataset without phantom protocol` |
| `phantom_protocol_applied` | true or false | always `false` |
| `num_samples` | integer | the number of training rows (26,048) |
| `num_features` | integer | the number of features of all clients (11) |
| `num_clients` | integer | the number of clients |
| `feature_split` | text | the number of features of each client, joined with `+` (`5+6`) |
| `vertical_split` | object | for each client (`client_0`, `client_1` …), its list of features |
| `label_distribution` | object | the number of rows of `class_0` and `class_1` |
| `preprocessing` | object | the text of the dataset config key `output_text.ground_truth_preprocessing` |
| `normalization` | text | the scaling text. It follows `scaler_columns`: `"all"` gives `RAW features saved, scalers fitted for ISLP compliance`; `"continuous_ordinal"` says that the scalers change the continuous and ordinal columns |
| `source_file` | text | the training CSV, as a path relative to the data folder |

**5. Ground-truth `config.json`, model form**

Path: `04_federated_datasets/<model split folder>/rs<seed>/ground_truth/config.json`. Stage 2 writes it.

| Key | Type | Meaning |
|---|---|---|
| `dataset_type` | text | `ground_truth_onehot` |
| `purpose` | text | the text of the dataset config key `output_text.model_purpose` |
| `onehot_features` | list | the nominal features that have one-hot columns |
| `continuous_features` | list | the continuous features |
| `binary_passthrough` | list | the binary features (they pass through as 0 or 1) |
| `ordinal_features` | list | the ordinal features. The key is present only if the dataset has an ordinal feature |
| `scaling` | text | the scaling text. It follows `scaler_columns` (`"all"` gives the text of the earlier framework) |
| `seed` | integer | the seed of the run |
| `source_split` | text | the text of the dataset config key `output_text.model_source_split`, with the split folder and the seed filled in |

## 16. Training with Flower: the app and the launcher

The training uses the Flower app in `flower_app/`. Each Flower run trains one model on one dataset of one seed: the ground truth, k\* or a k level. The app runs as a Flower simulation on one computer, with one node for each client and the server.

**What one run does**
- The server holds the labels. Each client holds the feature columns of its part of `dataset.json`.
- In each round, each client computes embeddings of all its training rows. The server joins them, runs the head model, computes the loss and sends each client the gradient of its embeddings. Each round is one gradient step on all training rows (full batch).
- The scaler of each client (stage 2, fitted on the ground truth) changes the training, test and validation features of that client.
- The server evaluates the model on the test set every 10 rounds and at the last round. With a validation set, it also evaluates the validation set at the same rounds.
- The seed (`random-seed`) selects the folder `rs<seed>` and sets the start weights of the models.

**Run-config keys** (defaults in `flower_app/pyproject.toml`, `[tool.flwr.app.config]`):

| Key | Default | Meaning |
|---|---|---|
| `num-server-rounds` | 250 | number of training rounds |
| `learning-rate` | 0.01 | learning rate of the client models |
| `out-feature-dim-clientapp` | 64 | width of the embedding of each client |
| `data-dir` | "" | data folder (required) |
| `dataset-config` | "" | path of `dataset.json` (required) |
| `phantom-dataset` | `k0` | the k folder: `ground_truth`, `k0` (k\*), `k5` … `k40` |
| `random-seed` | 42 | the seed of the split folder and of the start weights |
| `feature-splits` | "5,6" | the number of features of each client; the sum must equal the number of features in `dataset.json` |
| `loss-type` | `bce` | `bce`, `weighted_bce`, `focal` or `class_balanced` |
| `focal-alpha`, `focal-gamma` | 0.80, 2.0 | values of the focal loss |
| `cb-beta` | 0.9999 | value of the class-balanced loss |
| `results-tag` | "" | the name of the results folder; empty: the loss type |

- `weighted_bce` takes its weight of the positive class from the training labels of each k level.
- Fixed in the app (not run-config keys): the learning rate of the head model (0.01), the layer sizes of the head model (128 and 96), the evaluation interval (10 rounds) and the prediction threshold (0.5).
- The number of nodes is `options.num-supernodes` in `[tool.flwr.federations.local-simulation]` of `pyproject.toml` (2). For a dataset with more clients, change this value.
- The training runner (section 17) sets `data-dir`, `dataset-config`, `phantom-dataset`, `random-seed` and `feature-splits` itself.

**The launcher.** `framework/training/run_flower.py` starts one Flower run. Use it when you want one run by hand. For many runs, use the training runner.

```bash
python framework/training/run_flower.py --split-ratio 80_20_onehot \
  --run-config "data-dir='<full path of the data folder>' dataset-config='<full path of dataset.json>' phantom-dataset='k0' random-seed=42 loss-type='focal'"
```

- `--split-ratio` is required. It is the split folder name without `split_`: `80_20_onehot` (model form) or `80_20` (integer form); with a validation share, for example `60_20_20_onehot`.
- `--run-config` has the same text as for `flwr run`. Give full paths: the app runs in the folder `flower_app/`.
- The launcher sets the environment of the run: `SPLIT_RATIO`, `PYTHONPATH` (the code root first, so that the app finds `framework.training`) and `PYTHONUNBUFFERED`. It uses the program `flwr` next to the Python that runs it, or else the one on `PATH`.
- Start the launcher with the Python of the environment that has flwr.

**Run log.** The launcher copies every line of the run (server, clients and Ray) to two files in `<data-dir>/logs/`:
- `flower_<results tag>_<k folder>_rs<seed>_<date>-<time>.log`: all output.
- `flower_<results tag>_<k folder>_rs<seed>_<date>-<time>_warnings.txt`: the warning and error lines.
- A Flower line with the level WARNING, or a Python warning, gets the prefix `[!]`. A Flower line with the level ERROR, the first line of a traceback, or a line that reports a non-zero exit status gets the prefix `[ERROR]`. The launcher removes the colour codes of the Flower output.
- With an empty `data-dir`, there is no run log and no check of the result files.

**When a run fails.** The run fails when one of these is true:
- `flwr run` stops with an exit code other than 0;
- the output has at least one `[ERROR]` line;
- one of the three result files (section 19) is missing or older than the start of the run.

`flwr run` (flwr 1.23.0) returns 0 also when the simulation fails, so the launcher checks the output and the files itself. A failed run prints `[ERROR] The run failed: <reasons>`, and the launcher stops with exit code 1 (or the exit code of `flwr run`). A good run prints `Run completed. Result files:` and the three paths, and the launcher stops with 0.

**Known limits** of the app (full batch, the order of the evaluation, the gradient messages, the losses, the class-balance warning): `docs/flower_reference.md`, section 8. One of them matters for repeated runs: the server sends the gradients and does not wait for the clients. A client can then answer a later query with the weights of one round earlier. Runs with the same seed are identical in the first rounds, but can differ later. No known bias comes from this.

## 17. The training runner

The training runner starts the Flower runs of one `run_config.json`: each seed with each k level.

```bash
python framework/training/run_training.py --config <path>/run_config.json [--yes] [--resume]
```

**The `training` section** (optional; every key is optional):

```json
"training": {
  "seeds": [42, 50, 58],
  "k_levels": ["ground_truth", "kstar", 20],
  "feature_form": "model",
  "run_config": {"loss-type": "focal", "num-server-rounds": 250}
}
```

| Key | Default | Meaning |
|---|---|---|
| `seeds` | `generation.seeds` | the seeds of the runs |
| `k_levels` | `"ground_truth"`, then `generation.k_levels` | `"ground_truth"`, `"kstar"` (the folder `kstar_folder`) or a whole percentage from 1 to 99 (the folder `k<p>`) |
| `feature_form` | `"model"` | `"model"` (one-hot form) or `"integer"` |
| `run_config` | none | run-config keys of section 16, with the names and the value types of `pyproject.toml` (a whole number is accepted for a decimal value). A key that is not given keeps its default |

- `run_config` must not hold `data-dir`, `dataset-config`, `phantom-dataset`, `random-seed` or `feature-splits`. The runner sets them from `generation`, from the plan and from `dataset.json`.
- The split folder comes from `test_share` and `validation_share` of `data_preparation`, with `_onehot` for the model form.
- The data of every seed and k level must exist. Run the generation first (section 9).
- **Several seeds:** list them in `generation.seeds`, run the generation, then the training. Leave out `training.seeds` to train every generated seed, or list a part of them to run the seeds in batches (with `--resume`, below).

**What the runner does**
1. It checks the whole `run_config.json`. A fault stops it before the first run, with a list of all faults.
2. It makes the plan: each seed with each k level, in the given order. It reads each `--run-config` text back with Flower's reader and stops if a value differs from the plan.
3. It shows the plan, for example `runs: 10 planned, 0 kept from before, 10 to start`, and waits for Enter (the confirmation gate). Type `q` to stop. `--yes` runs without the gate.
4. It starts each run with the launcher (section 16). A run is completed when the launcher returns 0 and the test-metrics file has a last round equal to `num-server-rounds`. A failed run does not stop the runner.
5. At the end it prints `TRAINING RUNS: <n> completed, <f> failed (<k> kept from before)`, each failed run with its reason, the state folder and the summary file.

Exit code: 0 when every started run completed, 1 when a run failed, 2 when you stopped at the gate. Ctrl-C stops the runner. The interrupted run gets no row in the state file.

**State files and `--resume`.** The runner keeps the state of the runs of one results tag and split folder in `<data_dir>/06_flower_vfl_results/<results tag>/split_<ratio>/`:
- `training_settings.json`: the settings that every run of the folder shares (no seeds, no k levels). The runner writes it at the first start, after the gate.
- `training_runs.csv`: one row for each seed and k level. The runner writes it again after each run.

The rules:
- **Without `--resume`:** when `training_settings.json` exists, the runner stops. So a new start never overwrites results. To start again, use `--resume`, or choose another `results-tag`.
- **With `--resume`:** the runner stops when the settings differ from `training_settings.json`, and lists the keys that differ. It keeps each run that `training_runs.csv` shows as completed, if its test-metrics file still has the recorded checksum. It starts every other run of the plan: failed runs, runs that were not started, and runs whose file is missing or changed. With `--resume` and no state file, every run starts.
- Rows of seeds and k levels that are not in the plan stay in `training_runs.csv`. So you can run the seeds in batches: run with some seeds, then add more seeds to `training.seeds` and start the same command with `--resume`.
- A run that starts overwrites the result files of an earlier run with the same results tag, split, seed and k level.
- After a stop (Ctrl-C, a failed run, a fault of Ray, a computer that went to sleep), start the same command again with `--resume`. Only the runs that are not complete start.

**Summary file.** Each start of the runner writes `<data_dir>/logs/training_<date>-<time>_summary.csv`, again after each run. Columns: section 19.

**More than one loss on the same data.** The results tag keeps the runs of different losses apart: for example `weighted_bce` and `focal` on the same Bank Marketing data folder. Use one `run_config.json` for each loss, with the same `generation` and `data_preparation` sections. Run the generation once.

**Run time.** On the tested computer (section 3), one run takes about 77 seconds with 250 rounds and about 14 seconds with 20 rounds. A plan of 30 seeds and 10 datasets (the ground truth, k\* and 8 k levels) has 300 runs: about 6.5 hours with 250 rounds. On macOS, start a long run with `caffeinate -i` before the command, so that the computer does not sleep. To keep the screen output, add `2>&1 | tee -a <log file>` after the command.

**Runs at the same time.** You can start runners for different configs at the same time in different terminals, if each config has its own results tag or data folder. Do not start two runners with the same results tag and split folder: both write the same `training_runs.csv`, and rows are lost. Start the runners a few seconds apart (the summary file name has the time to the second). Never run the generation and the training in the same data folder at the same time. The speed of runs at the same time was not measured.

## 18. Tuning

The tuning driver searches values of run-config keys with a random search. It ranks the values on the validation set. It never reads a `test_*` column.

```bash
python framework/training/tune.py --config <path>/run_config.json [--yes] [--resume]
```

**Requirements**
- `data_preparation.validation_share` above 0. The driver uses the `val_*` metrics.
- The data of the seeds and k levels of the study. Run the generation first (section 9).

**The `tuning` section** (example: the focal loss of Bank Marketing):

```json
"tuning": {
  "study": "bm_focal_gt",
  "seeds": [1000, 1008, 1016],
  "k_levels": ["ground_truth"],
  "feature_form": "model",
  "trials": 60,
  "sampler_seed": 2026,
  "space": {
    "focal-alpha":   {"distribution": "uniform",     "low": 0.05,   "high": 0.95, "decimals": 3},
    "focal-gamma":   {"distribution": "uniform",     "low": 0.0,    "high": 5.0,  "decimals": 2},
    "learning-rate": {"distribution": "log_uniform", "low": 0.0001, "high": 0.1,  "significant": 3}
  },
  "run_config": {"loss-type": "focal", "num-server-rounds": 250},
  "metric": "val_f1",
  "tie_break": "val_pr_auc"
}
```

| Key | Default | Meaning |
|---|---|---|
| `study` | required | the name of the study: 1 to 64 letters, digits, `_` or `-` (first: a letter or a digit) |
| `seeds` | `generation.seeds` | the seeds of each trial |
| `k_levels` | `["ground_truth"]` | the datasets of each trial (as in section 17) |
| `feature_form` | `"model"` | as in section 17 |
| `trials` | 20 | the number of trials, 1 to 999 |
| `sampler_seed` | 0 | the seed of the random draws |
| `space` | required | the keys to search, one object for each key (below) |
| `run_config` | none | fixed values of every trial (as in section 17, without `results-tag` and without the keys of `space`) |
| `metric` | `"val_f1"` | the `val_*` column that ranks the trials |
| `tie_break` | `"val_pr_auc"` | the `val_*` column for equal values of `metric` |

**The search space**
- Only a key with a decimal default can be in `space` (for example `learning-rate`, `focal-alpha`, `focal-gamma`, `cb-beta`).
- `uniform`: a value from `low` to `high`, rounded to `decimals` places (default 3).
- `log_uniform`: a value from `low` to `high` on a log scale, rounded to `significant` digits (default 3). `low` must be above 0.
- `low` must be less than `high`. `low` and `high` must not change when they are rounded.
- The driver draws the values of all trials before the first run. The same settings give the same trials.

**What the driver does**
1. It checks the whole `run_config.json` and stops with a list of all faults.
2. Order of the runs: trial 1 with each seed and each k level, then trial 2, and so on. The runs of trial `<n>` use the results tag `tune_<study>_t<nnn>`.
3. It shows the plan and waits for Enter (`--yes` skips the gate).
4. For each run it reads the peak: the highest value of `metric` over the evaluated rounds (the earliest round if two rounds have the same value), and the value of `tie_break` at that round.
5. It ranks only the trials with every run completed: the mean of the peaks (highest first), then the mean of `tie_break` (highest first), then the trial number.
6. It writes the study files after each run (section 19).

Exit codes, the gate and Ctrl-C: as for the training runner (section 17).

**`--resume`.** Without `--resume`, an existing study folder stops the driver. With `--resume`, the driver stops when `study.json` or `trials_plan.csv` differs from the settings. It keeps each run that `runs.csv` shows as completed, if its test-metrics file still has the recorded checksum, and starts every other run. With `--resume` and no study folder, a new study starts.

**After the study**
- The driver does not change `run_config.json`. Copy the `run_config` of `best_run_config.json` into `training.run_config` yourself.
- Use other seeds for the training runs than for the tuning.
- `ranking.csv` has the mean and the standard deviation of the peaks of each trial. Trials within about one standard deviation of rank 1 are not clearly different.
- The ground truth does not depend on label sharing. So one study on the ground truth serves both label-sharing settings.
- Run time: trials × seeds × k levels runs (60 × 3 × 1 = 180 runs, about 4 hours with 250 rounds on the tested computer).

## 19. Keys of the training and tuning files

**1. Test metrics** `rs<seed>/<k folder>/<k folder>_test_metrics_rs<seed>.csv` (in `06_flower_vfl_results/<results tag>/split_<ratio>/`). One row for each evaluated round: every 10 rounds and the last round. The server app writes it.

| Column | Meaning |
|---|---|
| `round` | the training round |
| `test_f1` | F1 of the positive class (label code 1) |
| `test_auc_roc` | area under the ROC curve |
| `test_pr_auc` | area under the precision-recall curve (average precision) |
| `test_precision`, `test_recall`, `test_specificity` | of the positive class |
| `test_accuracy` | accuracy in percent |
| `test_loss` | the unweighted BCE, for every training loss |
| `test_tp`, `test_tn`, `test_fp`, `test_fn` | the confusion matrix |
| `val_f1` … `val_fn` | the same 12 metrics on the validation set (only with a validation share) |

- The prediction threshold is 0.5 for every metric except the two areas.
- Choose the round on the `val_*` columns, not on the `test_*` columns. The analysis does this (section 20).

**2. Training loss** `<k folder>_train_loss_rs<seed>.csv`. One row for each round. The server app writes it at the end of the run, so a stopped run has no loss file.

| Column | Meaning |
|---|---|
| `round` | the training round |
| `train_loss` | the loss of `loss-type` on the training rows. For `weighted_bce` and `focal` it is not comparable with `test_loss` |
| `grad_mean_abs_client<i>`, `grad_max_abs_client<i>` | the mean and the maximum absolute gradient of the embeddings of client `<i>` |

**3. Class-balance record** `<k folder>_class_balance_rs<seed>.csv`. One row. The server app writes it before the first round. "Positive" is the label code 1.

| Column | Meaning |
|---|---|
| `k`, `seed`, `split_ratio`, `loss_type` | the settings of the run |
| `n_train`, `positives_train`, `share_train` | the training labels of the k level |
| `n_ground_truth`, `positives_ground_truth`, `share_ground_truth` | the training labels of the ground truth |
| `n_validation`, `positives_validation`, `share_validation` | the validation labels (empty without a validation set) |
| `n_test`, `positives_test`, `share_test` | the test labels |
| `relative_difference` | the absolute difference of `share_train` and `share_ground_truth`, divided by `share_ground_truth` |
| `warning` | true when `relative_difference` is above 0.10. The warning does not stop the run |
| `pos_weight` | negatives divided by positives of the training labels (the weight of `weighted_bce`) |
| `focal_alpha`, `focal_gamma`, `cb_beta` | the run-config values of the run |
| `cb_weight_negative`, `cb_weight_positive` | the class weights of the `class_balanced` loss |

The record has every value for every loss type. `loss_type` tells which values the loss of the run uses.

**4. `training_settings.json`** (training runner; in `06_flower_vfl_results/<results tag>/split_<ratio>/`)

| Key | Meaning |
|---|---|
| `schema` | 1 |
| `results_tag`, `split_ratio`, `feature_form`, `feature_splits` | the settings of the runs |
| `dataset_config_md5` | the MD5 checksum of `dataset.json` |
| `run_config` | `training.run_config` |
| `app_defaults` | the defaults of `pyproject.toml` |

**5. `training_runs.csv`** (training runner; same folder). One row for each seed and k level.

| Column | Meaning |
|---|---|
| `seed`, `k` | the run |
| `status` | `completed` or `failed` |
| `exit_code` | the exit code of the launcher |
| `reason` | why the run failed (empty for a completed run) |
| `last_round` | the last round of the test-metrics file |
| `test_metrics_md5` | the MD5 checksum of the test-metrics file (`--resume` compares it) |
| `run_log` | the path of the run log |

**6. Summary file** `logs/training_<date>-<time>_summary.csv` (training runner). One row for each run of the plan: `run` (the number in the plan), `seed`, `k`, `results_tag`, `exit_code`, `status` (`completed`, `failed`, or `kept` for a run that `--resume` keeps), `reason`, `run_log`.

**7. Study files** (tuning driver; in `07_tuning/<study>/`)

| File | Content |
|---|---|
| `study.json` | the settings of the study (written before the first run; `--resume` compares them) |
| `trials_plan.csv` | `trial`, `results_tag` and one column for each key of `space` (written before the first run) |
| `runs.csv` | one row for each run: `trial`, `seed`, `k`, `results_tag`, `status`, `exit_code`, `reason`, `metric_peak`, `peak_round`, `tie_break_at_peak`, `last_round`, `test_metrics_md5`, `run_log` |
| `ranking.csv` | one row for each trial, rank 1 first: `rank`, `trial`, `results_tag`, the values of the trial, `runs_completed`, `runs_planned`, `mean_<metric>`, `sd_<metric>` (the sample standard deviation of the peaks), `mean_<tie_break>`, `peak_rounds` |
| `best_run_config.json` | the trial with rank 1: `study`, `trial`, `results_tag`, `mean_<metric>`, `sd_<metric>`, `trials_ranked`, `trials_planned`, `run_config` (the fixed values and the values of the trial) and a note |

## 20. Analysis of one run config

The analysis driver tests the results of one training plan (section 17). It compares each k level with the ground truth of the same seeds. Run it after the training of all seeds of the run config. The orchestrator runs it as the last step of a chain (section 23).

```bash
python framework/analysis/run_analysis.py --config <run_config.json> --yes
```

- It reads `generation`, `data_preparation` and `training` of the run config (the data folder, the seeds, the k levels, the split and the results tag) and the optional section `analysis` (below). It changes no input file.
- It writes to `<data_dir>/08_results_analysis/<results tag>/split_<ratio>/`. Section 22 gives the files. It does not write again into a folder that holds `analysis_settings.json`, unless you add `--overwrite`.
- It shows the plan and asks for confirmation. `--yes` skips the question.
- With `analysis.track_b` true (the default), it checks R and the R package emmeans before the question (section 3). If one of them is missing, it stops and writes nothing.
- It copies its screen output to a run log in `<data_dir>/logs/` (name `analysis`, section 10).
- Exit codes: 0 after a good run; 1 after a fault in the inputs (for example fewer complete seeds than `analysis.min_complete_seeds`); 2 when you answer no at the confirmation gate.
- `analysis_settings.json` is written first with `"complete": false` and last with `"complete": true`. A stopped analysis stays marked as not complete.

**The section `analysis`** (optional; every key is optional; the defaults are shown):

```json
"analysis": {
  "round_source": null,
  "selection_metric": "val_f1",
  "tie_break": "val_pr_auc",
  "metric": "test_f1",
  "extra_metrics": ["test_pr_auc", "test_auc_roc"],
  "alpha": 0.05,
  "bonferroni_family": null,
  "min_complete_seeds": 3,
  "sensitivity_gt_round": true,
  "track_b": true,
  "rscript": "Rscript",
  "output_folder": "08_results_analysis"
}
```

| Key | Meaning |
|---|---|
| `round_source` | where the round of each run is chosen: `"validation"` or `"test"`. `null`: `"validation"` when the split has a validation set, else `"test"` (below) |
| `selection_metric` | the validation column that selects the round of each run. A `val_*` column where a higher value is better (`round_source` `"validation"` only) |
| `tie_break` | the validation column for two rounds with the same `selection_metric`. It must differ from `selection_metric` (`round_source` `"validation"` only) |
| `metric` | the main test column. Track A, Track B and the ceiling use it |
| `extra_metrics` | further test columns. They get the descriptive tables and Track A only |
| `alpha` | the significance level of each group of tests. Every confidence interval has the level 1 − `alpha` |
| `bonferroni_family` | the number of tests of the Bonferroni correction of Track A. `null`: the number of k levels without the ground truth (9 with the default k levels) |
| `min_complete_seeds` | the smallest number of complete seeds (3 or more). With fewer, the analysis stops |
| `sensitivity_gt_round` | also write the tables of the second round rule (below) |
| `track_b` | `true`: run Track B in R. `false`: Track A only, without R |
| `rscript` | the `Rscript` command, or its full path |
| `output_folder` | the name of the output folder in the data folder |

**Inputs**
- The state files of the training runner: `training_runs.csv` and `training_settings.json` (section 19).
- A run counts only when its status is `completed` and the MD5 checksum of its test-metrics file is the recorded checksum.
- With `round_source` `"validation"`, the test-metrics files must have the `val_*` columns, so the training needs a validation set (`data_preparation.validation_share` above 0). Then the test set never chooses the round. With `round_source` `"test"`, the analysis needs no validation set.
- A complete seed has a counted run for the ground truth and for every k level. A seed that is not complete is left out of every test. `report.md` lists it.

**Two tracks of tests**
- **Track A** compares each k level with the ground truth, one pair at a time (Python, scipy).
- **Track B** compares all k levels together, with the seed as the repeated factor (R, emmeans).
- Each track gives its own viable ceiling.

**The round of each run.** Two rules choose a round:
- **Own round:** each run gets the round with the highest value of the selection column. For equal values, the higher `tie_break` value decides (`round_source` `"validation"` only), and then the earlier round.
- **Round of the ground truth:** every k level of a seed uses the own round of the ground truth of that seed.

`round_source` sets the selection column and the rule of the main tables. With `sensitivity_gt_round` true, a second folder holds the same tables with the other rule.

| `round_source` | Selection column | Main tables | Second folder |
|---|---|---|---|
| `"validation"` | `selection_metric`, then `tie_break` | own round | `sensitivity_gt_round/`: round of the ground truth |
| `"test"` | `metric` (a test column) | round of the ground truth | `sensitivity_own_round/`: own round |

The results of a run are the `test_*` values of the chosen round.
- `"test"` is the rule of the published results, for a split without a validation set (the default split 80-20). The test values of the chosen round are not independent of that choice. The run log and `report.md` state that the round is chosen on the test set.
- With a validation share (for example `validation_share` 0.2), the validation set chooses the round and the test set does not.

**Sign.** A difference is always the k level minus the ground truth (in a Track B pair: the later level minus the earlier level). A negative difference of F1 means that the k level is worse.

**Track A** (for `metric` and for each of `extra_metrics`). For each k level against the ground truth, over the complete seeds:
- The paired t-test (two-sided), with the Bonferroni correction: the corrected p-value is p × `bonferroni_family` (at most 1). The test is significant when the corrected p-value is below `alpha`.
- The mean difference with its confidence interval, and Cohen's d_z (the mean difference divided by the standard deviation of the differences).
- The Shapiro-Wilk test of the differences. It is a flag only: it does not change the test.
- The Wilcoxon signed-rank test with the same correction, and the Friedman test over all k levels. The analysis always reports them. The paired t-test stays the main test.

**Track B** (for `metric` only):
- The repeated-measures ANOVA over all k levels (the omnibus test), Mauchly's test of sphericity, and the Greenhouse-Geisser and Huynh-Feldt corrections. The p-value with the Greenhouse-Geisser correction decides (`omnibus_significant_gg`). Mauchly's test needs at least as many seeds as k levels (with the ground truth). With fewer seeds, the file gives a note.
- The Tukey test of all pairs of k levels (emmeans). The Tukey tests run also when the omnibus test is not significant. Then the column `omnibus_significant_gg` is false and the run log shows a warning.

**The viable ceiling** (`ceiling.csv`; for `metric`; one row for each track). A k level is separated when its test against the ground truth is significant (Track A: the corrected p-value of the paired t-test; Track B: the Tukey p-value).
- The levels are in the order of `generation.k_levels` (k\* first).
- The ceiling is the level before the first separated level. If the first level is separated, there is no ceiling (`none`).
- A level above the ceiling that is not separated is an exception. The file lists it. The column `literal` gives the highest level that is not separated.
- A level with no test result (for example, all differences equal) counts as separated. So a ceiling never rests on a missing test.
- "Not separated" means that the tests found no difference with these seeds. It is not a proof of equal performance (section 26).

**Empty values.** A test that cannot give a value (for example, all differences equal, or no residual variance in Track B) gets an empty value and a note. The driver writes a warning line and continues.

**The report.** `report.md` shows the settings, the seeds used and left out, Track A of the main metric, the omnibus test of Track B, the two ceiling rows and the warning lines of the run.

## 21. Summary across run configs and the optional seed-count advice

The summary command puts the analyses of several run configs (for example, the six combinations of dataset, label sharing and loss) into one set of tables.

```bash
python framework/analysis/summarise.py --config <summary_config.json> --yes
```

- Each run config must have a complete analysis (section 20) with the same `alpha` as the summary.
- Relative paths in the summary config are relative to the folder of the summary config.
- The confirmation gate, `--yes`, `--overwrite` and the exit codes are as in section 20. The run log (name `summary`) is in `<output_dir>/logs/`.

**A summary config** (in the folder that holds the data folders):

```json
{
  "schema_version": 1,
  "output_dir": "summary",
  "alpha": 0.05,
  "variants": [
    {"label": "ai_ls", "config": "ai_ls/config_ai_bce.json"},
    {"label": "ai_nls", "config": "ai_nls/config_ai_bce.json"}
  ]
}
```

| Key | Meaning |
|---|---|
| `schema_version` | 1 |
| `output_dir` | the folder of the summary files |
| `alpha` | the significance level (default 0.05). It must be the `analysis.alpha` of each run config |
| `variants` | the inputs. Each input has a `label` (a unique name) and one source: `config` (a run config with a complete analysis) or stored results (below) |
| `power` | optional: the seed-count advice (below) |

The summary files (section 22) hold the Track A, ceiling and omnibus rows of every run config. `track_a_all.csv` adds a second Bonferroni correction over all Track A tests of the main metric of all run configs (`p_bonf_all`).

**Optional: the seed-count advice.** The summary writes the advice only when its config has a `power` block. The advice is advice only. It changes no run and no result. The seeds of a run come only from `training.seeds` of its run config, and no part of the analysis reads the advice. You choose the number of seeds.

The advice answers one question: if a k level truly lowers the main metric by delta, with how many seeds do the tests detect the drop with the target power?
- **Track A:** the power of the paired t-test (the noncentral t distribution; where scipy gives no value, a numerical integration gives the same probability).
- **Track B:** a Monte Carlo simulation. It draws data with the covariance of the k levels, puts the drop on one k level at a time, and counts how often the omnibus test and the Tukey test of the ground truth against that level detect it.
- **A cautious spread:** the advice uses the upper confidence limit of the standard deviation of the input results (`sd_upper_confidence`), not the standard deviation itself.
- **Two readings:** `variant` uses the levels of one run config (Track A: `alpha` / the Bonferroni family; Track B: `alpha`). `all` corrects over all inputs in the target (Track A: `alpha` / all Track A tests; Track B: `alpha` / the number of inputs).
- The answer depends on delta and on the spread of the input results. Results of other settings (for example another split or other training values) give a rough first estimate only.
- Run time on the tested computer: about 6 minutes for 6 inputs with both tracks and 6 inputs with Track A only, each with 9 k levels and 4 deltas, with the default settings.

An input of the advice can also be stored results from another source. Stored results enter no table of results. Example (added to the summary config above):

```json
{
  "variants": [
    {"label": "ai_ls", "config": "ai_ls/config_ai_bce.json"},
    {"label": "earlier_ai_ls", "long_csv": "earlier/ai_ls_long.csv", "value_column": "f1",
     "note": "earlier results with other settings: a first estimate only"}
  ],
  "power": {"deltas": [0.01, 0.02], "target_power": 0.8, "sd_upper_confidence": 0.8,
            "replicates": 2000, "random_seed": 20261103, "n_min": 3, "n_max": 100}
}
```

| Key of an input | Meaning |
|---|---|
| `long_csv` | stored results with one row for each seed and k level: `seed`, `config` and the value column |
| `wide_csv` | stored results with one row for each seed and one column for each k level |
| `value_column` | (`long_csv`) the name of the value column (default `value`) |
| `configs`, `column_suffix` | (`wide_csv`) the k levels in order, and the end of their column names (for example `_f1` for the column `k5_f1`) |
| `reference` | (stored results) the reference level (default `ground_truth`) |
| `in_target` | `true` (default): the input counts for the largest numbers of seeds. `false`: its advice is shown beside the target |
| `power_tracks` | the tracks of the advice for this input (default `["track_a", "track_b"]`) |
| `note` | a text for the report |

A stored input needs at least 2 complete seeds. A seed with a missing value is left out, with a warning.

| Key of `power` | Meaning (default) |
|---|---|
| `deltas` | the drops of the main metric to detect (0.01 and 0.02). You choose them |
| `target_power` | the power to reach (0.8) |
| `sd_upper_confidence` | the confidence level of the upper limit of the standard deviation (0.8) |
| `replicates` | the simulated data sets for each number of seeds in Track B (2000) |
| `random_seed` | the random seed of the simulation (20261103). The same seed gives the same advice |
| `n_min`, `n_max` | the smallest and the largest number of seeds of the search (3 and 100) |

## 22. Keys of the analysis files

**1. `selection.csv`.** One row for each counted run.

| Column | Meaning |
|---|---|
| `seed`, `config` | the run. `config` is `ground_truth`, the k\* folder (`k0`) or `k<p>` |
| `complete_seed` | true when the seed has a counted run for every k level |
| `selected_round` | the chosen round (section 20). With `round_source` `"test"`, the main folder has the round of the ground truth of the seed |
| `last_round` | the last round of the test-metrics file |
| `val_f1`, `val_pr_auc` | the values of `selection_metric` and `tie_break` at the chosen round (the columns have the names of these settings). With `round_source` `"test"`, the selection column is `metric` and there is no tie-break column |
| `test_f1` … `test_fn` | the 12 test metrics at the selected round (section 19) |

**2. `wide_<metric>.csv` and `long_<metric>.csv`.** The values of the complete seeds. Wide: `seed` and one column for each k level (`ground_truth`, `k0`, `k5` …). Long: `seed`, `config`, `value`.

**3. `descriptive_<metric>.csv`.** One row for each k level: `config`, `n`, `mean`, `sd`, `median`, `min`, `max`, the confidence interval of the mean (`ci_level`, `ci_lower`, `ci_upper`) and `note`.

**4. `track_a_<metric>.csv`.** One row for each k level against the ground truth.

| Column | Meaning |
|---|---|
| `config` | the k level |
| `n` | the number of seed pairs |
| `diff_mean`, `diff_sd` | the mean and the standard deviation of the differences (k level minus ground truth) |
| `ci_level`, `ci_lower`, `ci_upper` | the confidence interval of the mean difference |
| `t_stat`, `p_value` | the paired t-test |
| `bonferroni_family`, `p_bonf_variant`, `significant_bonf_variant` | the Bonferroni correction of the run config: the number of tests, the corrected p-value, and true when it is below `alpha` |
| `cohens_dz` | Cohen's d_z |
| `shapiro_w`, `shapiro_p` | the Shapiro-Wilk test of the differences (a flag only) |
| `wilcoxon_n_nonzero`, `wilcoxon_stat`, `wilcoxon_p`, `wilcoxon_p_bonf_variant` | the Wilcoxon signed-rank test on the differences that are not zero, with the same correction |
| `note` | why a value is empty |

**5. `friedman_<metric>.csv`.** One row: `n_seeds`, `n_configs`, `chi2`, `df`, `p_value`, `note`.

**6. `track_b_omnibus.csv`.** One row. R writes `NA` for an empty value.

| Column | Meaning |
|---|---|
| `n_seeds`, `n_configs` | the complete seeds and the k levels (with the ground truth) |
| `f_stat`, `df1`, `df2`, `p_uncorrected` | the repeated-measures ANOVA |
| `mauchly_w`, `mauchly_p` | Mauchly's test of sphericity (empty with fewer seeds than k levels) |
| `gg_epsilon`, `p_gg` | the Greenhouse-Geisser correction |
| `hf_epsilon`, `p_hf` | the Huynh-Feldt correction |
| `alpha` | the significance level |
| `omnibus_significant_gg` | true when `p_gg` is below `alpha` |
| `note` | why a value is empty |

**7. `track_b_pairwise.csv`.** One row for each pair of k levels (45 with 10 levels).

| Column | Meaning |
|---|---|
| `group1`, `group2` | the pair, in the order of the k levels (the ground truth first) |
| `diff` | `group2` minus `group1`. In a row with the ground truth, this is the k level minus the ground truth |
| `se`, `df`, `t_ratio` | the standard error, the degrees of freedom and the t ratio of the difference |
| `p_tukey` | the Tukey-adjusted p-value |
| `ci_level`, `lower`, `upper` | the confidence interval of `diff` |
| `reject` | true when `p_tukey` is below `alpha` |
| `omnibus_significant_gg` | the result of the omnibus test (file 6), on every row |

**8. `ceiling.csv`.** One row for each track (only Track A with `analysis.track_b` false).

| Column | Meaning |
|---|---|
| `track` | `track_b_tukey` or `track_a_paired_t_bonferroni` |
| `ceiling` | the level before the first separated level, or `none` |
| `literal` | the highest level that is not separated |
| `exceptions` | the levels above the ceiling that are not separated |
| `untested` | the levels with no test result (they count as separated) |
| `first_separated` | the first separated level |
| `kstar_folder`, `kstar_separated` | the k\* folder and whether it is separated |
| `note` | an explanation, for example "no viable k level" |

**9. Other files of an analysis.** `r_session.txt`: the versions of R and emmeans. `report.md`: the main numbers and the warning lines of the run (main folder only). `analysis_settings.json`: `schema`, `complete`, `created_utc`, `run_config` and its MD5 checksum, `results_tag`, `split_ratio`, `configs`, `seeds_planned`, `seeds_used`, `seeds_left_out`, `analysis` (the section with the defaults), `ci_level`, the MD5 checksums of the training state files, `inputs` (each test-metrics file with its MD5 checksum), `files` and `versions` (Python, numpy, pandas, scipy; R and emmeans, or "not used"). `sensitivity_gt_round/` holds the same CSV files and `r_session.txt`.

**10. The summary files** (in `output_dir`).

| File | Content |
|---|---|
| `track_a_all.csv` | the rows of `track_a_<metric>.csv` (main metric) of each run config, with `variant` (the label) first and three more columns: `bonferroni_family_all` (the number of all these tests), `p_bonf_all` and `significant_bonf_all` |
| `ceilings_all.csv` | the rows of `ceiling.csv` of each run config, with `variant` first |
| `track_b_omnibus_all.csv` | the row of `track_b_omnibus.csv` of each run config with Track B, with `variant` first |
| `report.md` | the main numbers; with a `power` block also the advice, marked as advice only |
| `summary_settings.json` | the summary config with the defaults, the inputs with their MD5 checksums, the files and the versions |
| `seed_count_advice.csv` | (power only) one row for each input, k level, test (`track_a`, `track_b_omnibus`, `track_b_pair`), delta and reading (`variant`, `all`): `variant`, `in_target`, `level`, `test`, `delta`, `reading`, `alpha`, `n_data` (the seeds of the input), `sd_diff` and `sd_upper` (the standard deviation of the differences and its upper limit), `variance_factor`, `n_needed` (empty when `n_max` seeds do not reach the target power), `power_at_n`, `note` |
| `seed_count_advice_summary.csv` | (power only) one row for each `delta`, `reading` and `group` (`target` or `beside`): the largest `n_needed` of each test (`track_a`, `track_b_omnibus`, `track_b_pair`), `largest`, and `not_reached` (the tests that do not reach the target power with `n_max` seeds) |
| `power_curves.csv` | (power only) the Track B power for each input, k level, delta, number of seeds `n` and `alpha`: `power_omnibus`, `power_pair`, their Monte Carlo standard errors `mc_se_omnibus` and `mc_se_pair`, and `n_no_residual` (simulated data sets without residual variance) |

## 23. All steps with one command: the orchestrator

`framework/run_all.py` runs the chain of one or more run configs: the example prepare script when the cleaned CSV is missing, the generation, the training and the analysis. With `--summary`, it runs the summary at the end.

```bash
python framework/run_all.py --config <run_config.json> [--config <run_config.json> ...] \
  [--summary <summary_config.json>] [--steps <list>] [--yes] [--resume] [--overwrite]
python framework/run_all.py --new
```

**The steps of a chain** (in this order; `--steps` selects some of them, for example `--steps generation,training`):

| Step | Script | When it runs |
|---|---|---|
| `prepare` | `datasets/<name>/prepare.py` (section 7) | only when the cleaned CSV is missing; once for each data folder. A missing cleaned CSV with no prepare script beside `dataset.json` is a fault |
| `generation` | `framework/run_pipeline.py` (section 9) | once for each data folder and generation settings. A later run config with the same data folder and the same sections `generation` and `data_preparation` skips it |
| `training` | `framework/training/run_training.py` (section 17) | for each run config |
| `analysis` | `framework/analysis/run_analysis.py` (section 20) | when the training plan has at least `analysis.min_complete_seeds` seeds; else it is skipped with a note |
| summary | `framework/analysis/summarise.py` (section 21) | with `--summary`, after the last chain |

**Rules**
- The orchestrator reads and checks every run config before the first step. It lists all faults and stops.
- It makes each missing data folder (an empty folder) before the checks, and names it in the plan.
- It shows the plan of all steps once and waits for Enter; type q to stop. `--yes` runs without this gate. Each step runs with `--yes`, so the steps have no gates of their own.
- Each step runs as its own process with the same Python, from the code root, and writes its own run log in `<data_dir>/logs/` (section 10). The orchestrator writes its own run log (name `run_all`) in the logs folder of the first run config.
- `--resume` passes to the generation and the training (sections 14 and 17). `--overwrite` passes to the analysis and the summary.
- Existing training results need `--resume`, and an existing analysis needs `--overwrite`. Else the plan stops before the first step.
- The chain stops at the first step that fails. The orchestrator prints the step, its exit code and the command to continue.
- With `--summary`, an analysis step that would be skipped (too few seeds) is a fault.
- Exit codes: 0 when every step completed; 1 after a fault or a failed step; 2 when you stop at the gate.

**Questions** (`--new`). The orchestrator asks for the values of a new run config: optionally an example config to start from (its values become the defaults), the dataset config, label sharing, the data folder (default `data/<name>` in the code root), the seeds, the k levels, the test share and the validation share (default 80-20), the loss and its values, `num-server-rounds`, and Track B. Enter takes the value in brackets. Then it writes the run config, prints the command and asks whether to run it now. The questions write absolute paths.

## 24. Examples: the first test, the quick start and the full design

The folder `examples/` holds run configs that run as they are, from the code root. Their paths are relative to their own folder (section 8). They write their data and results to `runs/` in the code root. `examples/README.md` lists them.

**First test** (`examples/first_test.json`): the quick start (below) with 20 rounds instead of 250. Everything else is the same: the dataset, label sharing, the loss, the seeds, the k levels and the split. It writes to `runs/first_test_adult_income_ls/`. This is 30 Flower runs of about 14 seconds each; the whole chain takes about 8 minutes on the tested computer. Run it first. It checks that every part of the chain runs on your computer. Its numbers are not results: the tuned number of rounds is 250 (section 18).

```bash
python framework/run_all.py --config examples/first_test.json
```

**Quick start** (`examples/quickstart_adult_income_ls.json`): Adult Income, label sharing on, loss `bce`, seeds 42, 50 and 58, the ground truth, k\* and 8 k levels, 250 rounds, split 80-20 (no validation set). This is 30 Flower runs. It takes about 40 minutes on the tested computer. It uses the tuned values and is the check of a new installation (section 25).

```bash
python framework/run_all.py --config examples/quickstart_adult_income_ls.json
```

The chain downloads and cleans the dataset (section 7), makes the data, trains the 30 models and runs the analysis with Track B (R and emmeans, section 3). The report is `runs/quickstart_adult_income_ls/08_results_analysis/bce/split_80_20_onehot/report.md`. Without a validation set, the analysis chooses the round on the test set (section 20).

**Full design** (`examples/full_design/`): the six combinations of dataset (Adult Income, Bank Marketing), label sharing (on, off) and loss (Adult Income `bce`; Bank Marketing `weighted_bce` and `focal`), 10 seeds each (42, 50, … 114), 250 rounds, split 80-20, and a summary config for the six. This is 600 Flower runs, about 13 hours. The two Bank Marketing configs with the same label-sharing setting share one data folder, so the generation runs once for them.

```bash
python framework/run_all.py \
  --config examples/full_design/adult_income_bce_ls.json \
  --config examples/full_design/adult_income_bce_nls.json \
  --config examples/full_design/bank_marketing_weighted_bce_ls.json \
  --config examples/full_design/bank_marketing_focal_ls.json \
  --config examples/full_design/bank_marketing_weighted_bce_nls.json \
  --config examples/full_design/bank_marketing_focal_nls.json \
  --summary examples/full_design/summary_config.json
```

- The full design repeats the structure of the published experiments: the datasets, label sharing, the losses, the k levels, the seed numbers and the split. It does not repeat their numbers (section 1).
- The values of `run_config` are the results of three tuning studies (section 18) on the ground truth, with the split 60-20-20 and the one-hot form. The examples use them as they are with the split 80-20.

**How to cut the time**
- Fewer seeds: change `seeds` in `generation` and in `training`. The analysis needs at least 3 seeds.
- Fewer rounds: change `num-server-rounds` in `training.run_config`. The tuned value is 250.
- Fewer k levels: change `generation.k_levels`.
- Some steps only: `--steps` (section 23). After a stop, start the same command again with `--resume`.

## 25. Check a new installation

Run the first test of section 24 first (about 8 minutes). It uses every part of the framework: the prepare script, the three stages, the generation runner, the launcher, the training runner, the analysis with Track A and Track B, and the orchestrator. Then run the quick start. It uses the same parts with the tuned number of rounds, so the lines below are the full check.

**What to expect from the quick start** (with the tested environment of section 3):
- Plan: the steps `prepare`, `generation`, `training` and `analysis`, and `Flower runs in the plan: 30`. The orchestrator makes the data folder `runs/quickstart_adult_income_ls`.
- Prepare: `MATCH: the cleaned CSV is identical to the cleaned CSV of the published results.` (section 7).
- Generation: `ALL STAGES COMPLETED (3 seed(s))`. The natural budget k\* (`Natural budget k*`) is 2,790 (seed 42), 2,088 (seed 50) and 2,405 (seed 58). Expected warnings: the two level warnings of Adult Income in each seed (`workclass` 'Unemployed' and `occupation` 'Armed-Forces' below 1 %; in seed 50, 'Armed-Forces' is also not in the test rows) and, with label sharing on, label mismatches for each client and k level (section 10).
- Training: `runs: 30 planned, 0 kept from before, 30 to start` and, at the end, `TRAINING RUNS: 30 completed, 0 failed (0 kept from before)`. One Flower run takes about 75 seconds. Each test-metrics file has 25 rows (rounds 10 to 250) and 13 columns; it has no `val_*` columns, because the split has no validation set. The test set has 6,513 rows (the sum of `test_tp`, `test_tn`, `test_fp` and `test_fn` of a row).
- Analysis: `Runs: 30 planned, 30 used, 0 left out. Complete seeds: 3 of 3.`, the line `Track B: on (test_f1; <Rscript>, R <version>, emmeans <version>)`, and the warning that the round is chosen on the test set (section 20). With 3 seeds and 10 k levels, Track B has no Mauchly test (a warning).
- End: `ALL STEPS COMPLETED (4 step(s) run)` and the exit code 0.
- Then start the training again with `--resume`. It must start 0 Flower runs (`runs: 30 planned, 30 kept from before, 0 to start`):
  ```bash
  python framework/run_all.py --config examples/quickstart_adult_income_ls.json --steps generation,training --resume
  ```

**What to expect from the first test:** the same lines, with these differences:
- The data folder is `runs/first_test_adult_income_ls`.
- One Flower run takes about 14 seconds. Each test-metrics file has 2 rows (rounds 10 and 20) and 13 columns.
- The analysis prints the same lines and the same warnings.
- The command with `--resume` uses `examples/first_test.json`.

**Read the results as a check, not as a finding.** With 3 seeds the tests have little power, and two runs with the same seed are not exactly repeatable (section 26). So the test values, the ceilings and the other analysis warnings differ between installations.

**If something is wrong**
- A line that starts with `[ERROR]` means a fault. Read the warnings files in the `logs/` folder of the data folder (for example `runs/quickstart_adult_income_ls/logs/`).
- Exit codes: the orchestrator stops with 0 when every step completed, with 1 after a fault or a failed step, and with 2 when you stop at the gate. It prints the step that failed and the command to continue. For a failed Flower run, read its reason in the summary file and its run log (section 17), then start the same command again with `--resume`.
- **Other package versions** can give other numbers (section 3). Then the counts of rows and columns must still match, but k\* and the metric values can differ. Another pandas version can give `DIFFERENT` at the prepare step (a warning, section 7).
- The test scripts of the development compare this code with the earlier scripts. They are not part of this repository.

## 26. Known limits and links

- **`k0` is k\*.** `k0` is the default name of the k\* folder (`kstar_folder`, section 8). It is not "no phantoms". The generation, the training runner, the tuning driver and the analysis read the setting.
- **Data folder:** do not point `data_dir` at a folder with data that you want to keep. The stages replace files.
- **Analysis: no equivalence test.** "Not separated" at a k level means that the tests found no difference with the seeds of the plan. It is not a proof of equal performance.
- **Analysis: the Tukey tests of Track B** use the pooled error of all k levels. The Greenhouse-Geisser correction applies to the omnibus test only, so it does not protect the Tukey tests when the sphericity assumption fails. Track A tests each k level with its own error.
- **Analysis: Track B and the ceiling use the main metric only** (`analysis.metric`). The extra metrics get the descriptive tables and Track A.
- **Analysis: at least 3 complete seeds** (`analysis.min_complete_seeds`). A seed with a missing or failed run of one k level is left out of every test of the run config.
- **Summary: one round rule.** The summary command does not check that its run configs use the same `round_source`. Give all run configs of a summary the same split and the same round rule.
- **Seed-count advice:** advice only (section 21). It depends on the drop delta that you choose and on the spread of the input results.
- **Example prepare scripts** (section 7): they need network access and the Hugging Face copies at the fixed revisions. If a copy changes its history or leaves the Hub, the download fails. A cleaned CSV that you have stays valid.
- **Ordinal features:** the Phase 2 bivariate checks use the order (Spearman check; section 12). The example datasets declare no ordinal feature, so the real-data runs do not use this check. The tests use fake data with ordinal features.
- **Split:** the test share and the validation share are settings (section 8). The default is 80-20 with no validation set. Then the analysis chooses the round on the test set (section 20). The training runner and the tuning driver read the split folder of these settings. The tuning needs a validation set.
- **Scaler:** with the default rule (`continuous_ordinal`) the scaler changes only the continuous and ordinal columns. A model that trains on the integer form gets the integer codes of nominal features unscaled. `scaler_columns: "all"` gives the scalers of the earlier framework (section 12).
- **k\* stability:** k\* is repeatable for a seed (the same seed gives the same files). It is not stable: another random stream or another seed gives another k\*. `phase2_trials` reduces the spread only a little (section 12 has measured values). Each seed computes its own k\* with its own seeded search, so the k\* folder holds a different phantom rate in each seed. The analysis compares k\* with the ground truth of the same seed, as in the published results.
- **A column with one value stops Phase 2.** The Cramér's V check divides by zero for a categorical or ordinal column that has one value in a client. No example dataset has such a column. Remove it before the run.
- **Pair names in `budget_computation_history.json`.** The protocol names a pair from the first 3 letters of both features (for example `age_job`), and the file writer cuts the name again (`a_g`). Pairs with the same first letters overwrite each other in `bivariate_metrics` of this diagnostic file. No script reads the file.
- **Test labels:** text for Adult Income, numeric for Bank Marketing (`label.test_label_format`, section 6).
- **The Flower app:** full-batch training; the gradient messages without a wait, so runs with the same seed are not exactly repeatable; `train_loss` of a weighted loss is not comparable with `test_loss`; the fixed values of section 16. Details: `docs/flower_reference.md`, section 8.
- **Faults of Ray:** a run can stop because of a fault of Ray (1 run of 300 in a tuning study). The launcher reports it as a failed run. Start the same command again with `--resume`.
- **Relative paths** in `run_config.json` keep their `..` parts in the run config, the logs and the summary file. The folder is correct. The questions of the orchestrator (`--new`) write absolute paths.
- Links: `docs/flower_reference.md` (the Flower app: source, changes, known limits, licence checklist).
