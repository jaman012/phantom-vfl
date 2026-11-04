# Flower reference

This page records how the Phantom VFL framework uses Flower: where the Flower app comes from, what was changed, how the framework connects to it, and what the known limits are. Version: 3 Nov 2026 (first version: 31 Oct 2026).

## 1. Purpose

The training stage of Phantom VFL runs with Flower (https://flower.ai). The app in `flower_app/` is a modified version of the Flower example `examples/vertical-fl` (Apache License 2.0). The data loading, the losses, the metrics and the results folder are the author's own code in `framework/training/`. Flower itself is a dependency and is not part of this repository.

## 2. Upstream

| Item | Value |
|---|---|
| Repository | https://github.com/adap/flower |
| Folder | `examples/vertical-fl` |
| Version | Main branch after the v1.25.0 release (16 Dec 2025), taken on 26 January 2026 (date given by the author). The code is identical to the example at tags `v1.25.0` and `v1.26.0` (and `v1.26.1`). |
| Licence | Apache License 2.0 (`flower_app/LICENSE` = the repository `LICENSE` at tag `v1.26.0`). No copyright owner is named in the upstream files; the repository has no `NOTICE` file. |
| Citation | Beutel et al. (2020), "Flower: A Friendly Federated Learning Research Framework", arXiv:2007.14390 (BibTeX in `flower_app/ATTRIBUTION.md`). |

How the version was found: the MD5 of the upstream `client_app.py` (`5357a62343813949bce07e700ec83f55`), `server_app.py` (`1e3512d8d67c4647844baf1b162c57e3`) and `task.py` (`b1341207fe41cdd5c54041d633fd71d0`) is the same at tags `v1.25.0`, `v1.26.0` and `v1.26.1`; the Message-API form of the example first appears at tag `v1.25.0`; the author took the example on 26 January 2026 (flwr 1.25.0 was released on 16 Dec 2025 and 1.26.0 on 5 Feb 2026). The dependency list of the author's first copy is the list of the example's `pyproject.toml` at tag `v1.26.0`. The exact commit is not recorded.

## 3. What the upstream example does

Three clients each hold some columns of the Titanic dataset (loaded from the Hugging Face Hub and split by `VerticalSizePartitioner`); the server holds the label `Survived`. In each round the server asks each client for embeddings (`query.generate_embeddings`), joins them, runs the head model, computes the BCE loss, sends each client the gradient of its embedding block (`train.apply_gradients`, sent without waiting for replies) and updates the head. Every 25 rounds it logs the training accuracy.

## 4. Changes, file by file

`flower_app/MODIFICATIONS.md` has the full list. In short:

- `client_app.py`: new handlers `generate_test_embeddings` and `generate_validation_embeddings` (a copy for the validation set); seeded start weights for each client model (`framework.training.seeding`); otherwise the two upstream handlers are unchanged.
- `server_app.py`: labels from the phantom dataset of the k level; loss chosen with `loss-type`; test evaluation every 10 rounds with the metrics saved to a CSV file; with a validation set, the same evaluation on the validation set at the same rounds, with the `val_*` metrics in the same rows; loss and gradient size of each round logged and saved to a second CSV file; the shares of the positive labels and the loss values saved to a third CSV file before the first round; run settings passed to the clients as environment variables; seeded start weights for the head model. The training loop and the two message helpers are unchanged.
- `task.py`: the Titanic loading is replaced by wrappers around `framework.training.flower_data`; the models are the upstream models.
- `pyproject.toml`: 2 clients, new run-config keys, `publisher`, and the dependency list pinned to the tested versions (the same pins as `requirements.txt`).

Inherited from the upstream code (not changed, see section 8): the evaluation order and the gradient messages without a wait. The round-1 fault of the upstream handlers (two different random client models) does not occur, because both handlers build the client model from the same seed.

## 5. Own modules (`framework/training/`)

| Module | Content |
|---|---|
| `class_balance.py` | Class-balance record of a run: numbers and shares of the positive labels (training labels of the k level, ground truth, validation set, test set), relative difference to the ground truth, warning above 10 %, `pos_weight` and the loss values; one CSV row |
| `flower_data.py` | Reads the client features of the k level, the test set and the validation set (only with a validation share), applies the pickled scaler of each client (`scaler.transform`, float32), reads the training labels (`labels.csv`) and the test labels (label map of `dataset.json`; an unknown label stops the run); for the class-balance record also the ground-truth labels; the validation labels for the validation metrics and the class-balance record (`None` without a validation set) |
| `losses.py` | `bce`, `weighted_bce`, `focal`, `class_balanced`; `get_loss_function()` has no default values |
| `metrics.py` | Test metrics of one evaluated round; `validation_metrics()` gives the same metrics for the validation set, with the names `val_*` |
| `results.py` | Results folder and file names (test metrics; training loss and gradient size of each round; class-balance record) |
| `run_flower.py` | Launcher of one run (not part of the app): starts `flwr run` with `SPLIT_RATIO`, `PYTHONPATH` and `PYTHONUNBUFFERED`, copies all screen output to `<data-dir>/logs/` (`framework/run_log.py`), marks Flower warning and error lines, and checks the outcome of the run (section 6) |
| `run_training.py` | Training runner (not part of the app): reads the `training` section of `run_config.json`, checks it, calls the launcher for each seed and k level, and keeps the state of the runs for `--resume` (`RUNNING.md`, section 17) |
| `tune.py` | Tuning driver (not part of the app): random search of run-config values, ranked on the validation metrics (`RUNNING.md`, section 18) |
| `seeding.py` | Seeds of the start weights: `numpy.random.SeedSequence` key (run seed, tag 100, model role, client number); `seeded_start()` sets `torch.manual_seed` inside `torch.random.fork_rng`, so code outside the build gets the same random numbers |

## 6. How the framework connects to the app

**Run-config keys** (defaults in `flower_app/pyproject.toml`; the runner passes the values with `flwr run . --run-config "..."`):

| Key | Meaning |
|---|---|
| `data-dir` | Data folder of the dataset (required) |
| `dataset-config` | Path of `dataset.json` (required) |
| `phantom-dataset` | k folder: `k0` (k*), `k5` … `k40` or `ground_truth` |
| `random-seed` | Seed of the split and of the start weights of the models |
| `feature-splits` | Number of features of each client, for example "5,6" (the sum must equal the number of features in `dataset.json`; the client models take their input width from the data files) |
| `num-server-rounds`, `learning-rate`, `out-feature-dim-clientapp` | Training settings (250, 0.01, 64) |
| `loss-type` | `bce`, `weighted_bce`, `focal` or `class_balanced` |
| `focal-alpha`, `focal-gamma`, `cb-beta` | Loss values (0.80, 2.0, 0.9999) |
| `results-tag` | Folder name that keeps runs with different losses apart (empty: the loss type) |

**Environment variables.** `SPLIT_RATIO` selects the split folder and the form: for example `80_20` (integer form) or `80_20_onehot` (model form, one-hot); with a validation share, for example `60_20_20_onehot`. The runner sets it before `flwr run`. `PYTHONPATH` must contain the repository root, so that the app finds `framework.training`. The server app writes `PHANTOM_DATASET`, `RANDOM_SEED`, `SPLIT_RATIO`, `PHANTOM_DATA_DIR` and `PHANTOM_DATASET_CONFIG` for the client app.

**Folders read** (under `data-dir`): `04_federated_datasets/split_<ratio>/rs<seed>/<k>/client_<i>/features.csv` and `labels.csv`; `04_federated_datasets/split_<ratio>/rs<seed>/scalers/scaler_client_<i>.pkl`; `03_train_test_splits/split_<ratio>/rs<seed>/test/client_<i>_test_features.csv` and `test_labels.csv`. With a validation share also `03_train_test_splits/split_<ratio>/rs<seed>/validation/client_<i>_validation_features.csv` (scaled with the same scaler) and `validation_labels.csv` (validation metrics and class-balance record). For the class-balance record also `04_federated_datasets/split_<ratio>/rs<seed>/ground_truth/labels.csv`.

**Output** (folder `<data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/rs<seed>/<k>/`): `<k>_test_metrics_rs<seed>.csv` (one row for each evaluated round: `round`, 12 `test_*` columns and, with a validation set, 12 `val_*` columns of the same round) and `<k>_train_loss_rs<seed>.csv` (one row for each round: `round`, `train_loss`, `grad_mean_abs_client<i>`, `grad_max_abs_client<i>`; full precision; written at the end of the run) and `<k>_class_balance_rs<seed>.csv` (one row, written before the first round: the numbers and shares of the positive labels of the training labels, the ground truth, the validation set and the test set, the relative difference to the ground truth, a warning column, `pos_weight` and the loss values; `framework/training/class_balance.py`).

**Run log.** The launcher `framework/training/run_flower.py` writes the screen output of one run to `<data-dir>/logs/flower_<results-tag>_<k>_rs<seed>_<date>-<time>.log`, and the warning and error lines to a `_warnings.txt` file next to it. The run counts as failed when `flwr run` returns an exit code other than 0, when the output has an error line, or when one of the three result files is missing or older than the start of the run. The launcher then prints `[ERROR] The run failed: <reasons>` and stops with exit code 1 (section 8, "Exit code of `flwr run`"). A run that completes prints the three result paths, and the launcher stops with 0.

**Example with the launcher** (it sets `SPLIT_RATIO`, `PYTHONPATH` and `PYTHONUNBUFFERED` itself):

```
python <repository root>/framework/training/run_flower.py --split-ratio 80_20_onehot \
--run-config "data-dir='<data folder>' dataset-config='<repository root>/datasets/bank_marketing/dataset.json' phantom-dataset='k0' random-seed=42 loss-type='focal'"
```

**Training runner and tuning driver.** `framework/training/run_training.py` reads the `training` section of `run_config.json`, calls the launcher once for each seed and each k level, keeps the state of the runs and continues them with `--resume`. `framework/training/tune.py` runs a tuning study (the `tuning` section). `RUNNING.md` describes both: sections 17 and 18, with their keys, files and exit codes. The generation runner `framework/run_pipeline.py` accepts the `training` and `tuning` sections and does not read them.

**Example** (one run by hand, without the launcher, its run log and its result check):

```
cd flower_app
PYTHONPATH="<repository root>" SPLIT_RATIO=80_20_onehot \
flwr run . --run-config "data-dir='<data folder>' dataset-config='<repository root>/datasets/bank_marketing/dataset.json' phantom-dataset='k0' random-seed=42 loss-type='focal'" \
2>&1 | tee flwr_run.log
```

## 7. Tested environment and tests

The published results came from earlier copies of the modified example (three code sets; not in this repository). Those runs and the tests below used Python 3.10.19, torch 2.6.0, flwr 1.23.0, flwr-datasets 0.5.0, ray 2.31.0, numpy 2.2.5, pandas 2.2.3, scipy 1.15.3 and scikit-learn 1.5.0. flwr 1.23.0 has every API part that the app uses. These versions do not meet four pins of the upstream dependency list (flwr ≥ 1.25.0, torch ≥ 2.8.0, pandas == 2.3.3, scikit-learn ≥ 1.6.1); `flwr run` does not check the list.

Each test below ran the app side by side with the earlier code or with the app before the change, with a fake message grid in one process (both datasets, integer and one-hot form, 20 rounds). These tests do not cover the Flower runtime itself.

- **The app against the earlier code** (k\* and k20): bit-for-bit the same numbers with the same data and the same seed.
- **Weight seed** (k\* and the ground truth): the app without the seed, given the seeded start weights, and the app with the seed give bit-for-bit the same numbers. The seed changes only the start weights.
- **Training-loss file** (k\* and the ground truth): the same numbers and test-metrics files; the loss file is the only new file, and its values equal the loss and the gradient values of the run exactly.
- **Class-balance record** (k\* and the ground truth): the same numbers, test-metrics files and loss files; the record is the only new file, and its values equal an independent count of the label files.
- **Validation metrics** (k\* and the ground truth): without a validation folder, the same numbers, files and messages. With a validation folder, the only changes are the `val_*` columns and one validation query to each client at each evaluated round, after the test queries. The `val_*` values equal the `test_*` values when the validation files are a copy of the test files, and equal an independent computation for a validation set of every second test row.

Real runs through the launcher and the training runner (seed 42, one-hot form, 20 rounds; Adult Income `bce`, Bank Marketing `weighted_bce` and `focal`; splits 80/20 and 60/20/20; with label sharing on and off) completed and gave the expected files. The test scripts are not part of this repository.

## 8. Known limits

- **Full-batch training.** Each round is one gradient step on all training rows.
- **Round 1.** The upstream code builds two different random client models in round 1 (one for the embeddings, one for the gradient step). In this app both get the same seeded start weights, so the gradient goes to the model that made the embeddings. From round 2 both handlers use the saved model.
- **Evaluation order (upstream code).** The test result of round i uses the head after round i and the client models after round i − 1. The validation result of round i uses the same head and client models as the test result; in a real run, a late gradient message (next item) can in theory reach a client between the two queries.
- **Gradient messages without a wait (upstream code; not changed).** The server sends the gradients with `push_messages` and does not wait for the clients. In the Flower simulation, several worker processes handle the client messages, so the query of the next round can reach a client before it has saved its update. The client then answers with the weights of one round earlier, and the gradient of that round is applied to newer weights. A repeat test (real `flwr run`, 20 rounds, seed 42, three pairs) gave one identical pair; the other two pairs were identical up to round 4 and differed from round 5. The cause is a strong indication from the logs, not a proof. Effect: training noise in some rounds, and runs with the same seed are not exactly repeatable. No crash, no incomplete file and no known bias. A fix (`send_and_receive` with a wait for the empty replies) changes the message order and is not made.
- **Losses.** `weighted_bce` takes `pos_weight` from the training labels of each k level. The focal-loss defaults (`focal-alpha` 0.80, `focal-gamma` 2.0) are the values of the published runs. They were not tuned for the one-hot form. A tuning study gives values for it (`RUNNING.md`, section 18).
- **`feature-splits`** is not the input width; it gives the number of clients and the sum check.
- **Training-loss file.** `train_loss` is the loss of `loss-type`. For `weighted_bce` and `focal` it is not comparable with `test_loss` of the test file (always the unweighted BCE). The file is written at the end of the run, so a stopped run has no loss file; its run log has the values with 6 decimals.
- **Exit code of `flwr run`** (flwr 1.23.0). `flwr run` starts the simulation as a second program and returns 0 also when the simulation fails (it prints the error in red). The launcher checks the error lines and the result files itself; a run with `flwr run` by hand has no such check.
- **Class-balance record.** The warning (relative difference of the positive share of the training labels to the ground truth above 10 %) does not stop the run. The validation values are empty without a validation share.
- **Class-balanced loss: 32-bit `beta`.** `ClassBalancedLoss` computes `torch.pow` on a 32-bit tensor, so `cb-beta` 0.9999 acts as 0.99989998. With 20,000 to 40,000 training rows the class weights differ from the 64-bit formula by less than 0.01 %. The record holds the weights that the loss uses.
- **Training runner.** A start without `--resume` stops when the state of the results tag and split folder exists. With `--resume`, only the runs that are not complete start; a run that starts overwrites the result files of an earlier run with the same results tag, split, seed and k level. Relative paths in `run_config.json` keep their `..` parts in the run config, the logs and the summary file (the folder is correct).
- **Round selection.** The published analysis picked the best round on the test set. With a validation share, the test-metrics file has the `val_*` metrics of each evaluated round, so an analysis can choose the round on the validation set. The analysis of the framework does this with a validation set; without one, it chooses the round on the test set, as the published analysis did (`RUNNING.md`, section 20).
- **Threshold** 0.5; `test_loss` and `val_loss` are the unweighted BCE for every training loss.
- **Weight seed.** Each model gets its start weights from the run seed, its role and its client number. The first rounds of two runs with the same seed are identical; later rounds can differ because of the gradient messages (above).

## 9. Apache License 2.0, section 4: checklist

| Section | Requirement | How it is met |
|---|---|---|
| 4(a) | Give recipients a copy of the licence | `flower_app/LICENSE` (repository `LICENSE` at tag `v1.26.0`, unchanged) |
| 4(b) | Modified files carry prominent notices that they were changed | A change notice at the top of `task.py`, `client_app.py`, `server_app.py` and `pyproject.toml`; full list in `MODIFICATIONS.md` |
| 4(c) | Keep the copyright, patent, trademark and attribution notices of the source | The upstream example files have no such notices; `ATTRIBUTION.md` names the source |
| 4(d) | Include a readable copy of the NOTICE file, if the work has one | The repository has no `NOTICE` file at tag `v1.26.0` |

## 10. Not legal advice

This page records facts about the source and the changes. It is not legal advice. The licence of the whole repository is not decided yet.
