# Modifications to the Flower example `examples/vertical-fl`

This file lists every change of this app against the upstream example (Flower repository `adap/flower`, folder `examples/vertical-fl`, code identical to tag `v1.26.0`; see `ATTRIBUTION.md`). The list comes from a code-only diff (comments and docstrings removed) of the files against the files at tag `v1.26.0`. Modified by James Mansueto, 2026.

## Summary

| Area | Upstream example | This app |
|---|---|---|
| Data | Titanic dataset from the Hugging Face Hub, split into feature groups by `VerticalSizePartitioner`; preprocessing with its own `StandardScaler` and `OneHotEncoder` | Federated datasets of the Phantom VFL framework (one folder for each k level), the test set and the pickled scalers made by the framework; feature names, client split and label map from `dataset.json` |
| Clients | 3 (`feature-splits` "2,3,2", `num-supernodes` 3) | 2 (`feature-splits` from `dataset.json`, default "5,6"; `num-supernodes` 2) |
| Labels at the server | Column `Survived` of the Titanic data | `labels.csv` of the k level (phantom rows have the label of their substitute) |
| Loss of the head model | `BCELoss()` | Chosen with `loss-type`: `bce`, `weighted_bce`, `focal` or `class_balanced` |
| Evaluation | Accuracy on the training embeddings every 25 rounds | Test set every 10 rounds and at the last round (new message type); with a validation set, the validation set at the same rounds (second new message type); F1, AUC-ROC, PR-AUC, precision, recall, specificity, accuracy, unweighted BCE, confusion matrix |
| Output | Log only | Test metrics in a CSV file (with a validation set, also the validation metrics `val_*` in the same rows); loss and gradient size of each round in the log and in a second CSV file; shares of the positive labels and the loss values of the run in a third CSV file |
| Start weights | Not seeded | Seeded from `random-seed`, the model role and the client number (`framework.training.seeding`) |

## `vertical_fl/client_app.py`

1. New handler `@app.query("generate_test_embeddings")` (`get_test_emb`): loads the test features of the client, builds a `ClientModel` (seeded as in item 3), loads the saved client model from `context.state`, sets `model.eval()` and returns the embeddings (computed under `torch.no_grad()`) in the same message form as `generate_embeddings`.
2. Imports `load_test_data` and `load_validation_data` from `task.py` and `framework.training.seeding`.
3. `get_emb` (`generate_embeddings`) and `apply_grad` (`apply_gradients`): the `ClientModel` is built inside `seeding.seeded_start(...)` (start weights from `random-seed` and `partition-id`); the rest of the code is unchanged. Comments added.
4. New handler `@app.query("generate_validation_embeddings")` (`get_validation_emb`): a copy of `get_test_emb` that loads the validation features of the client (`load_validation_data`). The server sends this message only when the split has a validation set.

Round 1: `get_emb` makes the embeddings with a new client model and does not save it; `apply_grad` then builds a new client model and applies the gradient. In the upstream code the two models have different random start weights. In this app both models get the same seeded start weights, so the gradient goes to the model that made the embeddings. From round 2, both handlers use the saved model.

## `vertical_fl/server_app.py`

1. The Titanic loading (`datasets.load_dataset`) is removed. The training labels come from `load_train_labels()` (the k level of the run); the test labels from `load_test_labels()`.
2. New run-config keys: `data-dir` and `dataset-config` (required), `phantom-dataset`, `random-seed`, `loss-type`, `focal-alpha`, `focal-gamma`, `cb-beta`, `results-tag`. The split folder comes from the environment variable `SPLIT_RATIO` (for example `80_20` or `80_20_onehot`).
3. The server writes `PHANTOM_DATASET`, `RANDOM_SEED`, `SPLIT_RATIO`, `PHANTOM_DATA_DIR` and `PHANTOM_DATASET_CONFIG` into the environment for the client app.
4. The sum check of `feature-splits` uses the number of features in `dataset.json` (upstream: the Titanic column list).
5. Class counts of the training labels and `pos_weight` = negatives / positives (used by `weighted_bce`; logged).
6. The loss object comes from `framework.training.losses.get_loss_function()` with the values of the run config (no default values in code; the defaults are in `pyproject.toml`).
7. `eval_interval` 25 → 10. At each evaluated round, the new nested function `evaluate_on_test_set` asks the clients for test embeddings (new function `get_remote_test_embeddings`, message type `query.generate_test_embeddings`), runs the head model in `eval()` mode under `torch.no_grad()`, computes the metrics (`framework.training.metrics`) and sets the head back to `train()`. The training-accuracy evaluation of upstream and its final summary are removed.
8. Log of the loss of each round (6 decimals) and of the mean and maximum absolute gradient of each client. At the end of the run, the same values (full precision) go to `<k>_train_loss_rs<seed>.csv` in the results folder of the run (columns `round`, `train_loss`, `grad_mean_abs_client<i>`, `grad_max_abs_client<i>`; `framework.training.results`).
9. The test metrics of all evaluated rounds go to `<data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/rs<seed>/<k>/<k>_test_metrics_rs<seed>.csv` (`framework.training.results`). The final test results go to the log.
10. Unchanged upstream code: the training loop (one gradient step on all training rows in each round; head optimizer Adam, learning rate 0.01), `send_gradients_to_clients` (sent with `push_messages`, no wait) and `get_remote_embeddings`. `evaluate_head_model` is still imported but not called.
11. The head model is built inside `seeding.seeded_start(seeding.head_seed(random-seed))`.
12. Before the first round, the class-balance record goes to `<k>_class_balance_rs<seed>.csv` in the results folder (`framework.training.class_balance`): the numbers and shares of the positive labels of the training labels of the k level, of the ground truth, of the validation set (if any) and of the test set; the relative difference of the training share to the ground-truth share; `pos_weight`; the focal and class-balanced values of the run. A WARNING line goes to the log when the relative difference is more than 10 %. `WARNING` is imported from `logging`.
13. Validation metrics (only when the split has a validation folder): the validation labels are read once before the first round (`flower_data.load_validation_labels()`; the class-balance record takes them from the same variable). At each evaluated round, after the test evaluation, the new nested function `evaluate_on_validation_set` (a copy of `evaluate_on_test_set`) asks the clients for validation embeddings (new function `get_remote_validation_embeddings`, a copy of `get_remote_test_embeddings`; message type `query.generate_validation_embeddings`) and computes the metrics with `framework.training.metrics.validation_metrics()` (names `val_*`). At the end of the run, the `val_*` columns are joined to the test row of the same round in `<k>_test_metrics_rs<seed>.csv`. Without a validation folder, the server sends no validation message and logs one INFO line.

Inherited behaviour (upstream order, not changed): the evaluation runs after `optimizer.step()` of the head and before the gradients go to the clients, so the test result of round i uses the head after round i and the client models after round i − 1. The validation evaluation runs at the same point, after the test evaluation.

## `vertical_fl/task.py`

1. Removed: `FederatedDataset`, `VerticalSizePartitioner`, `load_and_preprocess` (Titanic preprocessing) and the cached `fds`.
2. `load_data`, `load_test_data`, `load_validation_data`, `load_train_labels`, `load_test_labels` and `feature_columns` are small wrappers around `framework.training.flower_data`. `load_data`, `load_test_data` and `load_validation_data` read the client features of the k level, the test set or the validation set and apply the pickled scaler of the client with `scaler.transform` (float32).
3. `ClientModel`: type hints and the default `out_feat_dim=64` added; layers unchanged. `ServerModel` and `evaluate_head_model`: unchanged (type hints).

## `pyproject.toml`

1. `version` 1.0.0 → 2.0.0; `description` changed; `publisher` "flwrlabs" → "James Mansueto".
2. `[tool.flwr.app.config]`: `feature-splits` "2,3,2" → "5,6"; new keys `data-dir`, `dataset-config`, `phantom-dataset`, `random-seed`, `loss-type`, `focal-alpha`, `focal-gamma`, `cb-beta`, `results-tag`.
3. `options.num-supernodes` 3 → 2.
4. The dependency list: the tested versions, pinned, with the same pins as `requirements.txt` at the code root (`flwr[simulation]` 1.23.0, `ray` 2.31.0, `numpy` 2.2.5, `pandas` 2.2.3, `scikit-learn` 1.5.0, `torch` 2.6.0); `flwr-datasets` removed (no file imports it).
5. Unchanged: `name`, `license`, `num-server-rounds` 250 (upstream value), `learning-rate` 0.01, `out-feature-dim-clientapp` 64, `[tool.flwr.app.components]`, the build system.

## New files (not in the upstream example; the author's own code, in `framework/training/` of this repository)

- `class_balance.py`: the class-balance record of a run (shares of the positive labels, loss values, warning).
- `flower_data.py`: paths, client features (k level, test set, validation set), scalers, training and test labels; the ground-truth labels for the class-balance record; the validation labels for the validation metrics and the class-balance record.
- `losses.py`: `bce`, `weighted_bce` (the BCE of each row times `pos_weight` for label 1, then the mean), `focal` (Lin et al. 2017), `class_balanced` (Cui et al. 2019).
- `metrics.py`: test metrics of one evaluated round (threshold 0.5); `validation_metrics()` gives the same metrics for the validation set, with the names `val_*`.
- `results.py`: results folder and file names.
- `run_flower.py`: launcher of one run (not part of the app): starts `flwr run`, sets `SPLIT_RATIO`, `PYTHONPATH` and `PYTHONUNBUFFERED`, and copies the screen output to `<data-dir>/logs/` (`framework/run_log.py`).
- `run_training.py`: training runner (not part of the app): reads the `training` section of `run_config.json` and calls the launcher for each seed and k level.
- `seeding.py`: seeds of the start weights (`numpy.random.SeedSequence` key: run seed, tag, model role, client number; `torch.random.fork_rng` around each build).

## History of this app

The dissertation runs used six copies of a modified example (one for each dataset and loss, with label sharing on and off; three distinct code sets). This app replaces them with one app and moves the dataset facts to `dataset.json`. A side-by-side test with the same seed showed that this app and the three old code sets give bit-for-bit the same losses, embeddings, gradients, model weights and test-metrics files (both datasets, integer and one-hot form, k* and k20, 20 rounds). The old copies stay unchanged as the dissertation version. A second side-by-side test showed that the seeded start weights are the only change of the weight seed. A third side-by-side test showed that the training-loss file is the only change of the output of the app, and that its values equal the loss and the gradient values of the run. A fourth side-by-side test showed that the class-balance record is the only change of the output of the app, and that its values equal an independent count of the label files. A fifth side-by-side test showed that, without a validation set, the app gives bit-for-bit the same numbers, files and messages as before; with a validation set, the `val_*` columns and one validation query to each client at each evaluated round are the only changes, and the `val_*` values equal the `test_*` values when the validation files are a copy of the test files, and an independent computation for another validation set.
