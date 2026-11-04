"""
Results folder and file names of a Flower run (James Mansueto's own code).

  <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/rs<seed>/<k>/<k>_test_metrics_rs<seed>.csv
  <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/rs<seed>/<k>/<k>_train_loss_rs<seed>.csv
  <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/rs<seed>/<k>/<k>_class_balance_rs<seed>.csv

The test-metrics file has one row for each evaluated round: round, the test_*
metrics and, when the split has a validation set, the val_* metrics of the same
round (framework.training.metrics). The loss file has
one row for each round: round, train_loss (the loss of the run-config key
loss-type), and grad_mean_abs_client<i> and grad_max_abs_client<i> (mean and
maximum absolute gradient of the embeddings of client i). The class-balance
file has one row with the shares of the positive labels and the loss values of
the run (framework.training.class_balance).

The results tag keeps the runs of different losses apart (for example
weighted_bce and focal on the same Bank Marketing data). When the run-config key
results-tag is empty, the server app uses the loss type as the tag.
"""

from pathlib import Path

RESULTS_FOLDER = "06_flower_vfl_results"


def results_tag(tag, loss_type):
    """The results tag of the run: the run-config value, or the loss type when it is empty."""
    return tag if tag else loss_type


def results_dir(data_dir, tag, split_ratio, random_seed, phantom_dataset):
    """Folder of the results of one run (the caller makes it)."""
    return Path(data_dir) / RESULTS_FOLDER / tag / f"split_{split_ratio}" / f"rs{random_seed}" / phantom_dataset


def test_metrics_file_name(phantom_dataset, random_seed):
    """File name of the test metrics of one run."""
    return f"{phantom_dataset}_test_metrics_rs{random_seed}.csv"


def train_loss_file_name(phantom_dataset, random_seed):
    """File name of the training loss and gradient size of each round of one run."""
    return f"{phantom_dataset}_train_loss_rs{random_seed}.csv"


def class_balance_file_name(phantom_dataset, random_seed):
    """File name of the class-balance record of one run (framework.training.class_balance)."""
    return f"{phantom_dataset}_class_balance_rs{random_seed}.csv"
