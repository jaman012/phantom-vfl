"""
Class-balance record of a Flower run (James Mansueto's own code).

The server app builds the record once, before the first round, and saves it as
one CSV row next to the test metrics (framework.training.results):

  <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/rs<seed>/<k>/<k>_class_balance_rs<seed>.csv

"Positive" is the label code 1 (label map of dataset.json). Columns:

  k, seed, split_ratio, loss_type          settings of the run
  n_train, positives_train, share_train    training labels of the k level
  n_ground_truth, positives_ground_truth, share_ground_truth
                                           training labels of the ground truth (no phantoms)
  n_validation, positives_validation, share_validation
                                           validation labels (empty without a validation set)
  n_test, positives_test, share_test       test labels
  relative_difference                      |share_train - share_ground_truth| / share_ground_truth
  warning                                  True when relative_difference > WARNING_LIMIT
  pos_weight                               negatives / positives of the training labels, as the
                                           server app computes it (the weight of weighted_bce;
                                           empty without positives)
  focal_alpha, focal_gamma, cb_beta        run-config values of the run
  cb_weight_negative, cb_weight_positive   class weights of the class_balanced loss
                                           (losses.ClassBalancedLoss; empty without both classes)

The record has every value for every loss type; the column loss_type tells which
values the loss of the run uses. A warning does not stop the run: the server app
logs it as a WARNING line, and the launcher copies it to the warnings file.
"""

import numpy as np
import pandas as pd
import torch

from framework.training.losses import ClassBalancedLoss

WARNING_LIMIT = 0.10  # largest relative difference of share_train to share_ground_truth without a warning


def label_counts(labels):
    """(number of rows, number of positives) of an array of label codes 0 and 1; (None, None) for None."""
    if labels is None:
        return None, None
    values = np.asarray(labels).reshape(-1)
    return int(values.size), int((values == 1).sum())


def positive_share(n_rows, n_positive):
    """Share of the positive rows; None without rows."""
    return n_positive / n_rows if n_rows else None


def class_balanced_weights(num_negative, num_positive, beta):
    """(weight of label 0, weight of label 1) of losses.ClassBalancedLoss; (None, None) without both classes."""
    if not num_negative or not num_positive:
        return None, None
    loss = ClassBalancedLoss(torch.tensor([num_negative, num_positive], dtype=torch.float32), beta=beta)
    weight_negative, weight_positive = loss.weights.tolist()
    return weight_negative, weight_positive


def build_record(phantom_dataset, random_seed, split_ratio, loss_type, train_labels, ground_truth_labels,
                 validation_labels, test_labels, pos_weight, focal_alpha, focal_gamma, cb_beta):
    """
    The record of one run as a dict, with the columns in the order of the module docstring.

    Args:
        train_labels, ground_truth_labels, test_labels: arrays of label codes 0 and 1
        validation_labels: array of label codes 0 and 1, or None without a validation set
        pos_weight: the weight of weighted_bce that the server app computed (None without positives)
        focal_alpha, focal_gamma, cb_beta: run-config values of the run
    """
    record = {"k": phantom_dataset, "seed": random_seed, "split_ratio": split_ratio, "loss_type": loss_type}
    for name, labels in (("train", train_labels), ("ground_truth", ground_truth_labels),
                         ("validation", validation_labels), ("test", test_labels)):
        n_rows, n_positive = label_counts(labels)
        record[f"n_{name}"] = n_rows
        record[f"positives_{name}"] = n_positive
        record[f"share_{name}"] = positive_share(n_rows, n_positive)
    share_train, share_ground_truth = record["share_train"], record["share_ground_truth"]
    if share_train is None or not share_ground_truth:
        record["relative_difference"] = None
    else:
        record["relative_difference"] = abs(share_train - share_ground_truth) / share_ground_truth
    record["warning"] = record["relative_difference"] is not None and record["relative_difference"] > WARNING_LIMIT
    record["pos_weight"] = pos_weight
    record["focal_alpha"] = focal_alpha
    record["focal_gamma"] = focal_gamma
    record["cb_beta"] = cb_beta
    num_positive = record["positives_train"]
    num_negative = record["n_train"] - num_positive
    record["cb_weight_negative"], record["cb_weight_positive"] = class_balanced_weights(
        num_negative, num_positive, cb_beta)
    return record


def write_record(path, record):
    """Save the record as a CSV file with one row (full precision)."""
    pd.DataFrame([record]).to_csv(path, index=False)


def warning_text(record):
    """Text of the warning line of a record whose column warning is True."""
    return (f"Class balance: the positive share of the training labels of {record['k']} "
            f"({record['share_train']:.5f}) differs from the ground truth ({record['share_ground_truth']:.5f}) "
            f"by {100 * record['relative_difference']:.1f} % (limit {100 * WARNING_LIMIT:.0f} %).")
