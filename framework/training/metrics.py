"""
Test metrics of the Phantom VFL experiments (James Mansueto's own code).

compute_comprehensive_metrics() gives the metrics of one evaluated round from the
predicted probabilities of the head model and the true test labels. The
prediction threshold is 0.5. test_loss is the unweighted BCE for every training
loss.

  Primary:    test_f1, test_auc_roc, test_pr_auc
  Supporting: test_precision, test_recall, test_specificity
  Secondary:  test_accuracy (percent), test_loss, confusion matrix (test_tp, test_tn, test_fp, test_fn)

validation_metrics() gives the same metrics for the validation set, with the
prefix val_ in place of test_ (val_f1, val_auc_roc, ..., val_fn).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def compute_comprehensive_metrics(predictions, true_labels, round_num):
    """
    Classification metrics of one evaluated round.

    Args:
        predictions: predicted probabilities (0 to 1), shape (N,) or (N, 1)
        true_labels: true binary labels (0 or 1), shape (N,) or (N, 1)
        round_num: training round

    Returns:
        dict with 'round' and the test metrics listed in the module docstring
    """
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if true_labels.ndim > 1:
        true_labels = true_labels.flatten()

    # binary predictions (threshold 0.5)
    pred_binary = (predictions > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_labels, pred_binary).ravel()

    f1 = f1_score(true_labels, pred_binary, zero_division=0)
    auc_roc = roc_auc_score(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)

    precision = precision_score(true_labels, pred_binary, zero_division=0)
    recall = recall_score(true_labels, pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    accuracy = accuracy_score(true_labels, pred_binary) * 100  # percent

    # unweighted BCE; predictions clipped to [1e-7, 1 - 1e-7] before the log
    epsilon = 1e-7
    predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
    bce_loss = -np.mean(
        true_labels * np.log(predictions_clipped) +
        (1 - true_labels) * np.log(1 - predictions_clipped)
    )

    return {
        'round': round_num,
        'test_f1': f1,
        'test_auc_roc': auc_roc,
        'test_pr_auc': pr_auc,
        'test_precision': precision,
        'test_recall': recall,
        'test_specificity': specificity,
        'test_accuracy': accuracy,
        'test_loss': bce_loss,
        'test_tp': int(tp),
        'test_tn': int(tn),
        'test_fp': int(fp),
        'test_fn': int(fn)
    }


def format_metrics_summary(metrics):
    """
    Text summary of the metrics of one round, for the screen.

    Args:
        metrics: dict from compute_comprehensive_metrics()

    Returns:
        multi-line string
    """
    summary = f"""
Round {metrics['round']} Test Results:
  PRIMARY METRICS:
    F1-Score:     {metrics['test_f1']:.4f}
    AUC-ROC:      {metrics['test_auc_roc']:.4f}
    PR-AUC:       {metrics['test_pr_auc']:.4f}
  
  SUPPORTING METRICS:
    Precision:    {metrics['test_precision']:.4f}
    Recall:       {metrics['test_recall']:.4f}
    Specificity:  {metrics['test_specificity']:.4f}
  
  SECONDARY METRICS:
    Accuracy:     {metrics['test_accuracy']:.2f}%
    Loss:         {metrics['test_loss']:.4f}
  
  CONFUSION MATRIX:
    TP: {metrics['test_tp']:>5}  FP: {metrics['test_fp']:>5}
    TN: {metrics['test_tn']:>5}  FN: {metrics['test_fn']:>5}
"""
    return summary


def validation_metrics(predictions, true_labels, round_num):
    """
    Classification metrics of one evaluated round on the validation set.

    The same computation as compute_comprehensive_metrics(); each key test_<name>
    becomes val_<name> (val_f1, val_auc_roc, ..., val_fn); 'round' stays.

    Args:
        predictions: predicted probabilities (0 to 1), shape (N,) or (N, 1)
        true_labels: true binary validation labels (0 or 1), shape (N,) or (N, 1)
        round_num: training round

    Returns:
        dict with 'round' and the val_* metrics, in the order of compute_comprehensive_metrics()
    """
    metrics = compute_comprehensive_metrics(predictions, true_labels, round_num)
    return {("val_" + key[len("test_"):] if key.startswith("test_") else key): value
            for key, value in metrics.items()}
