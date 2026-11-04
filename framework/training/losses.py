"""
Loss functions of the server (head) model (James Mansueto's own code).

The head model ends with a sigmoid, so every loss here takes predicted
probabilities (N, 1) and binary labels (N, 1).

  bce             torch.nn.BCELoss() (mean over the rows)
  weighted_bce    BCE of each row times a weight (pos_weight for label 1, 1.0 for
                  label 0), then the mean; pos_weight = negatives / positives of the
                  training labels of the k level
  focal           focal loss (Lin et al. 2017): alpha_t * (1 - p_t)^gamma * CE, mean
  class_balanced  BCE with class weights from the effective number of samples
                  (Cui et al. 2019), mean

get_loss_function() has no default values: the server app passes every value
from the run config, and pyproject.toml of the Flower app holds the defaults.
"""

import torch
import torch.nn as nn

LOSS_TYPES = ("bce", "weighted_bce", "focal", "class_balanced")


class FocalLoss(nn.Module):
    """
    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), mean over the rows.

    Args:
        alpha: weight of the positive class (0 to 1); the negative class gets 1 - alpha
        gamma: focusing value (>= 0); gamma = 0 gives a class-weighted BCE
    """

    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-7  # predictions are clipped to [eps, 1 - eps] before the log

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)
        # p_t: predicted probability of the true class
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device),
                              torch.tensor(1 - self.alpha, dtype=inputs.dtype, device=inputs.device))
        # cross entropy of each row: -[y*log(p) + (1-y)*log(1-p)]
        ce = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        focal_loss = alpha_t * focal_weight * ce
        return focal_loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    BCE with class weights from the effective number of samples
    E_n = (1 - beta^n) / (1 - beta). The weights are normalised so that they add up
    to the number of classes.

    Args:
        samples_per_class: tensor [number of negatives, number of positives]
        beta: value between 0 and 1 (for example 0.9999)
    """

    def __init__(self, samples_per_class: torch.Tensor, beta: float):
        super().__init__()
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        self.register_buffer('weights', weights / weights.sum() * len(weights))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sample_weights = self.weights[targets.long().squeeze()]
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        weighted_loss = (bce_loss.squeeze() * sample_weights).mean()
        return weighted_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE: BCE of each row times pos_weight (label 1) or 1.0 (label 0), then the mean.

    Args:
        pos_weight: weight of the positive class (negatives / positives of the training labels)
    """

    def __init__(self, pos_weight: float):
        super().__init__()
        self.pos_weight = pos_weight
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_per_sample = self.criterion(inputs, targets)
        sample_weights = torch.where(targets == 1, self.pos_weight, 1.0)
        return (loss_per_sample * sample_weights).mean()


def get_loss_function(
    loss_type: str,
    pos_weight: float,
    focal_alpha: float,
    focal_gamma: float,
    cb_beta: float,
    num_negative: int,
    num_positive: int,
) -> nn.Module:
    """
    Loss object for the run-config key loss-type.

    Args:
        loss_type: one of "bce", "weighted_bce", "focal", "class_balanced"
        pos_weight: weight of the positive class (weighted_bce)
        focal_alpha, focal_gamma: values of the focal loss (focal)
        cb_beta: beta of the class-balanced loss (class_balanced)
        num_negative, num_positive: label counts of the training labels (class_balanced)

    Returns:
        torch.nn.Module that takes (probabilities, labels) and returns the mean loss
    """
    if loss_type == "bce":
        return nn.BCELoss()
    if loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight)
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if loss_type == "class_balanced":
        samples_per_class = torch.tensor([num_negative, num_positive], dtype=torch.float32)
        return ClassBalancedLoss(samples_per_class, beta=cb_beta)
    raise ValueError(f"Unknown loss type: {loss_type}. Choose from: {', '.join(LOSS_TYPES)}")
