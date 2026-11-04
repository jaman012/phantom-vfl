"""
vertical_fl: models and data loading of the Phantom VFL Flower app.

Modified by James Mansueto, 2026. Based on vertical_fl/task.py of the Flower
example examples/vertical-fl (https://github.com/adap/flower, Apache License 2.0;
main branch after the v1.25.0 release, code identical to tag v1.26.0).
Changes: the Titanic data loading is replaced by wrappers around
framework.training.flower_data (federated datasets, test set and validation set
of the Phantom VFL framework); type hints and the default out_feat_dim=64 are
added to ClientModel.
ClientModel, ServerModel and evaluate_head_model are otherwise the upstream code.
See MODIFICATIONS.md.
"""

import numpy as np
import torch
import torch.nn as nn

from framework.training import flower_data


# ==============================================================================
# DATA LOADING (wrappers; the code is in framework/training/flower_data.py)
# ==============================================================================

def feature_columns() -> list[str]:
    """All feature names of the dataset, client by client (from dataset.json)."""
    return flower_data.feature_columns()


def load_data(partition_id: int, feature_splits: list[int]) -> np.ndarray:
    """Training features of one client at the k level of the run, scaled, float32."""
    return flower_data.load_client_features(partition_id, feature_splits)


def load_test_data(partition_id: int, feature_splits: list[int]) -> np.ndarray:
    """Test features of one client (no phantoms), scaled with the same scaler, float32."""
    return flower_data.load_client_test_features(partition_id, feature_splits)


def load_validation_data(partition_id: int, feature_splits: list[int]) -> np.ndarray:
    """Validation features of one client (no phantoms), scaled with the same scaler, float32."""
    return flower_data.load_client_validation_features(partition_id, feature_splits)


def load_train_labels() -> np.ndarray:
    """Training labels of the k level of the run, float32, shape (N, 1)."""
    return flower_data.load_train_labels()


def load_test_labels() -> np.ndarray:
    """Test labels of the run, mapped with the label map of dataset.json, float32, shape (N, 1)."""
    return flower_data.load_test_labels()


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

class ClientModel(nn.Module):
    """
    Client model: local features -> embedding that goes to the server.
    Architecture: input -> Linear(128) -> ReLU -> Linear(out_feat_dim)

    Args:
        input_size: number of feature columns of the client (from the data file)
        out_feat_dim: embedding width (default 64)
    """
    def __init__(self, input_size: int, out_feat_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, out_feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return self.fc2(x)


class ServerModel(nn.Module):
    """
    Server (head) model: joined client embeddings -> predicted probability.
    Architecture: input -> Linear(96) -> ReLU -> BatchNorm -> Linear(1) -> Sigmoid

    Args:
        input_size: total embedding width (out_feat_dim x number of clients)
    """
    def __init__(self, input_size: int):
        super(ServerModel, self).__init__()
        self.hidden = nn.Linear(input_size, 96)
        self.fc = nn.Linear(96, 1)
        self.bn = nn.BatchNorm1d(96)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = nn.functional.relu(x)
        x = self.bn(x)
        x = self.fc(x)
        return self.sigmoid(x)


def evaluate_head_model(
    head: ServerModel,
    embeddings: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Accuracy (percent) of the head model on the given embeddings, threshold 0.5.
    The server app of Phantom VFL does not call it (it evaluates on the test set).
    """
    head.eval()
    with torch.no_grad():
        correct = 0
        embeddings_eval = embeddings.detach()
        output = head(embeddings_eval)
        predicted = (output > 0.5).float()
        correct += (predicted == labels).sum().item()
        accuracy = correct / len(labels) * 100

    return accuracy
