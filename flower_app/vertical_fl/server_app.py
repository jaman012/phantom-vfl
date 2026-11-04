"""
vertical_fl: ServerApp of the Phantom VFL Flower app.

Modified by James Mansueto, 2026. Based on vertical_fl/server_app.py of the
Flower example examples/vertical-fl (https://github.com/adap/flower, Apache
License 2.0; main branch after the v1.25.0 release, code identical to tag
v1.26.0). Changes: the training labels come from the phantom dataset of the k
level and the run settings go to the clients as environment variables; the loss
of the head model is chosen with the run-config key loss-type; test evaluation
every 10 rounds (new message type "query.generate_test_embeddings"), with the
metrics saved to a CSV file; when the split has a validation set, the same
evaluation on the validation set at the same rounds (new message type
"query.generate_validation_embeddings"), with the val_* metrics in the same rows
of that CSV file; the loss and the gradient size of each round are
logged and saved to a second CSV file; before the first round, the shares of
the positive labels and the loss values are saved to a third CSV file
(framework.training.class_balance); the head model gets seeded start weights
(framework.training.seeding).
The training loop, send_gradients_to_clients and get_remote_embeddings are the
upstream code. See MODIFICATIONS.md.
"""

from logging import INFO, WARNING
import os

import torch
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from vertical_fl.task import ServerModel, evaluate_head_model, feature_columns  # evaluate_head_model: upstream import, not called

import logging
import pandas as pd

from framework.training import class_balance, flower_data, results, seeding
from framework.training.losses import get_loss_function
from framework.training.metrics import compute_comprehensive_metrics, format_metrics_summary
from framework.training.metrics import validation_metrics

logger = logging.getLogger(__name__)

# Create ServerApp; Flower calls the function with @app.main() and gives it `grid` and `context`
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    phantom_dataset: str = context.run_config["phantom-dataset"]  # k folder, for example "k0" (k*) or "k20"
    random_seed = context.run_config.get("random-seed", 42)
    split_ratio = os.getenv("SPLIT_RATIO", "80_20")  # 80_20 (integer form) or 80_20_onehot (model form); set by the runner
    data_dir: str = context.run_config["data-dir"]
    dataset_config: str = context.run_config["dataset-config"]
    if not data_dir or not dataset_config:
        raise ValueError(
            "The run-config keys data-dir and dataset-config are required "
            "(flwr run . --run-config \"data-dir='...' dataset-config='...'\")."
        )

    # Set environment variables for the clients and for framework.training.flower_data
    os.environ["PHANTOM_DATASET"] = phantom_dataset
    os.environ["RANDOM_SEED"] = str(random_seed)
    os.environ["SPLIT_RATIO"] = split_ratio
    os.environ[flower_data.ENV_DATA_DIR] = data_dir
    os.environ[flower_data.ENV_DATASET_CONFIG] = dataset_config

    log(INFO, f"Using phantom dataset: {phantom_dataset} (split={split_ratio}, seed={random_seed})")
    num_rounds: int = context.run_config["num-server-rounds"]
    feature_splits: str = context.run_config["feature-splits"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]  # for example "5,6" -> [5, 6]
    n_features = len(feature_columns())  # number of features in dataset.json
    if sum(in_feature_dim_clientapp) != n_features:
        raise ValueError(
            "The sum of feature splits must equal the total number of features "
            f"(got {sum(in_feature_dim_clientapp)} vs. {n_features})."
        )
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]

    # Training labels of the k level
    from vertical_fl.task import load_train_labels
    labels = torch.from_numpy(load_train_labels()).float()

    # Test labels (the same test set for every k level)
    from vertical_fl.task import load_test_labels
    test_labels = torch.from_numpy(load_test_labels()).float()
    log(INFO, f"Loaded test set: {len(test_labels)} samples")

    # Validation labels (only with a validation share; None without a validation folder)
    validation_labels_np = flower_data.load_validation_labels()
    validation_labels = torch.from_numpy(validation_labels_np).float() if validation_labels_np is not None else None
    if validation_labels is None:
        log(INFO, "No validation set: validation metrics are not computed")
    else:
        log(INFO, f"Loaded validation set: {len(validation_labels)} samples")

    # Serverapp model (made in round 1, when the number of clients is known)
    head = None
    optimizer = None

    # Class counts of the training labels; pos_weight = negatives / positives
    num_negative = (labels == 0).sum().item()
    num_positive = (labels == 1).sum().item()
    pos_weight = num_negative / num_positive if num_positive else None
    if pos_weight is not None:
        log(INFO, f"Class imbalance - pos_weight: {pos_weight:.2f}")

    # Loss of the head model (run-config key loss-type; the values come from the run config)
    loss_type: str = context.run_config["loss-type"]
    if loss_type == "weighted_bce" and pos_weight is None:
        raise ValueError("loss-type weighted_bce needs at least one positive training label")
    criterion = get_loss_function(
        loss_type=loss_type,
        pos_weight=pos_weight,
        focal_alpha=context.run_config["focal-alpha"],
        focal_gamma=context.run_config["focal-gamma"],
        cb_beta=context.run_config["cb-beta"],
        num_negative=num_negative,
        num_positive=num_positive,
    )
    log(INFO, f"Using loss function: {loss_type}")

    # Track metrics
    eval_interval = 10  # test evaluation every 10 rounds and at the last round
    round_values: list[dict] = []  # training loss and gradient size of each round (saved at the end)
    test_results: list[dict] = []  # test metrics of each evaluated round
    validation_results: list[dict] = []  # validation metrics of each evaluated round (only with a validation set)

    # Results folder: <data-dir>/06_flower_vfl_results/<results-tag>/split_<ratio>/rs<seed>/<k>/
    tag = results.results_tag(context.run_config.get("results-tag", ""), loss_type)
    results_dir = results.results_dir(data_dir, tag, split_ratio, random_seed, phantom_dataset)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Class-balance record of the run: shares of the positive labels and the loss values
    balance = class_balance.build_record(
        phantom_dataset=phantom_dataset,
        random_seed=random_seed,
        split_ratio=split_ratio,
        loss_type=loss_type,
        train_labels=labels.numpy(),
        ground_truth_labels=flower_data.load_ground_truth_labels(),
        validation_labels=validation_labels_np,
        test_labels=test_labels.numpy(),
        pos_weight=pos_weight,
        focal_alpha=context.run_config["focal-alpha"],
        focal_gamma=context.run_config["focal-gamma"],
        cb_beta=context.run_config["cb-beta"],
    )
    balance_csv_path = results_dir / results.class_balance_file_name(phantom_dataset, random_seed)
    class_balance.write_record(balance_csv_path, balance)
    log(INFO, f"Class-balance record saved: {balance_csv_path}")
    if balance["warning"]:
        log(WARNING, class_balance.warning_text(balance))

    def evaluate_on_test_set(round_num, head_model, node_ids, node_pos_mapping):
        """
        Evaluate the model on the TEST set. The clients get the message type
        "query.generate_test_embeddings" and return the embeddings of their test
        features.

        Returns:
            dict of test metrics (framework.training.metrics)
        """
        head_model.eval()  # BatchNorm uses its running statistics

        # Get TEST embeddings from clients
        test_embeddings, _ = get_remote_test_embeddings(
            grid=grid,
            node_ids=node_ids,
            num_test_nodes=len(test_labels),  # number of test rows
            embedding_dim=out_feature_dim_clientapp,
        )

        # Forward pass (no gradient needed)
        with torch.no_grad():
            test_output = head_model(test_embeddings)
            test_predictions = test_output.numpy().flatten()
            test_labels_np = test_labels.numpy().flatten()

        metrics = compute_comprehensive_metrics(
            predictions=test_predictions,
            true_labels=test_labels_np,
            round_num=round_num
        )

        head_model.train()  # back to training mode
        return metrics

    def evaluate_on_validation_set(round_num, head_model, node_ids, node_pos_mapping):
        """
        Evaluate the model on the VALIDATION set. A copy of evaluate_on_test_set
        for the validation rows: the clients get the message type
        "query.generate_validation_embeddings" and return the embeddings of
        their validation features.

        Returns:
            dict of validation metrics (framework.training.metrics.validation_metrics)
        """
        head_model.eval()  # BatchNorm uses its running statistics

        # Get VALIDATION embeddings from clients
        validation_embeddings, _ = get_remote_validation_embeddings(
            grid=grid,
            node_ids=node_ids,
            num_validation_nodes=len(validation_labels),  # number of validation rows
            embedding_dim=out_feature_dim_clientapp,
        )

        # Forward pass (no gradient needed)
        with torch.no_grad():
            validation_output = head_model(validation_embeddings)
            validation_predictions = validation_output.numpy().flatten()
            validation_labels_flat = validation_labels.numpy().flatten()

        metrics = validation_metrics(
            predictions=validation_predictions,
            true_labels=validation_labels_flat,
            round_num=round_num
        )

        head_model.train()  # back to training mode
        return metrics

    for i in range(1, num_rounds + 1):
        log(INFO, "")
        log(INFO, f"--- ServerApp Round {i} / {num_rounds} ---")

        node_ids = list(grid.get_node_ids())

        if len(node_ids) != len(in_feature_dim_clientapp):
            raise ValueError(
                "The number of feature splits must equal the number of nodes "
                f"(got {len(in_feature_dim_clientapp)} vs. {len(node_ids)})."
            )

        if head is None:
            # The server model's input size is determined by the number of clients
            # and the output feature dimension of each embedding produced by the clients
            with seeding.seeded_start(seeding.head_seed(random_seed)):  # start weights from the run seed
                head = ServerModel(input_size=out_feature_dim_clientapp * len(node_ids))
            optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

        # 1. Get embeddings from all clients (TRAINING data)
        embeddings, node_pos_mapping = get_remote_embeddings(
            grid=grid,
            node_ids=node_ids,
            num_nodes=len(labels),  # number of training rows
            embedding_dim=out_feature_dim_clientapp,
        )
        # 2. Complete forward pass and compute loss
        optimizer.zero_grad()
        output = head(embeddings)
        loss = criterion(output, labels)

        log(INFO, f"Round {i}, Loss: {loss.item():.6f}")
        round_row = {"round": i, "train_loss": loss.item()}  # one row of the loss file

        loss.backward()

        # 3. Extract gradients w.r.t. embeddings (one block for each client)
        embeddings_grad = embeddings.grad.split(
            [out_feature_dim_clientapp] * len(node_ids), dim=1
        )

        # Log the size of the gradients (a mean near 0 shows vanishing gradients)
        for idx, grad in enumerate(embeddings_grad):
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()
            log(INFO, f"Round {i} - Client {idx} gradient: mean={grad_mean:.6f}, max={grad_max:.10f}")
            round_row[f"grad_mean_abs_client{idx}"] = grad_mean
            round_row[f"grad_max_abs_client{idx}"] = grad_max

        # Update the head model
        optimizer.step()

        # 4. Evaluate on the test set every eval_interval rounds and at the last round
        # (head after this round's update; clients still with the weights of the previous round)
        if i % eval_interval == 0 or i == num_rounds:
            test_metrics = evaluate_on_test_set(i, head, node_ids, node_pos_mapping)
            test_results.append(test_metrics)

            log(INFO, "")
            log(INFO, f"Round {i} Test Results:")
            log(INFO, f"  F1-Score: {test_metrics['test_f1']:.4f}")
            log(INFO, f"  AUC-ROC:  {test_metrics['test_auc_roc']:.4f}")
            log(INFO, f"  PR-AUC:   {test_metrics['test_pr_auc']:.4f}")
            log(INFO, f"  Accuracy: {test_metrics['test_accuracy']:.2f}%")

            # Evaluate on the validation set at the same rounds, after the test set
            # (the head after this round's update, as in the test evaluation; only with a validation set)
            if validation_labels is not None:
                val_metrics = evaluate_on_validation_set(i, head, node_ids, node_pos_mapping)
                validation_results.append(val_metrics)

                log(INFO, f"Round {i} Validation Results:")
                log(INFO, f"  F1-Score: {val_metrics['val_f1']:.4f}")
                log(INFO, f"  AUC-ROC:  {val_metrics['val_auc_roc']:.4f}")
                log(INFO, f"  PR-AUC:   {val_metrics['val_pr_auc']:.4f}")
                log(INFO, f"  Accuracy: {val_metrics['val_accuracy']:.2f}%")

        round_values.append(round_row)

        # 5. Send gradients to clients
        send_gradients_to_clients(grid, node_pos_mapping, embeddings_grad)

    # Save the training loss and gradient size of each round, and the test metrics, to CSV files
    log(INFO, "")
    log(INFO, "=== Saving Results ===")

    if round_values:
        df_rounds = pd.DataFrame(round_values)
        loss_csv_path = results_dir / results.train_loss_file_name(phantom_dataset, random_seed)
        df_rounds.to_csv(loss_csv_path, index=False)
        log(INFO, f"Training loss and gradient values saved: {loss_csv_path}")

    if test_results:
        df_test = pd.DataFrame(test_results)
        if validation_results:
            # val_* columns next to the test_* columns, in the row of the same round
            df_validation = pd.DataFrame(validation_results)
            df_test = df_test.merge(df_validation, on="round", how="left", validate="one_to_one")
        test_csv_path = results_dir / results.test_metrics_file_name(phantom_dataset, random_seed)
        df_test.to_csv(test_csv_path, index=False)
        log(INFO, f"Test metrics saved: {test_csv_path}")

        final = test_results[-1]
        log(INFO, "")
        log(INFO, "=== Final Test Results ===")
        log(INFO, f"Round {final['round']}:")
        log(INFO, f"  F1-Score: {final['test_f1']:.4f}")
        log(INFO, f"  AUC-ROC:  {final['test_auc_roc']:.4f}")
        log(INFO, f"  PR-AUC:   {final['test_pr_auc']:.4f}")
        log(INFO, f"  Accuracy: {final['test_accuracy']:.2f}%")


def send_gradients_to_clients(
    grid: Grid,
    node_pos_mapping: dict[int, int],
    embeddings_grad: list[torch.Tensor],
) -> None:
    """Send gradients to clients."""

    # Create messages that target method in ClientApp with
    # @app.train("apply_gradients") decorator
    messages = []
    for node_id, pos in node_pos_mapping.items():
        arrc = ArrayRecord({"local-gradients": Array(embeddings_grad[pos].numpy())})
        message = Message(
            content=RecordDict({"gradients": arrc}),
            message_type="train.apply_gradients",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages, but don't wait for results (no replies expected)
    log(INFO, "Sending gradients to %s nodes...", len(messages))
    grid.push_messages(messages)


def get_remote_embeddings(
    grid: Grid,
    node_ids: list[str],
    num_nodes: int,  # number of training rows
    embedding_dim: int,  # embedding width of each client
) -> tuple[torch.Tensor, dict[int, int]]:
    """Get embeddings from sampled remote nodes (TRAINING data)."""

    # Create messages that target method in ClientApp with
    # @app.query("generate_embeddings") decorator
    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=RecordDict(),
            message_type="query.generate_embeddings",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    log(INFO, "Requesting embeddings from %s nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %s/%s results", len(replies), len(messages))

    embeddings = torch.zeros((num_nodes, embedding_dim * len(node_ids)))

    # Convert all embeddings back to pytorch tensors
    # and place them in the corresponding feature segment
    node_pos_mapping: dict[int, int] = {}
    for reply in replies:
        if reply.has_content():
            arr = reply.content["arrays"]["embedding"]
            embd = torch.from_numpy(arr.numpy())
            pos = reply.content["config"]["pos"]
            node_pos_mapping[reply.metadata.src_node_id] = pos
            embeddings[:, pos * embedding_dim : (pos + 1) * embedding_dim] = embd

    return embeddings.requires_grad_(), node_pos_mapping


def get_remote_test_embeddings(
    grid: Grid,
    node_ids: list[str],
    num_test_nodes: int,  # number of test rows
    embedding_dim: int,
) -> tuple[torch.Tensor, dict[int, int]]:
    """
    Get TEST embeddings from sampled remote nodes. As get_remote_embeddings(),
    with the message type "query.generate_test_embeddings".
    """

    messages = []
    for node_id in node_ids:
        message = Message(
            content=RecordDict(),
            message_type="query.generate_test_embeddings",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    log(INFO, "Requesting TEST embeddings from %s nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %s/%s test results", len(replies), len(messages))

    embeddings = torch.zeros((num_test_nodes, embedding_dim * len(node_ids)))

    node_pos_mapping: dict[int, int] = {}
    for reply in replies:
        if reply.has_content():
            arr = reply.content["arrays"]["embedding"]
            embd = torch.from_numpy(arr.numpy())
            pos = reply.content["config"]["pos"]
            node_pos_mapping[reply.metadata.src_node_id] = pos
            embeddings[:, pos * embedding_dim : (pos + 1) * embedding_dim] = embd

    return embeddings.requires_grad_(), node_pos_mapping  # requires_grad_ is not used here (the evaluation runs under torch.no_grad())


def get_remote_validation_embeddings(
    grid: Grid,
    node_ids: list[str],
    num_validation_nodes: int,  # number of validation rows
    embedding_dim: int,
) -> tuple[torch.Tensor, dict[int, int]]:
    """
    Get VALIDATION embeddings from sampled remote nodes. A copy of
    get_remote_test_embeddings() with the message type
    "query.generate_validation_embeddings".
    """

    messages = []
    for node_id in node_ids:
        message = Message(
            content=RecordDict(),
            message_type="query.generate_validation_embeddings",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    log(INFO, "Requesting VALIDATION embeddings from %s nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %s/%s validation results", len(replies), len(messages))

    embeddings = torch.zeros((num_validation_nodes, embedding_dim * len(node_ids)))

    node_pos_mapping: dict[int, int] = {}
    for reply in replies:
        if reply.has_content():
            arr = reply.content["arrays"]["embedding"]
            embd = torch.from_numpy(arr.numpy())
            pos = reply.content["config"]["pos"]
            node_pos_mapping[reply.metadata.src_node_id] = pos
            embeddings[:, pos * embedding_dim : (pos + 1) * embedding_dim] = embd

    return embeddings.requires_grad_(), node_pos_mapping  # requires_grad_ is not used here (the evaluation runs under torch.no_grad())
