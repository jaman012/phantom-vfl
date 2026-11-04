"""
vertical_fl: ClientApp of the Phantom VFL Flower app.

Modified by James Mansueto, 2026. Based on vertical_fl/client_app.py of the
Flower example examples/vertical-fl (https://github.com/adap/flower, Apache
License 2.0; main branch after the v1.25.0 release, code identical to tag
v1.26.0). Changes: the new handlers "generate_test_embeddings" and
"generate_validation_embeddings" (embeddings of the test set and of the
validation set with the trained client model); seeded start weights for each client
model (framework.training.seeding); comments. Apart from the seed, the handlers
"generate_embeddings" and "apply_gradients" are the upstream code. See
MODIFICATIONS.md.
"""

import io  # in-memory bytes buffer for the optimizer state

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from vertical_fl.task import ClientModel, load_data, load_test_data, load_validation_data
from framework.training import seeding

# Flower ClientApp; the decorators below register the message handlers
app = ClientApp()


@app.query("generate_embeddings")  # message type "query.generate_embeddings", sent by the server in every round
def get_emb(msg: Message, context: Context):
    """Generate embeddings from TRAINING data."""

    # Read from config
    partition_id = context.node_config["partition-id"]  # node number given by Flower (0 or 1 for 2 clients); selects the client's features
    feature_splits: str = context.run_config["feature-splits"]  # for example "5,6"
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]  # for example 64
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]  # for example [5, 6]
    data = load_data(partition_id, in_feature_dim_clientapp)  # training features of this client at the k level of the run, scaled

    data = torch.from_numpy(data).float()  # float32 tensor (rows, features of this client)
    run_seed: int = context.run_config["random-seed"]  # seed of the run; also selects the start weights
    with seeding.seeded_start(seeding.client_seed(run_seed, partition_id)):
        model = ClientModel(
            input_size=data.shape[1], out_feat_dim=out_feature_dim_clientapp
        )  # input width from the data file; start weights from the run seed and the client number
    # Load model from state if available
    if model_record := context.state.get("model", None):  # no saved model in round 1: the model keeps its seeded start weights
        model.load_state_dict(model_record.to_torch_state_dict())
    # Do forward pass
    embedding = model(data)  # (rows, out_feature_dim_clientapp)

    # Construct and return reply Message
    model_record = ArrayRecord({"embedding": Array(embedding.detach().numpy())})
    content = RecordDict(
        {
            "arrays": model_record,
            "config": ConfigRecord({"pos": partition_id}),  # position of this client's block in the joined embeddings
        }
    )
    return Message(content=content, reply_to=msg)


@app.query("generate_test_embeddings")  # message type "query.generate_test_embeddings", sent at each evaluated round
def get_test_emb(msg: Message, context: Context):
    """
    Generate embeddings from TEST data.

    The server sends this message at each evaluated round. The handler loads the
    test features of this client (the test set has no phantoms) and returns their
    embeddings from the current client model.
    """

    # Read from config (as in the training handler)
    partition_id = context.node_config["partition-id"]
    feature_splits: str = context.run_config["feature-splits"]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]

    # Load TEST data (the same test set for every k level)
    test_data = load_test_data(partition_id, in_feature_dim_clientapp)

    test_data = torch.from_numpy(test_data).float()
    run_seed: int = context.run_config["random-seed"]  # seed of the run; also selects the start weights
    with seeding.seeded_start(seeding.client_seed(run_seed, partition_id)):
        model = ClientModel(
            input_size=test_data.shape[1], out_feat_dim=out_feature_dim_clientapp
        )

    # Load model from state if available
    if model_record := context.state.get("model", None):  # the weights saved by "apply_gradients"
        model.load_state_dict(model_record.to_torch_state_dict())

    # Do forward pass (evaluation mode, no gradients)
    model.eval()
    with torch.no_grad():
        embedding = model(test_data)

    # Construct and return reply Message (same form as the training handler)
    model_record = ArrayRecord({"embedding": Array(embedding.detach().numpy())})
    content = RecordDict(
        {
            "arrays": model_record,
            "config": ConfigRecord({"pos": partition_id}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.query("generate_validation_embeddings")  # message type "query.generate_validation_embeddings", sent at each evaluated round when the split has a validation set
def get_validation_emb(msg: Message, context: Context):
    """
    Generate embeddings from VALIDATION data.

    A copy of the test handler "generate_test_embeddings" for the validation
    set. The server sends this message at each evaluated round, after the test
    message, only when the split has a validation set. The handler loads the
    validation features of this client (the validation set has no phantoms)
    and returns their embeddings from the current client model.
    """

    # Read from config (as in the training handler)
    partition_id = context.node_config["partition-id"]
    feature_splits: str = context.run_config["feature-splits"]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]

    # Load VALIDATION data (the same validation set for every k level)
    validation_data = load_validation_data(partition_id, in_feature_dim_clientapp)

    validation_data = torch.from_numpy(validation_data).float()
    run_seed: int = context.run_config["random-seed"]  # seed of the run; also selects the start weights
    with seeding.seeded_start(seeding.client_seed(run_seed, partition_id)):
        model = ClientModel(
            input_size=validation_data.shape[1], out_feat_dim=out_feature_dim_clientapp
        )

    # Load model from state if available
    if model_record := context.state.get("model", None):  # the weights saved by "apply_gradients"
        model.load_state_dict(model_record.to_torch_state_dict())

    # Do forward pass (evaluation mode, no gradients)
    model.eval()
    with torch.no_grad():
        embedding = model(validation_data)

    # Construct and return reply Message (same form as the training handler)
    model_record = ArrayRecord({"embedding": Array(embedding.detach().numpy())})
    content = RecordDict(
        {
            "arrays": model_record,
            "config": ConfigRecord({"pos": partition_id}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.train("apply_gradients")  # message type "train.apply_gradients", sent by the server after the loss of each round
def apply_grad(msg: Message, context: Context):
    """Apply gradients to local model."""

    # Read from config
    lr = context.run_config["learning-rate"]
    partition_id = context.node_config["partition-id"]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    feature_splits: str = context.run_config["feature-splits"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]
    data = load_data(partition_id, in_feature_dim_clientapp)  # the forward pass is computed again for the backward pass

    data = torch.from_numpy(data).float()
    run_seed: int = context.run_config["random-seed"]  # seed of the run; also selects the start weights
    with seeding.seeded_start(seeding.client_seed(run_seed, partition_id)):
        model = ClientModel(
            input_size=data.shape[1], out_feat_dim=out_feature_dim_clientapp
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # Load model from state if available
    # (round 1: no saved model, so this model keeps the seeded start weights,
    # the weights that made the embeddings of round 1)
    if model_record := context.state.get("model", None):
        model.load_state_dict(model_record.to_torch_state_dict())

    # Load optimizer state from state if available (Adam moment estimates of earlier rounds)
    if optimizer_state_record := context.state.get("optimizer_state", None):
        buffer = io.BytesIO(optimizer_state_record["serialized"])
        optimizer.load_state_dict(torch.load(buffer))

    # Do forward pass
    embedding = model(data)

    # Get gradients from message and apply them: the server sends d(loss)/d(embedding);
    # backward() gives d(loss)/d(weights) of this client model
    embedding.backward(
        torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())
    )
    optimizer.step()
    # Save updated model in state for next round
    context.state["model"] = ArrayRecord(model.state_dict())
    # (Optional) Save the optimizer state. Not all optimizers have state, but Adam does.
    torch.save(optimizer.state_dict(), buffer := io.BytesIO())
    context.state["optimizer_state"] = ConfigRecord({"serialized": buffer.getvalue()})

    # Construct and return reply Message
    return Message(content=RecordDict(), reply_to=msg)
