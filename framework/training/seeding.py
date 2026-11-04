"""
Seeds of the start weights of the Flower models (Phantom VFL framework,
James Mansueto's own code).

Each model of the Flower app gets its start weights from a seed. The seed
depends on the run seed (run-config key random-seed), the role of the model
(client or head) and the client number. The seed is derived with
numpy.random.SeedSequence, the method of the stage seeds of the phantom
generation (framework.phantom_protocol.stage_seed).

seeded_start() sets the torch random stream for the build of one model only.
torch.random.fork_rng saves the random state before the block and restores it
after the block, so code outside the block gets the same random numbers as
without the seed.
"""

import contextlib
from typing import Iterator

import numpy as np
import torch

WEIGHT_SEED_TAG = 100  # second part of the key; marks a start-weight seed (the generation stages use 1 to 4)
ROLE_CLIENT = 1        # client model; the last part of the key is the client number
ROLE_HEAD = 2          # head (server) model; the last part of the key is 0


def model_seed(run_seed: int, role: int, index: int) -> int:
    """
    Seed of the start weights of one model.

    Key: (run seed, WEIGHT_SEED_TAG, role, index). index is the client number
    (partition-id) for a client model and 0 for the head model.
    Returns a 32-bit integer for torch.manual_seed.
    """
    key = [int(run_seed), WEIGHT_SEED_TAG, int(role), int(index)]
    return int(np.random.SeedSequence(key).generate_state(1)[0])


def client_seed(run_seed: int, client: int) -> int:
    """Seed of the start weights of the client model with this client number."""
    return model_seed(run_seed, ROLE_CLIENT, client)


def head_seed(run_seed: int) -> int:
    """Seed of the start weights of the head model."""
    return model_seed(run_seed, ROLE_HEAD, 0)


@contextlib.contextmanager
def seeded_start(seed: int) -> Iterator[None]:
    """
    Block for the build of one model. Inside the block, the torch random stream
    (CPU) starts from the seed. After the block, the random state is the state
    before the block.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield
