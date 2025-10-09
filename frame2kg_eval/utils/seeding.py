"""Utilities for deterministic seeding.

At evaluation startup we fix RNG seeds so that any downstream components
relying on randomness (e.g. embedding batching, tie-breaking inside
external libraries) behave deterministically across runs.
"""

from typing import Optional
import random

import numpy as np

# Default seed used across matching/evaluation entry points
MATCHING_SEED: int = 42


def seed_matching(seed: Optional[int] = None) -> None:
    """Seed Python/NumPy/Torch RNGs for deterministic matching workflows."""

    value = MATCHING_SEED if seed is None else seed
    random.seed(value)
    np.random.seed(value)

    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)

        # Keep CUDA kernels deterministic when possible
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch not installed; nothing else to do
        pass
