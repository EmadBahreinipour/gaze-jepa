"""
Shared utilities: seeding, device selection, I/O helpers.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str | None = None) -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        device_str: "cuda", "cpu", "mps", or None for auto-detect.

    Returns:
        torch.device
    """
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
