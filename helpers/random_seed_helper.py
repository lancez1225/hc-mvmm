"""Reproducibility helpers: seeds Python, NumPy, PyTorch and cuDNN."""

import random

import numpy as np
import torch


def set_random_seed(seed):
    """Sets every relevant RNG with derived seeds for reproducibility.

    cuDNN's benchmark mode is disabled and deterministic mode is enabled.
    Note that some CUDA ops remain non-deterministic regardless.

    Args:
        seed: Base integer seed; derived seeds are powers of this value to
            diversify state between Python, NumPy and PyTorch RNGs.
    """
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
