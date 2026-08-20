"""Lightweight numpy/torch interoperability helpers."""

import numpy as np
import torch


def check_numpy_to_torch(x):
    """Converts a numpy array to a float torch tensor when needed.

    Args:
        x: Either a numpy ndarray or a torch tensor.

    Returns:
        A tuple ``(tensor, is_numpy)`` where ``tensor`` is always a torch
        tensor (float32 if ``x`` was a numpy array) and ``is_numpy``
        indicates whether ``x`` was originally a numpy array.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset, period):
    """Wraps periodic values such as headings into ``[-offset*P, (1-offset)*P)``.

    Args:
        val: Numpy array or torch tensor of values to wrap.
        offset: Offset that shifts the wrapping interval.
        period: Period length.

    Returns:
        Wrapped values with the same backend (numpy or torch) as ``val``.
    """
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans
