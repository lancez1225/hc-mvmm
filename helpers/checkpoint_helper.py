"""Checkpoint save/load helpers compatible with the previous file format."""

import os

import torch


def save_checkpoint(file_name, model, optimizer=None, epoch=None):
    """Serialises model (and optionally optimizer) state to ``file_name``.

    Args:
        file_name: Destination ``.pth`` path.
        model: Module whose ``state_dict`` will be saved.
        optimizer: Optional optimizer whose ``state_dict`` will also be saved.
        epoch: Optional epoch counter stored alongside the weights.
    """
    state_dict = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
    }
    torch.save(state_dict, file_name)


def load_checkpoint(file_name, model, optimizer, map_location, logger=None):
    """Loads weights and optimizer state from ``file_name`` in-place.

    Args:
        file_name: Source ``.pth`` path.
        model: Module to load the weights into; may be ``None`` to skip.
        optimizer: Optimizer to load state into; may be ``None`` to skip.
        map_location: Device or callable forwarded to :func:`torch.load`.
        logger: Optional logger used to report progress.

    Returns:
        The epoch number stored in the checkpoint, or ``None`` if absent.

    Raises:
        FileNotFoundError: If ``file_name`` does not exist.
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError(file_name)

    if logger is not None:
        logger.info('==> Loading from the checkpoint "{}"...'.format(file_name))

    checkpoint = torch.load(file_name, map_location)
    if model is not None and checkpoint['model_state'] is not None:
        model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and checkpoint['optimizer_state'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if logger is not None:
        logger.info('==> Done.')

    return checkpoint.get('epoch')
