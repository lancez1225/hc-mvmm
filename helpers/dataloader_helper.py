"""Factory helpers that build the train/test PyTorch DataLoaders."""

import numpy as np
from torch.utils.data import DataLoader

from data.kitti_dataset import KittiDataset


def _worker_init_fn(worker_id):
    """Re-seeds NumPy per worker so augmentations are not identical."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _build_dataset(cfg, split, is_training, augment_data):
    if cfg['type'] == 'KITTI':
        return KittiDataset(cfg, split, is_training=is_training, augment_data=augment_data)
    raise NotImplementedError(f"Unknown dataset type: {cfg['type']}")


def build_train_loader(cfg, split, num_workers=4):
    """Builds the training dataloader.

    Args:
        cfg: ``dataset`` sub-dictionary of the YAML config.
        split: Split name (e.g. ``train`` or ``trainval``).
        num_workers: Number of CPU workers feeding the loader.

    Returns:
        A :class:`torch.utils.data.DataLoader` with shuffle/drop_last enabled.
    """
    dataset = _build_dataset(cfg, split, is_training=True, augment_data=True)
    return DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        collate_fn=dataset.collate_batch,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )


def build_test_loader(cfg, split, num_workers=4):
    """Builds the evaluation dataloader.

    Args:
        cfg: ``dataset`` sub-dictionary of the YAML config.
        split: Split name (e.g. ``val`` or ``test``).
        num_workers: Number of CPU workers feeding the loader.

    Returns:
        A :class:`torch.utils.data.DataLoader` with shuffle disabled.
    """
    dataset = _build_dataset(cfg, split, is_training=False, augment_data=False)
    return DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        collate_fn=dataset.collate_batch,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
