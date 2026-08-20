"""Optimizer / LR-scheduler factory used by the training entry-point."""

from functools import partial

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from utils.optim_wrapper_utils import OneCycle
from utils.optim_wrapper_utils import OptimWrapper


class CosineAnnealingWarmup:
    """Linear warmup followed by cosine annealing to ``min_lr``."""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 total_iters_each_epoch, max_lr, min_lr=1e-5):
        """Initialises the schedule.

        Args:
            optimizer: PyTorch optimizer whose ``param_groups['lr']`` is updated.
            warmup_epochs: Number of warmup epochs.
            total_epochs: Total number of training epochs.
            total_iters_each_epoch: Iterations per epoch (decides the step count).
            max_lr: Peak learning rate reached at the end of warmup.
            min_lr: Minimum learning rate at the end of training.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * total_iters_each_epoch
        self.total_steps = total_epochs * total_iters_each_epoch
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self, step=None):
        """Advances the schedule and writes the new LR into ``optimizer``.

        Args:
            step: Optional absolute iteration index. When omitted, the
                schedule advances by one step internally.
        """
        self.current_step = step if step is not None else self.current_step + 1

        if self.current_step <= self.warmup_steps:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (
                self.current_step / self.warmup_steps
            )
        else:
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def build_optimizer(cfg, model, total_iters_each_epoch, total_epochs):
    """Builds the optimizer and LR scheduler described by ``cfg``.

    Three optimizer types are supported (matching the YAML configs):
    ``AdamOneCycle`` (default for HC-MVMM), ``AdamCosine`` and plain ``Adam``.

    Args:
        cfg: ``optimizer`` sub-dictionary of the YAML config.
        model: The model whose parameters will be optimised.
        total_iters_each_epoch: Iteration count per epoch.
        total_epochs: Total epoch count.

    Returns:
        Tuple ``(optimizer, lr_scheduler)``. The scheduler may be ``None``.

    Raises:
        NotImplementedError: If ``cfg['type']`` is unknown.
    """
    if cfg['type'] == 'AdamOneCycle':
        def _children(m):
            return list(m.children())

        def _flatten(m):
            return sum(map(_flatten, m.children()), []) if _children(m) else [m]

        def _layer_groups(m):
            return [nn.Sequential(*_flatten(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, cfg['lr'], _layer_groups(model),
            wd=cfg['weight_decay'], true_wd=True, bn_wd=True,
        )
        total_steps = total_iters_each_epoch * total_epochs

        moms = cfg.get('moms', [0.95, 0.85])
        div_factor = cfg.get('div_factor', 10)
        pct_start = cfg.get('pct_start', 0.4)
        lr_scheduler = OneCycle(
            optimizer, total_steps, cfg['lr'], moms, div_factor, pct_start
        )
        return optimizer, lr_scheduler

    if cfg['type'] == 'AdamCosine':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
            betas=cfg.get('betas', [0.9, 0.999]),
        )
        warmup_epochs = cfg.get('warmup_epochs', 5)
        min_lr = cfg.get('min_lr', 1e-5)
        lr_scheduler = CosineAnnealingWarmup(
            optimizer, warmup_epochs, total_epochs, total_iters_each_epoch,
            cfg['lr'], min_lr,
        )
        return optimizer, lr_scheduler

    if cfg['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
            betas=cfg.get('betas', [0.9, 0.999]),
        )
        if 'lr_scheduler' in cfg:
            scheduler_cfg = cfg['lr_scheduler']
            if scheduler_cfg['type'] == 'StepLR':
                lr_scheduler = StepLR(
                    optimizer,
                    step_size=scheduler_cfg['step_size'] * total_iters_each_epoch,
                    gamma=scheduler_cfg['gamma'],
                )
            elif scheduler_cfg['type'] == 'CosineAnnealingLR':
                lr_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs * total_iters_each_epoch,
                    eta_min=scheduler_cfg.get('min_lr', 1e-5),
                )
            else:
                raise NotImplementedError(
                    f"Scheduler {scheduler_cfg['type']} not implemented"
                )
        else:
            lr_scheduler = None
        return optimizer, lr_scheduler

    raise NotImplementedError(f"Optimizer type {cfg['type']} not implemented")
