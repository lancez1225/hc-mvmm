"""Adam One-Cycle optimizer wrapper.

This is the same wrapper used in OpenPCDet/Fastai-style training pipelines.
``OptimWrapper`` separates batch-norm and non-batch-norm parameter groups
so that true weight decay (AdamW-style) and step-wise LR/momentum updates
can be applied uniformly.
"""

try:  # pragma: no cover - Python 3.10+ compatibility shim
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from functools import partial

import numpy as np
from torch import nn


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def split_bn_bias(layer_groups):
    """Splits each layer group into a (non-BN, BN) pair."""
    split_groups = []
    for layer in layer_groups:
        non_bn, bn = [], []
        for child in layer.children():
            if isinstance(child, BN_TYPES):
                bn.append(child)
            else:
                non_bn.append(child)
        split_groups += [nn.Sequential(*non_bn), nn.Sequential(*bn)]
    return split_groups


def listify(p=None, q=None):
    """Coerces ``p`` into a list whose length matches ``q``."""
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if isinstance(q, int) else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def trainable_params(module):
    """Returns an iterator over ``module``'s trainable parameters."""
    return filter(lambda p: p.requires_grad, module.parameters())


def _is_tuple(x):
    return isinstance(x, tuple)


class OptimWrapper:
    """Thin wrapper around a base ``torch.optim`` optimizer.

    Adds a true-weight-decay step (a la AdamW) and convenient setters for
    learning rate, momentum, beta and weight decay over multiple param groups.
    """

    def __init__(self, opt, wd, true_wd=False, bn_wd=True):
        """Initialises with an already-built base optimizer.

        Args:
            opt: A ``torch.optim`` optimizer instance.
            wd: Initial weight decay value.
            true_wd: When True, weight decay is applied outside the optimizer step.
            bn_wd: Whether to also apply weight decay to BN parameter groups.
        """
        self.opt = opt
        self.true_wd = true_wd
        self.bn_wd = bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func, lr, layer_groups, **kwargs):
        """Creates an ``OptimWrapper`` from a layer-group description."""
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func(
            [{'params': trainable_params(lg), 'lr': 0} for lg in split_groups]
        )
        opt = cls(opt, **kwargs)
        opt.lr = listify(lr, layer_groups)
        opt.opt_func = opt_func
        return opt

    def new(self, layer_groups):
        """Clones this wrapper onto a different set of layer groups."""
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        split_groups = split_bn_bias(layer_groups)
        opt_func([{'params': trainable_params(lg), 'lr': 0} for lg in split_groups])
        return self.create(
            opt_func, self.lr, layer_groups,
            wd=self.wd, true_wd=self.true_wd, bn_wd=self.bn_wd,
        )

    def __repr__(self):
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    def step(self):
        """Applies true weight decay (if enabled) then steps the optimizer."""
        if self.true_wd:
            iterator = zip(
                self._lr, self._wd,
                self.opt.param_groups[::2], self.opt.param_groups[1::2],
            )
            for lr, wd, pg1, pg2 in iterator:
                for p in pg1['params']:
                    if p.requires_grad is False:
                        continue
                    p.data.mul_(1 - wd * lr)
                if self.bn_wd:
                    for p in pg2['params']:
                        if p.requires_grad is False:
                            continue
                        p.data.mul_(1 - wd * lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self):
        """Clears optimizer gradients."""
        self.opt.zero_grad()

    def __getattr__(self, k):
        return getattr(self.opt, k, None)

    def clear(self):
        """Resets the internal state of the wrapped optimizer."""
        sd = self.state_dict()
        sd['state'] = {}
        self.load_state_dict(sd)

    @property
    def lr(self):
        return self._lr[-1]

    @lr.setter
    def lr(self, val):
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self):
        return self._mom[-1]

    @mom.setter
    def mom(self, val):
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:
            self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self):
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val):
        """Sets beta (or alpha, depending on the optimizer family)."""
        if val is None:
            return
        if 'betas' in self.opt_keys:
            self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self):
        return self._wd[-1]

    @wd.setter
    def wd(self, val):
        """Sets weight decay (forwarded to the optimizer when ``true_wd`` is off)."""
        if not self.true_wd:
            self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    def read_defaults(self):
        """Reads the current hyper-parameter values from the inner optimizer."""
        self._beta = None
        if 'lr' in self.opt_keys:
            self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys:
            self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys:
            self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys:
            self._mom, self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys:
            self._wd = self.read_val('weight_decay')

    def set_val(self, key, val, bn_groups=True):
        """Writes ``val`` to ``key`` across (non-BN, BN) parameter group pairs."""
        if _is_tuple(val):
            val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(val, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups:
                pg2[key] = v
        return val

    def read_val(self, key):
        """Reads ``key`` from (non-BN, BN) parameter group pairs."""
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if _is_tuple(val[0]):
            val = [o[0] for o in val], [o[1] for o in val]
        return val


class LRSchedulerStep:
    """Base class for piecewise schedule of learning rate and momentum."""

    def __init__(self, fai_optimizer, total_step, lr_phases, mom_phases):
        """Initialises the phase boundaries.

        Args:
            fai_optimizer: An :class:`OptimWrapper` instance.
            total_step: Total number of optimization steps.
            lr_phases: Iterable of ``(start_fraction, lambda)`` tuples.
            mom_phases: Same structure as ``lr_phases`` but for momentum.
        """
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = self._materialize_phases(lr_phases, total_step)
        assert self.lr_phases[0][0] == 0
        self.mom_phases = self._materialize_phases(mom_phases, total_step)
        assert self.mom_phases[0][0] == 0

    @staticmethod
    def _materialize_phases(phases, total_step):
        out = []
        for i, (start, lambda_func) in enumerate(phases):
            if out:
                assert out[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(phases) - 1:
                end = int(phases[i + 1][0] * total_step)
            else:
                end = total_step
            out.append((int(start * total_step), end, lambda_func))
        return out

    def step(self, step):
        """Updates ``optimizer.lr`` and ``optimizer.mom`` at iteration ``step``."""
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    """Cosine annealing helper used by :class:`OneCycle`."""
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    """Smith's one-cycle LR schedule used for Adam-based training."""

    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor, pct_start):
        """Builds cosine-annealed LR and momentum phases.

        Args:
            fai_optimizer: An :class:`OptimWrapper`.
            total_step: Total number of optimization steps.
            lr_max: Maximum learning rate reached at the warmup peak.
            moms: Tuple ``(mom_max, mom_min)`` for the momentum schedule.
            div_factor: ``lr_initial = lr_max / div_factor``.
            pct_start: Fraction of the schedule spent in the warmup phase.
        """
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        low_lr = self.lr_max / self.div_factor
        lr_phases = (
            (0, partial(annealing_cos, low_lr, self.lr_max)),
            (self.pct_start, partial(annealing_cos, self.lr_max, low_lr / 1e4)),
        )
        mom_phases = (
            (0, partial(annealing_cos, *self.moms)),
            (self.pct_start, partial(annealing_cos, *self.moms[::-1])),
        )
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)
