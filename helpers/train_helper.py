"""Trainer driving epoch / iteration loops for HC-MVMM."""

import os

import numpy as np
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

from helpers.checkpoint_helper import load_checkpoint
from helpers.checkpoint_helper import save_checkpoint


class Trainer:
    """Standard training driver with optional gradient accumulation."""

    def __init__(self, cfg, model, optimizer, lr_scheduler,
                 data_loader, logger, tb_logger):
        """Builds the trainer state.

        Args:
            cfg: ``trainer`` sub-dictionary of the YAML config.
            model: The model to train.
            optimizer: Already-built optimizer.
            lr_scheduler: LR scheduler (may be ``None``).
            data_loader: Training :class:`DataLoader`.
            logger: Standard logger for periodic info messages.
            tb_logger: TensorBoardX :class:`SummaryWriter` for scalars.
        """
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.logger = logger
        self.tb_logger = tb_logger
        self.epoch = 0
        self.iter = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints')

        self.gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', 1)
        self.logger.info(
            f'Using gradient accumulation with {self.gradient_accumulation_steps} steps'
        )
        self.logger.info(f'Checkpoint directory: {self.checkpoint_dir}')
        if self.gradient_accumulation_steps > 1:
            physical_bs = cfg.get(
                'batch_size',
                data_loader.batch_size if hasattr(data_loader, 'batch_size') else 1,
            )
            effective_batch_size = physical_bs * self.gradient_accumulation_steps
            self.logger.info(
                f'Physical batch size: {physical_bs}, '
                f'Effective batch size: {effective_batch_size}'
            )

        if cfg.get('resume_checkpoint') is not None:
            assert os.path.exists(cfg['resume_checkpoint'])
            self.epoch = load_checkpoint(
                file_name=cfg['resume_checkpoint'],
                model=self.model,
                optimizer=self.optimizer,
                map_location=self.device,
                logger=self.logger,
            )
            assert self.epoch is not None
            self.iter = self.epoch * len(self.data_loader)

    def train(self):
        """Runs the outer epoch loop and periodic checkpointing."""
        start_epoch = self.epoch
        progress_bar = tqdm.tqdm(
            range(start_epoch, self.cfg['epochs']),
            dynamic_ncols=True, leave=True, desc='epochs',
        )
        for epoch in range(start_epoch, self.cfg['epochs']):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_one_epoch()
            self.epoch += 1

            if self.epoch % self.cfg['save_frequency'] == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                ckpt_name = os.path.join(
                    self.checkpoint_dir,
                    f'checkpoint_epoch_{self.epoch}.pth',
                )
                save_checkpoint(ckpt_name, self.model, self.optimizer, self.epoch)
            progress_bar.update()
        progress_bar.close()

    def train_one_epoch(self):
        """Runs one epoch with optional gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad()

        progress_bar = tqdm.tqdm(
            total=len(self.data_loader),
            dynamic_ncols=True,
            leave=(self.epoch + 1 == self.cfg['epochs']),
            desc='iters',
        )

        idx = -1
        for idx, batch_dict in enumerate(self.data_loader):
            batch_dict = self.data_loader.dataset.load_data_to_gpu(
                batch_dict, self.device
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.iter)

            try:
                cur_lr = float(self.optimizer.lr)
            except (AttributeError, TypeError):
                cur_lr = self.optimizer.param_groups[0]['lr']

            total_loss, stats_dict = self.model(batch_dict)
            if self.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.gradient_accumulation_steps

            total_loss.backward()

            if (idx + 1) % self.gradient_accumulation_steps == 0:
                clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.gradient_accumulation_steps > 1:
                total_loss = total_loss * self.gradient_accumulation_steps

            self.tb_logger.add_scalar('learning_rate/learning_rate', cur_lr, self.iter)
            self.tb_logger.add_scalar('loss/loss', total_loss.item(), self.iter)
            for key, val in stats_dict.items():
                self.tb_logger.add_scalar('sub_loss/' + key, val, self.iter)

            log_interval = self.cfg.get('log_interval', 50)
            if (idx + 1) % log_interval == 0:
                msg = (
                    f"epoch {self.epoch + 1}/{self.cfg['epochs']}, "
                    f"iter {idx + 1}/{len(self.data_loader)}, "
                    f"lr: {cur_lr:.6f}, total_loss: {total_loss.item():.4f}"
                )
                for key, val in stats_dict.items():
                    msg += f", {key}: {val:.4f}"
                self.logger.info(msg)

            self.iter += 1
            progress_bar.update()

        if (self.gradient_accumulation_steps > 1
                and idx >= 0
                and (idx + 1) % self.gradient_accumulation_steps != 0):
            clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.logger.info('Applied remaining accumulated gradients at epoch end')
        progress_bar.close()
