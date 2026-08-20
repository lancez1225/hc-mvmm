"""Training entry point for HC-MVMM.

Reads a YAML config (defaults to ``configs/hc_mvmm.yaml``), builds
train/test dataloaders, the model and the optimizer, runs the trainer,
and finally launches a single evaluation pass on the validation split.
"""

import argparse
import datetime
import os

import yaml
from tensorboardX import SummaryWriter

from helpers.dataloader_helper import build_test_loader
from helpers.dataloader_helper import build_train_loader
from helpers.logger_helper import create_logger
from helpers.logger_helper import log_cfg
from helpers.optimizer_helper import build_optimizer
from helpers.random_seed_helper import set_random_seed
from helpers.test_helper import Tester
from helpers.train_helper import Trainer
from hc_mvmm import build_model


def parse_config():
    """Parses command-line arguments for ``train.py``."""
    parser = argparse.ArgumentParser(description='HC-MVMM training entry point.')
    parser.add_argument(
        '--cfg_file', type=str, default='configs/hc_mvmm.yaml',
        help='Path to the YAML config file.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override the batch size declared in the YAML config.',
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override the number of training epochs.',
    )
    parser.add_argument(
        '--result_dir', type=str, default='outputs/data',
        help='Directory for the KITTI-format detection results.',
    )
    parser.add_argument(
        '--resume_checkpoint', type=str, default=None,
        help='Resume training from this checkpoint path.',
    )
    return parser.parse_args()


def main():
    args = parse_config()
    assert os.path.exists(args.cfg_file), args.cfg_file

    with open(args.cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if args.batch_size is not None:
        cfg['dataset']['batch_size'] = args.batch_size
    if args.epochs is not None:
        cfg['trainer']['epochs'] = args.epochs
        cfg['trainer']['save_frequency'] = min(
            args.epochs, cfg['trainer']['save_frequency']
        )
        cfg['tester']['checkpoint'] = f'checkpoints/checkpoint_epoch_{args.epochs}.pth'
    if args.resume_checkpoint is not None:
        cfg['trainer']['resume_checkpoint'] = args.resume_checkpoint

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f'log_train_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
    )
    logger = create_logger(log_file)
    log_cfg(args, cfg, logger)

    tb_logger = SummaryWriter(log_dir=log_dir)

    logger.info('###################  Training  ###################')
    set_random_seed(cfg['random_seed'])

    train_loader = build_train_loader(cfg['dataset'], cfg['trainer']['split'])
    test_loader = build_test_loader(cfg['dataset'], cfg['tester']['split'])

    model = build_model(cfg['model'], dataset=train_loader.dataset)

    total_iters_each_epoch = len(train_loader)
    total_epochs = cfg['trainer']['epochs']
    optimizer, lr_scheduler = build_optimizer(
        cfg['optimizer'], model, total_iters_each_epoch, total_epochs
    )

    trainer = Trainer(
        cfg=cfg['trainer'],
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        data_loader=train_loader,
        logger=logger,
        tb_logger=tb_logger,
    )
    trainer.train()

    logger.info('###################  Evaluation  ###################')
    tester = Tester(
        cfg=cfg['tester'],
        model=model,
        data_loader=test_loader,
        result_dir=args.result_dir,
        logger=logger,
    )
    tester.test()


if __name__ == '__main__':
    main()
