"""Evaluation entry point for HC-MVMM.

Loads a trained checkpoint and runs detection + KITTI evaluation on the
validation split (or just inference on the test split).
"""

import argparse
import datetime
import os

import yaml

from helpers.dataloader_helper import build_test_loader
from helpers.logger_helper import create_logger
from helpers.logger_helper import log_cfg
from helpers.random_seed_helper import set_random_seed
from helpers.test_helper import Tester
from hc_mvmm import build_model


def parse_config():
    """Parses command-line arguments for ``test.py``."""
    parser = argparse.ArgumentParser(description='HC-MVMM evaluation entry point.')
    parser.add_argument(
        '--cfg_file', type=str, default='configs/hc_mvmm.yaml',
        help='Path to the YAML config file.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override the batch size declared in the YAML config.',
    )
    parser.add_argument(
        '--result_dir', type=str, default='outputs/data',
        help='Directory for the KITTI-format detection results.',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Override the checkpoint path declared in the YAML config.',
    )
    return parser.parse_args()


def main():
    args = parse_config()
    assert os.path.exists(args.cfg_file), args.cfg_file

    with open(args.cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if args.batch_size is not None:
        cfg['dataset']['batch_size'] = args.batch_size
    if args.checkpoint is not None:
        cfg['tester']['checkpoint'] = args.checkpoint

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f'log_eval_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
    )
    logger = create_logger(log_file)
    log_cfg(args, cfg, logger)

    logger.info('###################  Evaluation Only  ###################')
    set_random_seed(cfg['random_seed'])

    test_loader = build_test_loader(cfg['dataset'], cfg['tester']['split'])
    model = build_model(cfg['model'], dataset=test_loader.dataset)

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
