"""Console / file logging helpers."""

import logging


def create_logger(log_file):
    """Creates a stream + file logger and returns the named instance.

    Args:
        log_file: Path of the log file that receives ``INFO``-level messages.

    Returns:
        A :class:`logging.Logger` whose name matches this module.
    """
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def log_cfg(args, cfg, logger):
    """Pretty-prints CLI arguments and the YAML config tree.

    Args:
        args: Parsed argparse namespace.
        cfg: Nested config dictionary (depth up to 2 is rendered).
        logger: A :class:`logging.Logger` used for printing.
    """
    logger.info('Arguments:')
    for key, val in vars(args).items():
        logger.info('  {:20} {}'.format(key, val))

    logger.info('Configuration:')
    for key, val in cfg.items():
        if not isinstance(val, dict):
            logger.info('  {:20} {}'.format(key, val))
        else:
            logger.info('  {}'.format(key))
            for sub_key, sub_val in val.items():
                logger.info('    {:18} {}'.format(sub_key, sub_val))
