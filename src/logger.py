import logging

from aim import Run
from omegaconf import DictConfig


def setup_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)")

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def setup_aim_logger(name: str, config: DictConfig) -> Run:
    import aim

    experiment_tracker = aim.Run(
        experiment=name,
        repo="aim://server.aim.mjanicki.org:53800",
    )
    experiment_tracker['config'] = config

    return experiment_tracker


logger = setup_logger(__name__)
