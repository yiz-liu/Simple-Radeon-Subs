import logging
import os
import sys


def setup_logger(name: str = "SimpleRadeonSubs") -> logging.Logger:
    """
    Sets up a logger with a standard format.
    """
    logger = logging.getLogger(name)

    # improved idempotency: check if handlers already exist
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Create a default logger instance for easy import
logger = setup_logger()


def configure_vllm_logging() -> None:
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    os.environ["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"
    native = logging.getLogger("vllm")
    native.setLevel(logging.WARNING)
    for handler in native.handlers:
        handler.setLevel(logging.WARNING)
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)
