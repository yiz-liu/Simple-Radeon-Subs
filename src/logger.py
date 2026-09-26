import ctypes
import logging
import os
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from typing import BinaryIO, TextIO

from tqdm import tqdm

from src.config import NATIVE_OUTPUT_TAIL_BYTES


class _ProgressLogHandler(logging.StreamHandler[TextIO]):
    def emit(self, record: logging.LogRecord) -> None:
        with tqdm.external_write_mode(file=self.stream):
            super().emit(record)


def setup_logger(name: str = "SimpleRadeonSubs") -> logging.Logger:
    """
    Sets up a logger with a standard format.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = _ProgressLogHandler(sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Create a default logger instance for easy import
logger = setup_logger()


@contextmanager
def project_output(stream: TextIO) -> Generator[None, None, None]:
    handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, _ProgressLogHandler)
    ]
    previous = [handler.stream for handler in handlers]
    propagate = logger.propagate
    try:
        logger.propagate = False
        for handler in handlers:
            handler.setStream(stream)
        yield
    finally:
        logger.propagate = propagate
        for handler, original in zip(handlers, previous, strict=True):
            handler.setStream(original)


def native_output_tail(handle: BinaryIO) -> str:
    size = os.fstat(handle.fileno()).st_size
    return (
        os.pread(
            handle.fileno(),
            NATIVE_OUTPUT_TAIL_BYTES,
            max(0, size - NATIVE_OUTPUT_TAIL_BYTES),
        )
        .decode("utf-8", errors="replace")
        .strip()
    )


class NativeOutput:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.failed = False


def _flush_output() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    for name in ("stdout", "stderr"):
        if libc.fflush(ctypes.c_void_p.in_dll(libc, name)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))


@contextmanager
def native_output(label: str) -> Generator[NativeOutput, None, None]:
    _flush_output()
    with (
        tempfile.TemporaryFile() as captured,
        os.fdopen(os.dup(1), "w", encoding="utf-8") as stdout,
        os.fdopen(os.dup(2), "w", encoding="utf-8") as stream,
        project_output(stream),
    ):
        output = NativeOutput(stream)
        try:
            os.dup2(captured.fileno(), 1)
            os.dup2(captured.fileno(), 2)
            yield output
        except BaseException:
            output.failed = True
            raise
        finally:
            try:
                _flush_output()
            except OSError as error:
                logger.warning("Unable to flush %s diagnostics: %s", label, error)
            finally:
                os.dup2(stdout.fileno(), 1)
                os.dup2(stream.fileno(), 2)
            if output.failed:
                detail = native_output_tail(captured)
                if detail:
                    logger.warning(
                        "%s native diagnostics (last %d bytes):\n%s",
                        label,
                        NATIVE_OUTPUT_TAIL_BYTES,
                        detail,
                    )


def configure_vllm_logging() -> None:
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    os.environ["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"
    native = logging.getLogger("vllm")
    native.setLevel(logging.WARNING)
    for handler in native.handlers:
        handler.setLevel(logging.WARNING)
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)
