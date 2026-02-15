import logging
import logging.handlers
import sys
from typing import Optional


def stderr(
    name: Optional[str] = None,
    level: Optional[int] = logging.INFO,
    fmt: Optional[str] = "-",
) -> logging.Logger:
    """
    Create or retrieve a logger that writes to the standard error stream.

    This helper ensures you always get a logger configured for `sys.stderr`,
    with sensible defaults and safe updates if a logger already exists.

    Parameters
    ----------
    name : str, optional
        Name of the logger. If not provided, a default unnamed logger is used.
        Supplying a name allows you to reuse or distinguish loggers across modules.
    level : int, optional
        Logging level (e.g., `logging.DEBUG`, `logging.INFO`). If the existing
        logger has a higher level, it will be lowered to the new value. Passing
        `None` leaves the current level unchanged.
    fmt : str, optional
        Format string for log messages. Passing `None` removes any existing
        formatter. Passing "-" leaves the current formatter unchanged.

    Returns
    -------
    logging.Logger
        A logger instance that writes to `sys.stderr`.

    Example
    -------------
    >>> from utilitybarn.log import stderr
    >>> logger = stderr(
    ...     name="myapp", level=logging.DEBUG, fmt="%(levelname)s: %(message)s"
    ... )
    >>> logger.info("Application started")
    >>> logger.error("Something went wrong")
    """

    logger = logging.getLogger(name)
    if level is not None and logger.level > level:
        # Set the level to required for whole logger
        logger.setLevel(level)

    for h in logger.handlers:
        if not isinstance(h, logging.StreamHandler):
            continue

        # Ignore handler that do not log to stderr
        if hasattr(h.stream, "fileno") and h.stream.fileno() != 2:
            continue

        if fmt is None:
            # Remove existing formatter
            h.setFormatter(None)
        elif fmt != "-":
            # Set new formatter
            formatter = logging.Formatter(fmt=fmt)
            h.setFormatter(formatter)

        if level is not None and h.level > level:
            # Drop level to lower for this handler
            h.setLevel(level)
        return logger
    # Add new handler with required properties
    handler = logging.StreamHandler(stream=sys.stderr)
    if fmt is not None and fmt != "-":
        formatter = logging.Formatter(fmt=fmt)
        handler.setFormatter(formatter)
    if level is not None:
        handler.setLevel(level)
    logger.addHandler(handler)
    return logger


def nolog() -> logging.Logger:
    """Get logger without any handler so logs are not routed anywhere"""
    return logging.getLogger(f"{__name__}.nolog")


class BlockingQueueHandler(logging.handlers.QueueHandler):
    def enqueue(self, record):
        """Enqueue a record in blocking mode."""
        self.queue.put(record, block=True)


__all__ = ["stderr", "BlockingQueueHandler"]
