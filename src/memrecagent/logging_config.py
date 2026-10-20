"""Logging configuration, called explicitly by application entry points."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging for a standalone process without forcing handlers."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
