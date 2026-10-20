"""Minimal startup command; recommendation and Agent services arrive in later phases."""

import argparse
import logging

from memrecagent.logging_config import configure_logging


def main() -> None:
    """Validate startup options and emit a scaffold status message."""
    parser = argparse.ArgumentParser(description="MemRecAgent Phase 0 scaffold")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Console log threshold (default: INFO)",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    logging.getLogger("memrecagent").info("Phase 0 scaffold ready; no recommendation model loaded.")


if __name__ == "__main__":
    main()
