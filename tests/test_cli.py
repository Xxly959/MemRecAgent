"""Exercise the installed package in real subprocesses, without external services."""

import subprocess
import sys

import pytest


def test_module_starts_with_info_log() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "memrecagent"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "INFO" in result.stderr
    assert "memrecagent" in result.stderr
    assert result.stdout == ""


def test_warning_level_suppresses_startup_info() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "memrecagent", "--log-level", "WARNING"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == ""


def test_invalid_log_level_is_rejected() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "memrecagent", "--log-level", "INVALID"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


@pytest.mark.parametrize("module", ["memrecagent", "memrecagent.logging_config"])
def test_import_does_not_configure_root_logging(module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, logging, sys; "
            "before = (list(logging.root.handlers), logging.root.level); "
            "importlib.import_module(sys.argv[1]); "
            "assert (list(logging.root.handlers), logging.root.level) == before",
            module,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == ""
