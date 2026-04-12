"""Shared pytest fixtures and vcrpy configuration.

VCR cassettes are stored in tests/cassettes/. Record a cassette by running
pytest with a live API key once; subsequent runs play back the recording.
"""

from __future__ import annotations

from pathlib import Path

import pytest


CASSETTE_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture(scope="session", autouse=True)
def cassette_dir() -> Path:
    """Ensure the cassettes directory exists and return its path.

    Returns:
        Path to the cassettes directory.
    """
    CASSETTE_DIR.mkdir(exist_ok=True)
    return CASSETTE_DIR
