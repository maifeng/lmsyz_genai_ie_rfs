"""Configure pytest so live API tests run only with explicit --live opt-in.

Input: pytest command-line options and collected tests.
Output: live tests are skipped by default, regardless of available credentials.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the flag for requests to real provider APIs.

    Args:
        parser: Pytest command-line parser.
    """
    parser.addoption(
        "--live", action="store_true", default=False,
        help="Run tests that call real provider APIs and may incur charges.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live tests unless the caller opted in.

    Args:
        config: Parsed pytest configuration.
        items: Collected tests to mark.
    """
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="Live API tests require --live.")
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip_live)
