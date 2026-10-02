"""Check explicit live-test selection using isolated pytest sessions.

Input: synthetic offline/live tests; output: verified passed/skipped counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("live", [False, True])
def test_live_requires_opt_in(pytester: pytest.Pytester, live: bool) -> None:
    """Verify credentials alone cannot opt tests into real API calls.

    Args:
        pytester: Isolated pytest project fixture.
        live: Whether the invocation explicitly opts in.
    """
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text())
    pytester.makeini("[pytest]\nmarkers = live: provider API test\n")
    pytester.makepyfile('''
import pytest

@pytest.mark.live
def test_provider() -> None:
    """Stand in for a live provider call."""
    assert True

def test_offline() -> None:
    """Stand in for an offline check."""
    assert True
''')
    result = pytester.runpytest_subprocess(*(["--live"] if live else []), "-q")
    result.assert_outcomes(passed=2 if live else 1, skipped=0 if live else 1)
