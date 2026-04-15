"""Tests for SqliteCache: get, put, all_ids, and persistence behaviour.

Uses pytest's tmp_path fixture so every test gets a fresh temporary directory.
No API calls are made.

Input: temporary SQLite database paths.
Output: assertion results only; no files written outside tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lmsyz_genai_ie_rfs.dataframe import SqliteCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache(tmp_path: Path, name: str = "test_cache.db") -> SqliteCache:
    """Return a fresh SqliteCache backed by a temp file.

    Args:
        tmp_path: pytest-provided temporary directory.
        name: Filename for the SQLite database.

    Returns:
        A SqliteCache instance pointing at tmp_path/name.
    """
    return SqliteCache(tmp_path / name)


def _sample_result(row_id: str) -> dict[str, Any]:
    """Return a minimal result dict for testing round-trips.

    Args:
        row_id: Identifier to embed in the result dict.

    Returns:
        Dict with input_id, label, and confidence keys.
    """
    return {"input_id": row_id, "label": "positive", "confidence": 0.9}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSqliteCacheGet:
    """Tests for SqliteCache.get."""

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        """get() on an empty cache should return None, not raise."""
        cache = _make_cache(tmp_path)
        assert cache.get("nonexistent-id") is None

    def test_get_after_put_returns_value(self, tmp_path: Path) -> None:
        """get() should return the dict previously stored by put()."""
        cache = _make_cache(tmp_path)
        result = _sample_result("row-1")
        cache.put("row-1", result)
        assert cache.get("row-1") == result

    def test_get_returns_correct_type(self, tmp_path: Path) -> None:
        """get() must return a dict, not a string."""
        cache = _make_cache(tmp_path)
        cache.put("row-1", _sample_result("row-1"))
        value = cache.get("row-1")
        assert isinstance(value, dict)

    def test_get_preserves_nested_values(self, tmp_path: Path) -> None:
        """put/get round-trip must preserve all value types, including floats."""
        cache = _make_cache(tmp_path)
        result = {"input_id": "r1", "score": 0.75, "tags": ["a", "b"], "nested": {"k": 1}}
        cache.put("r1", result)
        assert cache.get("r1") == result


class TestSqliteCachePut:
    """Tests for SqliteCache.put."""

    def test_put_overwrites_existing(self, tmp_path: Path) -> None:
        """A second put() on the same key should replace the value."""
        cache = _make_cache(tmp_path)
        cache.put("row-1", {"input_id": "row-1", "label": "positive"})
        cache.put("row-1", {"input_id": "row-1", "label": "negative"})
        assert cache.get("row-1")["label"] == "negative"

    def test_put_multiple_keys(self, tmp_path: Path) -> None:
        """Putting different keys should not interfere with each other."""
        cache = _make_cache(tmp_path)
        cache.put("a", {"v": 1})
        cache.put("b", {"v": 2})
        assert cache.get("a") == {"v": 1}
        assert cache.get("b") == {"v": 2}


class TestSqliteCacheAllIds:
    """Tests for SqliteCache.all_ids."""

    def test_all_ids_empty(self, tmp_path: Path) -> None:
        """all_ids() on a fresh cache should return an empty set."""
        cache = _make_cache(tmp_path)
        assert cache.all_ids() == set()

    def test_all_ids_after_single_put(self, tmp_path: Path) -> None:
        """all_ids() after one put should return a set with that ID."""
        cache = _make_cache(tmp_path)
        cache.put("row-1", _sample_result("row-1"))
        assert cache.all_ids() == {"row-1"}

    def test_all_ids_after_multiple_puts(self, tmp_path: Path) -> None:
        """all_ids() should reflect all inserted IDs."""
        cache = _make_cache(tmp_path)
        for i in range(5):
            cache.put(f"row-{i}", _sample_result(f"row-{i}"))
        assert cache.all_ids() == {f"row-{i}" for i in range(5)}

    def test_all_ids_after_overwrite(self, tmp_path: Path) -> None:
        """Overwriting a key should not duplicate it in all_ids()."""
        cache = _make_cache(tmp_path)
        cache.put("row-1", {"v": 1})
        cache.put("row-1", {"v": 2})
        assert cache.all_ids() == {"row-1"}

    def test_all_ids_returns_set(self, tmp_path: Path) -> None:
        """all_ids() must return a set, not a list or other type."""
        cache = _make_cache(tmp_path)
        cache.put("x", {})
        assert isinstance(cache.all_ids(), set)


class TestSqliteCachePersistence:
    """Tests that SqliteCache persists data across instances."""

    def test_data_survives_new_instance(self, tmp_path: Path) -> None:
        """Data written by one SqliteCache instance must be readable by another."""
        db_path = tmp_path / "persistent.db"
        result = _sample_result("row-99")

        # Write with first instance.
        cache1 = SqliteCache(db_path)
        cache1.put("row-99", result)

        # Read with a fresh instance pointing at the same file.
        cache2 = SqliteCache(db_path)
        assert cache2.get("row-99") == result

    def test_all_ids_survives_new_instance(self, tmp_path: Path) -> None:
        """all_ids() must return the correct set even after a fresh SqliteCache init."""
        db_path = tmp_path / "persistent2.db"

        cache1 = SqliteCache(db_path)
        for i in range(3):
            cache1.put(f"id-{i}", {"v": i})

        cache2 = SqliteCache(db_path)
        assert cache2.all_ids() == {"id-0", "id-1", "id-2"}


# ---------------------------------------------------------------------------
# Prompt-hash gating (added 2026-04)
# ---------------------------------------------------------------------------


class TestPromptHashGating:
    """``put`` and ``get`` / ``all_ids`` respect ``prompt_hash`` filtering."""

    def test_get_matches_hash(self, tmp_path: Path) -> None:
        """Get returns the row when the hash matches."""
        cache = _make_cache(tmp_path)
        cache.put("r1", {"x": 1}, prompt_hash="h_a")
        assert cache.get("r1", prompt_hash="h_a") == {"x": 1}

    def test_get_rejects_wrong_hash(self, tmp_path: Path) -> None:
        """Get returns None when the hash differs."""
        cache = _make_cache(tmp_path)
        cache.put("r1", {"x": 1}, prompt_hash="h_a")
        assert cache.get("r1", prompt_hash="h_b") is None

    def test_get_without_hash_returns_regardless(self, tmp_path: Path) -> None:
        """``prompt_hash=None`` is 'any hash' and returns the stored row."""
        cache = _make_cache(tmp_path)
        cache.put("r1", {"x": 1}, prompt_hash="h_a")
        assert cache.get("r1") == {"x": 1}

    def test_all_ids_filters_by_hash(self, tmp_path: Path) -> None:
        """``all_ids(prompt_hash=h)`` returns only IDs stamped with ``h``."""
        cache = _make_cache(tmp_path)
        cache.put("r1", {"x": 1}, prompt_hash="h_a")
        cache.put("r2", {"x": 2}, prompt_hash="h_b")
        cache.put("r3", {"x": 3}, prompt_hash="h_a")
        assert cache.all_ids(prompt_hash="h_a") == {"r1", "r3"}
        assert cache.all_ids(prompt_hash="h_b") == {"r2"}
        assert cache.all_ids() == {"r1", "r2", "r3"}

    def test_put_overwrites_hash(self, tmp_path: Path) -> None:
        """Re-putting the same row_id updates both result and hash."""
        cache = _make_cache(tmp_path)
        cache.put("r1", {"x": 1}, prompt_hash="h_a")
        cache.put("r1", {"x": 2}, prompt_hash="h_b")
        assert cache.get("r1", prompt_hash="h_a") is None
        assert cache.get("r1", prompt_hash="h_b") == {"x": 2}

    def test_legacy_rows_without_hash_are_null(self, tmp_path: Path) -> None:
        """A legacy put (no hash) stores NULL; does not match any specific hash."""
        cache = _make_cache(tmp_path)
        cache.put("r1", {"x": 1})  # no prompt_hash kwarg
        assert cache.get("r1") == {"x": 1}
        assert cache.get("r1", prompt_hash="h_a") is None
        assert cache.all_ids(prompt_hash="h_a") == set()
        assert cache.all_ids() == {"r1"}


class TestPromptHashHelper:
    """``compute_prompt_hash`` is stable and 16 hex chars."""

    def test_hash_is_deterministic(self) -> None:
        """Same input → same hash."""
        from lmsyz_genai_ie_rfs.dataframe import compute_prompt_hash
        assert compute_prompt_hash("hello") == compute_prompt_hash("hello")

    def test_hash_differs_per_input(self) -> None:
        """Different inputs → different hashes."""
        from lmsyz_genai_ie_rfs.dataframe import compute_prompt_hash
        assert compute_prompt_hash("a") != compute_prompt_hash("b")

    def test_hash_length(self) -> None:
        """Hash is 16 hex characters."""
        from lmsyz_genai_ie_rfs.dataframe import compute_prompt_hash
        h = compute_prompt_hash("anything")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)
