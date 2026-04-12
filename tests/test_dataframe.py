"""Tests for dataframe.py: DataFrameIterator chunking logic.

No API calls required. Validates chunk sizes, edge cases (empty DataFrame,
non-divisible lengths), and column renaming behaviour.
"""

from __future__ import annotations

import pandas as pd
import pytest

from genai_batch_ie_rfs.dataframe import DataFrameIterator


def _make_df(n: int) -> pd.DataFrame:
    """Build a simple test DataFrame with n rows.

    Args:
        n: Number of rows.

    Returns:
        DataFrame with columns 'id' (int) and 'text' (str).
    """
    return pd.DataFrame({"id": range(n), "text": [f"text_{i}" for i in range(n)]})


class TestDataFrameIterator:
    """Chunking logic and edge-case tests for DataFrameIterator."""

    def test_len_exact_multiple(self) -> None:
        """len() should equal n/chunk_size when evenly divisible."""
        it = DataFrameIterator(_make_df(10), id_col="id", text_col="text", chunk_size=5)
        assert len(it) == 2

    def test_len_non_divisible(self) -> None:
        """len() should round up when the row count is not divisible by chunk_size."""
        it = DataFrameIterator(_make_df(11), id_col="id", text_col="text", chunk_size=5)
        assert len(it) == 3

    def test_len_single_row(self) -> None:
        """A single-row DataFrame should yield exactly one chunk."""
        it = DataFrameIterator(_make_df(1), id_col="id", text_col="text", chunk_size=5)
        assert len(it) == 1

    def test_len_empty_df(self) -> None:
        """An empty DataFrame should yield zero chunks."""
        it = DataFrameIterator(_make_df(0), id_col="id", text_col="text", chunk_size=5)
        assert len(it) == 0

    def test_chunk_content_and_keys(self) -> None:
        """Each chunk item should have the formatted id and text keys."""
        it = DataFrameIterator(_make_df(3), id_col="id", text_col="text", chunk_size=5)
        chunks = list(it)
        assert len(chunks) == 1
        first_chunk = chunks[0]
        assert len(first_chunk) == 3
        for item in first_chunk:
            assert "input_id" in item
            assert "input_text" in item

    def test_custom_formatted_cols(self) -> None:
        """Custom formatted_id_col and formatted_text_col names should appear in output."""
        it = DataFrameIterator(
            _make_df(2),
            id_col="id",
            text_col="text",
            chunk_size=5,
            formatted_id_col="my_id",
            formatted_text_col="my_text",
        )
        chunks = list(it)
        assert "my_id" in chunks[0][0]
        assert "my_text" in chunks[0][0]

    def test_chunking_splits_correctly(self) -> None:
        """10 rows with chunk_size=3 should produce 3 full chunks and 1 partial."""
        it = DataFrameIterator(_make_df(10), id_col="id", text_col="text", chunk_size=3)
        chunks = list(it)
        assert len(chunks) == 4
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        assert len(chunks[3]) == 1

    def test_all_rows_covered(self) -> None:
        """All row IDs should appear exactly once across all chunks."""
        df = _make_df(7)
        it = DataFrameIterator(df, id_col="id", text_col="text", chunk_size=3)
        seen_ids = [item["input_id"] for chunk in it for item in chunk]
        assert sorted(seen_ids) == [str(i) for i in range(7)]

    def test_ids_are_strings(self) -> None:
        """input_id values should be strings even when the source column is int."""
        it = DataFrameIterator(_make_df(2), id_col="id", text_col="text", chunk_size=5)
        chunks = list(it)
        for item in chunks[0]:
            assert isinstance(item["input_id"], str)

    def test_iteration_restarts(self) -> None:
        """Calling iter() again should restart from the beginning."""
        df = _make_df(4)
        it = DataFrameIterator(df, id_col="id", text_col="text", chunk_size=2)
        first_pass = list(it)
        second_pass = list(it)  # __iter__ resets _start
        assert [item["input_id"] for c in first_pass for item in c] == \
               [item["input_id"] for c in second_pass for item in c]

    def test_stop_iteration_on_empty(self) -> None:
        """Calling next() on a fully-consumed iterator should raise StopIteration."""
        it = DataFrameIterator(_make_df(2), id_col="id", text_col="text", chunk_size=5)
        list(it)  # consume
        with pytest.raises(StopIteration):
            next(it)
