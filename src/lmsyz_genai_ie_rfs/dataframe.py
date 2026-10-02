"""Small helpers shared by the concurrent and batch paths.

Contents:
    DataFrameIterator    - chunks a DataFrame into formatted dicts.
    SqliteCache          - get / put / all_ids for resume-on-interrupt.
    compute_prompt_hash  - short stable hash of a prompt string for cache gating.

The main concurrent entry point ``extract_df`` lives in ``client.py``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd


def compute_prompt_hash(prompt: str) -> str:
    """Return a short stable hash of ``prompt`` for cache invalidation.

    Uses SHA-256 truncated to 16 hex chars. Stable across Python versions
    and machines, unlike the built-in ``hash()``.

    Args:
        prompt: Prompt text.

    Returns:
        16-character lowercase hex digest.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def validate_positive_integer(value: int, name: str) -> None:
    """Reject invalid sizes before iteration or worker creation.

    Args:
        value: Requested count.
        name: Parameter name to include in the error.

    Raises:
        ValueError: If the count is a boolean, non-integer, or non-positive.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_input_dataframe(df: pd.DataFrame, id_col: str, text_col: str) -> None:
    """Validate columns and IDs used to align extraction results and cache entries.

    Args:
        df: Input observations.
        id_col: Identifier column.
        text_col: Text column.

    Raises:
        ValueError: If required columns are absent or repeated, or IDs are null,
            empty, or nonunique after string conversion.
    """
    for column in {id_col, text_col}:
        if list(df.columns).count(column) != 1:
            raise ValueError(f"Input must contain exactly one {column!r} column.")
    ids = df[id_col]
    if ids.isna().any():
        raise ValueError(f"{id_col!r} must contain non-null row IDs.")
    if ids.empty:
        return
    normalized = ids.map(str)
    if normalized.str.strip().eq("").any():
        raise ValueError(f"{id_col!r} must contain non-empty row IDs.")
    duplicates = normalized[normalized.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(
            f"{id_col!r} must contain unique row IDs after string conversion; "
            f"duplicate IDs: {duplicates!r}. Create a unique ID for each observation."
        )


class DataFrameIterator:
    """Chunk a DataFrame into formatted dicts for LLM input.

    Attributes:
        dataframe: Source DataFrame.
        chunk_size: Rows per chunk.
        id_col: Source column name for identifiers.
        text_col: Source column name for text.
        formatted_id_col: Key used in each output dict for the identifier.
        formatted_text_col: Key used in each output dict for the text.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_col: str,
        text_col: str,
        chunk_size: int = 5,
        formatted_id_col: str = "input_id",
        formatted_text_col: str = "input_text",
    ) -> None:
        """Store the chunking configuration.

        Args:
            dataframe: The DataFrame to iterate.
            id_col: Column name with row identifiers.
            text_col: Column name with text content.
            chunk_size: Rows per chunk. Default 5.
            formatted_id_col: Output-dict key for the identifier. Default "input_id".
            formatted_text_col: Output-dict key for the text. Default "input_text".
        """
        validate_positive_integer(chunk_size, "chunk_size")
        validate_input_dataframe(dataframe, id_col, text_col)
        self.dataframe = dataframe
        self.chunk_size = chunk_size
        self.id_col = id_col
        self.text_col = text_col
        self.formatted_id_col = formatted_id_col
        self.formatted_text_col = formatted_text_col
        self._start = 0

    def __iter__(self) -> DataFrameIterator:
        """Reset and return self."""
        self._start = 0
        return self

    def __next__(self) -> list[dict[str, str]]:
        """Return the next chunk of formatted dicts.

        Raises:
            StopIteration: When all rows have been yielded.
        """
        if self._start >= len(self.dataframe):
            raise StopIteration
        end = self._start + self.chunk_size
        chunk = self.dataframe.iloc[self._start:end]
        self._start = end
        return [
            {
                self.formatted_id_col: str(row_id),
                self.formatted_text_col: str(text),
            }
            for row_id, text in zip(
                chunk[self.id_col].map(str), chunk[self.text_col], strict=True
            )
        ]

    def __len__(self) -> int:
        """Number of chunks (ceiling division)."""
        n = len(self.dataframe)
        return (n + self.chunk_size - 1) // self.chunk_size


class SqliteCache:
    """SQLite-backed get/put cache for resuming interrupted runs.

    Each row is stored with the hash of the prompt that produced it. On
    read, callers can pass ``prompt_hash=`` to restrict the result set to
    rows produced by the same prompt: if the prompt changed, cached rows
    are invisible, and ``extract_df`` will re-run them.

    Backward compatible: caches created before the ``prompt_hash`` column
    existed are migrated with ``ALTER TABLE`` on first open. Rows whose
    hash is NULL are treated as not-matching any specific hash.
    """

    _CREATE_SQL = (
        "CREATE TABLE IF NOT EXISTS results ("
        "row_id TEXT PRIMARY KEY, "
        "json_result TEXT NOT NULL, "
        "prompt_hash TEXT"
        ")"
    )

    def __init__(self, db_path: Path) -> None:
        """Create the DB file and table if absent; migrate legacy schema.

        Args:
            db_path: Path to the SQLite file.
        """
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as con:
            con.execute(self._CREATE_SQL)
            # Migrate pre-hash schema by adding the column if missing.
            try:
                con.execute("ALTER TABLE results ADD COLUMN prompt_hash TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists

    def get(self, row_id: str, prompt_hash: str | None = None) -> dict[str, Any] | None:
        """Return the cached row dict, or None.

        Args:
            row_id: Row identifier.
            prompt_hash: If given, only return the row if it was stored
                under this exact prompt hash. If None, return whatever is
                there regardless of hash.

        Returns:
            The cached dict, or None.
        """
        with sqlite3.connect(self.db_path) as con:
            if prompt_hash is None:
                row = con.execute(
                    "SELECT json_result FROM results WHERE row_id = ?", (row_id,)
                ).fetchone()
            else:
                row = con.execute(
                    "SELECT json_result FROM results WHERE row_id = ? AND prompt_hash = ?",
                    (row_id, prompt_hash),
                ).fetchone()
        if row is None:
            return None
        result: dict[str, Any] = json.loads(row[0])
        return result

    def put(
        self,
        row_id: str,
        result: dict[str, Any],
        prompt_hash: str | None = None,
    ) -> None:
        """Upsert a result for ``row_id`` under ``prompt_hash``.

        Args:
            row_id: Row identifier.
            result: The row dict to cache.
            prompt_hash: Hash of the prompt that produced ``result``.
        """
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO results (row_id, json_result, prompt_hash) "
                "VALUES (?, ?, ?)",
                (row_id, json.dumps(result), prompt_hash),
            )

    def all_ids(self, prompt_hash: str | None = None) -> set[str]:
        """Return cached row IDs, optionally filtered by prompt hash.

        Args:
            prompt_hash: If given, return only IDs whose cached row was
                stored under this hash. If None, return all IDs regardless.

        Returns:
            Set of row IDs (possibly empty).
        """
        with sqlite3.connect(self.db_path) as con:
            if prompt_hash is None:
                rows = con.execute("SELECT row_id FROM results").fetchall()
            else:
                rows = con.execute(
                    "SELECT row_id FROM results WHERE prompt_hash = ?", (prompt_hash,)
                ).fetchall()
        return {r[0] for r in rows}
