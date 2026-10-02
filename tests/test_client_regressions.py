"""Exercise extraction integrity with synthetic inputs and provider responses.

Input: temporary cache files and in-memory SDK stand-ins. Output: assertions on
validated results, cache contents, request counts, and omission diagnostics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from lmsyz_genai_ie_rfs.client import _load_schema, extract_df
from lmsyz_genai_ie_rfs.dataframe import DataFrameIterator, SqliteCache, compute_prompt_hash


def _sdk(payload: Any = None) -> Any:
    """Create a recording OpenAI stand-in, echoing rows unless payload is set.

    Args:
        payload: Fixed response content, or None to echo input IDs.

    Returns:
        SDK-shaped object with a mock request method.
    """
    def respond(**kwargs: Any) -> Any:
        """Build an SDK response for a recorded request.

        Args:
            **kwargs: Completion request arguments.

        Returns:
            Response object with JSON message content.
        """
        result = payload
        if result is None:
            rows = json.loads(kwargs["messages"][1]["content"])
            result = {"all_results": [{"input_id": row["input_id"], "label": "ok"} for row in rows]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=respond))))


def _run(tmp_path: Path, client: Any, df: pd.DataFrame | None = None, **kwargs: Any) -> pd.DataFrame:
    """Run extraction against a temporary cache.

    Args:
        tmp_path: Directory for the SQLite cache.
        client: Test provider client.
        df: Optional input frame.
        **kwargs: Extraction argument overrides.

    Returns:
        Extracted result frame.
    """
    if df is None:
        df = pd.DataFrame({"id": ["a", "b"], "text": ["first", "second"]})
    options = {"prompt": "Extract JSON", "model": "test", "cache_path": tmp_path / "cache.db", "client": client}
    options.update(kwargs)
    return extract_df(df, **options)


@pytest.mark.parametrize("ids", [["a", "a"], [1, "1"], [None, "a"], ["", "a"], [" ", "a"]])
def test_invalid_ids_rejected_before_requests(tmp_path: Path, ids: list[Any]) -> None:
    """Reject ambiguous IDs before creating caches or sending requests.

    Args:
        tmp_path: Temporary directory.
        ids: Invalid identifier values.
    """
    sdk = _sdk()
    with pytest.raises(ValueError, match="row IDs"):
        _run(tmp_path, sdk, pd.DataFrame({"id": ids, "text": ["one", "two"]}))
    sdk.chat.completions.create.assert_not_called()
    assert not (tmp_path / "cache.db").exists()


@pytest.mark.parametrize("parameter", ["chunk_size", "max_workers"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_sizes_rejected_with_completed_cache(tmp_path: Path, parameter: str, value: Any) -> None:
    """Validate sizes even when every row is already cached.

    Args:
        tmp_path: Temporary directory.
        parameter: Extraction size parameter.
        value: Invalid size.
    """
    sdk = _sdk()
    _run(tmp_path, sdk)
    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        _run(tmp_path, sdk, **{parameter: value})
    assert sdk.chat.completions.create.call_count == 1


@pytest.mark.parametrize("value", [0, -2, True, 1.5])
def test_iterator_rejects_invalid_size(value: Any) -> None:
    """Reject sizes that previously caused invalid iteration.

    Args:
        value: Invalid chunk size.
    """
    with pytest.raises(ValueError, match="positive integer"):
        DataFrameIterator(pd.DataFrame({"id": [1], "text": ["x"]}), "id", "text", value)


def test_iterator_preserves_integer_ids_with_numeric_text() -> None:
    """Preserve identifier representation when numeric columns have mixed types."""
    chunks = list(DataFrameIterator(pd.DataFrame({"id": [1], "text": [1.5]}), "id", "text"))
    assert chunks == [[{"input_id": "1", "input_text": "1.5"}]]


@pytest.mark.parametrize("column", ["id", "text"])
def test_missing_columns_rejected_with_completed_cache(tmp_path: Path, column: str) -> None:
    """Require both columns on cache-only calls too.

    Args:
        tmp_path: Temporary directory.
        column: Required column to omit.
    """
    sdk = _sdk()
    _run(tmp_path, sdk)
    df = pd.DataFrame({"id": ["a", "b"], "text": ["first", "second"]}).drop(columns=column)
    with pytest.raises(ValueError, match="exactly one"):
        _run(tmp_path, sdk, df)


def test_backend_validated_with_injected_client(tmp_path: Path) -> None:
    """Reject a backend typo before using an injected client.

    Args:
        tmp_path: Temporary directory.
    """
    sdk = _sdk()
    with pytest.raises(ValueError, match="Unknown backend"):
        _run(tmp_path, sdk, backend="opneai")
    sdk.chat.completions.create.assert_not_called()


def test_only_unambiguous_results_are_returned_and_cached(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Drop malformed, foreign and all duplicate rows while preserving valid rows.

    Args:
        tmp_path: Temporary directory.
        caplog: Captured diagnostics.
    """
    rows = ["bad", {"label": "missing"}, {"input_id": None}, {"input_id": []}, {"input_id": "foreign"},
            {"input_id": "a", "label": "first"}, {"input_id": "a", "label": "second"},
            {"input_id": "b", "label": "valid"}]
    with caplog.at_level(logging.WARNING):
        result = _run(tmp_path, _sdk({"all_results": rows}))
    assert result.to_dict("records") == [{"input_id": "b", "label": "valid"}]
    assert SqliteCache(tmp_path / "cache.db").all_ids() == {"b"}
    assert "duplicate input_ids: ['a']" in caplog.text
    assert "missing results for input_ids: ['a']" in caplog.text
    assert "foreign" in caplog.text
    sdk = _sdk()
    assert len(_run(tmp_path, sdk)) == 2
    request_rows = json.loads(sdk.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert [row["input_id"] for row in request_rows] == ["a"]


@pytest.mark.parametrize("bad_payload", [[], {"all_results": "bad"}, {"all_results": 7}, {}])
def test_bad_chunk_preserves_other_completed_chunks(tmp_path: Path, bad_payload: Any) -> None:
    """Keep successful chunks when another response envelope is malformed.

    Args:
        tmp_path: Temporary directory.
        bad_payload: Invalid provider payload for one chunk.
    """
    sdk = _sdk()
    original = sdk.chat.completions.create.side_effect

    def respond(**kwargs: Any) -> Any:
        """Corrupt only the chunk containing observation a.

        Args:
            **kwargs: Completion request arguments.

        Returns:
            Synthetic SDK response.
        """
        rows = json.loads(kwargs["messages"][1]["content"])
        if rows[0]["input_id"] == "a":
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(bad_payload)))])
        return original(**kwargs)

    sdk.chat.completions.create.side_effect = respond
    result = _run(tmp_path, sdk, chunk_size=1)
    assert result["input_id"].tolist() == ["b"]
    assert SqliteCache(tmp_path / "cache.db").all_ids() == {"b"}


def test_cached_and_empty_calls_do_not_construct_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow valid completed-cache and empty-input calls without credentials.

    Args:
        tmp_path: Temporary directory.
        monkeypatch: Attribute replacement fixture.
    """
    _run(tmp_path, _sdk())
    factory = Mock(side_effect=AssertionError("unexpected SDK construction"))
    monkeypatch.setattr("lmsyz_genai_ie_rfs.client._make_client", factory)
    assert len(_run(tmp_path, None)) == 2
    assert _run(tmp_path, None, pd.DataFrame(columns=["id", "text"])).empty
    factory.assert_not_called()


def test_prompt_cache_and_fresh_failure_policy(tmp_path: Path) -> None:
    """Preserve prompt invalidation and prior cache data after failed refresh.

    Args:
        tmp_path: Temporary directory.
    """
    sdk = _sdk()
    _run(tmp_path, sdk)
    _run(tmp_path, sdk, prompt="changed")
    assert sdk.chat.completions.create.call_count == 2
    _run(tmp_path, sdk, prompt="cosmetic", ignore_prompt_hash=True)
    assert sdk.chat.completions.create.call_count == 2
    assert _run(tmp_path, _sdk({"all_results": []}), fresh=True, prompt="changed").empty
    cached = _run(tmp_path, sdk, prompt="changed")
    assert len(cached) == 2
    assert sdk.chat.completions.create.call_count == 2


def test_invalid_old_cache_entry_is_reprocessed(tmp_path: Path) -> None:
    """Reprocess a cache row whose embedded ID does not match its cache key.

    Args:
        tmp_path: Temporary directory.
    """
    cache = SqliteCache(tmp_path / "cache.db")
    cache.put("a", {"input_id": "foreign"}, compute_prompt_hash("Extract JSON"))
    result = _run(tmp_path, _sdk())
    assert set(result["input_id"]) == {"a", "b"}
    assert cache.get("a") == {"input_id": "a", "label": "ok"}


@pytest.mark.parametrize("mode", ["text", "tool"])
def test_anthropic_validation_matches_openai(tmp_path: Path, mode: str) -> None:
    """Apply result alignment checks in both Anthropic response modes.

    Args:
        tmp_path: Temporary directory.
        mode: Free-text or forced-tool response mode.
    """
    payload = {"all_results": [{"input_id": "a"}, "bad", {"input_id": "foreign"}]}
    block = (SimpleNamespace(type="text", text=json.dumps(payload)) if mode == "text" else
             SimpleNamespace(type="tool_use", name="extract_results", input=payload))
    sdk = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=SimpleNamespace(content=[block]))))
    schema = None if mode == "text" else {"type": "object", "properties": {"all_results": {"type": "array"}}}
    result = _run(tmp_path, sdk, backend="anthropic", schema=schema)
    assert result["input_id"].tolist() == ["a"]
    assert SqliteCache(tmp_path / "cache.db").all_ids() == {"a"}


@pytest.mark.parametrize("payload", [[], "not an object", 42, None])
def test_schema_file_requires_json_object(tmp_path: Path, payload: Any) -> None:
    """Reject valid JSON with an unsupported schema container.

    Args:
        tmp_path: Temporary directory.
        payload: Non-object JSON value to write.
    """
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Schema file must contain a JSON object"):
        _load_schema(schema_path)


def test_schema_file_preserves_response_object(tmp_path: Path) -> None:
    """Load the complete response schema without inferring or rewriting it.

    Args:
        tmp_path: Temporary directory.
    """
    schema = {"type": "object", "properties": {"all_results": {"type": "array"}}}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    assert _load_schema(schema_path) == schema



def test_datetime_ids_match_submissions_and_cache_resume(tmp_path: Path) -> None:
    """Use one string representation for date IDs in submissions and cache reads.

    Args:
        tmp_path: Temporary directory.
    """
    df = pd.DataFrame({
        "id": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "text": ["first", "second"],
    })
    expected = {"2026-01-01 00:00:00", "2026-01-02 00:00:00"}
    chunks = list(DataFrameIterator(df, "id", "text", chunk_size=1))
    assert {row["input_id"] for chunk in chunks for row in chunk} == expected
    sdk = _sdk()
    first = _run(tmp_path, sdk, df, chunk_size=1)
    assert set(first["input_id"]) == expected
    assert SqliteCache(tmp_path / "cache.db").all_ids() == expected
    assert sdk.chat.completions.create.call_count == 2
    resumed = _run(tmp_path, sdk, df, chunk_size=1)
    assert set(resumed["input_id"]) == expected
    assert sdk.chat.completions.create.call_count == 2



def test_mixed_datetime_ids_stay_stable_across_chunks_and_subsets(tmp_path: Path) -> None:
    """Keep midnight IDs stable when other observations contain non-midnight times.

    Args:
        tmp_path: Temporary directory.
    """
    df = pd.DataFrame({
        "id": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-02 12:00:00"]),
        "text": ["midnight", "noon"],
    })
    expected = {"2026-01-01 00:00:00", "2026-01-02 12:00:00"}
    chunks = list(DataFrameIterator(df, "id", "text", chunk_size=1))
    assert {row["input_id"] for chunk in chunks for row in chunk} == expected
    midnight = df.iloc[:1]
    assert list(DataFrameIterator(midnight, "id", "text"))[0][0]["input_id"] == "2026-01-01 00:00:00"
    cache = SqliteCache(tmp_path / "cache.db")
    cache.put("2026-01-02 12:00:00", {"input_id": "2026-01-02 12:00:00", "label": "prior"},
              compute_prompt_hash("Extract JSON"))
    sdk = _sdk()
    first = _run(tmp_path, sdk, df, chunk_size=1)
    assert set(first["input_id"]) == expected
    assert sdk.chat.completions.create.call_count == 1
    submitted = json.loads(sdk.chat.completions.create.call_args.kwargs["messages"][1]["content"])
    assert submitted[0]["input_id"] == "2026-01-01 00:00:00"
    assert cache.all_ids() == expected
    subset_result = _run(tmp_path, sdk, midnight, chunk_size=1)
    assert subset_result["input_id"].tolist() == ["2026-01-01 00:00:00"]
    assert set(_run(tmp_path, sdk, df, chunk_size=1)["input_id"]) == expected
    assert sdk.chat.completions.create.call_count == 1


def test_timestamp_and_equivalent_string_ids_are_rejected(tmp_path: Path) -> None:
    """Use the same scalar conversion when validating heterogeneous source IDs.

    Args:
        tmp_path: Temporary directory.
    """
    df = pd.DataFrame({
        "id": pd.Series([pd.Timestamp("2026-01-01"), "2026-01-01 00:00:00"], dtype=object),
        "text": ["first", "second"],
    })
    with pytest.raises(ValueError, match="unique row IDs after string conversion"):
        _run(tmp_path, _sdk(), df)
