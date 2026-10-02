"""Check batch recovery using SDK-shaped fixtures as input and assertions as output."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse
from openai.types import Batch

from lmsyz_genai_ie_rfs.anthropic_batch import AnthropicBatchExtractor
from lmsyz_genai_ie_rfs.batch import OpenAIBatchExtractor


def _extractor(tmp_path: Path) -> OpenAIBatchExtractor:
    """Build an extractor with an offline client.

    Args:
        tmp_path: Directory for test batch files.

    Returns:
        Extractor backed by a mocked SDK client.
    """
    with patch("openai.OpenAI"):
        return OpenAIBatchExtractor(batch_root_dir=str(tmp_path))


def _status(status: str, **kwargs: Any) -> Batch:
    """Build a validated OpenAI batch status.

    Args:
        status: Provider status string.
        **kwargs: Optional batch fields.

    Returns:
        SDK batch object with real optional-field defaults.
    """
    return Batch.model_validate({
        "id": "batch-test", "completion_window": "24h", "created_at": 1,
        "endpoint": "/v1/chat/completions", "input_file_id": "input-test",
        "object": "batch", "status": status, **kwargs,
    })


def _manifest(tmp_path: Path) -> Path:
    """Write a submission manifest.

    Args:
        tmp_path: Directory for batch files.

    Returns:
        Output directory containing the manifest.
    """
    output = tmp_path / "job" / "batch_output"
    output.mkdir(parents=True)
    (output / "submission_test.json").write_text('{"id": "batch-test"}')
    return output


@pytest.mark.parametrize("existing_error", [False, True])
def test_mixed_batch_downloads_output_despite_errors(tmp_path: Path, existing_error: bool) -> None:
    """Retrieve successes both on first polling and after an earlier error-only download."""
    extractor = _extractor(tmp_path)
    output = _manifest(tmp_path)
    if existing_error:
        (output / "batch_error_batch-test.txt").write_bytes(b"errors")
    extractor.client.batches.retrieve = Mock(return_value=_status(
        "completed", output_file_id="output", error_file_id="errors", completed_at=2,
    ))
    extractor.client.files.content = Mock(side_effect=[
        SimpleNamespace(content=b"results")
    ] if existing_error else [SimpleNamespace(content=b"errors"), SimpleNamespace(content=b"results")])
    extractor.check_batch_status("job", continuous=True)
    assert (output / "batch_result_batch-test.jsonl").read_bytes() == b"results"
    assert (output / "batch_error_batch-test.txt").read_bytes() == b"errors"
    assert extractor.client.files.content.call_count == (1 if existing_error else 2)
    extractor.check_batch_status("job")
    assert extractor.client.files.content.call_count == (1 if existing_error else 2)


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled", "completed"])
@pytest.mark.parametrize("partial_output", [False, True])
def test_terminal_status_stops_polling(
    tmp_path: Path, status: str, partial_output: bool,
) -> None:
    """Terminal states stop even without files and retain any available partial output."""
    extractor = _extractor(tmp_path)
    output = _manifest(tmp_path)
    extractor.client.batches.retrieve = Mock(return_value=_status(
        status, output_file_id="output" if partial_output else None,
    ))
    extractor.client.files.content = Mock(return_value=SimpleNamespace(content=b"partial"))
    with patch("lmsyz_genai_ie_rfs.batch.time.sleep", side_effect=AssertionError("Unexpected poll")):
        extractor.check_batch_status("job", continuous=True)
    assert extractor.client.batches.retrieve.call_count == 1
    assert (output / "batch_result_batch-test.jsonl").exists() == partial_output


def test_resume_uses_named_id_when_label_matches_another_input(tmp_path: Path) -> None:
    """Reordered result fields must not silently omit a different observation."""
    extractor = _extractor(tmp_path)
    output = _manifest(tmp_path)
    record = {"response": {"body": {"choices": [{"message": {
        "content": json.dumps({"all_results": [{"label": 1, "input_id": "2"}]}),
    }}]}}}
    (output / "batch_result_test.jsonl").write_text(json.dumps(record) + "\n")
    extractor.create_batch_jsonl(
        pd.DataFrame({"id": ["1", "2"], "text": ["unfinished", "finished"]}),
        "id", "text", "Extract", "job", "gpt-4.1-mini", chunk_size=1,
    )
    requests = [json.loads(line) for path in (tmp_path / "job" / "batch_input").glob("*.jsonl")
                for line in path.read_text().splitlines()]
    assert len(requests) == 1
    assert json.loads(requests[0]["body"]["messages"][1]["content"])[0]["input_id"] == "1"


def test_resume_reports_missing_input_id(tmp_path: Path) -> None:
    """Results without identifiers cannot be used to skip source rows."""
    extractor = _extractor(tmp_path)
    _manifest(tmp_path)
    with patch.object(extractor, "retrieve_results_as_dataframe", return_value=pd.DataFrame({"label": [1]})):
        with pytest.raises(ValueError, match="input_id"):
            extractor.create_batch_jsonl(
                pd.DataFrame({"id": ["1"], "text": ["text"]}),
                "id", "text", "Extract", "job", "gpt-4.1-mini",
            )


def _openai_record(payload: Any) -> dict[str, Any]:
    """Wrap extraction output as an OpenAI batch record.

    Args:
        payload: JSON output from the model.

    Returns:
        JSON-serializable batch response record.
    """
    return {"custom_id": "request-test", "response": {"body": {"choices": [
        {"message": {"content": json.dumps(payload)}}
    ]}}}


@pytest.mark.parametrize("bad_record", [
    {"response": None},
    {"response": {"body": {"choices": []}}},
    {"response": {"body": {"choices": [{"message": {"content": None, "refusal": "Declined"}}]}}},
    {"response": {"body": {"choices": [{"message": {"content": "invalid JSON"}}]}}},
    _openai_record([{"input_id": "wrong-shape"}]),
    _openai_record({"all_results": "invalid"}),
    _openai_record({"all_results": None}),
    _openai_record({"unexpected": []}),
    [],
])
def test_unusable_record_preserves_later_valid_results(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad_record: Any,
) -> None:
    """Refusals and malformed records must not prevent parsing later observations."""
    extractor = _extractor(tmp_path)
    output = _manifest(tmp_path)
    valid = _openai_record({"all_results": [{"input_id": "valid", "label": 1}]})
    (output / "batch_result_test.jsonl").write_text(
        json.dumps(bad_record) + "\n" + json.dumps(valid) + "\n"
    )
    results = extractor.retrieve_results_as_dataframe("job")
    assert results is not None
    assert results["input_id"].tolist() == ["valid"]
    assert "line 1" in caplog.text


def test_invalid_row_preserves_valid_siblings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A malformed row within a response must not discard valid sibling rows."""
    extractor = _extractor(tmp_path)
    output = _manifest(tmp_path)
    record = _openai_record({"all_results": [{"input_id": "a"}, "invalid", {"input_id": "b"}]})
    (output / "batch_result_test.jsonl").write_text(json.dumps(record))
    results = extractor.retrieve_results_as_dataframe("job")
    assert results is not None
    assert results["input_id"].tolist() == ["a", "b"]
    assert "request-test" in caplog.text


def _anthropic_extractor(tmp_path: Path) -> AnthropicBatchExtractor:
    """Create an offline Anthropic extractor and submission manifest.

    Args:
        tmp_path: Directory for batch files.

    Returns:
        Extractor with a mocked SDK client.
    """
    with patch("anthropic.Anthropic"):
        extractor = AnthropicBatchExtractor(batch_root_dir=str(tmp_path))
    directory = tmp_path / "job" / "batch_input"
    directory.mkdir(parents=True)
    (directory / "submission.json").write_text('{"id": "msgbatch-test"}')
    return extractor


def _anthropic_entry() -> MessageBatchIndividualResponse:
    """Build one validated SDK result.

    Returns:
        Successful result with a structured extraction tool response.
    """
    return MessageBatchIndividualResponse.model_validate({
        "custom_id": "request-1", "result": {"type": "succeeded", "message": {
            "id": "msg-test", "type": "message", "role": "assistant", "model": "claude-test",
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
            "content": [{"type": "tool_use", "id": "tool-test", "name": "extract_results",
                         "input": {"all_results": [{"input_id": "1", "label": "ok"}]}}],
        }},
    })


@pytest.mark.parametrize("fail_after_first", [False, True])
def test_anthropic_failed_stream_preserves_previous_file(tmp_path: Path, fail_after_first: bool) -> None:
    """A failed request or partial stream must preserve a previous complete download."""
    extractor = _anthropic_extractor(tmp_path)
    output = tmp_path / "job" / "batch_output"
    output.mkdir()
    results_path = output / "results.jsonl"
    results_path.write_bytes(b"previous complete results\n")

    def interrupted_stream() -> Iterator[MessageBatchIndividualResponse]:
        """Yield an optional result and simulate a stream failure.

        Yields:
            One validated SDK result before an interrupted download.
        """
        assert results_path.read_bytes() == b"previous complete results\n"
        if fail_after_first:
            yield _anthropic_entry()
        raise OSError("stream interrupted")

    extractor.client.messages.batches.results = Mock(return_value=interrupted_stream())
    with pytest.raises(OSError, match="stream interrupted"):
        extractor.retrieve_results_as_dataframe("job")
    assert results_path.read_bytes() == b"previous complete results\n"
    assert list(output.iterdir()) == [results_path]


def test_anthropic_success_replaces_file_after_stream_completion(tmp_path: Path) -> None:
    """A completed download atomically replaces prior raw output and returns parsed rows."""
    extractor = _anthropic_extractor(tmp_path)
    output = tmp_path / "job" / "batch_output"
    output.mkdir()
    results_path = output / "results.jsonl"
    results_path.write_text("old results")
    entry = _anthropic_entry()

    def completed_stream() -> Iterator[MessageBatchIndividualResponse]:
        """Inspect the old file until streaming finishes.

        Yields:
            One successful batch record.
        """
        assert results_path.read_text() == "old results"
        yield entry
        assert results_path.read_text() == "old results"

    extractor.client.messages.batches.results = Mock(return_value=completed_stream())
    results = extractor.retrieve_results_as_dataframe("job")
    assert results is not None
    assert results["input_id"].tolist() == ["1"]
    assert json.loads(results_path.read_text()) == json.loads(entry.model_dump_json())
    assert list(output.iterdir()) == [results_path]


def test_anthropic_canceling_waits_for_ended(tmp_path: Path) -> None:
    """Continuous polling must continue through cancellation until results are final."""
    extractor = _anthropic_extractor(tmp_path)
    statuses = [MessageBatch.model_validate({
        "id": "msgbatch-test", "type": "message_batch", "processing_status": status,
        "created_at": "2026-01-01T00:00:00Z", "expires_at": "2026-01-02T00:00:00Z",
        "request_counts": {"processing": 0, "succeeded": 1, "errored": 0, "canceled": 1, "expired": 0},
    }) for status in ["canceling", "ended"]]
    extractor.client.messages.batches.retrieve = Mock(side_effect=statuses)
    with patch("lmsyz_genai_ie_rfs.anthropic_batch.time.sleep") as sleep:
        assert extractor.check_batch_status("job", continuous=True, interval=1) == "ended"
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_invalid_input_keeps_existing_batch_files(tmp_path: Path, provider: str) -> None:
    """Invalid source IDs must be rejected before replacing a prepared batch input."""
    extractor = _extractor(tmp_path) if provider == "openai" else _anthropic_extractor(tmp_path)
    input_dir = tmp_path / "job" / "batch_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    original = input_dir / ("batch_0.jsonl" if provider == "openai" else "requests.json")
    original.write_text("prepared input")
    create = extractor.create_batch_jsonl if provider == "openai" else extractor.create_batch_requests
    with pytest.raises(ValueError, match="unique"):
        create(pd.DataFrame({"id": ["1", 1], "text": ["a", "b"]}),
               "id", "text", "Extract", "job", "model")
    assert original.read_text() == "prepared input"


@pytest.mark.parametrize("payload", ["invalid", 7, None, ["invalid", {"input_id": "sibling"}]])
def test_anthropic_text_rows_keep_valid_results(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, payload: Any,
) -> None:
    """Malformed text-result rows are logged without poisoning valid sibling output."""
    extractor = _anthropic_extractor(tmp_path)
    entry_data = _anthropic_entry().model_dump()
    entry_data["result"]["message"]["content"] = [
        {"type": "text", "text": json.dumps({"all_results": payload})},
        {"type": "text", "text": json.dumps({"all_results": [{"input_id": "later"}]})},
    ]
    extractor.client.messages.batches.results = Mock(return_value=[
        MessageBatchIndividualResponse.model_validate(entry_data)
    ])
    results = extractor.retrieve_results_as_dataframe("job")
    assert results is not None
    expected = ["sibling", "later"] if isinstance(payload, list) else ["later"]
    assert results["input_id"].dropna().tolist() == expected
    assert "request-1" in caplog.text


@pytest.mark.parametrize("text", ["plain assistant text", "```json\n{broken json}\n```"])
def test_anthropic_unparsed_text_preserves_fallback(tmp_path: Path, text: str) -> None:
    """Unparsed free-form text remains available in the public text-column fallback."""
    extractor = _anthropic_extractor(tmp_path)
    entry_data = _anthropic_entry().model_dump()
    entry_data["result"]["message"]["content"] = [{"type": "text", "text": text}]
    extractor.client.messages.batches.results = Mock(return_value=[
        MessageBatchIndividualResponse.model_validate(entry_data)
    ])
    results = extractor.retrieve_results_as_dataframe("job")
    assert results is not None
    assert results.to_dict("records") == [{"custom_id": "request-1", "text": text}]
