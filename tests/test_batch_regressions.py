"""Check batch recovery using SDK-shaped fixtures as input and assertions as output."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from openai.types import Batch

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
