"""Tests for batch.py: OpenAIBatchExtractor.create_batch_jsonl and retrieve_results_as_dataframe.

No live API calls. Uses real file I/O against tmp_path fixtures. Verifies:
- JSONL file count from large DataFrames.
- Correct JSON structure in each JSONL line.
- role="system" fix (P0 bug).
- custom_id format.
- Temperature override for o1/o3/gpt-5 models.
- Schema dict vs default response_format.
- retrieve_results_as_dataframe parsing from fixture files.
- exclude_processed skips already-done rows.

Input: small synthetic DataFrames and tmp_path directories.
Output: assertion results only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lmsyz_genai_ie_rfs.batch import OpenAIBatchExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int, id_col: str = "id", text_col: str = "text") -> pd.DataFrame:
    """Build a test DataFrame with n rows.

    Args:
        n: Number of rows.
        id_col: Column name for row IDs.
        text_col: Column name for text.

    Returns:
        DataFrame with id_col and text_col columns.
    """
    return pd.DataFrame(
        {
            id_col: [f"doc-{i}" for i in range(n)],
            text_col: [f"content of document {i}" for i in range(n)],
        }
    )


def _make_classifier(tmp_path: Path) -> OpenAIBatchExtractor:
    """Return a OpenAIBatchExtractor backed by a temp directory.

    The internal openai.OpenAI client is replaced with a MagicMock so that
    no API keys are needed.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        OpenAIBatchExtractor with mocked API client.
    """
    with patch("openai.OpenAI"):
        clf = OpenAIBatchExtractor(
            batch_root_dir=str(tmp_path / "batches"),
            max_requests_per_batch=4,
        )
    return clf


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all non-empty lines from a JSONL file as parsed dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed dicts, one per line.
    """
    lines = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


_PROMPT = "Extract culture signals."
_JOB_ID = "test-job-001"


# ---------------------------------------------------------------------------
# create_batch_jsonl: file count and structure
# ---------------------------------------------------------------------------


class TestCreateBatchJsonlFileCount:
    """Tests for the number of JSONL files created by create_batch_jsonl."""

    def test_15_rows_chunk1_max4_produces_4_files(self, tmp_path: Path) -> None:
        """15 rows, chunk_size=1, max_requests_per_batch=4 should produce 4 files.

        4 files: [0-3], [4-7], [8-11], [12-14] (last file has 3 requests).
        """
        clf = _make_classifier(tmp_path)
        df = _make_df(15)
        clf.create_batch_jsonl(
            dataframe=df,
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )
        input_dir = tmp_path / "batches" / _JOB_ID / "batch_input"
        jsonl_files = sorted(input_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 4

    def test_4_rows_chunk1_max4_produces_1_file(self, tmp_path: Path) -> None:
        """Exactly max_requests_per_batch rows should produce exactly 1 file."""
        clf = _make_classifier(tmp_path)
        df = _make_df(4)
        clf.create_batch_jsonl(
            dataframe=df,
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )
        input_dir = tmp_path / "batches" / _JOB_ID / "batch_input"
        jsonl_files = sorted(input_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

    def test_total_request_count_matches_chunks(self, tmp_path: Path) -> None:
        """Total requests across all JSONL files must equal number of chunks."""
        clf = _make_classifier(tmp_path)
        n_rows = 12
        chunk_size = 3
        # 12 rows / 3 per chunk = 4 chunks; max 4 per file => 1 file
        df = _make_df(n_rows)
        clf.create_batch_jsonl(
            dataframe=df,
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=chunk_size,
            exclude_processed=False,
        )
        input_dir = tmp_path / "batches" / _JOB_ID / "batch_input"
        all_requests: list[dict] = []
        for path in input_dir.glob("*.jsonl"):
            all_requests.extend(_read_jsonl(path))
        expected_chunks = n_rows // chunk_size
        assert len(all_requests) == expected_chunks


# ---------------------------------------------------------------------------
# create_batch_jsonl: JSON structure and P0 bug check
# ---------------------------------------------------------------------------


class TestCreateBatchJsonlStructure:
    """Tests for the internal structure of JSONL request objects."""

    def _get_all_requests(self, tmp_path: Path, job_id: str) -> list[dict[str, Any]]:
        """Return all request dicts from all JSONL files for a job.

        Args:
            tmp_path: The pytest temp directory.
            job_id: Job identifier used in create_batch_jsonl.

        Returns:
            List of all request dicts.
        """
        input_dir = tmp_path / "batches" / job_id / "batch_input"
        requests: list[dict] = []
        for path in sorted(input_dir.glob("*.jsonl")):
            requests.extend(_read_jsonl(path))
        return requests

    def test_required_top_level_keys(self, tmp_path: Path) -> None:
        """Each JSONL line must have custom_id, method, url, and body."""
        clf = _make_classifier(tmp_path)
        clf.create_batch_jsonl(
            dataframe=_make_df(2),
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )
        requests = self._get_all_requests(tmp_path, _JOB_ID)
        for req in requests:
            assert "custom_id" in req
            assert "method" in req
            assert "url" in req
            assert "body" in req

    def test_role_is_system_not_assistant(self, tmp_path: Path) -> None:
        """P0 bug fix: the system prompt message must have role='system'."""
        clf = _make_classifier(tmp_path)
        clf.create_batch_jsonl(
            dataframe=_make_df(3),
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )
        requests = self._get_all_requests(tmp_path, _JOB_ID)
        for req in requests:
            first_msg = req["body"]["messages"][0]
            assert first_msg["role"] == "system", (
                f"Expected role='system', got {first_msg['role']!r}. "
                "P0 bug fix regression."
            )
            assert first_msg["content"] == _PROMPT

    def test_custom_id_format(self, tmp_path: Path) -> None:
        """custom_id must follow the pattern {job_id}-{batch_counter}-{request_counter}."""
        clf = _make_classifier(tmp_path)
        clf.create_batch_jsonl(
            dataframe=_make_df(5),
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )
        requests = self._get_all_requests(tmp_path, _JOB_ID)
        for req in requests:
            custom_id = req["custom_id"]
            parts = custom_id.split("-")
            # job_id itself contains a hyphen, so there will be 5 parts total:
            # "test", "job", "001", batch_counter, request_counter
            assert custom_id.startswith(_JOB_ID + "-"), (
                f"{custom_id!r} does not start with job_id prefix"
            )

    def test_method_is_post(self, tmp_path: Path) -> None:
        """HTTP method must be POST."""
        clf = _make_classifier(tmp_path)
        clf.create_batch_jsonl(
            dataframe=_make_df(2),
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )
        requests = self._get_all_requests(tmp_path, _JOB_ID)
        for req in requests:
            assert req["method"] == "POST"

    def test_url_is_chat_completions(self, tmp_path: Path) -> None:
        """URL must be /v1/chat/completions."""
        clf = _make_classifier(tmp_path)
        clf.create_batch_jsonl(
            dataframe=_make_df(2),
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=_JOB_ID,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )
        requests = self._get_all_requests(tmp_path, _JOB_ID)
        for req in requests:
            assert req["url"] == "/v1/chat/completions"


# ---------------------------------------------------------------------------
# Temperature override tests
# ---------------------------------------------------------------------------


class TestTemperatureOverride:
    """Tests for model-specific temperature handling in JSONL output."""

    def _get_temperatures(self, tmp_path: Path, model_name: str) -> list[float]:
        """Run create_batch_jsonl and collect temperature values from output.

        Args:
            tmp_path: pytest temp directory.
            model_name: Model name to pass to create_batch_jsonl.

        Returns:
            List of temperature values found in the JSONL body dicts.
        """
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(
                batch_root_dir=str(tmp_path / "batches"),
                max_requests_per_batch=10,
            )
        clf.create_batch_jsonl(
            dataframe=_make_df(3),
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id="temp-test",
            model_name=model_name,
            temperature=0.0,
            chunk_size=1,
            exclude_processed=False,
        )
        input_dir = tmp_path / "batches" / "temp-test" / "batch_input"
        temps = []
        for path in input_dir.glob("*.jsonl"):
            for req in _read_jsonl(path):
                temps.append(req["body"]["temperature"])
        return temps

    def test_o1_preview_forces_temp_one(self, tmp_path: Path) -> None:
        """o1-preview model must produce temperature=1.0 in every request."""
        temps = self._get_temperatures(tmp_path, "o1-preview")
        assert all(t == 1.0 for t in temps), f"Got temperatures: {temps}"

    def test_o3_mini_forces_temp_one(self, tmp_path: Path) -> None:
        """o3-mini model must produce temperature=1.0 in every request."""
        temps = self._get_temperatures(tmp_path, "o3-mini")
        assert all(t == 1.0 for t in temps), f"Got temperatures: {temps}"

    def test_gpt5_forces_temp_one(self, tmp_path: Path) -> None:
        """gpt-5 model must produce temperature=1.0 in every request."""
        temps = self._get_temperatures(tmp_path, "gpt-5")
        assert all(t == 1.0 for t in temps), f"Got temperatures: {temps}"

    def test_gpt41mini_uses_provided_temp(self, tmp_path: Path) -> None:
        """gpt-4.1-mini must use the provided temperature (0.0)."""
        temps = self._get_temperatures(tmp_path, "gpt-4.1-mini")
        assert all(t == 0.0 for t in temps), f"Got temperatures: {temps}"

    def test_gpt4o_uses_provided_temp(self, tmp_path: Path) -> None:
        """gpt-4o must use the provided temperature (0.0)."""
        temps = self._get_temperatures(tmp_path, "gpt-4o")
        assert all(t == 0.0 for t in temps), f"Got temperatures: {temps}"


# ---------------------------------------------------------------------------
# schema_dict / response_format tests
# ---------------------------------------------------------------------------


class TestSchemaDict:
    """Tests for the schema_dict / response_format passthrough."""

    def _get_response_formats(
        self, tmp_path: Path, schema_dict: dict | None
    ) -> list[dict]:
        """Run create_batch_jsonl and collect response_format values.

        Args:
            tmp_path: pytest temp directory.
            schema_dict: Value to pass as schema_dict parameter.

        Returns:
            List of response_format dicts from JSONL bodies.
        """
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(
                batch_root_dir=str(tmp_path / "batches"),
                max_requests_per_batch=10,
            )
        clf.create_batch_jsonl(
            dataframe=_make_df(2),
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id="fmt-test",
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
            schema_dict=schema_dict,
        )
        input_dir = tmp_path / "batches" / "fmt-test" / "batch_input"
        formats = []
        for path in input_dir.glob("*.jsonl"):
            for req in _read_jsonl(path):
                formats.append(req["body"]["response_format"])
        return formats

    def test_none_schema_dict_uses_json_object(self, tmp_path: Path) -> None:
        """schema_dict=None should produce response_format={"type": "json_object"}."""
        formats = self._get_response_formats(tmp_path, schema_dict=None)
        for fmt in formats:
            assert fmt == {"type": "json_object"}

    def test_custom_schema_dict_passed_through(self, tmp_path: Path) -> None:
        """A custom schema_dict should appear verbatim as response_format."""
        custom = {"type": "json_schema", "json_schema": {"name": "CultureBatch"}}
        formats = self._get_response_formats(tmp_path, schema_dict=custom)
        for fmt in formats:
            assert fmt == custom


# ---------------------------------------------------------------------------
# retrieve_results_as_dataframe tests
# ---------------------------------------------------------------------------


class TestRetrieveResults:
    """Tests for OpenAIBatchExtractor.retrieve_results_as_dataframe."""

    def _write_result_jsonl(self, output_dir: Path, filename: str, lines: list[str]) -> None:
        """Write a JSONL fixture file into output_dir.

        Args:
            output_dir: Target directory (must already exist).
            filename: Filename for the JSONL file.
            lines: List of raw JSON strings, one per line.
        """
        path = output_dir / filename
        path.write_text("\n".join(lines) + "\n")

    def _make_openai_result_line(
        self, custom_id: str, content_dict: dict[str, Any]
    ) -> str:
        """Build a JSON string resembling one line of an OpenAI batch result file.

        Args:
            custom_id: The request identifier.
            content_dict: The parsed JSON content to embed as the message content.

        Returns:
            JSON string for one result line.
        """
        record = {
            "id": f"batch_result_{custom_id}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(content_dict)
                            }
                        }
                    ]
                },
            },
        }
        return json.dumps(record)

    def test_returns_none_when_no_output_dir(self, tmp_path: Path) -> None:
        """retrieve_results_as_dataframe should return None if no output dir exists."""
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(batch_root_dir=str(tmp_path / "batches"))
        result = clf.retrieve_results_as_dataframe("nonexistent-job")
        assert result is None

    def test_parses_valid_jsonl_into_dataframe(self, tmp_path: Path) -> None:
        """Valid result JSONL should be parsed into a DataFrame with correct rows."""
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(batch_root_dir=str(tmp_path / "batches"))

        job_id = "parse-test"
        output_dir = tmp_path / "batches" / job_id / "batch_output"
        output_dir.mkdir(parents=True)

        rows = [
            {"input_id": "doc-0", "culture_type": "innovation_adaptability", "tone": "positive"},
            {"input_id": "doc-1", "culture_type": "performance_oriented", "tone": "neutral"},
        ]
        line = self._make_openai_result_line(
            "job-0-0", {"all_results": rows}
        )
        self._write_result_jsonl(output_dir, "batch_result_abc.jsonl", [line])

        df = clf.retrieve_results_as_dataframe(job_id)
        assert df is not None
        assert len(df) == 2
        assert set(df["input_id"]) == {"doc-0", "doc-1"}

    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON lines should be skipped with a warning, not crash."""
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(batch_root_dir=str(tmp_path / "batches"))

        job_id = "malformed-test"
        output_dir = tmp_path / "batches" / job_id / "batch_output"
        output_dir.mkdir(parents=True)

        good_line = self._make_openai_result_line(
            "job-0-0", {"all_results": [{"input_id": "doc-0", "v": 1}]}
        )
        self._write_result_jsonl(
            output_dir,
            "batch_result_xyz.jsonl",
            [good_line, "this is not valid json at all {{{{"],
        )

        df = clf.retrieve_results_as_dataframe(job_id)
        assert df is not None
        assert len(df) == 1

    def test_supports_results_key_alias(self, tmp_path: Path) -> None:
        """The parser should also accept 'results' as an alternative to 'all_results'."""
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(batch_root_dir=str(tmp_path / "batches"))

        job_id = "alias-test"
        output_dir = tmp_path / "batches" / job_id / "batch_output"
        output_dir.mkdir(parents=True)

        line = self._make_openai_result_line(
            "job-0-0", {"results": [{"input_id": "doc-0", "label": "x"}]}
        )
        self._write_result_jsonl(output_dir, "batch_result_alias.jsonl", [line])

        df = clf.retrieve_results_as_dataframe(job_id)
        assert df is not None
        assert len(df) == 1

    def test_returns_none_when_no_valid_rows(self, tmp_path: Path) -> None:
        """If all lines fail to parse, retrieve_results_as_dataframe returns None."""
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(batch_root_dir=str(tmp_path / "batches"))

        job_id = "empty-test"
        output_dir = tmp_path / "batches" / job_id / "batch_output"
        output_dir.mkdir(parents=True)

        self._write_result_jsonl(
            output_dir, "batch_result_bad.jsonl", ["{not json}", "also bad"]
        )

        df = clf.retrieve_results_as_dataframe(job_id)
        assert df is None


# ---------------------------------------------------------------------------
# exclude_processed tests
# ---------------------------------------------------------------------------


class TestExcludeProcessed:
    """Tests for the exclude_processed logic in create_batch_jsonl."""

    def _write_mock_result(
        self, output_dir: Path, done_ids: list[str], job_id: str
    ) -> None:
        """Write a mock batch result JSONL file containing the given IDs.

        Args:
            output_dir: The batch_output directory for the job.
            done_ids: Row IDs to mark as already-done.
            job_id: Job identifier used in custom_id fields.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"input_id": rid, "label": "done"} for rid in done_ids]
        record = {
            "id": "batch_xyz",
            "custom_id": f"{job_id}-0-0",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {"message": {"content": json.dumps({"all_results": rows})}}
                    ]
                },
            },
        }
        path = output_dir / "batch_result_done.jsonl"
        path.write_text(json.dumps(record) + "\n")

    def test_exclude_processed_skips_done_rows(self, tmp_path: Path) -> None:
        """With exclude_processed=True and 5 done IDs, only remaining rows appear."""
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(
                batch_root_dir=str(tmp_path / "batches"),
                max_requests_per_batch=100,
            )

        job_id = "excl-test"
        n_total = 10
        n_done = 5
        done_ids = [f"doc-{i}" for i in range(n_done)]
        df = _make_df(n_total)

        output_dir = tmp_path / "batches" / job_id / "batch_output"
        self._write_mock_result(output_dir, done_ids, job_id)

        clf.create_batch_jsonl(
            dataframe=df,
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=job_id,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=True,
        )

        input_dir = tmp_path / "batches" / job_id / "batch_input"
        all_requests: list[dict] = []
        for path in sorted(input_dir.glob("*.jsonl")):
            all_requests.extend(_read_jsonl(path))

        # Should only have chunks for the 5 remaining rows.
        assert len(all_requests) == n_total - n_done

    def test_exclude_processed_false_includes_all(self, tmp_path: Path) -> None:
        """With exclude_processed=False, all rows appear even if prior results exist."""
        with patch("openai.OpenAI"):
            clf = OpenAIBatchExtractor(
                batch_root_dir=str(tmp_path / "batches"),
                max_requests_per_batch=100,
            )

        job_id = "no-excl-test"
        n_total = 6
        done_ids = [f"doc-{i}" for i in range(3)]
        df = _make_df(n_total)

        output_dir = tmp_path / "batches" / job_id / "batch_output"
        self._write_mock_result(output_dir, done_ids, job_id)

        clf.create_batch_jsonl(
            dataframe=df,
            id_col="id",
            text_col="text",
            prompt=_PROMPT,
            job_id=job_id,
            model_name="gpt-4.1-mini",
            chunk_size=1,
            exclude_processed=False,
        )

        input_dir = tmp_path / "batches" / job_id / "batch_input"
        all_requests: list[dict] = []
        for path in sorted(input_dir.glob("*.jsonl")):
            all_requests.extend(_read_jsonl(path))

        assert len(all_requests) == n_total
