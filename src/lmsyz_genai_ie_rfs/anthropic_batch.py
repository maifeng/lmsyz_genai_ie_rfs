"""Anthropic Message Batches API: batch job submission, polling, and result retrieval.

Anthropic's batch API is architecturally different from OpenAI's:
    - Input: a JSON body with a list of requests, sent in a single POST.
      (NOT a JSONL file upload like OpenAI.)
    - Limits: up to 100,000 requests or 256 MB per batch, whichever comes first.
    - Results: streamed back as JSONL via ``client.messages.batches.results(batch_id)``.
    - Retention: results are available for 29 days.

Use this when you have thousands of rows and can tolerate up to 24 hours of
turnaround. For small jobs or when you need results immediately, use the
concurrent ``extract_df`` function instead.

Input: a pandas DataFrame with id and text columns, plus a prompt string.
Output: local JSON and JSONL manifests on disk, then a pandas DataFrame of
parsed results.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import anthropic
import pandas as pd
from tqdm import tqdm

from lmsyz_genai_ie_rfs.dataframe import (
    DataFrameIterator,
    validate_input_dataframe,
    validate_positive_integer,
)

log = logging.getLogger(__name__)


def _object_rows(payload: Any, custom_id: str) -> list[dict[str, Any]]:
    """Retain object rows and report unusable extraction output.

    Args:
        payload: Untrusted model output under a result-container key.
        custom_id: Request identifier used in warnings.

    Returns:
        Valid object rows, preserving their original order.
    """
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        log.warning("Ignoring malformed rows for request %s.", custom_id)
        return []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if isinstance(row, dict):
            rows.append(row)
        else:
            log.warning("Ignoring non-object row %s for request %s.", index, custom_id)
    return rows


class AnthropicBatchExtractor:
    """Submit, monitor, and retrieve Anthropic Message Batches API jobs from a DataFrame.

    Lifecycle (parallels ``OpenAIBatchExtractor`` but with a different wire format):

    1. ``create_batch_requests``: builds an in-memory list of requests and writes
       it to ``batch_input/requests.json`` for inspection / reproducibility.
    2. ``submit_batch``: posts the request list via ``client.messages.batches.create``,
       saves the submission manifest.
    3. ``check_batch_status``: polls with ``client.messages.batches.retrieve``.
    4. ``retrieve_results_as_dataframe``: streams results via
       ``client.messages.batches.results`` and flattens tool_use outputs into a DataFrame.

    Directory layout created under batch_root_dir/job_id/:

        batch_input/
            requests.json           -- serialized list of request dicts
            submission.json         -- returned batch manifest (id, status, ...)
        batch_output/
            results.jsonl           -- raw streamed results from Anthropic

    Failed requests remain in the raw results file and are reported through logging.

    Attributes:
        batch_root_dir: Root directory for all batch jobs.
        client: An ``anthropic.Anthropic`` client instance.
    """

    def __init__(
        self,
        batch_root_dir: str = "anthropic_batch_jobs",
        api_key: str | None = None,
    ) -> None:
        """Initialise the extractor.

        Args:
            batch_root_dir: Root directory for batch jobs. Created if absent.
            api_key: Optional Anthropic API key override; otherwise read from
                the standard ``ANTHROPIC_API_KEY`` environment variable.
        """
        self.batch_root_dir = Path(batch_root_dir)
        self.batch_root_dir.mkdir(parents=True, exist_ok=True)
        self.client = anthropic.Anthropic(api_key=api_key)

    def create_batch_requests(
        self,
        dataframe: pd.DataFrame,
        id_col: str,
        text_col: str,
        prompt: str,
        job_id: str,
        model_name: str,
        chunk_size: int = 5,
        schema_dict: dict[str, Any] | None = None,
        tool_name: str = "extract_results",
        max_tokens: int = 32000,
    ) -> Path:
        """Build the request list and write it to ``batch_input/requests.json``.

        The system prompt is passed as a list-block with
        ``cache_control={"type": "ephemeral"}`` so the long prompt is cached
        across the many chunk requests.

        If ``schema_dict`` is provided, the request uses ``tool_use`` with that
        schema as the tool's ``input_schema``, forcing structured output. If
        ``schema_dict`` is None, the request omits the tool definition and the
        model returns free-form text.

        Args:
            dataframe: Input DataFrame. Must contain id_col and text_col.
            id_col: Column name for non-null row IDs, unique after string conversion.
            text_col: Column name for text content.
            prompt: System prompt text (cached via cache_control).
            job_id: Unique job identifier. Used as the subdirectory name.
            model_name: Anthropic model identifier (e.g., "claude-haiku-4-5-20251001").
            chunk_size: Rows per request chunk. Default 5.
            schema_dict: Optional JSON schema dict describing the tool's ``input_schema``.
                If None, no tool is used and the model returns free-form text.
            tool_name: Name of the tool when schema_dict is provided. Default "extract_results".
            max_tokens: Max tokens per response. Default 32000 (well within
                Claude 4.x model output limits of 64K).

        Returns:
            Path to the written ``requests.json`` file.
        """
        validate_positive_integer(chunk_size, "chunk_size")
        validate_input_dataframe(dataframe, id_col, text_col)
        job_dir = self.batch_root_dir / job_id
        input_dir = job_dir / "batch_input"
        output_dir = job_dir / "batch_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_iter = DataFrameIterator(
            dataframe=dataframe,
            id_col=id_col,
            text_col=text_col,
            chunk_size=chunk_size,
        )

        requests: list[dict[str, Any]] = []
        for i, chunk in enumerate(tqdm(df_iter, desc="Building Anthropic batch")):
            params: dict[str, Any] = {
                "model": model_name,
                "max_tokens": max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {"role": "user", "content": json.dumps(chunk)},
                ],
            }
            if schema_dict is not None:
                params["tools"] = [
                    {
                        "name": tool_name,
                        "description": "Return structured extraction results for all input rows.",
                        "input_schema": schema_dict,
                    }
                ]
                params["tool_choice"] = {"type": "tool", "name": tool_name}

            requests.append(
                {
                    "custom_id": f"{job_id}-chunk-{i:06d}",
                    "params": params,
                }
            )

        path = input_dir / "requests.json"
        path.write_text(json.dumps(requests, indent=2))
        print(f"Wrote {len(requests)} Anthropic batch requests to {path}.")
        return path

    def submit_batch(self, job_id: str) -> str:
        """Submit the prebuilt request list to Anthropic Message Batches.

        Args:
            job_id: The job identifier whose ``batch_input/requests.json`` to submit.

        Returns:
            The Anthropic batch ID (``msgbatch_...``).
        """
        job_dir = self.batch_root_dir / job_id
        requests_path = job_dir / "batch_input" / "requests.json"
        requests = json.loads(requests_path.read_text())

        batch = self.client.messages.batches.create(requests=requests)
        manifest = job_dir / "batch_input" / "submission.json"
        manifest.write_text(batch.model_dump_json(indent=2))
        print(f"Submitted Anthropic batch {batch.id} ({len(requests)} requests).")
        return batch.id

    def check_batch_status(
        self,
        job_id: str,
        continuous: bool = False,
        interval: int = 30,
        timeout: int | None = None,
    ) -> str:
        """Poll batch status and optionally wait until the batch ends.

        Args:
            job_id: The job identifier.
            continuous: If True, keep polling until the batch ends.
            interval: Seconds between polls when continuous=True. Default 30.
            timeout: Optional upper bound on polling time in seconds.

        Returns:
            The latest ``processing_status``: "in_progress", "canceling", or
            "ended". Continuous polling returns only "ended"; cancellation
            remains in progress while the status is "canceling".

        Raises:
            TimeoutError: If ``timeout`` elapses before the batch ends.
            FileNotFoundError: If no submission manifest exists for this job.
        """
        job_dir = self.batch_root_dir / job_id
        manifest_path = job_dir / "batch_input" / "submission.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No Anthropic batch submission found for job {job_id!r}. "
                f"Run submit_batch first."
            )
        batch_id = json.loads(manifest_path.read_text())["id"]

        start = time.monotonic()
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts.model_dump()
            print(
                f"Anthropic batch {batch_id}: status={batch.processing_status} counts={counts}"
            )
            if batch.processing_status == "ended":
                return batch.processing_status
            if not continuous:
                return batch.processing_status
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Anthropic batch {batch_id} did not end within {timeout}s."
                )
            time.sleep(interval)

    def retrieve_results_as_dataframe(
        self,
        job_id: str,
        tool_name: str = "extract_results",
    ) -> pd.DataFrame | None:
        """Stream results and flatten tool_use outputs into a DataFrame.

        Raw results replace the local file only after the stream completes.
        A failed download preserves the previous file and raises the error.

        Args:
            job_id: The job identifier.
            tool_name: Tool name used at submission time. Must match whatever
                was passed to ``create_batch_requests``.

        Returns:
            DataFrame of parsed result rows, or None if no results yet.

        Note:
            When the original request had no tool (free-form text mode), this
            method falls back to writing the raw assistant text into the
            returned DataFrame's ``text`` column.
        """
        job_dir = self.batch_root_dir / job_id
        manifest_path = job_dir / "batch_input" / "submission.json"
        if not manifest_path.exists():
            return None
        batch_id = json.loads(manifest_path.read_text())["id"]

        output_dir = job_dir / "batch_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.jsonl"

        rows: list[dict[str, Any]] = []
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir, prefix=".results-",
                suffix=".tmp", delete=False,
            ) as results_file:
                temporary_path = Path(results_file.name)
                for entry in self.client.messages.batches.results(batch_id):
                    # Persist raw entry for reproducibility.
                    results_file.write(entry.model_dump_json() + "\n")

                    if entry.result.type != "succeeded":
                        log.warning(
                            "Anthropic batch %s: request %s did not succeed (%s).",
                            batch_id,
                            entry.custom_id,
                            entry.result.type,
                        )
                        continue

                    message = entry.result.message
                    for block in message.content:
                        if block.type == "tool_use" and block.name == tool_name:
                            payload = block.input
                            batch_rows = (
                                payload.get("all_results")
                                if isinstance(payload, dict)
                                else None
                            ) or []
                            rows.extend(_object_rows(batch_rows, entry.custom_id))
                        elif block.type == "text":
                            # Free-form path: tolerate ```json fences and mild preamble
                            # by extracting the outermost JSON object.
                            raw = block.text.strip()
                            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
                            start, end = raw.find("{"), raw.rfind("}")
                            inner: Any = []
                            if start != -1 and end > start:
                                try:
                                    parsed = json.loads(raw[start : end + 1])
                                    inner = next(
                                        (parsed[key] for key in ("all_results", "results")
                                         if key in parsed),
                                        [],
                                    )
                                except json.JSONDecodeError:
                                    inner = []
                            valid_rows = _object_rows(inner, entry.custom_id)
                            if valid_rows:
                                rows.extend(valid_rows)
                            else:
                                rows.append({"custom_id": entry.custom_id, "text": block.text})

            temporary_path.replace(results_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return pd.DataFrame(rows) if rows else None
