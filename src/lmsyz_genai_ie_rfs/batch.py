"""Batch API path: OpenAI Batch API job submission, status polling, and result retrieval.

Lifted and modernized from gpt_funcs.py:GPTBatchJobClassifier (renamed to
OpenAIBatchExtractor to reflect that the package is general-purpose
extraction, not classification).

P0 BUG FIX (original gpt_funcs.py:303):
    The original code passed the system prompt as {"role": "assistant", ...}.
    The correct role is "system". This was causing the model to receive the
    instruction template as a simulated assistant turn, degrading instruction-
    following quality and bypassing system-prompt caching. THIS FILE USES
    {"role": "system"} EVERYWHERE. Do not revert this.

Input: a pandas DataFrame with id and text columns, plus a prompt string.
Output: batch JSONL files on disk, then a pandas DataFrame of parsed results.

# MODEL-SWITCHING GUIDE
#
# To use a different OpenAI model, pass model_name= to create_batch_jsonl:
#     clf.create_batch_jsonl(df, ..., model_name="gpt-4.1")
#
# Temperature is forced to 1.0 automatically for o1, o3, and gpt-5 model
# families. All other models use the temperature= argument (default 0.0).
#
# For Anthropic Message Batches, use AnthropicBatchExtractor from
# anthropic_batch.py. For OpenAI-compatible providers, use concurrent
# extract_df with backend="openai" and the provider's base_url; this class
# targets the native OpenAI Batch API.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import openai
import pandas as pd
from tqdm import tqdm

from lmsyz_genai_ie_rfs.dataframe import (
    DataFrameIterator,
    validate_input_dataframe,
    validate_positive_integer,
)

log = logging.getLogger(__name__)


def _requires_temp_one(model_name: str) -> bool:
    """Return True if the model only accepts temperature=1.

    Covers o1, o3, and gpt-5 model families. gpt-4o is intentionally excluded:
    it accepts temperature=0, so the original function name
    "is_gpt4o_or_gpt5_model" was misleading (P0-3 in code review).

    Args:
        model_name: The model identifier string.

    Returns:
        True for o1/o3/gpt-5 models, False otherwise.
    """
    lower = model_name.lower()
    return lower.startswith("o1") or lower.startswith("o3") or "gpt-5" in lower


class OpenAIBatchExtractor:
    """Submit, monitor, and retrieve OpenAI Batch API jobs from a DataFrame.

    Handles the full lifecycle:
    1. create_batch_jsonl: builds JSONL input files from a DataFrame.
    2. submit_batches: uploads and submits to OpenAI Batch API.
    3. check_batch_status: polls until completion (optionally continuous).
    4. retrieve_results_as_dataframe: assembles results into a DataFrame.

    Directory layout created under batch_root_dir/job_id/:
        batch_input/   -- JSONL files ready for submission
        batch_output/  -- submission manifests, result files, error logs

    Attributes:
        batch_root_dir: Root directory for all batch jobs.
        max_requests_per_batch: Cap on requests per individual batch file.
        client: An openai.OpenAI client instance.
    """

    def __init__(
        self,
        batch_root_dir: str = "batch_jobs",
        max_requests_per_batch: int = 5000,
        api_key: str | None = None,
    ) -> None:
        """Initialise the extractor.

        Args:
            batch_root_dir: Root directory for batch jobs. Created if absent.
            max_requests_per_batch: Max requests per JSONL file. The output file
                cannot exceed 5 GB and the input file cannot exceed 200 MB;
                reduce this value if you hit those limits.
            api_key: Optional OpenAI API key override.
        """
        validate_positive_integer(max_requests_per_batch, "max_requests_per_batch")
        self.batch_root_dir = Path(batch_root_dir)
        self.batch_root_dir.mkdir(parents=True, exist_ok=True)
        self.max_requests_per_batch = max_requests_per_batch
        self.client = openai.OpenAI(api_key=api_key)

    def create_batch_jsonl(
        self,
        dataframe: pd.DataFrame,
        id_col: str,
        text_col: str,
        prompt: str,
        job_id: str,
        model_name: str,
        temperature: float = 0.0,
        chunk_size: int = 5,
        exclude_processed: bool = True,
        schema_dict: dict[str, Any] | None = None,
    ) -> None:
        """Build JSONL batch input files from a DataFrame.

        *** P0 BUG FIX ***
        The original gpt_funcs.py:303 set role="assistant" for the system prompt.
        This method uses role="system" as required by the OpenAI Chat Completions API.
        Sending the system prompt as an assistant turn degrades instruction-following
        and prevents system-prompt caching. See code review P0-1.

        Args:
            dataframe: Input DataFrame. Must contain id_col and text_col.
            id_col: Column name for non-null row IDs, unique after string conversion.
            text_col: Column name for text content.
            prompt: System prompt text. Passed as role="system" (P0 fix).
            job_id: Unique job identifier. Used as the subdirectory name.
            model_name: OpenAI model identifier (e.g., "gpt-4.1-mini").
            temperature: Sampling temperature. Overridden to 1 for o1/o3/gpt-5 models.
            chunk_size: Rows per LLM request chunk. Default 5.
            exclude_processed: If True, skip rows already present in batch_output/.
            schema_dict: Optional JSON schema dict for response_format. If None,
                uses {"type": "json_object"}.
        """
        validate_positive_integer(chunk_size, "chunk_size")
        validate_input_dataframe(dataframe, id_col, text_col)
        job_dir = self.batch_root_dir / job_id
        input_dir = job_dir / "batch_input"
        output_dir = job_dir / "batch_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        if exclude_processed and any(output_dir.iterdir()):
            prior = self.retrieve_results_as_dataframe(job_id=job_id)
            if prior is not None:
                if "input_id" not in prior.columns:
                    raise ValueError(
                        "Prior batch results must contain 'input_id' to resume. "
                        "Inspect the saved results or use exclude_processed=False."
                    )
                done_ids = set(prior["input_id"].dropna().astype(str))
                before = len(dataframe)
                dataframe = dataframe[~dataframe[id_col].astype(str).isin(done_ids)]
                print(
                    f"Excluded {before - len(dataframe)} already-processed rows; "
                    f"{len(dataframe)} remain."
                )
            else:
                print(f"No prior results found in {output_dir}.")

        # Clear stale input files before regenerating.
        for stale in input_dir.glob("*"):
            stale.unlink()

        # Shuffle for better load distribution across workers.
        dataframe = dataframe.sample(frac=1, random_state=1).reset_index(drop=True)

        df_iter = DataFrameIterator(
            dataframe=dataframe,
            id_col=id_col,
            text_col=text_col,
            chunk_size=chunk_size,
        )

        response_format: dict[str, Any] = schema_dict or {"type": "json_object"}

        batch_counter = 0
        request_counter = 0
        batch_data: list[dict[str, Any]] = []

        for chunk in tqdm(df_iter, desc="Building batch JSONL", total=len(df_iter)):
            effective_temp = 1.0 if _requires_temp_one(model_name) else temperature

            # *** P0 BUG FIX: role must be "system", not "assistant" ***
            # Original gpt_funcs.py:303 used role="assistant" which caused the
            # model to treat the system prompt as a prior assistant turn.
            task = {
                "custom_id": f"{job_id}-{batch_counter}-{request_counter}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "temperature": effective_temp,
                    "response_format": response_format,
                    "messages": [
                        {"role": "system", "content": prompt},  # FIX: was "assistant"
                        {"role": "user", "content": json.dumps(chunk)},
                    ],
                },
            }
            batch_data.append(task)
            request_counter += 1

            if request_counter >= self.max_requests_per_batch:
                self._write_batch_file(input_dir, batch_counter, batch_data)
                batch_counter += 1
                request_counter = 0
                batch_data = []

        if batch_data:
            self._write_batch_file(input_dir, batch_counter, batch_data)

    def _write_batch_file(
        self, input_dir: Path, counter: int, data: list[dict[str, Any]]
    ) -> None:
        """Write a list of batch request dicts to a JSONL file.

        Args:
            input_dir: Directory to write into.
            counter: Batch file index (used in the filename).
            data: List of request dicts to serialize.
        """
        path = input_dir / f"batch_{counter}.jsonl"
        with open(path, "w") as fh:
            for item in data:
                fh.write(json.dumps(item) + "\n")
        print(f"Wrote {len(data)} requests to {path}.")

    def submit_batches(self, job_id: str) -> None:
        """Upload JSONL files and submit each as an OpenAI Batch job.

        Args:
            job_id: The job identifier whose batch_input/ files to submit.
        """
        job_dir = self.batch_root_dir / job_id
        input_dir = job_dir / "batch_input"
        output_dir = job_dir / "batch_output"

        for batch_file in sorted(input_dir.glob("*.jsonl")):
            with open(batch_file, "rb") as fh:
                uploaded = self.client.files.create(file=fh, purpose="batch")

            submission = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"Batch job {job_id}"},
            )

            manifest = output_dir / f"submission_{submission.id}.json"
            manifest.write_text(json.dumps(submission.model_dump()))
            print(f"Submitted {batch_file.name} as batch {submission.id}.")

    def check_batch_status(
        self,
        job_id: str,
        continuous: bool = False,
        interval: int = 300,
    ) -> None:
        """Poll batch status and download available output and error files.

        Continuous polling stops when every batch is completed, failed, expired,
        or cancelled. Successful partial output remains available for parsing.

        Args:
            job_id: The job identifier to check.
            continuous: If True, keep polling until all batches finish.
            interval: Seconds between polls when continuous=True. Default 300.
        """
        job_dir = self.batch_root_dir / job_id
        output_dir = job_dir / "batch_output"

        while True:
            manifests = list(output_dir.glob("submission_*.json"))
            done = 0

            for manifest in manifests:
                info = json.loads(manifest.read_text())
                batch_id = info["id"]
                result_path = output_dir / f"batch_result_{batch_id}.jsonl"
                error_path = output_dir / f"batch_error_{batch_id}.txt"

                status = self.client.batches.retrieve(batch_id)
                counts = (status.model_dump().get("request_counts") or {})
                print(
                    f"Batch {batch_id}: "
                    f"{counts.get('completed', '?')}/{counts.get('total', '?')} completed."
                )

                if status.error_file_id and not error_path.exists():
                    raw = self.client.files.content(status.error_file_id).content
                    error_path.write_bytes(raw)
                    log.warning("Errors for batch %s written to %s.", batch_id, error_path)
                if status.output_file_id and not result_path.exists():
                    raw = self.client.files.content(status.output_file_id).content
                    result_path.write_bytes(raw)
                    print(f"Results for batch {batch_id} written to {result_path}.")
                if status.status in {"completed", "failed", "expired", "cancelled"}:
                    done += 1
                    if status.status != "completed":
                        log.warning("Batch %s ended with status %s.", batch_id, status.status)
                elif status.status == "finalizing":
                    print(f"Batch {batch_id} is finalizing.")
                else:
                    print(f"Batch {batch_id} is still in progress.")

            if not continuous or done == len(manifests):
                if done == len(manifests):
                    print(f"All {done} batches reached a terminal status.")
                break

            print(f"Waiting {interval}s before next poll.")
            time.sleep(interval)

    def retrieve_results_as_dataframe(self, job_id: str) -> pd.DataFrame | None:
        """Parse completed batch result JSONL files into a DataFrame.

        Unusable records and non-object rows are logged and skipped. Valid rows
        from the same file remain available; raw records are preserved on disk.

        Args:
            job_id: The job identifier whose results to retrieve.

        Returns:
            DataFrame of parsed result rows, or None if no results exist yet.
        """
        output_dir = self.batch_root_dir / job_id / "batch_output"

        if not output_dir.exists() or not any(output_dir.iterdir()):
            return None

        rows: list[dict[str, Any]] = []
        for path in tqdm(list(output_dir.glob("*.jsonl")), desc="Parsing results"):
            with open(path) as fh:
                for line_number, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        content = (
                            record["response"]["body"]["choices"][0]["message"]["content"]
                        )
                        if not isinstance(content, str):
                            raise ValueError("response has no text content (possibly a refusal)")
                        parsed = json.loads(content)
                        if not isinstance(parsed, dict):
                            raise ValueError("response JSON must be an object")
                        # Preserve the supported legacy result-key conventions.
                        batch_rows: Any = next(
                            (parsed[key] for key in ("all_results", "all results", "results")
                             if key in parsed),
                            None,
                        )
                        if isinstance(batch_rows, dict):
                            batch_rows = [batch_rows]
                        if not isinstance(batch_rows, list):
                            raise ValueError("result rows must be an array or object")
                        for row_number, row in enumerate(batch_rows, start=1):
                            if isinstance(row, dict):
                                rows.append(row)
                            else:
                                log.warning(
                                    "Ignoring non-object row %s in %s line %s (request %s).",
                                    row_number, path.name, line_number, record.get("custom_id", "?"),
                                )
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        log.warning(
                            "Could not parse %s line %s: %s", path.name, line_number, exc,
                        )

        return pd.DataFrame(rows) if rows else None
