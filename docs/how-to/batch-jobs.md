# Run a batch job

## Problem

You have 100,000 rows. You can wait up to 24 hours. You want to use the provider batch
API instead of the concurrent path to get roughly 50% cheaper token prices.

## Solution

Both providers follow the same four-step lifecycle: build the input, submit, poll until
done, retrieve as a DataFrame.

### OpenAI batch (JSONL file upload)

```python
import pandas as pd
from lmsyz_genai_ie_rfs import OpenAIBatchExtractor

df = pd.read_csv("my_corpus.csv")   # must have "id" and "text" columns

prompt = """
For each input row, extract:
- input_id: copy verbatim.
- culture_type: one of collaboration_people, customer_oriented, innovation_adaptability,
  integrity_risk, performance_oriented, miscellaneous.
- tone: "positive", "neutral", or "negative".
- confidence: float in [0.0, 1.0].

Return {"all_results": [...]}.
"""

# Optional: strict JSON schema for structured outputs.
import json
schema_dict = json.loads(open("culture_batch_schema.json").read())  # OpenAI wrapper form

ext = OpenAIBatchExtractor(batch_root_dir="runs/openai_job")

# Step 1: write JSONL input files to disk.
ext.create_batch_jsonl(
    dataframe=df,
    id_col="id",
    text_col="text",
    prompt=prompt,
    job_id="my_job",
    model_name="gpt-4.1-mini",
    chunk_size=5,
    schema_dict=schema_dict,   # or None for free-form json_object
)

# Optional: inspect the JSONL before spending money.
# (See "Verifying the input before submission" below.)

# Step 2: upload each JSONL file and submit to the OpenAI Batch API.
ext.submit_batches("my_job")

# Step 3: poll until all batches complete (blocks, checks every 5 minutes by default).
ext.check_batch_status("my_job", continuous=True, interval=300)

# Step 4: parse results into a DataFrame.
out = ext.retrieve_results_as_dataframe("my_job")
if out is None:
    print("Batch not finished yet; check status and retry.")
else:
    print(out.head())
    out.to_csv("runs/openai_output.csv", index=False)
```

### Anthropic batch (JSON body, no file upload)

```python
import pandas as pd
from lmsyz_genai_ie_rfs import AnthropicBatchExtractor

df = pd.read_csv("my_corpus.csv")

prompt = """
For each input row, extract:
- input_id: copy verbatim.
- culture_type: one of collaboration_people, customer_oriented, innovation_adaptability,
  integrity_risk, performance_oriented, miscellaneous.
- tone: "positive", "neutral", or "negative".
- confidence: float in [0.0, 1.0].

Return {"all_results": [...]}.
"""

# For Anthropic, schema_dict is the INNER JSON schema (not the OpenAI wrapper).
import json
full_schema = json.loads(open("culture_batch_schema.json").read())
inner_schema = full_schema["json_schema"]["schema"]   # unwrap the OpenAI envelope

ext = AnthropicBatchExtractor(batch_root_dir="runs/anthropic_job")

# Step 1: build the request list and write it to batch_input/requests.json.
ext.create_batch_requests(
    dataframe=df,
    id_col="id",
    text_col="text",
    prompt=prompt,
    job_id="my_job",
    model_name="claude-haiku-4-5-20251001",
    chunk_size=5,
    schema_dict=inner_schema,   # or None for free-form text
)

# Step 2: POST the request list to Anthropic's batch endpoint.
ext.submit_batch("my_job")

# Step 3: poll until complete. Raises TimeoutError if not done within timeout seconds.
status = ext.check_batch_status("my_job", continuous=True, interval=20, timeout=1800)
print("Final status:", status)   # "ended" on success

# Step 4: stream results and assemble into a DataFrame.
out = ext.retrieve_results_as_dataframe("my_job")
if out is None:
    print("Batch not finished yet; check status and retry.")
else:
    print(out.head())
    out.to_csv("runs/anthropic_output.csv", index=False)
```

## Explanation

### Verifying the input before submission (OpenAI)

After `create_batch_jsonl` writes the JSONL files to `batch_input/`, you can inspect
them before submitting:

```python
import json
from pathlib import Path

job_dir = Path("runs/openai_job/my_job/batch_input")
for jsonl_file in sorted(job_dir.glob("*.jsonl")):
    with open(jsonl_file) as fh:
        first = json.loads(fh.readline())
    print(f"\n--- {jsonl_file.name} ---")
    print("model   :", first["body"]["model"])
    print("temp    :", first["body"]["temperature"])
    print("rf type :", first["body"]["response_format"]["type"])
    print("sys role:", first["body"]["messages"][0]["role"])   # must be "system"
    print("content :", first["body"]["messages"][0]["content"][:80], "...")
```

This verifies the prompt, model, temperature, and schema before any money is spent.

### Directory layout

After the full lifecycle, the on-disk layout looks like this:

```
runs/openai_job/my_job/
    batch_input/
        batch_0.jsonl           # chunk requests, ready for upload
        batch_1.jsonl           # (if more than max_requests_per_batch chunks)
    batch_output/
        submission_batch_abc.json       # submission manifest from OpenAI
        batch_result_batch_abc.jsonl    # raw results (written after completion)
        batch_error_batch_abc.txt       # error log (written only if errors occurred)

runs/anthropic_job/my_job/
    batch_input/
        requests.json           # full request list (all chunks, JSON array)
        submission.json         # Anthropic batch manifest (id, status, ...)
    batch_output/
        results.jsonl           # raw streamed results from Anthropic
        errors.txt              # per-request error payloads, if any
```

Every intermediate file is on disk so you can audit what was sent and what came back.
Nothing is hidden in memory.

### `max_requests_per_batch` and large inputs

OpenAI's batch API limits input files to 200 MB and 50,000 requests per file.
`OpenAIBatchExtractor` defaults to `max_requests_per_batch=5000`. If your DataFrame has
more chunks than that, the extractor writes multiple JSONL files and submits each
separately. The results are merged automatically by `retrieve_results_as_dataframe`.

If you are using a small `chunk_size` (e.g., 1 or 2), you will have more requests for
the same number of rows. In that case, reduce `max_requests_per_batch` further if you
approach the 200 MB file size limit.

```python
ext = OpenAIBatchExtractor(
    batch_root_dir="runs/large_job",
    max_requests_per_batch=2000,   # write a new file every 2000 chunks
)
```

### Provider comparison

| Feature | OpenAI | Anthropic |
|---|---|---|
| Input format | JSONL file upload | JSON body (list of requests) |
| File upload required | Yes (`client.files.create`) | No |
| Schema enforcement | `response_format` (`json_schema`) | `tool_use` (`input_schema`) |
| Max requests per batch | 50,000 (200 MB input limit) | 100,000 (256 MB) |
| Result retention | 7 days | 29 days |
| Polling | `check_batch_status(interval=300)` | `check_batch_status(interval=20)` |
| Timeout control | Manual (poll loop) | `timeout=` parameter |
| Gemini batch support | Hybrid SDK required (not wrapped) | N/A |

### Gemini and batch

Gemini's OpenAI-compat endpoint supports `batches.create`, but the file upload step
requires the native `google-generativeai` SDK. Because of this hybrid requirement,
`OpenAIBatchExtractor` does not support Gemini out of the box. Use the concurrent path
instead. See [Switch providers: Gemini via OpenAI-compatible endpoint](switch-providers.md#3-gemini-via-openai-compatible-endpoint)
for the full example.

## Related

- [Switch providers](switch-providers.md): running the same prompt on OpenAI, Anthropic,
  and Gemini with `extract_df`.
- [Resume after a crash](resume-after-crash.md): crash recovery with `cache_path=` on
  the concurrent path.
- [Inspect the results database](inspect-results-db.md): auditing SQLite cache contents.
- [Reference: `OpenAIBatchExtractor`](../reference/batch-extractors.md)
- [Reference: `AnthropicBatchExtractor`](../reference/batch-extractors.md)
