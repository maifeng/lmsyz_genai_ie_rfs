# Troubleshooting

---

## My DataFrame came back with fewer rows than I put in

**Symptoms:** `extract_df` returns a DataFrame with fewer rows than the input. No error was raised.

**Cause:** One or more chunks failed. When a chunk exhausts all retries, `extract_df` logs the exception via `log.exception` and skips that chunk. The remaining chunks' results are still returned.

**Fix:**

1. Enable logging to see the stack trace:
   ```python
   import logging
   logging.basicConfig(level=logging.WARNING)
   ```
2. Look for lines starting with `extract_df: chunk failed; results for this chunk skipped.`
3. Common causes:
   - **Malformed prompt:** the model returned text that is not valid JSON, or returned JSON without the expected `all_results` key. Inspect the error message for a JSON parse error.
   - **Rate limits exhausted:** all 5 retry attempts hit `RateLimitError`. Reduce `max_workers` or add a delay by using a smaller `chunk_size`.
   - **Context overflow:** one or more rows in the chunk are too long for the model's context window. Use a smaller `chunk_size` (so fewer rows per call) or truncate the input text.
4. After fixing the cause, rerun the same call with the same `cache_path`. The rows that already succeeded will be skipped; only the failed rows will be retried.

---

## Rerun silently returned cached stale results

**Symptoms:** You changed the prompt and reran `extract_df`, but the output looks the same as before.

**Cause:** Prompt-hash gating should prevent this by default. If you are seeing stale results, check:

1. `ignore_prompt_hash` is `False` (the default). If it is `True`, the library is explicitly ignoring the hash and reusing whatever is cached.
2. The prompt text actually changed. Verify by printing the prompt and checking its hash:
   ```python
   from lmsyz_genai_ie_rfs.dataframe import compute_prompt_hash
   print(compute_prompt_hash(old_prompt))
   print(compute_prompt_hash(new_prompt))
   ```
   If the two hashes are the same, the prompts are identical (perhaps a whitespace difference you cannot see).
3. Inspect the `prompt_hash` column in the SQLite file:
   ```bash
   sqlite3 runs/cache.sqlite "SELECT DISTINCT prompt_hash FROM results;"
   ```
   If you see the new prompt's hash appearing, the gating is working. If only the old hash appears, the new prompt is not producing cache misses (the prompts may be the same).

**Fix:** If the prompts are genuinely different and you still see stale results, pass `fresh=True` to force a full re-run ignoring all cached rows.

---

## `TypeError: extract_df() missing 1 required keyword-only argument: 'cache_path'`

**Cause:** `cache_path` is required. There is no default value. The library enforces this because skipping persistence on a long run is almost always a mistake.

**Fix:** Pass a path:

```python
from lmsyz_genai_ie_rfs import extract_df

out = extract_df(
    df,
    prompt=my_prompt,
    backend="openai",
    model="gpt-4.1-mini",
    cache_path="runs/my_experiment.sqlite",   # required
)
```

If you genuinely want no persistent file, use a temporary path:

```python
import uuid, os
cache = f"/tmp/ephemeral_{uuid.uuid4()}.sqlite"
out = extract_df(df, prompt=..., cache_path=cache, backend="openai", model="gpt-4.1-mini")
os.unlink(cache)
```

See [Results database](../concepts/results-db.md) for the rationale.

---

## My strict JSON schema keeps being rejected by OpenAI

**Symptoms:** OpenAI returns a validation error or the model refuses to produce output matching the schema.

**Cause:** OpenAI's structured outputs (`strict: true`) have specific requirements that are easy to violate:

- Every object must have `"additionalProperties": false`.
- Every property must appear in the `"required"` array.
- No `$ref` references are allowed (the schema must be fully inlined).
- Limited type set: `string`, `number`, `integer`, `boolean`, `array`, `object`, `null`. No `anyOf` on primitives.

**Fix:** Use the pattern from `tests/data/culture_batch_schema.json`, which is a known-good example:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "culture_batch",
    "strict": true,
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": ["all_results"],
      "properties": {
        "all_results": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["input_id", "culture_type", "tone", "confidence"],
            "properties": {
              "input_id":     {"type": "string"},
              "culture_type": {"type": "string", "enum": ["collaboration_people", "customer_oriented",
                               "innovation_adaptability", "integrity_risk",
                               "performance_oriented", "miscellaneous"]},
              "tone":         {"type": "string", "enum": ["positive", "neutral", "negative"]},
              "confidence":   {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
          }
        }
      }
    }
  }
}
```

Check that every nested `object` has both `additionalProperties: false` and all its properties in `required`.

---

## Claude returns ```json fenced output when I use `schema=None`

**Symptoms:** With `backend="anthropic"` and no schema, the returned text contains markdown fences like ` ```json ... ``` `.

**Cause:** In free-form text mode, Anthropic returns a text block. The library's `_call_anthropic` already strips fences automatically with a regex:

```python
raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
```

If you are still seeing fenced output in the final DataFrame, it likely means the JSON inside the fences could not be parsed (e.g., the model returned multiple JSON objects or a non-JSON preamble that confused the fence stripper).

**Fix:** Add an explicit instruction to the prompt:

```
Respond with ONLY a JSON object, no preamble, no markdown fences.
```

Alternatively, pass a `schema=` argument. With a schema, Anthropic uses forced `tool_use` and the response is always structured.

---

## Anthropic batch is stuck

**Symptoms:** `check_batch_status` is reporting `in_progress` and has been for a long time.

**Cause:** Anthropic Message Batches process asynchronously. Small jobs (20 rows) typically take 5-30 minutes. Larger jobs can take several hours. This is expected behavior, not a bug.

**Fix:** Use the `continuous=True` option with an appropriate `interval` and `timeout`:

```python
ext.check_batch_status(
    "my_job",
    continuous=True,
    interval=30,      # poll every 30 seconds
    timeout=1800,     # give up after 30 minutes
)
```

If the job has been running for more than a few hours on a small input (< 1,000 rows), check the Anthropic console for errors. The batch may have been cancelled or hit a content policy block.

---

## `ValueError: No tool_use block in Anthropic response`

**Symptoms:** `_call_anthropic` raises `ValueError: No tool_use block in Anthropic response: [...]`.

**Cause:** The model returned a text block instead of calling the tool. This usually happens when:

- The content triggered Anthropic's content policy and the model refused to process it.
- The model was confused by the input and responded in a way that bypassed tool calling.

**Fix:**

1. Inspect the raw response by passing a custom client with verbose logging:
   ```python
   import anthropic, logging
   logging.basicConfig(level=logging.DEBUG)
   client = anthropic.Anthropic()
   out = extract_df(df, prompt=..., client=client, backend="anthropic", model="claude-haiku-4-5-20251001",
                    cache_path="debug.sqlite")
   ```
2. If it is a content policy refusal, identify the offending row and remove or redact it.
3. If the model is responding in text mode despite `tool_choice` being forced, this may indicate a temporary API issue. Retry the run; the row will be reprocessed (the cache miss will trigger it).

---

## My OpenAI batch job ended with 19/20 rows in `batch_result_*.jsonl` and 1 row in `batch_error_*.txt`

**Symptoms:** `check_batch_status` writes both a `batch_result_*.jsonl` and a `batch_error_*.txt` file. The error file contains one or more failed requests.

**Cause:** Each line in the error file corresponds to one chunk that failed at the provider level. Common reasons:

- The chunk's text content exceeded the model's context window.
- The chunk triggered a content filter.
- A transient internal error at OpenAI (rare).

**Fix:**

1. Read the error file:
   ```bash
   cat batch_output/batch_error_<batch_id>.txt
   ```
2. Each entry contains a `custom_id` in the form `{job_id}-{batch_counter}-{request_counter}`. Use that to identify the failing chunk.
3. Fix the offending row (truncate the text, remove it, or substitute a placeholder).
4. Resubmit as a one-row or one-chunk top-up batch.

---

## Temperature 0 but I am getting different results on rerun

**Symptoms:** You are using a non-o1/o3/gpt-5 model (so `temperature=0.0`) but the output changes between runs on the same input.

**Cause:** Two possibilities:

1. **The model does not strictly honor temperature=0.** Some models have inherent non-determinism at the token-sampling level even at temperature=0, particularly for long or ambiguous completions.
2. **The prompt introduces non-determinism.** If the prompt references context that varies (e.g., "today's date is..."), results will differ between runs.

**Fix:**

- Pass `seed=` via a pre-built client if your provider supports it:
  ```python
  import openai
  client = openai.OpenAI()
  # seed is passed in the API call; use a custom client wrapper or extend _call_openai
  ```
- If exact reproducibility is required, pin the results by running once, caching, and always loading from cache (`fresh=False`). The cached rows are returned verbatim on all subsequent calls.

---

## I do not want the SQLite file

**Cause:** `cache_path` is required. The design is deliberate.

**Fix:** Pass a temporary path and delete it after:

```python
import uuid, os
from lmsyz_genai_ie_rfs import extract_df

cache = f"/tmp/ephemeral_{uuid.uuid4()}.sqlite"
try:
    out = extract_df(
        df, prompt=...,
        backend="openai", model="gpt-4.1-mini",
        cache_path=cache,
    )
finally:
    os.unlink(cache)
```

See [Why `cache_path` is required](../explanation/design-decisions.md#why-cache_path-is-required) for the reasoning.
