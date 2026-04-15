# Design decisions

This page explains the "why" behind choices that might surprise a new user. Most of these decisions traded generality for clarity: the library does one thing and makes that one thing hard to accidentally break.

---

## Why one function, not a class

The public entry point is `extract_df`, a module-level function. There is no `Extractor` class to instantiate, no `LLMClient.run()` method, no object to configure before calling.

This matches the shape of the original research code (`gpt_funcs.py:run_gpt_on_df`), which was also a function. Researchers working in notebooks naturally call functions; they do not want to manage object lifecycle. Keeping the API flat means:

- One import: `from lmsyz_genai_ie_rfs import extract_df`.
- No `__init__` arguments to remember.
- No state shared across calls unless explicitly passed.

The `client=` parameter is the escape hatch for power users who need to pass a pre-configured SDK client (custom timeouts, proxies, organization IDs). It accepts any OpenAI or Anthropic client instance. The library then uses it directly instead of building its own. This gives expert users full control without complicating the default path.

---

## Why `cache_path` is required

Lost results are worse than the friction of passing a file path.

A researcher running 100,000 rows for two hours should never accidentally skip persistence because they forgot a keyword argument. Making `cache_path` required with no default means the error is a `TypeError` at call time, not a silent data loss at crash time.

This mirrors `gpt_funcs.py:run_gpt_on_df` where the equivalent `db_path` was also required. The original authors made this choice deliberately, and it was preserved here.

If you genuinely do not want a persistent file, pass a temporary path and clean it up after:

```python
import uuid, os
from lmsyz_genai_ie_rfs import extract_df

cache = f"/tmp/ephemeral_{uuid.uuid4()}.sqlite"
out = extract_df(df, prompt=..., cache_path=cache, backend="openai", model="gpt-4.1-mini")
os.unlink(cache)
```

---

## Why the schema is optional

The prompt already describes the output shape. Adding a Pydantic class duplicates that description in a different language. If the prompt says "return a `culture_type` field with one of six values," a `BaseModel` subclass with a `culture_type: Literal[...]` field says the same thing twice.

The optional JSON schema file exists for a different purpose: it asks the provider to enforce the shape at the model level, so a bad response raises a structured error rather than producing a misparse. This matters in production runs (100k rows, overnight batch) where you want hard failures on bad rows rather than silent garbage in your DataFrame.

The key point is that both paths use the same format. The JSON schema file is a standard OpenAI `response_format` object; the same file works for Anthropic without any modification. No Python class definitions required in either case.

---

## Why hashing the prompt

Without a hash, there is no way to tell whether a cached row was produced by the current prompt or an older version.

Consider a researcher iterating on a classification prompt. They run 1,000 rows, examine the results, revise the prompt, and rerun. Without hashing, the library would see 1,000 cached rows, skip all of them, and return the old results. The researcher would think the new prompt was applied when it was not.

Hashing the prompt and stamping each cached row with that hash means the library can detect the mismatch. Changing the prompt produces a different hash; cached rows under the old hash are invisible; the new prompt is applied to all rows.

The escape hatch `ignore_prompt_hash=True` handles the legitimate case where a prompt edit is non-semantic (typo fix, whitespace change, heading rename). Use it consciously, not by default.

---

## Why `OpenAIBatchExtractor` and `AnthropicBatchExtractor` are separate classes

Anthropic's batch API is structurally different from OpenAI's. A shared abstraction would hide important details:

| | OpenAI | Anthropic |
|---|---|---|
| Input format | JSONL file on disk, uploaded via `files.create` | JSON body: a list of request dicts in a single POST |
| Submission | `client.batches.create(input_file_id=...)` | `client.messages.batches.create(requests=[...])` |
| Result retrieval | Download output file by ID | Stream via `client.messages.batches.results(batch_id)` |
| Error handling | Separate error file | Per-request `result.type` field |

Forcing these into one class with a `provider=` argument would require either a large `if provider == "openai" / else` block in every method, or an ABC that the concrete classes satisfy independently. The first is worse than two separate classes; the second adds layers without adding clarity.

Keeping them separate means each class's method signatures and documentation match its provider's actual API. A user reading `OpenAIBatchExtractor` sees the OpenAI lifecycle. A user reading `AnthropicBatchExtractor` sees the Anthropic lifecycle. Neither is burdened with the other's details.

---

## Why no ABC on the concurrent backends

`_call_openai` and `_call_anthropic` are private module-level functions with identical signatures:

```python
def _call_openai(client, system_prompt, user_message, response_format, model): ...
def _call_anthropic(client, system_prompt, user_message, input_schema, model): ...
```

An ABC would require: an abstract base class, two subclasses, `__init__` methods, and method dispatch. The current approach has: two functions. `extract_df` selects the right one based on `backend=`:

```python
if backend == "openai":
    call_fn = _call_openai
    call_args = {"response_format": schema_dict}
else:
    call_fn = _call_anthropic
    call_args = {"input_schema": input_schema}
```

Duck typing at the function level. Fewer layers, easier to read, easier to test individually.

---

## Why retries happen on individual chunks, not entire jobs

A transient `RateLimitError` on one chunk should not fail the other 99 chunks.

`@retry_api_call` wraps `_call_openai` and `_call_anthropic` individually. Each chunk gets 5 attempts with exponential backoff. If a chunk exhausts all 5 attempts, `extract_df` catches the exception in the `as_completed` loop, logs it via `log.exception`, and continues. The other chunks' results are still written to cache and returned.

The alternative (retry the entire job) would mean one bad row's transient error re-processes everything. That is the wrong trade-off for a library where a "job" can be thousands of rows.

If a chunk fails permanently (bad content, context overflow, content policy), the partial results are still returned and the missing rows are visible in the log. The researcher can inspect the failing chunk, fix the input, and rerun.

---

## Why shuffle the working DataFrame

Before chunking, `extract_df` shuffles the rows with a fixed seed:

```python
working = working.sample(frac=1, random_state=42).reset_index(drop=True)
```

Input DataFrames often have structure (sorted by date, grouped by company, ordered by length). Without shuffling, the first chunks might all be very long rows (or very short ones), which concentrates latency and rate-limit pressure at the start of the job. Shuffling distributes that pressure more evenly across the workers.

---

## Why `chunk_size` defaults to 5

Five is a deliberate compromise. Smaller chunks reduce blast radius when a single chunk fails: at `chunk_size=5`, a bad chunk loses at most 5 rows; at `chunk_size=50`, it loses 50. Smaller chunks also parallelize better across `max_workers` threads: with 20 workers and 100 rows, `chunk_size=5` gives 20 chunks (full parallelism); `chunk_size=50` gives 2 chunks (most workers idle).

Larger chunks amortize the system prompt token cost better, especially with Anthropic prompt caching. Once the prompt is cached, each chunk pays only the user-message tokens, so packing more rows per chunk saves on the cache-hit overhead.

The default of 5 is biased toward the safer, more-parallel end of this trade-off. For small jobs with short system prompts, it is rarely the bottleneck. For large jobs with very long system prompts, bumping to 10 or 20 can meaningfully reduce cost. Benchmark on your actual prompt before tuning.

---

## Why you do not need Pydantic

The library uses `pydantic-settings` internally to load `.env` files. That is the only Pydantic dependency from your perspective: it is transitive, handled by the library, and does not require you to write Pydantic classes.

Schemas for structured output are plain JSON dicts or JSON files. A user opening `tests/data/culture_batch_schema.json` sees a standard JSON schema, not a Python class. Same file works on both OpenAI (as `response_format`) and Anthropic (as forced `tool_use` `input_schema`).

An earlier version of this library required users to define a `BaseModel` subclass per extraction task. That design was rejected because it duplicated the prompt's shape description in a second language. The current design makes Pydantic an implementation detail you never touch.
