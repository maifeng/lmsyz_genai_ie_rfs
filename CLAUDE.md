# CLAUDE.md — lmsyz_genai_ie_rfs

Project-local instructions for future Claude sessions. Read this before touching any code.

## What this package does

`lmsyz_genai_ie_rfs` provides concurrent and batch LLM-based information extraction on
pandas DataFrames. Input: DataFrame of text rows and a prompt. Output: DataFrame of
structured information shaped by the prompt. The package was extracted from the research
codebase for Li, Mai, Shen, Yang & Zhang (2026), "Dissecting Corporate Culture Using
Generative AI," *RFS* 39(1):253-296, but is itself domain-agnostic.

Two execution paths:

1. **Concurrent path** (`client.py`): `ThreadPoolExecutor` fans chunks out to the API in
   parallel. Every completed row is persisted to a SQLite cache (`cache_path` is
   required). The public entry point is `extract_df`.
2. **Batch path**: `OpenAIBatchExtractor` (`batch.py`) for OpenAI's `/v1/batches`
   endpoint (JSONL file upload); `AnthropicBatchExtractor` (`anthropic_batch.py`) for
   Anthropic Message Batches (JSON body, no file upload). ~50% cheaper per token, up
   to 24 h turnaround.

## Public API (stable)

```python
from lmsyz_genai_ie_rfs import (
    extract_df,               # primary concurrent entry point
    OpenAIBatchExtractor,     # OpenAI Batch API lifecycle
    AnthropicBatchExtractor,  # Anthropic Message Batches lifecycle
    SqliteCache,              # the results DB class
)
```

`extract_df` signature (keyword-only after `df`):

- Required: `prompt`, `cache_path`, `model`.
- Optional: `schema` (None, dict, or path to JSON file in OpenAI response_format shape),
  `backend` ("openai" or "anthropic"), `id_col`, `text_col`, `chunk_size=5`,
  `max_workers=20`, `fresh=False`, `ignore_prompt_hash=False`, `api_key`, `base_url`,
  `client` (pre-built SDK client for advanced use).

## Internal modules

- `client.py`: `extract_df`, `_call_openai`, `_call_anthropic`, `_load_schema`,
  `_make_client`, `_requires_temp_one`. No ABCs, no facade classes. Duck typing between
  the two `_call_*` functions.
- `batch.py`: `OpenAIBatchExtractor` class with `create_batch_jsonl`, `submit_batches`,
  `check_batch_status`, `retrieve_results_as_dataframe`. System prompt role is
  `"system"` (the original `gpt_funcs.py:303` had a bug using `"assistant"`; fixed and
  documented with a prominent comment at the top of `batch.py`).
- `anthropic_batch.py`: `AnthropicBatchExtractor` class with `create_batch_requests`,
  `submit_batch`, `check_batch_status`, `retrieve_results_as_dataframe`. `schema_dict`
  takes the INNER JSON schema (not the OpenAI wrapper form).
- `dataframe.py`: `DataFrameIterator` (chunker), `SqliteCache` (three-column
  `results(row_id, json_result, prompt_hash)`), `compute_prompt_hash` (sha256 hex[:16]).
- `retry.py`: `retry_api_call` tenacity decorator (5 attempts, exponential 2-30 s
  backoff, triggers on `RateLimitError` and `APIError`).
- `settings.py`: `Settings` (pydantic-settings, reads `.env`). No user-facing Pydantic
  models. `pydantic-settings` is used internally only.

## Load-bearing design decisions (do not re-derive)

- **`cache_path` is required.** Matches original `gpt_funcs.py:run_gpt_on_df:db_path`.
  Users who want ephemeral persistence pass a temp path and `os.unlink` after.
- **Schema is optional and is a JSON file, not a Pydantic class.** Rejected "define a
  BaseModel per task." Accepted: pass a path to a standard OpenAI `response_format` JSON,
  a dict, or None (free-form `json_object` on OpenAI, plain-text JSON on Anthropic).
  Same file works on both providers.
- **Prompt-hash invalidation is the default.** Each cached row is stamped with
  `sha256(prompt)[:16]`. Prompt edits auto-invalidate affected rows. Override with
  `ignore_prompt_hash=True`.
- **One function, not a class.** `extract_df` is a module-level function. No
  `Extractor` class, no `LLMClient` facade, no ABC on backends. Researchers writing in
  notebooks don't want to manage object lifecycle.
- **Anthropic's system prompt always gets `cache_control={"type": "ephemeral"}`**
  automatically for prompt caching across chunks.
- **Temperature is forced to 1.0 for o1, o3, gpt-5 families**; 0.0 otherwise. See
  `_requires_temp_one` in `client.py`.
- **Temporary shuffling** (`random_state=42`) before chunking, to distribute long rows
  across workers rather than bunching them at the start.

## How to run tests

```bash
pip install -e ".[dev,docs]"
pytest                               # offline, no API keys needed
pytest -m "live and not slow"        # concurrent live tests, ~30 s, spends cents
pytest -m "live and slow"            # batch API live tests, up to 30 min each
```

Test layout:

- `tests/test_cache.py` (22 tests): SqliteCache get/put/all_ids, prompt-hash gating,
  legacy migration.
- `tests/test_batch.py`: OpenAI batch path offline.
- `tests/test_dataframe.py`: DataFrameIterator chunking.
- `tests/test_live_api.py`: 11 live tests (concurrent with / without schema for both
  providers; relation extraction; cache resume; batch path with / without schema for
  both providers). Artifacts persist to `test_artifacts/<test_name>/`.
- `tests/data/`: shared fixtures (20-row CSV, prompts, JSON schema).

## Build docs

```bash
pip install -e ".[docs]"
mkdocs serve                         # http://localhost:8000
mkdocs build --strict                # must exit 0
```

Docs layout:

- `docs/index.md` includes `../README.md` verbatim via the include-markdown plugin,
  then adds a short navigation section. Edit the README; the docs index updates.
- `docs/concepts/` (architecture, prompts-and-schemas, results-db).
- `docs/how-to/` (resume, change prompt, switch providers, batch jobs, inspect DB,
  troubleshooting).
- `docs/reference/` (mkdocstrings autogen for extract_df, batch extractors,
  SqliteCache, Settings).
- `docs/explanation/` (design decisions only; troubleshooting lives under How-to).

## Coding conventions

- Python 3.11+, `from __future__ import annotations` everywhere.
- Google-style docstrings on all public functions / classes / methods.
- **No em-dashes** in prose or docstrings. Use colon, comma, or period. Hard rule.
- **No emojis** in docs or code.
- **No "X, not Y" rhetorical constructions.** Feng dislikes the negative framing.
  Describe what things ARE, not what they are not.
- Pin deps in `pyproject.toml` with both lower and upper bounds.
- Keep files under 250 lines where possible.

## Fixed bugs worth remembering

- **`batch.py` system prompt role**: the original `gpt_funcs.py:303` passed the system
  prompt as `{"role": "assistant", ...}`. This package fixes it to `{"role": "system"}`.
  Prominent comment in `batch.py` flags the fix.

## Related plans

- Scaffold plan: `plan/16_lmsyz_genai_ie_scaffold.md`.
- Test + multi-provider research PRD: `plan/prd_genai_batch_tests.md`.
- Overall decision log: `plan/04_decisions.md`.
