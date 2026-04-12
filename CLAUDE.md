# CLAUDE.md — genai_batch_ie_rfs

Project-local instructions for future Claude sessions. Read this before touching any code.

## What this package does

`genai_batch_ie_rfs` provides concurrent and batch LLM-based information extraction on
pandas DataFrames, with Pydantic-typed structured outputs. It was extracted from the
research codebase for Li, Mai, Shen, Yang & Zhang (2026), "Dissecting Corporate Culture
Using Generative AI," *RFS* 39(1):253-296.

Two execution paths:

1. **Concurrent path** (`dataframe.py`, `client.py`): `ThreadPoolExecutor`-based, sends
   chunks to the API in parallel, stores results, supports resume via `SqliteCache`.
2. **Batch path** (`batch.py`): OpenAI Batch API submission and retrieval for large jobs
   (cheaper, async, up to 24h turnaround).

## Architecture

```
LLMClient (client.py)
  |- OpenAIBackend  -> uses client.chat.completions.parse + Pydantic schema
  |- AnthropicBackend -> uses tool-use with cache_control on system prompt

DataFrameIterator (dataframe.py)   chunking + formatting
classify_df (dataframe.py)         concurrent path orchestrator
SqliteCache (dataframe.py)         resume / get-put cache (stub)

GPTBatchJobClassifier (batch.py)   OpenAI Batch API full lifecycle
retry_api_call (retry.py)          tenacity decorator
Settings (settings.py)             pydantic-settings from .env
CultureRow (schema.py)             example Pydantic output schema
```

## How to run tests

```bash
# from repo root
pip install -e ".[dev]"
pytest
```

Tests in `tests/` cover:
- `test_schema.py`: Pydantic roundtrip validation (no API calls).
- `test_dataframe.py`: `DataFrameIterator` chunking logic (no API calls).
- `conftest.py`: vcrpy cassette directory setup.

## Known stubs (need real implementation)

The following methods raise `NotImplementedError` and need Feng's implementation:

- `OpenAIBackend._call` in `client.py`: wire `client.chat.completions.parse`.
- `AnthropicBackend._call` in `client.py`: wire `client.messages.create` with
  `cache_control` on the system prompt block and tool-use for structured output.
- `SqliteCache.get` / `SqliteCache.put` in `dataframe.py`: SQLite resume logic.
- `classify_df` in `dataframe.py`: orchestration loop (iterator + executor + cache).

## Open design questions for Feng

See `plan/16_genai_ie_scaffold.md` for the full list.

## Coding conventions

- Python 3.11+, `from __future__ import annotations` everywhere.
- Google-style docstrings on all public functions/classes/methods.
- No em-dashes in prose or docstrings.
- Pin deps in `pyproject.toml` (both lower and upper bounds).
- Keep files under 200 lines. Split if a file grows past that.

## P0 bug already fixed

**`batch.py` line with role "assistant"**: The original `gpt_funcs.py:303` passed the
system prompt as `{"role": "assistant", ...}`. This package fixes it to `{"role": "system"}`.
See the prominent comment in `GPTBatchJobClassifier.create_batch_jsonl`.
