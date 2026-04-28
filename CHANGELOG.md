# Changelog

All notable changes to `lmsyz_genai_ie_rfs` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.2] - 2026-04-28

### Added

- Reference page `docs/reference/draft_prompt.md` with mkdocstrings autodoc and
  narrative intro; added to mkdocs.yml nav under Reference.
- `CHANGELOG.md` (this file).

### Changed

- `pyproject.toml` description rewritten to reflect that no user-facing Pydantic
  schema is required: "Concurrent and batch LLM extraction over a pandas DataFrame,
  no schema boilerplate."
- Stale `classify_df` references in `settings.py` and `anthropic_batch.py` updated
  to `extract_df`.
- Stale `LLMClient(backend='openai', ...)` reference in `batch.py` updated to the
  current `extract_df(..., provider='openai', ...)` form.
- `__init__` docstrings in both batch extractor classes: "Initialise the classifier"
  changed to "Initialise the extractor".
- `docs/explanation/troubleshooting.md` moved to `docs/how-to/troubleshooting.md`;
  mkdocs.yml nav updated to match.
- README "All knobs" section extended with a `draft_prompt knobs` subsection covering
  `goal`, `backend`, `model`, `api_key`, `base_url`, and `print_prompt`.

---

## [0.1.1] - 2026-04-28

### Added

- `draft_prompt(goal=...)`: one-shot meta-prompt helper that generates a candidate
  `extract_df` prompt in the house style (numbered steps, strict `all_results` JSON
  envelope, closing field-list sentence). Commits `32bc9ef`, `d3ce4c6`.

---

## [0.1.0] - 2026-04-27

### Added

- Stable release promoting 0.1.0a3. Single `pip install` (no `--pre`) picks this up.
- `extract_df`: concurrent ThreadPoolExecutor extraction over OpenAI, Anthropic, and
  OpenRouter with required SQLite cache and sha256(prompt)[:16] invalidation.
- `OpenAIBatchExtractor`: full lifecycle for OpenAI `/v1/batches` (JSONL file upload,
  submit, poll, retrieve).
- `AnthropicBatchExtractor`: full lifecycle for Anthropic Message Batches (JSON body
  POST, no file upload, tool_use structured output).
- Optional JSON schema support: pass a path, a dict, or None for free-form JSON.
- 55 offline tests + 11 live tests (cassette-backed where possible). Commit `3845167`.

### Fixed

- System prompt role in `batch.py`: original upstream code (`gpt_funcs.py:303`) passed
  the system prompt as `{"role": "assistant"}`. Fixed to `{"role": "system"}`.

---

## [0.1.0a3] - 2026-04-27

### Added

- `examples/news_extraction_schema.json`: schema example matching the Quickstart task
  (entities + causal_triples + sentiment).

### Changed

- README rewrite for mechanism-led prose; removed promotional framing.
- Quickstart row 2 rephrased to contain explicit causal language so the expected output
  shows a non-empty `causal_triple`. Commit `06b67a8`.

---

## [0.1.0a2] - 2026-04-27

### Changed

- Progress messages switched from `log.info()` to `print()` so they appear without
  `logging.basicConfig`.
- Repository URL corrected to `github.com/maifeng/lmsyz_genai_ie_rfs`.
- README: reorganized sections, added OpenRouter `base_url` example. Commit `298bc2c`.

---

## [0.1.0a1] - 2026-04-15

### Added

- Initial PyPI release under the `lmsyz_genai_ie_rfs` name (renamed from
  `genai_batch_ie_rfs`).
- `extract_df`, `OpenAIBatchExtractor`, `AnthropicBatchExtractor`, `SqliteCache`
  public API.
- 15-page MkDocs site (concepts / how-to / reference / explanation). Commit `41ec4b4`.
