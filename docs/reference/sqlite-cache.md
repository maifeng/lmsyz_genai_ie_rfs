# SqliteCache

This page documents the small helpers under `lmsyz_genai_ie_rfs.dataframe`. Two concerns live together here: **caching** (the SQLite-backed results store that makes `extract_df` resumable) and **chunking** (the iterator that splits a DataFrame into per-call input blocks).

For a conceptual walkthrough of the results database, see [Results database](../concepts/results-db.md). For hands-on inspection of a live cache file, see [Inspect the results database](../how-to/inspect-results-db.md).

---

## Caching

`SqliteCache` is the on-disk results store used by `extract_df`. Every completed row is written to it as it finishes. The cache is also prompt-hash aware: rows produced under a different prompt are invisible by default, so changing the prompt automatically re-executes affected rows.

::: lmsyz_genai_ie_rfs.SqliteCache

---

### compute_prompt_hash

`extract_df` stamps every cached row with `compute_prompt_hash(prompt)`. Lookups are gated on this hash so a prompt change produces a cache miss and re-runs the row.

::: lmsyz_genai_ie_rfs.dataframe.compute_prompt_hash

---

## Chunking

`DataFrameIterator` is an internal helper that splits a DataFrame into fixed-size chunks of `{input_id, input_text}` dicts, ready to be serialized as the user message for each LLM call. Most users never touch it directly; it is documented here because it shows up in the public attribute `chunk_size` on `extract_df`.

::: lmsyz_genai_ie_rfs.dataframe.DataFrameIterator

---

## See also

- [Results database](../concepts/results-db.md): the conceptual picture, schema, hash gating.
- [Inspect the results database](../how-to/inspect-results-db.md): SQL recipes and Python patterns.
- [Resume after a crash](../how-to/resume-after-crash.md): the most common reason to care about the cache.
