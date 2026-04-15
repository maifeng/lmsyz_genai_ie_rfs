# extract_df

`extract_df` is the primary entry point for concurrent extraction. It chunks a DataFrame,
sends each chunk to OpenAI or Anthropic in parallel via a `ThreadPoolExecutor`, and
returns a flat DataFrame of results. Every completed row is written to a SQLite file as
it finishes. An interrupted run resumes from where it stopped on the next call with the
same `cache_path`.

::: lmsyz_genai_ie_rfs.extract_df

---

## See also

- [Resume after a crash](../how-to/resume-after-crash.md): how `cache_path` enables
  zero-loss restarts.
- [Change the prompt safely](../how-to/change-prompt-safely.md): how prompt-hash
  invalidation works and when to use `ignore_prompt_hash=True`.
- [Switch providers](../how-to/switch-providers.md): switching between OpenAI, Anthropic,
  and Gemini.
