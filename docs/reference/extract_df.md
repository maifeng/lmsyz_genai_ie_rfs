# extract_df

`extract_df` is the primary entry point for concurrent extraction. It chunks a DataFrame,
sends each chunk to OpenAI or Anthropic in parallel via a `ThreadPoolExecutor`, and
returns a flat DataFrame of results. Every completed row is written to a SQLite file as
it finishes. An interrupted run resumes from where it stopped on the next call with the
same `cache_path`.

::: lmsyz_genai_ie_rfs.extract_df

---

## Temperature rules

The `_requires_temp_one` function returns `True` for model families that only accept
`temperature=1.0`. All other models use `temperature=0.0`.

| Model family | Temperature enforced |
|---|---|
| `o1`, `o1-mini`, `o1-preview`, ... | 1.0 |
| `o3`, `o3-mini`, ... | 1.0 |
| `gpt-5`, `gpt-5-mini`, ... | 1.0 |
| Everything else | 0.0 (deterministic) |

The check uses the model name string: `lower.startswith(("o1", "o3"))` or `"gpt-5" in lower`.
You cannot override this in the concurrent path; it is automatic. In the batch path,
`create_batch_jsonl` accepts a `temperature` argument but overrides it to 1.0 for the
affected families.

---

## See also

- [Resume after a crash](../how-to/resume-after-crash.md): how `cache_path` enables
  zero-loss restarts.
- [Change the prompt safely](../how-to/change-prompt-safely.md): how prompt-hash
  invalidation works and when to use `ignore_prompt_hash=True`.
- [Switch providers](../how-to/switch-providers.md): switching between OpenAI, Anthropic,
  and Gemini.
