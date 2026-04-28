# draft_prompt

`draft_prompt` generates a candidate `extract_df` prompt from a plain-English goal
description. It sends a one-shot meta-prompt to any supported LLM and returns a
prompt string in the house style that `extract_df` expects: numbered step-by-step
instructions, a strict `{"all_results": [...]}` JSON envelope, and a closing
field-list sentence.

The result is a **starting point**. Read it, edit field names, tighten type
constraints, and then pass the edited string to `extract_df(prompt=...)`.

::: lmsyz_genai_ie_rfs.draft_prompt

---

## See also

- [extract_df](extract_df.md): the primary extraction entry point that consumes the
  prompt produced here.
- [Prompts and schemas](../concepts/prompts-and-schemas.md): house-style rules that
  `draft_prompt` enforces automatically.
