# Settings

::: lmsyz_genai_ie_rfs.settings.Settings

---

## Environment variables and .env

All `Settings` fields map directly to environment variables of the same name
(case-insensitive). Set them in the shell or drop a `.env` file in the working directory:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=gpt-4.1-mini
DEFAULT_BACKEND=openai
OPENAI_BASE_URL=                  # optional: OpenRouter or Gemini compat URL
MAX_WORKERS=20
CHUNK_SIZE=5
```

The `.env` file is loaded automatically by `pydantic-settings` when the `Settings` object
is instantiated. Values passed explicitly to `extract_df` (e.g., `api_key=`, `base_url=`)
always take precedence over settings.
