# Settings

::: lmsyz_genai_ie_rfs.settings.Settings

---

## Environment variables and .env

All `Settings` fields map directly to environment variables of the same name.
`pydantic-settings` is configured with `case_sensitive=False`, so the env var
`DEFAULT_MODEL` binds to the `default_model` attribute; upper-case env var names and
lower-case attribute names are equivalent. Set them in the shell or drop a `.env` file in
the working directory:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=gpt-4.1-mini
DEFAULT_BACKEND=openai
# OPENAI_BASE_URL=https://openrouter.ai/api/v1  # optional custom endpoint
MAX_WORKERS=20
CHUNK_SIZE=5
```

The `.env` file is loaded when the package settings singleton is instantiated at import.
`extract_df` and `draft_prompt` use its provider credentials and OpenAI base URL when
explicit arguments are omitted. Explicit `api_key=` and `base_url=` take precedence.
Batch constructors use explicit credentials or the SDK's process environment lookup.

The `default_model`, `default_backend`, `max_workers`, and `chunk_size` fields are
available to callers but do not override function signatures automatically. `extract_df`
requires `model=` and otherwise uses `backend="openai"`, `max_workers=20`, and
`chunk_size=5`. To use stored values, pass them explicitly:

```python
from lmsyz_genai_ie_rfs import extract_df
from lmsyz_genai_ie_rfs.settings import settings

out = extract_df(
    df, prompt=prompt, cache_path="run.sqlite",
    model=settings.default_model, backend=settings.default_backend,
    max_workers=settings.max_workers, chunk_size=settings.chunk_size,
)
```

After changing a `.env` file in a running notebook, restart the kernel or pass the new
values explicitly.
