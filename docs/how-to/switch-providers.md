# Switch providers

## Problem

You want to try the same prompt on OpenAI, Anthropic, and Gemini to compare outputs or
costs before committing to a provider.

## Solution

The three variants below all call `extract_df` with the same `prompt` and `schema` file.
Only `backend`, `model`, `api_key`, and `base_url` differ.

```python
import os
import pandas as pd
from lmsyz_genai_ie_rfs import extract_df

df = pd.DataFrame({
    "id": [f"doc_{i}" for i in range(5)],
    "text": [
        "The team delivered the product on time.",
        "Management values transparency and ethics.",
        "Rapid iteration drives our innovation culture.",
        "Customer feedback shapes every roadmap decision.",
        "Performance targets are reviewed quarterly.",
    ],
})

prompt = """
For each input row, extract:
- input_id: copy verbatim.
- culture_type: one of collaboration_people, customer_oriented, innovation_adaptability,
  integrity_risk, performance_oriented, miscellaneous.
- tone: "positive", "neutral", or "negative".
- confidence: float in [0.0, 1.0].

Return {"all_results": [...]}.
"""

# Optional: the same JSON schema file works for all three providers.
SCHEMA = "culture_batch_schema.json"   # path to a JSON schema file, or None
```

### 1. OpenAI

```python
out_openai = extract_df(
    df,
    prompt=prompt,
    cache_path="runs/openai.sqlite",       # separate file per provider
    model="gpt-4.1-mini",
    backend="openai",
    schema=SCHEMA,
)
out_openai.to_csv("runs/openai_results.csv", index=False)
```

### 2. Anthropic

The system prompt is automatically cached via `cache_control={"type": "ephemeral"}`.
No extra configuration is needed.

```python
out_anthropic = extract_df(
    df,
    prompt=prompt,
    cache_path="runs/anthropic.sqlite",    # separate file: no hash collisions
    model="claude-haiku-4-5-20251001",
    backend="anthropic",
    schema=SCHEMA,
)
out_anthropic.to_csv("runs/anthropic_results.csv", index=False)
```

### 3. Gemini via OpenAI-compatible endpoint

```python
out_gemini = extract_df(
    df,
    prompt=prompt,
    cache_path="runs/gemini.sqlite",
    model="gemini-2.5-flash",
    backend="openai",                      # uses the OpenAI SDK under the hood
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
    schema=SCHEMA,
)
out_gemini.to_csv("runs/gemini_results.csv", index=False)
```

### Comparing results

```python
import pandas as pd

oai  = pd.read_csv("runs/openai_results.csv").rename(columns={"culture_type": "ct_oai", "tone": "tone_oai"})
ant  = pd.read_csv("runs/anthropic_results.csv").rename(columns={"culture_type": "ct_ant", "tone": "tone_ant"})
gem  = pd.read_csv("runs/gemini_results.csv").rename(columns={"culture_type": "ct_gem", "tone": "tone_gem"})

cmp = oai[["input_id", "ct_oai", "tone_oai"]] \
        .merge(ant[["input_id", "ct_ant", "tone_ant"]], on="input_id") \
        .merge(gem[["input_id", "ct_gem", "tone_gem"]], on="input_id")

print(cmp)
```

## Explanation

### Schema file: one file, three providers

The `schema=` argument accepts a path to a JSON file in the OpenAI `response_format`
wrapper format:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "my_schema",
    "strict": true,
    "schema": { ... }
  }
}
```

For OpenAI, the library passes the full wrapper as `response_format`. For Anthropic, it
unwraps the `schema` object and passes it as the `input_schema` of a forced `tool_use`
tool. For Gemini via the OpenAI-compat endpoint, the full wrapper is passed as
`response_format`, just like OpenAI. All three work from the same file.

### Anthropic prompt caching

Anthropic's `claude-haiku-4-5-20251001` and other models support prompt caching when the
system block carries `cache_control={"type": "ephemeral"}`. The library applies this
automatically on every `backend="anthropic"` call. No user action is needed. For long
prompts (typically over a few hundred tokens), this reduces the cost of subsequent chunks
significantly, because the cached system prompt tokens are billed at a lower rate.

### Gemini batch limitation

Gemini's OpenAI-compatible `/v1/batches` endpoint supports batch submission, but the
file upload step (`client.files.create`) requires the native `google-generativeai` SDK,
not the OpenAI SDK. Because of this hybrid requirement, the recommended approach for
Gemini users is the concurrent path shown above. If you need Gemini batch, you must
upload the file using the `google-generativeai` SDK, then pass the resulting file ID to
`openai_client.batches.create(input_file_id=...)` manually.

### Use separate `cache_path` files per provider

Using different SQLite files per provider (as in the examples above) avoids hash
collisions and makes side-by-side comparison straightforward. Row IDs are the same across
files, so a merge on `input_id` works cleanly. Using the same file is also valid if you
want all results in one place, but you must use different `prompt_hash` values to
distinguish results (which is automatic as long as the prompts differ).

### Temperature note

For `o1`, `o3`, and `gpt-5` model families, the library forces `temperature=1.0`
automatically. All other models use `temperature=0.0`. This is handled internally; you
do not need to pass a temperature argument to `extract_df`.

## Related

- [Run a batch job](batch-jobs.md): using `OpenAIBatchExtractor` and
  `AnthropicBatchExtractor` for large async jobs.
- [Resume after a crash](resume-after-crash.md): `cache_path` and crash recovery.
- [Reference: `extract_df`](../reference/extract_df.md)
- [Concepts: prompts and schemas](../concepts/prompts-and-schemas.md)
