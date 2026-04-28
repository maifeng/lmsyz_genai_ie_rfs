# lmsyz_genai_ie_rfs

**Turn a prompt into a DataFrame of structured information.**

Code from Li, Kai, Feng Mai, Rui Shen, Chelsea Yang, and Tengfei Zhang (2026), "Dissecting Corporate Culture Using Generative AI," *Review of Financial Studies* 39(1):253–296, [doi.org/10.1093/rfs/hhaf081](https://doi.org/10.1093/rfs/hhaf081). The library itself is domain-agnostic: any prompt, any DataFrame.

---

## The idea in one example

You have this DataFrame:

| id | text |
|---|---|
| 1 | Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025. |
| 2 | Tesla acquired SolarCity in 2016 for $2.6 billion to enter the solar market. |
| 3 | Pfizer's decision to spin off its consumer health unit was driven by activist pressure from Trian Partners. |

You want this DataFrame out:

| input_id | entities | causal_triples | sentiment |
|---|---|---|---|
| 1 | `[{Tim Cook, PERSON}, {Apple, ORG}, {iPhone 17, PRODUCT}, {WWDC, EVENT}, {June 2025, DATE}]` | `[]` | neutral |
| 2 | `[{Tesla, ORG}, {SolarCity, ORG}, {2016, DATE}, {$2.6 billion, MONEY}]` | `[]` | neutral |
| 3 | `[{Pfizer, ORG}, {Trian Partners, ORG}]` | `[[activist pressure, drove, spin-off]]` | neutral |

One function call gets you there:

```python
from lmsyz_genai_ie_rfs import extract_df

out = extract_df(
    df, prompt=my_prompt,
    backend="openai", model="gpt-4.1-mini",
    cache_path="runs/my_results.sqlite",
)
```

The full runnable code with the exact prompt is in the [90-second tour](#90-second-tour-entity--relation-extraction) below.

---

## Install

```bash
pip install lmsyz_genai_ie_rfs
```

Set at least one provider key (or drop them in a `.env` file next to your notebook):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 90-second tour: entity + relation extraction

An information-extraction task with the JSON output shape spelled out in the prompt, matching the style of the research prompts in the RFS 2026 paper.

```python
import pandas as pd
from lmsyz_genai_ie_rfs import extract_df

df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025.",
        "Tesla acquired SolarCity in 2016 for $2.6 billion to enter the solar market.",
        "Pfizer's decision to spin off its consumer health unit was driven by activist pressure from Trian Partners.",
    ],
})

prompt = """
You are an information-extraction assistant. For each input row, analyze the text and extract structured information.

Step-by-step instructions:

1. input_id: Copy the input_id from the row verbatim.
2. entities: List every named entity mentioned in the text. For each entity give:
   - name: the surface form as it appears in the text.
   - type: one of "PERSON", "ORG", "PRODUCT", "DATE", "MONEY".
3. causal_triples: If the text explicitly states a cause and effect, list each as a
   three-element array ["cause", "relation", "effect"]. If there is no explicit
   causation, return an empty list [].
4. sentiment: One of "positive", "neutral", or "negative".

Return a JSON object with this EXACT structure:

{
  "all_results": [
    {
      "input_id": "1",
      "entities": [
        {"name": "Apple",    "type": "ORG"},
        {"name": "Tim Cook", "type": "PERSON"}
      ],
      "causal_triples": [],
      "sentiment": "neutral"
    },
    {
      "input_id": "2",
      "entities": [...],
      "causal_triples": [...],
      "sentiment": "..."
    }
  ]
}

Do not include any fields besides input_id, entities, causal_triples, and sentiment.
"""

out = extract_df(
    df, prompt=prompt,
    backend="openai", model="gpt-4.1-mini",
    cache_path="runs/demo.sqlite",
    id_col="id", text_col="text",
)
print(out)
```

Save the result:

```python
out.to_csv("extraction.csv", index=False)                          # lists/dicts as stringified cells
out.to_json("extraction.jsonl", orient="records", lines=True)      # round-trips cleanly
```

---

## Speed: `chunk_size` and `max_workers`

These two arguments are how you scale a one-row demo to a hundred-thousand-row job.

```python
out = extract_df(
    df, prompt=...,
    chunk_size=5,        # rows packed into one API call (amortizes the system prompt)
    max_workers=20,      # API calls in flight at once (a thread pool)
    backend="openai", model="gpt-4.1-mini",
    cache_path="runs/big.sqlite",
)
```

The two speedups compose. With ~1 second per call and 100,000 rows:

| Approach | API calls | Wall-clock time |
|---|---|---|
| One row per call, serial (a plain `for` loop) | 100,000 | ~28 hours |
| One row per call, `max_workers=20` | 100,000 | ~83 minutes |
| `chunk_size=5`, `max_workers=20` | 20,000 | **~17 minutes** |
| `chunk_size=10`, `max_workers=60` | 10,000 | **~2.8 minutes** |

**Tuning:**

- **`chunk_size`** — start at 5. Smaller chunks parallelize better and limit blast radius if one chunk's JSON is malformed. Larger chunks (10–20) amortize the system prompt better, especially with Anthropic prompt caching. Past ~30 the model starts truncating or skipping rows.
- **`max_workers`** — start at 20. Raise it until the log shows rate-limit retries. OpenAI tier-3 accounts comfortably handle 60–100; Anthropic tier-2 handles 20–50. If you start hitting HTTP 429, drop it.

Need it ~50% cheaper and can wait hours instead of minutes? See [the batch path](#the-batch-path-when-you-have-a-lot-of-rows) below.

---

## Resilience: cache + prompt hashing

Two things you get without doing anything:

- **Resume on failure.** Every completed row is written to SQLite as it finishes. When an API times out, rate-limits you, or returns a 5xx, the library retries with exponential backoff. If the run dies altogether, rerun the same cell and only the unfinished rows are processed.
- **Safe prompt edits.** The library remembers which prompt produced each row (a 16-char hash). Edit the prompt and rerun: affected rows are re-executed automatically.

Details: [The results DB](#the-results-db-cache_path).

---

## Optional: a JSON schema file

The default is prompt only. The model returns a JSON object described by the prompt. For 95% of research use cases this is enough.

You may want stricter guarantees when running the same prompt over 100,000 rows overnight. A malformed row in that setting is expensive: it corrupts your DataFrame, and you only notice later during analysis. A JSON schema file asks the provider to enforce the output shape at the API layer, so the model is mechanically prevented from returning a bad row. OpenAI accepts it as `response_format={"type": "json_schema", ...}`; Anthropic uses the same schema as a forced `tool_use` input schema.

Drop a JSON schema file next to your notebook and pass it:

```python
out = extract_df(
    df, prompt=my_prompt, schema="my_schema.json",
    backend="openai", model="gpt-4.1-mini",
    cache_path="runs/strict.sqlite",
)
```

The file contains a standard OpenAI `response_format` object. One example, taken directly from `tests/data/culture_batch_schema.json`:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "culture_batch",
    "strict": true,
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": ["all_results"],
      "properties": {
        "all_results": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["input_id", "culture_type", "tone", "confidence"],
            "properties": {
              "input_id":     {"type": "string"},
              "culture_type": {"type": "string", "enum": ["collaboration_people", "customer_oriented", "innovation_adaptability", "integrity_risk", "performance_oriented", "miscellaneous"]},
              "tone":         {"type": "string", "enum": ["positive", "neutral", "negative"]},
              "confidence":   {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
          }
        }
      }
    }
  }
}
```

The same file works for Claude: the library passes the inner `schema` object to Anthropic as a forced `tool_use` input schema. One file, both providers, identical validation.

`schema=` accepts a file path, a dict with the contents above, or `None` (the default).

---

## Provider support

| Capability | OpenAI | Anthropic | Gemini (via OpenAI compat) |
|---|---|---|---|
| Concurrent, with schema   | yes: structured outputs | yes: `tool_use` + prompt caching | yes: structured outputs |
| Concurrent, no schema     | yes: `json_object` | yes: plain text (parsed tolerantly) | yes: `json_object` |
| Batch API (~50% cheaper)  | yes: `OpenAIBatchExtractor` | yes: `AnthropicBatchExtractor` | partial: native SDK needed for upload |
| Long-prompt caching       | automatic (1024+ tokens) | explicit `cache_control` on system | automatic |

For Gemini, pass the OpenAI-compat endpoint as a base URL:

```python
out = extract_df(
    df, prompt=my_prompt,
    backend="openai", model="gemini-2.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
    cache_path="runs/gemini.sqlite",
)
```

For OpenRouter (Llama, Mistral, DeepSeek, Qwen, ...): same pattern with `base_url="https://openrouter.ai/api/v1"` and `api_key=os.environ["OPENROUTER_API_KEY"]`.

---

## The results DB (cache_path)

`cache_path` is **required**. It is the path to a regular SQLite file on disk where every completed row is written as it finishes. Nothing is ever auto-deleted.

```python
out = extract_df(
    df, prompt=...,
    backend="openai", model="gpt-4.1-mini",
    cache_path="runs/my_experiment.sqlite",    # required
)
```

If your process dies at row 60,000 of 100,000, those 60,000 rows are on disk. Rerun the same cell: the library reads what's already there, processes the remaining 40,000, returns all 100,000.

The schema is three columns:

```sql
results(row_id TEXT PRIMARY KEY, json_result TEXT, prompt_hash TEXT)
```

Open it in any SQLite browser or `sqlite3 my_experiment.sqlite` to inspect.

Force a full re-run ignoring the DB: `fresh=True`.

### What if I change the prompt?

Each cached row is stamped with `sha256(prompt)[:16]`. On rerun, rows whose stamp matches the current prompt are skipped; rows stamped with a different prompt are re-executed and **the old row is overwritten** (`INSERT OR REPLACE` on `row_id`).

Three escape hatches:

- Want to compare prompts side by side? Use **a different `cache_path`** for each prompt version. Each file keeps its own copy.
- Fixed a typo and want to keep the cached rows? Pass `ignore_prompt_hash=True`. The library reuses the existing rows even though the hash differs.
- Want to wipe and redo? `fresh=True`, or delete the `.sqlite` file.

---

## The batch path (when you have a lot of rows)

The batch API is ~50% cheaper per token but asynchronous. You submit a blob, the provider works through it, you collect results up to 24 hours later. Use it when the savings matter and the wait does not.

### OpenAI batch

```python
from lmsyz_genai_ie_rfs import OpenAIBatchExtractor

ext = OpenAIBatchExtractor(batch_root_dir="runs/my_job/")

ext.create_batch_jsonl(               # 1. write JSONL input files
    dataframe=df, id_col="id", text_col="text",
    prompt=prompt, job_id="my_job",
    model_name="gpt-4.1-mini",
    schema_dict=None,                 # or a strict json_schema dict
)
ext.submit_batches("my_job")          # 2. upload + submit
ext.check_batch_status("my_job",      # 3. poll
                       continuous=True, interval=60)
out = ext.retrieve_results_as_dataframe("my_job")   # 4. results as DataFrame
```

### Anthropic batch

Same lifecycle, different wire format (JSON body, not JSONL upload):

```python
from lmsyz_genai_ie_rfs import AnthropicBatchExtractor

ext = AnthropicBatchExtractor(batch_root_dir="runs/my_job/")
ext.create_batch_requests(..., schema_dict=my_schema)
ext.submit_batch("my_job")
ext.check_batch_status("my_job", continuous=True)
out = ext.retrieve_results_as_dataframe("my_job")
```

Every intermediate file (JSONL input, submission manifest, raw results) is on disk under `batch_root_dir/` so you can inspect exactly what went to the provider and what came back. Nothing is hidden.

---

## All knobs

```python
extract_df(
    df,
    prompt=...,                 # required
    cache_path="path.sqlite",   # required: results DB on disk
    backend="openai",           # or "anthropic"
    model="gpt-4.1-mini",       # required
    schema=None,                # optional: path to JSON schema file, dict, or None
    id_col="id",
    text_col="text",
    chunk_size=5,               # rows per API call; 5-20 is the sweet spot
    max_workers=20,             # parallel threads; tune to your rate limit
    fresh=False,                # True to ignore prior DB contents
    ignore_prompt_hash=False,   # True to reuse rows produced by a different prompt
    api_key=None,               # overrides .env / environment
    base_url=None,              # for OpenRouter / Gemini compat
    client=None,                # BYO preconfigured SDK client
)
```

Retries are automatic (tenacity, exponential backoff, up to 5 attempts) for rate limits and transient 5xx errors. If a single chunk keeps failing, it is logged and skipped; the other chunks' results are still returned.

---

## FAQ

**Why does my DataFrame have fewer rows than I expect?** A chunk failed. Check the log for the stack trace. Nothing is silently swallowed.

**How do I restart a batch job when I lost the `job_id`?** Look under `batch_root_dir/`. The directories ARE the `job_id`s.

**Is there retry logic?** Yes: tenacity, 5 attempts with exponential backoff 2-30s for `RateLimitError` and `APIError`.

**What about nested lists and dicts in the output?** Works. Pandas renders nested fields as stringified lists in CSV; save to JSONL (`df.to_json(..., orient="records", lines=True)`) for clean round-tripping.

---

## Development

```bash
git clone https://github.com/maifeng/lmsyz_genai_ie_rfs
cd lmsyz_genai_ie_rfs
pip install -e ".[dev]"
pytest                               # offline tests, no keys needed
pytest -m "live and not slow"        # concurrent, ~30 seconds, spends cents
pytest -m "live and slow"            # batch, up to 30 minutes
```

Docs:

```bash
pip install -e ".[docs]"
mkdocs serve                         # http://localhost:8000
```

Test artifacts land in `test_artifacts/<test_name>/` (gitignored).

---

## License

MIT.
