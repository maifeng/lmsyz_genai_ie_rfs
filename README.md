# GenAI Information Extraction for DataFrames

A general-purpose library for **prompt-based information extraction over DataFrames**, with concurrent and batch execution against OpenAI, Anthropic, and models on OpenRouter.

If you find this library useful in your research, please cite:

Li, Mai, Shen, Yang, Zhang (2026), "Dissecting Corporate Culture Using Generative AI," *Review of Financial Studies* 39(1):253–296, [doi.org/10.1093/rfs/hhaf081](https://doi.org/10.1093/rfs/hhaf081).

---

## Install

```bash
pip install -U lmsyz_genai_ie_rfs
```

Set at least one provider key (or drop them in a `.env` file next to your notebook):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quickstart: entity + relation extraction

Three short news sentences in:

| id | text |
|---|---|
| 1 | Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025. |
| 2 | Tesla acquired SolarCity in 2016 for $2.6 billion to enter the solar market. |
| 3 | Pfizer's decision to spin off its consumer health unit was driven by activist pressure from Trian Partners. |

Structured DataFrame out:

| input_id | entities | causal_triples | sentiment |
|---|---|---|---|
| 1 | `[{Tim Cook, PERSON}, {Apple, ORG}, {iPhone 17, PRODUCT}, {WWDC, EVENT}, {June 2025, DATE}]` | `[]` | neutral |
| 2 | `[{Tesla, ORG}, {SolarCity, ORG}, {2016, DATE}, {$2.6 billion, MONEY}]` | `[[acquisition, enabled, market entry]]` | positive |
| 3 | `[{Pfizer, ORG}, {Trian Partners, ORG}]` | `[[activist pressure, drove, spin-off]]` | neutral |

One function call:

```python
import pandas as pd
from lmsyz_genai_ie_rfs import extract_df

df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025.",
        "Tesla’s $2.6 billion acquisition of SolarCity in 2016 enabled its entry into the solar market.",
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
   causation, return an empty list []. All elements should be concisely summarized, in three words or less. 
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
      "causal_triples": [[cause_1, relation_1, effect_1], [cause_2, relation_2, effect_2], ...],
      "sentiment": "positive/neutral/negative"
    }
  ]
}

Do not include any fields besides input_id, entities, causal_triples, and sentiment.
"""

out = extract_df(
    df, prompt=prompt,
    chunk_size=5, 
    max_workers=20,
    backend="openai", 
    model="gpt-4.1-mini",
    cache_path="demo.sqlite",
    id_col="id", text_col="text",
    api_key=os.environ["OPENAI_API_KEY"],
)

print(out)
entities = out[['input_id', 'entities']].explode("entities")
print(entities)
```

Save:

```python
out.to_csv("extraction.csv", index=False)                          
```

---

## Speed up: `chunk_size` and `max_workers`

`max_workers` sets the size of the threadpool that issues API calls in parallel. `chunk_size` sets how many DataFrame rows are packed into each call (one shared system prompt, `chunk_size` user inputs). Combining them can speed up execution by 100× compared to a naive `for` loop with one row per call. 

```python
out = extract_df(
    df, prompt=...,
    chunk_size=5,        # rows packed into one API call (amortizes the system prompt)
    max_workers=20,      # API calls in flight at once (a thread pool)
    backend="openai", model="gpt-4.1-mini",
    cache_path="big.sqlite",
)
```

At ~1 second per call, 100,000 rows:

| Approach | API calls | Wall-clock |
|---|---|---|
| One row per call, serial (a plain `for` loop) | 100,000 | ~28 hours |
| One row per call, `max_workers=20` | 100,000 | ~83 minutes |
| `chunk_size=5`, `max_workers=20` | 20,000 | **~17 minutes** |

**Tuning:** start `chunk_size=5`, `max_workers=20`. Raise `max_workers` until the log shows rate-limit retries (OpenAI tier-3 handles 60–100; Anthropic tier-2 handles 20–50). 

For ~50% lower per-token cost in exchange for asynchronous execution, see [the batch path](#the-batch-path) below.

---

## The results cache DB (`cache_path`)

API calls cost money and time. The cache exists so that an interrupted, edited, or replayed run does not re-send rows that already returned.

`cache_path` is **required**. It is the path to a SQLite file. Each completed row is written to it as soon as it returns from the model. The file is not auto-deleted.

On rerun with the same `cache_path`, rows already in the cache are skipped; only missing rows go to the model.

The schema is three columns:

```sql
results(row_id TEXT PRIMARY KEY, json_result TEXT, prompt_hash TEXT)
```

Open in any SQLite browser or `sqlite3 my_experiment.sqlite` to inspect. Force a full re-run: `fresh=True`.

### What if I change the prompt?

Each cached row is stamped with `sha256(prompt)[:16]`. On rerun, rows whose stamp matches the current prompt are skipped; rows stamped with a different prompt are re-executed and **the old row is overwritten** (`INSERT OR REPLACE` on `row_id`).

Three escape hatches:

- Compare prompts side by side: use a different `cache_path` for each prompt version.
- Fixed a typo and want to keep the cached rows: pass `ignore_prompt_hash=True`.
- Wipe and redo: `fresh=True`, or delete the `.sqlite` file.

---

## Optional: a JSON schema file

By default, output structure is enforced only by the prompt; the model returns a JSON object whose shape it infers from the instructions.

Pass `schema=` to enforce structure at the API layer. OpenAI consumes it as `response_format={"type": "json_schema", ...}`; Anthropic consumes it as the `input_schema` of a forced `tool_use`.

```python
out = extract_df(
    df, prompt=my_prompt, schema="my_schema.json",
    backend="openai", model="gpt-4.1-mini",
    cache_path="strict.sqlite",
)
```

The file contains a standard OpenAI `response_format` object. Here is one matching the Quickstart task above:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "news_extraction",
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
            "required": ["input_id", "entities", "causal_triples", "sentiment"],
            "properties": {
              "input_id": {"type": "string"},
              "entities": {
                "type": "array",
                "items": {
                  "type": "object",
                  "additionalProperties": false,
                  "required": ["name", "type"],
                  "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["PERSON", "ORG", "PRODUCT", "DATE", "MONEY", "EVENT"]}
                  }
                }
              },
              "causal_triples": {
                "type": "array",
                "items": {
                  "type": "array",
                  "minItems": 3, "maxItems": 3,
                  "items": {"type": "string"}
                }
              },
              "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
            }
          }
        }
      }
    }
  }
}
```

The same file works on Anthropic: the library extracts the inner `schema` object and passes it as the `input_schema` of a `tool_use`.

`schema=` accepts a file path, a dict with the contents above, or `None` (the default).

---

## Provider support

| Capability | OpenAI | Anthropic | Gemini (via OpenAI compat) |
|---|---|---|---|
| Concurrent, with schema   | yes: structured outputs | yes: `tool_use` + prompt caching | yes: structured outputs |
| Concurrent, no schema     | yes: `json_object` | yes: plain text (parsed tolerantly) | yes: `json_object` |
| Batch API (~50% cheaper)  | yes: `OpenAIBatchExtractor` | yes: `AnthropicBatchExtractor` | partial: native SDK needed for upload |

Gemini and OpenRouter speak the OpenAI chat-completions API, so you reach them by setting `base_url=`:

```python
# Gemini
out = extract_df(
    df, prompt=my_prompt,
    backend="openai", model="gemini-2.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
    cache_path="gemini.sqlite",
)

# OpenRouter (Llama, Mistral, DeepSeek, Qwen, ...)
out = extract_df(
    df, prompt=my_prompt,
    backend="openai", model="meta-llama/llama-3.3-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    cache_path="llama.sqlite",
)
```

---

## The batch path

OpenAI and Anthropic both expose a Batch API at approximately 50% of the per-token cost. Execution is asynchronous with a stated 24-hour SLA (typical completion is faster). Lifecycle: build request files, submit, poll status, retrieve.

### OpenAI batch

```python
from lmsyz_genai_ie_rfs import OpenAIBatchExtractor

ext = OpenAIBatchExtractor(batch_root_dir="my_job/")

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

Same lifecycle, different wire format (a JSON request body, not a JSONL file upload):

```python
from lmsyz_genai_ie_rfs import AnthropicBatchExtractor

ext = AnthropicBatchExtractor(batch_root_dir="my_job/")
ext.create_batch_requests(..., schema_dict=my_schema)
ext.submit_batch("my_job")
ext.check_batch_status("my_job", continuous=True)
out = ext.retrieve_results_as_dataframe("my_job")
```

All intermediate files (JSONL input, submission manifest, raw results) are written under `batch_root_dir/<job_id>/` and can be inspected directly.

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
    chunk_size=5,               # rows per API call; 5-20 is typical
    max_workers=20,             # parallel threads; tune to your rate limit
    fresh=False,                # True to ignore prior DB contents
    ignore_prompt_hash=False,   # True to reuse rows produced by a different prompt
    api_key=None,               # overrides .env / environment
    base_url=None,              # for OpenRouter / Gemini compat
    client=None,                # BYO preconfigured SDK client
)
```

Failed API calls are retried automatically (tenacity, exponential backoff, up to 5 attempts) on `RateLimitError` and `APIError`. If a chunk still fails after retries, the failure is logged and the chunk's rows are omitted from the returned DataFrame.

---

## FAQ

**Why does my DataFrame have fewer rows than I expect?** A chunk failed after retries. Check the log for the stack trace.

**How do I restart a batch job when I lost the `job_id`?** Look under `batch_root_dir/`. The directories ARE the `job_id`s.

**Is there retry logic?** Yes: tenacity, 5 attempts with exponential backoff 2-30s for `RateLimitError` and `APIError`.

**What about nested lists and dicts in the output?** Flatten with `df.explode("entities")` (each list item becomes its own row), followed by `pd.json_normalize(df["entities"])` (each dict's keys become columns). CSV stringifies nested fields; save as JSONL (`df.to_json(..., orient="records", lines=True)`) for clean round-tripping.

---

## License

MIT.
