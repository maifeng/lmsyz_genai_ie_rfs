# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # lmsyz_genai_ie_rfs: Quickstart
#
# Prompt-based information extraction over a pandas DataFrame.
# One function, one DataFrame in, one DataFrame out.
#
# Originally developed for:
# Li, Kai, Feng Mai, Rui Shen, Chelsea Yang, and Tengfei Zhang (2026),
# "Dissecting Corporate Culture Using Generative AI,"
# *Review of Financial Studies* 39(1):253-296.
# https://doi.org/10.1093/rfs/hhaf081
#
# **Corpus**: 2,000 Glassdoor reviews.

# %% [markdown]
# ## 1. Install
#
# In Colab, uncomment and run. Pinned to the workshop alpha to keep
# everyone on the same version.

# %%
# !pip install -q -U lmsyz_genai_ie_rfs

# %% [markdown]
# ## 2. Set your API key
#
# The library reads `OPENAI_API_KEY` / `OPENROUTER_API_KEY` /
# `ANTHROPIC_API_KEY` from the environment. If you don't have one yet,
# create a key at the provider's key page (you may need to add a few
# dollars of credit before the key actually works):
#
# - OpenAI: https://platform.openai.com/api-keys
# - OpenRouter (one key, hundreds of models: Llama, Gemini, DeepSeek...): https://openrouter.ai/keys
# - Anthropic (Claude): https://console.anthropic.com/settings/keys
#
# **Uncomment the line that matches your provider below and paste your
# key**, otherwise the extraction cell will fail with an authentication
# error.

# %%
import os

# %%
import pandas as pd

# os.environ["OPENAI_API_KEY"] = "sk-..."          # uncomment and fill in for OpenAI
# os.environ["OPENROUTER_API_KEY"] = "sk-or-..."   # OR this one for OpenRouter
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."   # OR this one for Claude

# %% [markdown]
# ## 3. Load the corpus
#
# 2,000 Glassdoor reviews: 1,000 pros + 1,000 cons, shuffled, 945 firms.
# Loaded directly from the GitHub raw URL, so the cell works in Colab
# without uploading anything. We run the live demo on a 20-row sample
# to keep cost and wait time low.

# %%
CORPUS_URL = "https://raw.githubusercontent.com/maifeng/lmsyz_genai_ie_rfs/main/data/glassdoor_culture_2000.csv"
corpus = pd.read_csv(CORPUS_URL)
demo = corpus.sample(20, random_state=42).reset_index(drop=True)
demo[["review_id", "text"]].head(5)

# %% [markdown]
# ## 4. Write the culture extraction prompt
#
# Adapting the extraction task from Li, Mai, Shen, Yang, Zhang (2026) to
# the shorter, bullet-list shape of Glassdoor reviews. We classify the
# culture type, judge tone, and pull out specific aspects of work each
# reviewer mentions, **each grounded in a quoted evidence phrase from
# the text**. The evidence field is the workshop's anti-hallucination
# guardrail: if the phrase is not literally in the input, the extraction
# is wrong, and you can see it.

# %%
CULTURE_PROMPT = """\
For each input row, classify the corporate culture described in the text and
extract grounded evidence. Be specific and use phrases from the text.

Steps:

1. culture_type: pick ONE of the six types (use "Miscellaneous" only if none
   fit):
   - Innovation / Adaptability: innovation, creativity, technology, agility,
     willingness to experiment, fast-moving, resilience to change.
   - Collaboration / People-Focused: teamwork, communication, employee
     well-being, diversity, inclusion, empowerment, talent.
   - Customer-Oriented: customer service, customer satisfaction, brand-driven,
     quality of product or service.
   - Integrity / Risk Management: ethical standards, transparency,
     accountability, compliance, financial prudence.
   - Performance-Oriented: high expectations, sales growth, hard work,
     efficiency, productivity, operational excellence.
   - Miscellaneous: non-specific, or does not fit any of the above.

2. tone: "positive", "negative", or "neutral" toward this culture.

3. aspects: a list of 1 to 6 specific aspects the reviewer discusses, each
   with an "aspect" label and an "evidence" phrase quoted from the text.
   Aspects can cover anything the reviewer talks about, positive or negative.

Return a JSON object with key "all_results" mapping to a list of per-row
results.

JSON output structure (placeholders shown; replace with actual values):

{
  "all_results": [
    {
      "input_id": <integer review_id>,
      "culture_type": "<one of the six types>",
      "tone": "positive" / "negative" / "neutral",
      "aspects": [
        {"aspect": "<short lowercase label>", "evidence": "<phrase quoted from the text>"},
        {"aspect": "<short lowercase label>", "evidence": "<phrase quoted from the text>"}
      ]
    },
    {
      "input_id": <integer review_id>,
      "culture_type": "<one of the six types>",
      "tone": "positive" / "negative" / "neutral",
      "aspects": [
        {"aspect": "<short lowercase label>", "evidence": "<phrase quoted from the text>"}
      ]
    }
  ]
}

Notes:
- input_id: copy the review_id field verbatim (integer).
- aspect: lowercase, hyphenated, 1 to 3 words.
- evidence: quote or near-quote a phrase from the text. Do not paraphrase.
"""
# Note: the library parses the LLM response by looking for the top-level
# key "all_results". If you write your own prompt, keep this key name.

# %% [markdown]
# ## 5. Extract (20-row demo)
#
# Call `extract_df`. The model returns JSON; the library lands it in a
# DataFrame. Each completed row is persisted to the SQLite cache, so
# a crash loses nothing.
#
# Three knobs worth seeing once explicitly (rather than as defaults):
#
# - **`chunk_size=5`**: how many rows are packed into a single LLM call.
#   Higher chunks are cheaper (one prompt amortized across N rows) but
#   risk truncation and JSON-parsing failures on very long inputs.
# - **`max_workers=20`**: how many chunks run in parallel via a thread
#   pool. Drop this if you hit the provider's rate limit (HTTP 429).
# - The combination of the two leads to 100x results faster than a naive loop over rows.
# - **`api_key=os.environ["OPENAI_API_KEY"]`**: passed explicitly so we
#   bypass the package's `pydantic-settings` singleton, which caches the
#   key at import time. If your environment variable changes after the
#   first import, only an explicit `api_key=` will pick up the new value.

# %%
from lmsyz_genai_ie_rfs import extract_df

out = extract_df(
    demo,
    prompt=CULTURE_PROMPT,
    backend="openai",
    model="gpt-4.1-mini",
    id_col="review_id",
    text_col="text",
    cache_path="glassdoor_demo.sqlite",
    chunk_size=5,  # How many rows to pack into each LLM call. Higher is faster but worse performance on long inputs.
    max_workers=20,  # How many chunks to run in parallel. Drop if you hit API rate limits.
    api_key=os.environ["OPENAI_API_KEY"],
)

# %%
out

# %% [markdown]
# The output is a DataFrame with one row per input. Columns depend on the
# prompt: here you will see `culture_type`, `tone`, and `aspects` (a list
# of `{aspect, evidence}` dicts). Because LLM output is stochastic, your
# exact results may differ slightly from a colleague's, but the
# `evidence` strings should always appear in the source text.

# %% [markdown]
# Explode the `aspects` list so each (review, aspect) pair gets its own
# row. This is the standard shape for downstream analysis: aspect
# frequencies, aspect × tone crosstabs, joining to firm metadata, etc.

# %%
flat = out.explode("aspects", ignore_index=True)
flat = pd.concat(
    [flat.drop(columns="aspects"), pd.json_normalize(flat["aspects"])], axis=1
)
flat.head(10)

# %%
out.to_csv("glassdoor_extraction_20.csv", index=False)
flat.to_csv("glassdoor_aspects_20.csv", index=False)
print(
    "Saved nested form to glassdoor_extraction_20.csv and flat form to glassdoor_aspects_20.csv"
)

# %% [markdown]
# ## 6. (Optional) Enforce a JSON schema
#
# Pass a dict or a path to a JSON file to `schema=`. The library uses it as
# OpenAI `response_format` or Anthropic `tool_use` input_schema. The same
# file works on both providers, so swapping `backend=` works without
# touching the schema.
#
# Strict mode constrains `culture_type` to the six allowed labels and
# `tone` to three values. The model can no longer drift to "Innovative
# Culture" or "Positive". This is the structured-outputs guarantee.

# %%
CULTURE_TYPES = [
    "Innovation / Adaptability",
    "Collaboration / People-Focused",
    "Customer-Oriented",
    "Integrity / Risk Management",
    "Performance-Oriented",
    "Miscellaneous",
]

my_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "all_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input_id": {"type": "integer"},
                            "culture_type": {"type": "string", "enum": CULTURE_TYPES},
                            "tone": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                            },
                            "aspects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "aspect": {"type": "string"},
                                        "evidence": {"type": "string"},
                                    },
                                    "required": ["aspect", "evidence"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["input_id", "culture_type", "tone", "aspects"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["all_results"],
            "additionalProperties": False,
        },
    },
}

out_strict = extract_df(
    demo,
    prompt=CULTURE_PROMPT,
    schema=my_schema,
    backend="openai",
    model="gpt-4.1-mini",
    id_col="review_id",
    text_col="text",
    cache_path="glassdoor_strict.sqlite",
    chunk_size=5,
    max_workers=20,
    api_key=os.environ["OPENAI_API_KEY"],
)
out_strict.head()

# %% [markdown]
# ## 7. Switch providers in one line
#
# Same prompt, same schema, same DataFrame. Just flip ``backend`` and pick a
# Claude model. Set ``ANTHROPIC_API_KEY`` first.

# %%
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

# out_claude = extract_df(
#     demo, prompt=CULTURE_PROMPT,
#     backend="anthropic", model="claude-haiku-4-5-20251001",
#     id_col="review_id", text_col="text",
#     cache_path="glassdoor_claude.sqlite",
#     chunk_size=5,
#     max_workers=20,
#     api_key=os.environ["ANTHROPIC_API_KEY"],
# )

# %% [markdown]
# ## 8. Use any OpenAI-compatible endpoint (OpenRouter, Together, vLLM, ...)
#
# The OpenAI SDK speaks to anything that exposes the chat-completions API:
# OpenRouter, Together AI, Groq, DeepSeek, a self-hosted vLLM, etc. Pass
# ``base_url=`` plus the right model identifier and ``extract_df`` works
# unchanged. This is how you run the same task on Llama, Mistral, Gemini,
# DeepSeek, or any other OpenRouter-hosted model without leaving the package.
#
# Caveat: OpenRouter forwards `response_format` to the underlying provider,
# but JSON-mode support varies by model.
#
# Get an OpenRouter key at https://openrouter.ai/keys, then:

# %%
out_ds = extract_df(
    demo,
    prompt=CULTURE_PROMPT,
    backend="openai",  # OpenAI-compatible client
    base_url="https://openrouter.ai/api/v1",  # point it at OpenRouter
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="deepseek/deepseek-v4-flash",  # any OpenRouter slug
    id_col="review_id",
    text_col="text",
    cache_path="glassdoor_ds.sqlite",
    chunk_size=5,
    max_workers=20,
)

# %%
out_ds.head()

# %% [markdown]
# ## 9. Resume on interrupt, and what happens when you edit the prompt
#
# Because `cache_path=` is required, your run is always resumable. If the
# notebook crashes after 60k of 100k rows, those 60k rows are already on
# disk. Rerun the same cell and it picks up where it left off.
#
# Each cached row is stamped with `sha256(prompt)[:16]`. If you edit the
# prompt and rerun, the new hash will not match the stored hash, so the
# library treats every row as not-yet-done and re-extracts. Because the
# table uses `INSERT OR REPLACE` keyed on `row_id`, **the rerun overwrites
# the old cached row**: the previous extraction (with the previous prompt
# hash) is destroyed when the new one lands. SQLite has no version history
# for it.
#
# The practical implications:
#
# - **Want to compare prompts side by side?** Use a different `cache_path=`
#   for each prompt version (e.g. `glassdoor_v1.sqlite` /
#   `glassdoor_v2.sqlite`). Each file keeps its own copy.
# - **Edited the prompt by accident** and want the old extractions back?
#   You cannot recover them from the cache. Restore from a backup of the
#   `.sqlite` file, or re-run with the original prompt.
# - **Want to reuse cached rows even though the prompt changed** (e.g.
#   you only fixed a typo)? Pass `ignore_prompt_hash=True`. The library
#   ignores the hash mismatch and treats the existing rows as valid.
# - **Want to wipe the cache and start fresh?** Pass `fresh=True`, or
#   delete the `.sqlite` file.


# %% [markdown]
# ## 10. Batch API (50% cheaper, up to 24h turnaround)
#
# For large jobs the OpenAI Batch API cuts the per-token cost in half in
# exchange for slower turnaround (up to 24 hours, usually much faster).
# The lifecycle has four steps. Everything is keyed by a string `job_id`,
# which is also the subdirectory name under `batch_jobs/`. Submitting
# twice with the same `job_id` resumes, skipping rows already done.
#
# Demo on the first 100 reviews to keep wait time short.
# Note that the Batch API has its limitation, for OpenAI:
#
#     - Maximum file size: typically up to ~200 MB per batch input file
#     - Maximum number of requests per file: up to 50,000 requests (i.e., 50k JSONL lines)
# %% [markdown]
# **Step 1: build the JSONL request files.**

# %%
from lmsyz_genai_ie_rfs import OpenAIBatchExtractor

batch = OpenAIBatchExtractor(api_key=os.environ["OPENAI_API_KEY"])

batch.create_batch_jsonl(
    dataframe=corpus.head(100),
    id_col="review_id",
    text_col="text",
    prompt=CULTURE_PROMPT,
    job_id="culture_v1",
    model_name="gpt-4o-mini",
    chunk_size=10000,
)

# %% [markdown]
# **Step 2: submit the JSONL files to OpenAI.**

# %%
batch.submit_batches(job_id="culture_v1")

# %% [markdown]
# **Step 3: poll status.** Returns immediately with current state. Pass
# `continuous=True` to block until all batches complete.

# %%
batch.check_batch_status(job_id="culture_v1")

# %% [markdown]
# **Step 4: retrieve results once complete.**

# %%
results = batch.retrieve_results_as_dataframe(job_id="culture_v1")
results.head()

# %% [markdown]
# ## 11. Your turn: extraction exercise (breakout)
#
# The pattern works for *any* structured extraction. Below is a 3-row
# DataFrame of business news. Your job: write a prompt that turns the
# input into the output shown.
#
# **Input**
#
# | id | text |
# |----|------|
# | 1 | Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025. |
# | 2 | Tesla acquired SolarCity in 2016 for $2.6 billion to enter the solar market. |
# | 3 | Pfizer's decision to spin off its consumer health unit was driven by activist pressure from Trian Partners. |
#
# **Target output**
#
# | input_id | entities                                                                                          | causal_triples                                   | sentiment |
# |----------|---------------------------------------------------------------------------------------------------|--------------------------------------------------|-----------|
# | 1        | `[{Tim Cook, PERSON}, {Apple, ORG}, {iPhone 17, PRODUCT}, {WWDC, EVENT}, {June 2025, DATE}]`      | `[]`                                             | neutral   |
# | 2        | `[{Tesla, ORG}, {SolarCity, ORG}, {2016, DATE}, {$2.6 billion, MONEY}]`                           | `[]`                                             | neutral   |
# | 3        | `[{Pfizer, ORG}, {Trian Partners, ORG}]`                                                          | `[["activist pressure", "drove", "spin-off"]]`   | neutral   |
#
# Fill in `MY_PROMPT` and run. Compare with your breakout group.

# %%
news = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "text": [
            "Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025.",
            "Tesla acquired SolarCity in 2016 for $2.6 billion to enter the solar market.",
            "Pfizer's decision to spin off its consumer health unit was driven by activist pressure from Trian Partners.",
        ],
    }
)

MY_PROMPT = """\
For each input row, ...   # TODO: write your prompt here

Return a JSON object with key "all_results" whose value is a list of
per-row results.
"""

# %%
my_out = extract_df(
    news,
    prompt=MY_PROMPT,
    backend="openai",
    model="gpt-4.1-mini",
    id_col="id",
    text_col="text",
    cache_path="my_extraction.sqlite",
    chunk_size=5,
    max_workers=20,
    api_key=os.environ["OPENAI_API_KEY"],
)
my_out

# %% [markdown]
# ## 12. Citation
#
# If you use this package in research, please cite:
#
# ```
# Li, Kai, Feng Mai, Rui Shen, Chelsea Yang, and Tengfei Zhang. 2026.
# "Dissecting Corporate Culture Using Generative AI."
# Review of Financial Studies 39(1):253-296.
# https://doi.org/10.1093/rfs/hhaf081
# ```

# %%
import lmsyz_genai_ie_rfs

print(lmsyz_genai_ie_rfs.__paper__)
