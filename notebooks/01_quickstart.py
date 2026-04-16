# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
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
# **Corpus**: 2,000 Glassdoor "pros" reviews about corporate culture,
# sampled from the RFS 2026 validation dataset. The same corpus is used
# across all three workshop notebooks for comparability.

# %% [markdown]
# ## 1. Install and download the corpus

# %%
# !pip install -q lmsyz_genai_ie_rfs    # uncomment and run in Colab

# %%
# Download the shared workshop corpus (uncomment in Colab):
# !wget -q https://raw.githubusercontent.com/maifeng/culture-llm-workshop/main/data/glassdoor_culture_2000.csv

# %% [markdown]
# ## 2. Set your API key
#
# The library reads `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) from the
# environment. **You must uncomment one of the lines below and paste your
# key**, otherwise the extraction cell will fail with an authentication error.

# %%
import os

# os.environ["OPENAI_API_KEY"] = "sk-..."          # uncomment and fill in
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."   # OR this one for Claude

# %% [markdown]
# ## 3. Load the corpus
#
# 2,000 Glassdoor "pros" reviews filtered for culture-related content
# (2+ culture keywords, 200-1000 chars, 970 unique firms, 2016-2022).
# We run the live demo on a 20-row sample to keep API costs and wait
# times low; Section 9 shows how to run on the full corpus.

# %%
import pandas as pd

CORPUS_PATH = "glassdoor_culture_2000.csv"
if not os.path.exists(CORPUS_PATH):
    CORPUS_PATH = "../../../data/glassdoor_culture_2000.csv"  # local dev

corpus = pd.read_csv(CORPUS_PATH)
print(f"Full corpus: {len(corpus)} reviews, {corpus['firm_id'].nunique()} firms")

demo = corpus.sample(20, random_state=42).reset_index(drop=True)
demo[["review_id", "text"]].head(5)

# %% [markdown]
# ## 4. Write the culture extraction prompt
#
# This is the extraction task from Li, Mai, Shen, Yang, Zhang (2026).
# Given a Glassdoor review, classify the culture type, extract causes
# and consequences, and identify causal triples.

# %%
CULTURE_PROMPT = """\
For each input row, classify the corporate culture described in the text.

Extract:
- input_id: copy the review_id verbatim.
- culture_type: one of "Collaboration / People-Focused", "Customer-Oriented",
  "Innovation / Adaptability", "Integrity / Risk Management",
  "Performance-Oriented", or "Miscellaneous".
- tone: "positive", "negative", or "neutral".
- causes: list of strings describing what drives this culture.
- consequences: list of strings describing the effects of this culture.
- causal_triples: list of [cause, relation, effect] triples if explicit
  causation is stated.

Return a JSON object with key "all_results" whose value is the list of
per-row objects.
"""
# Note: the library parses the LLM response by looking for the top-level
# key "all_results". If you write your own prompt, keep this key name.

# %% [markdown]
# ## 5. Extract (20-row demo)
#
# Call `extract_df`. The model returns JSON; the library lands it in a
# DataFrame. Each completed row is persisted to the SQLite cache, so
# a crash loses nothing.

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
)
out

# %% [markdown]
# The output is a DataFrame with one row per input. Columns depend on the
# prompt: here you will see `culture_type`, `tone`, `causes`,
# `consequences`, and `causal_triples`. Because LLM output is stochastic,
# your exact results may differ slightly from a colleague's.

# %%
out.to_csv("glassdoor_extraction_20.csv", index=False)
print("Saved to glassdoor_extraction_20.csv")

# %% [markdown]
# ## 6. (Optional) Enforce a JSON schema
#
# Pass a dict or a path to a JSON file to `schema=`. The library uses it as
# OpenAI `response_format` or Anthropic `tool_use` input_schema. The same
# file works on both providers.

# %%
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
                            "culture_type": {"type": "string"},
                            "tone": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        },
                        "required": ["input_id", "culture_type", "tone"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["all_results"],
            "additionalProperties": False,
        },
    },
}

# Uncomment to try:
# out_strict = extract_df(
#     demo, prompt=CULTURE_PROMPT, schema=my_schema,
#     backend="openai", model="gpt-4.1-mini",
#     id_col="review_id", text_col="text",
#     cache_path="glassdoor_strict.sqlite",
# )

# %% [markdown]
# ## 7. Switch providers in one line

# %%
# out_claude = extract_df(
#     demo, prompt=CULTURE_PROMPT,
#     backend="anthropic", model="claude-haiku-4-5-20251001",
#     id_col="review_id", text_col="text",
#     cache_path="glassdoor_claude.sqlite",
# )

# %% [markdown]
# ## 8. Resume on interrupt is automatic
#
# Because `cache_path=` is required, your run is always resumable. If the
# notebook crashes after 60k of 100k rows, those 60k rows are already on disk.
# Rerun the same cell and it picks up where it left off.
#
# Changing the prompt automatically invalidates affected rows, so you cannot
# accidentally reuse stale results after editing the prompt.
#
# Force a full re-run:  ``fresh=True``.
# Reuse rows even when the prompt changed: ``ignore_prompt_hash=True``.

# %% [markdown]
# ## 9. Run on the full corpus (2,000 reviews)
#
# The 20-row demo above costs a fraction of a cent. Running all 2,000
# reviews costs roughly **$0.50-$1.00** on `gpt-4.1-mini` at real-time
# prices, or half that via the Batch API. The cache means you only pay once.

# %%
# Uncomment to run on the full corpus:
# full_out = extract_df(
#     corpus,
#     prompt=CULTURE_PROMPT,
#     backend="openai",
#     model="gpt-4.1-mini",
#     id_col="review_id",
#     text_col="text",
#     cache_path="glassdoor_full.sqlite",
# )
# full_out.to_csv("glassdoor_extraction_full.csv", index=False)

# %% [markdown]
# ## 10. Batch API (50% cheaper, up to 24h turnaround)
#
# For large jobs (thousands of rows), the OpenAI Batch API cuts cost in half.
# Submit a JSONL file, poll for completion, retrieve results.

# %%
# from lmsyz_genai_ie_rfs import OpenAIBatchExtractor
#
# batch = OpenAIBatchExtractor(model="gpt-4.1-mini")
# jsonl_path = batch.create_batch_jsonl(
#     corpus, prompt=CULTURE_PROMPT,
#     id_col="review_id", text_col="text",
#     output_path="glassdoor_batch.jsonl",
# )
# batch_id = batch.submit_batches([jsonl_path])
# # Poll later:
# # status = batch.check_batch_status(batch_id)
# # results = batch.retrieve_results_as_dataframe(batch_id)

# %% [markdown]
# ## 11. Citation
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

# %% [markdown]
# ## 12. Related packages
#
# This workshop covers three tools on the **same 2,000 Glassdoor reviews**.
# Pick the one that fits your research question:
#
# | Package | Best for | Runtime |
# |---|---|---|
# | **`lmsyz_genai_ie_rfs`** (this notebook) | Structured extraction: culture type, causes, consequences, causal triples | Requires an LLM API key |
# | **`spar_measure`** | Scoring short texts on a custom semantic scale (e.g., CVF dimensions) | Local CPU/GPU, no API key |
# | **`lmsy_w2v_rfs`** | Historical, deterministic 5-dimension culture scores from word2vec | Local CPU, no API key |
