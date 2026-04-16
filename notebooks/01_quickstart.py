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

# %% [markdown]
# ## 1. Install (run once in Colab)

# %%
# Uncomment and run in Colab:
# !pip install lmsyz_genai_ie_rfs

# %% [markdown]
# ## 2. Set your API key
#
# The library reads `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) from the environment.

# %%
import os

# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

# %% [markdown]
# ## 3. Build a DataFrame

# %%
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025.",
        "Tesla acquired SolarCity in 2016 for $2.6 billion to enter the solar market.",
        "Pfizer's decision to spin off its consumer health unit was driven by activist pressure from Trian Partners.",
    ],
})
print(df)

# %% [markdown]
# ## 4. Write the prompt
#
# The prompt describes the output. No Python classes required.

# %%
PROMPT = """
For each input row, extract:
- input_id: copy verbatim.
- entities: list of {"name": str, "type": "PERSON" | "ORG" | "PRODUCT" | "DATE" | "MONEY"}.
- events:   list of {"actor": str, "action": str, "object": str, "time": str | null}.
- causal_triples: list of ["cause", "relation", "effect"] if explicit causation is stated.
- sentiment: "positive" | "neutral" | "negative".

Return a JSON object with key "all_results" whose value is the list of per-row objects.
"""

# %% [markdown]
# ## 5. Extract
#
# Call `extract_df`. The model returns JSON; the library lands it in a DataFrame.

# %%
from lmsyz_genai_ie_rfs import extract_df

out = extract_df(
    df, prompt=PROMPT,
    backend="openai", model="gpt-4.1-mini",
    id_col="id", text_col="text",
    cache_path="quickstart_results.sqlite",   # required: every row is persisted here as it completes
)
print(out)

# %% [markdown]
# ## 6. (Optional) Enforce a JSON schema
#
# If you want the provider to reject malformed rows, point `schema=` at a JSON
# schema file. The same file works for both OpenAI (as `response_format`) and
# Anthropic (as a forced `tool_use` schema). No code changes needed.

# %%
# out_strict = extract_df(
#     df, prompt=PROMPT, schema="my_schema.json",
#     backend="openai", model="gpt-4.1-mini",
#     id_col="id", text_col="text",
# )

# %% [markdown]
# ## 7. Switch providers in one line

# %%
# out_claude = extract_df(
#     df, prompt=PROMPT,
#     backend="anthropic", model="claude-haiku-4-5-20251001",
#     id_col="id", text_col="text",
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
# ## 9. Case study: RFS 2026 culture pipeline in one cell
#
# This is the extraction task from Li, Mai, Shen, Yang, Zhang (2026).
# Given a chunk of analyst report text, classify the corporate culture type,
# extract causes and consequences, and identify causal triples.

# %%
culture_df = pd.DataFrame({
    "id": ["10233684", "10260257", "10302283"],
    "text": [
        "The new CEO has initiated a complete restructuring of the R&D division, "
        "replacing legacy processes with agile sprints and cross-functional pods. "
        "Engineers report higher autonomy but also describe a 'move fast' pressure "
        "that has shortened product review cycles from six months to six weeks.",

        "Following the data breach in Q2, the board mandated a company-wide overhaul "
        "of information security protocols. Compliance headcount doubled, and every "
        "product release now requires sign-off from the newly created Chief Risk "
        "Officer. Employees describe the environment as cautious but trustworthy.",

        "Management has publicly committed to a 'customer obsession' philosophy, "
        "tying 30% of executive compensation to Net Promoter Score. Support teams "
        "now operate 24/7, and the company has opened regional service centers in "
        "twelve cities. Analyst consensus is that retention metrics have improved.",
    ],
})

CULTURE_PROMPT = """\
For each input row, classify the corporate culture described in the text.

Extract:
- input_id: copy the id verbatim.
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

# %%
culture_out = extract_df(
    culture_df,
    prompt=CULTURE_PROMPT,
    backend="openai",
    model="gpt-4.1-mini",
    id_col="id",
    text_col="text",
    cache_path="culture_demo.sqlite",
)
culture_out

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
#     culture_df, prompt=CULTURE_PROMPT,
#     id_col="id", text_col="text",
#     output_path="culture_batch.jsonl",
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
