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
# ## 9. Citation
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
