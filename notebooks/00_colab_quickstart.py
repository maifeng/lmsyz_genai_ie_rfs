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
# # lmsyz_genai_ie_rfs: 60-second Colab quickstart
#
# Run a prompt over a 3-row pandas DataFrame and get a structured DataFrame
# back. Companion code to Li, Mai, Shen, Yang, Zhang (2026), *RFS*
# 39(1):253–296.
#
# For the full workshop notebook (Glassdoor corpus, schema, batch API,
# OpenRouter), see `notebooks/01_quickstart.ipynb` in this repo.

# %% [markdown]
# ## 1. Install

# %%
# !pip install -q lmsyz_genai_ie_rfs==0.1.0a3

# %% [markdown]
# ## 2. Set your OpenAI key

# %%
import os

# os.environ["OPENAI_API_KEY"] = "sk-..."   # uncomment and paste

# %% [markdown]
# ## 3. Build the input DataFrame

# %%
import pandas as pd

df = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "text": [
            "Apple CEO Tim Cook announced the iPhone 17 at WWDC in June 2025.",
            "Tesla's $2.6 billion acquisition of SolarCity in 2016 enabled its entry into the solar market.",
            "Pfizer's decision to spin off its consumer health unit was driven by activist pressure from Trian Partners.",
        ],
    }
)
df

# %% [markdown]
# ## 4. Write the extraction prompt

# %%
prompt = """
For each input row, extract:
- input_id: copy the id verbatim.
- entities: list every named entity, each as {"name": ..., "type": "PERSON|ORG|PRODUCT|DATE|MONEY|EVENT"}.
- causal_triples: explicit cause-effect relations as ["cause", "relation", "effect"]; empty list if none.
- sentiment: "positive", "neutral", or "negative".

Return a JSON object: {"all_results": [<one object per row>]}.
"""

# %% [markdown]
# ## 5. Run the extraction

# %%
from lmsyz_genai_ie_rfs import extract_df

out = extract_df(
    df,
    prompt=prompt,
    backend="openai",
    model="gpt-4.1-mini",
    cache_path="demo.sqlite",
    chunk_size=5,
    max_workers=20,
    id_col="id",
    text_col="text",
    api_key=os.environ["OPENAI_API_KEY"],
)
out

# %% [markdown]
# ## 6. Explode the entities column
#
# Each row's `entities` field is a list of `{name, type}` dicts. Use
# `df.explode` to turn each list into multiple rows (one entity per row).

# %%
flat = out.explode("entities", ignore_index=True)
flat

# %% [markdown]
# Then `pd.json_normalize` flattens each dict's keys into separate columns.

# %%
flat = pd.concat(
    [flat.drop(columns="entities"), pd.json_normalize(flat["entities"])],
    axis=1,
)
flat
