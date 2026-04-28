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
# !pip install -q -U lmsyz_genai_ie_rfs

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

# %% [markdown]
# ## 7. (Optional) Draft a prompt from a one-line goal
#
# `draft_prompt(goal=...)` turns a plain-English description into a
# candidate prompt in the same house style as the one in section 4.
# The drafted prompt is printed below; read it and copy what you like.
# Different shape from the basic example: here we ask for a closed-enum
# business-event type and a 1-5 impact score.

# %%
from lmsyz_genai_ie_rfs import draft_prompt

draft_prompt(
    goal=(
        "From a short business news sentence, classify the event type as "
        "one of [acquisition, product_launch, partnership, leadership_change, "
        "lawsuit, financial_result, other] and rate the impact on the primary "
        "company on a 1-5 integer scale."
    ),
    api_key=os.environ["OPENAI_API_KEY"],
)
