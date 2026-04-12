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
# # genai_batch_ie_rfs: Quickstart
#
# Concurrent and batch LLM information extraction on a pandas DataFrame.
#
# Originally developed for:
# Li, Mai, Shen, Yang & Zhang (2026), "Dissecting Corporate Culture Using Generative AI,"
# *Review of Financial Studies* 39(1):253-296.
# https://doi.org/10.1093/rfs/hhaf081
#
# **What this notebook shows:**
# 1. Define a Pydantic output schema.
# 2. Build a tiny hard-coded DataFrame (no data download needed).
# 3. Call `LLMClient.classify_df` to extract structured fields from each row.
# 4. Inspect the typed results.

# %% [markdown]
# ## 1. Install (run once in Colab)

# %%
# Uncomment and run in Colab:
# !pip install genai_batch_ie_rfs

# %% [markdown]
# ## 2. Set your API key
#
# The library reads `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) from the environment.
# In Colab, use Secrets or set it directly:

# %%
import os

# os.environ["OPENAI_API_KEY"] = "sk-..."   # paste your key here
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

# %% [markdown]
# ## 3. Define an output schema
#
# Any Pydantic `BaseModel` works. Here we use the built-in `CultureRow` from the package.

# %%
from genai_batch_ie_rfs.schema import CultureRow

# Preview the schema fields:
print(CultureRow.model_json_schema())

# %% [markdown]
# ## 4. Build a sample DataFrame
#
# Five rows drawn from fictional analyst reports. No real data required.

# %%
import pandas as pd

sample_texts = [
    "Management strongly emphasizes integrity and transparent reporting to shareholders.",
    "The firm rewards top performers with significant bonuses and promotes a cut-throat culture.",
    "Engineers are encouraged to experiment and fail fast; innovation is the company's core identity.",
    "Customer satisfaction scores drive every product decision; clients always come first.",
    "Cross-functional teamwork and psychological safety are central to how the company operates.",
    "The company maintains strict risk controls and conservative credit standards.",
    "Speed of execution matters above all else; quarterly targets are non-negotiable.",
]

df = pd.DataFrame({
    "id": [f"seg_{i:03d}" for i in range(len(sample_texts))],
    "text": sample_texts,
})

print(df)

# %% [markdown]
# ## 5. Run classification
#
# `LLMClient.classify_df` sends each chunk to the LLM and returns a typed DataFrame.
#
# **Note:** The `classify_df` method currently raises `NotImplementedError`
# because the API call implementation is a stub. This cell is ready to run
# once the backend is wired up.

# %%
from genai_batch_ie_rfs import LLMClient

SYSTEM_PROMPT = (
    "You are a corporate-culture analyst. "
    "For each input row, identify the culture type and tone based on the text. "
    "Return a JSON object with key 'all_results' containing a list of results. "
    "Each result must have: input_id, culture_type (one of: collaboration_people, "
    "customer_oriented, innovation_adaptability, integrity_risk, performance_oriented, "
    "miscellaneous), tone (brief label), confidence (0.0-1.0)."
)

client = LLMClient(backend="openai", model="gpt-4.1-mini")

# Uncomment once classify_df is implemented:
# results = client.classify_df(
#     df=df,
#     schema=CultureRow,
#     prompt=SYSTEM_PROMPT,
#     id_col="id",
#     text_col="text",
#     chunk_size=3,
#     max_workers=4,
# )
# print(results)

# %% [markdown]
# ## 6. Inspect results (placeholder)
#
# Once `classify_df` is implemented, the results DataFrame will have columns
# matching `CultureRow`: `input_id`, `culture_type`, `tone`, `confidence`.

# %%
# Example of what results would look like:
import json

mock_results = [
    {"input_id": "seg_000", "culture_type": "integrity_risk", "tone": "formal", "confidence": 0.91},
    {"input_id": "seg_001", "culture_type": "performance_oriented", "tone": "competitive", "confidence": 0.87},
    {"input_id": "seg_002", "culture_type": "innovation_adaptability", "tone": "energetic", "confidence": 0.95},
    {"input_id": "seg_003", "culture_type": "customer_oriented", "tone": "service", "confidence": 0.89},
    {"input_id": "seg_004", "culture_type": "collaboration_people", "tone": "warm", "confidence": 0.88},
    {"input_id": "seg_005", "culture_type": "integrity_risk", "tone": "conservative", "confidence": 0.93},
    {"input_id": "seg_006", "culture_type": "performance_oriented", "tone": "urgent", "confidence": 0.84},
]

mock_df = pd.DataFrame(mock_results)
print(mock_df.to_string(index=False))

# %% [markdown]
# ## 7. Validate results with Pydantic
#
# You can validate the full result set using `CultureRow`:

# %%
validated = [CultureRow(**row) for row in mock_results]
for row in validated:
    print(f"{row.input_id}: {row.culture_type} ({row.tone}, conf={row.confidence:.2f})")

# %% [markdown]
# ## 8. Citation
#
# If you use this package in research, please cite:
#
# ```
# Li, Kai, Feng Mai, Rui Shen, Zilong Yang, and Xinlei Zhang. 2026.
# "Dissecting Corporate Culture Using Generative AI."
# Review of Financial Studies 39(1):253-296.
# https://doi.org/10.1093/rfs/hhaf081
# ```

# %%
import genai_batch_ie_rfs
print(genai_batch_ie_rfs.__paper__)
