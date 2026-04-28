"""Prompt-based multi-thread and batch LLM extraction over pandas DataFrames.

Public API:
    extract_df                    - concurrent (threadpool) extraction
    OpenAIBatchExtractor          - OpenAI Batch API lifecycle helper
    AnthropicBatchExtractor       - Anthropic Message Batches lifecycle helper
    SqliteCache                   - small get/put/all_ids resume cache

The framework is domain-agnostic. Provide your own Pydantic schema (or
``schema=None`` for free-form JSON) and any prompt that describes the
output shape. Originally developed for Li, Mai, Shen, Yang & Zhang (2026),
"Dissecting Corporate Culture Using Generative AI," RFS 39(1):253-296.
https://doi.org/10.1093/rfs/hhaf081
"""

from __future__ import annotations

from lmsyz_genai_ie_rfs.anthropic_batch import AnthropicBatchExtractor
from lmsyz_genai_ie_rfs.batch import OpenAIBatchExtractor
from lmsyz_genai_ie_rfs.client import extract_df
from lmsyz_genai_ie_rfs.dataframe import SqliteCache

__version__ = "0.1.0a2"

__paper__ = (
    "Li, Kai, Feng Mai, Rui Shen, Chelsea Yang, and Tengfei Zhang. 2026. "
    "'Dissecting Corporate Culture Using Generative AI.' "
    "Review of Financial Studies 39(1):253-296. "
    "https://doi.org/10.1093/rfs/hhaf081"
)

__all__ = [
    "extract_df",
    "OpenAIBatchExtractor",
    "AnthropicBatchExtractor",
    "SqliteCache",
    "__version__",
    "__paper__",
]
