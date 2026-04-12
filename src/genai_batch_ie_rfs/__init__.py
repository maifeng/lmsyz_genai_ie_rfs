"""genai_batch_ie_rfs: concurrent and batch LLM information extraction on pandas DataFrames.

Extracted from the research codebase for Li, Mai, Shen, Yang & Zhang (2026),
"Dissecting Corporate Culture Using Generative AI," RFS 39(1):253-296.
https://doi.org/10.1093/rfs/hhaf081
"""

from __future__ import annotations

from genai_batch_ie_rfs.client import AnthropicBackend, LLMClient, OpenAIBackend

__version__ = "0.1.0a1"

__paper__ = (
    "Li, Kai, Feng Mai, Rui Shen, Zilong Yang, and Xinlei Zhang. 2026. "
    "'Dissecting Corporate Culture Using Generative AI.' "
    "Review of Financial Studies 39(1):253-296. "
    "https://doi.org/10.1093/rfs/hhaf081"
)

__all__ = [
    "LLMClient",
    "OpenAIBackend",
    "AnthropicBackend",
    "__version__",
    "__paper__",
]
