"""Schema helpers: Pydantic BaseModel definitions for structured LLM outputs.

This module provides example schemas used in tests, the README, and the Colab
quickstart notebook. Users should subclass BaseModel directly for custom tasks.

Input: structured JSON from the LLM response.
Output: validated Pydantic model instances.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Culture types derived from Li, Mai, Shen, Yang & Zhang (2026) RFS taxonomy.
CultureTypeLiteral = Literal[
    "collaboration_people",
    "customer_oriented",
    "innovation_adaptability",
    "integrity_risk",
    "performance_oriented",
    "miscellaneous",
]


class CultureRow(BaseModel):
    """Structured output schema for a single culture-classification result.

    This schema is used as the reference example throughout tests, the README,
    and the quickstart notebook. It mirrors the six-type taxonomy from
    Li, Mai, Shen, Yang & Zhang (2026).

    Attributes:
        input_id: Identifier of the source row. Matches the id_col passed to classify_df.
        culture_type: One of the six RFS 2026 culture categories.
        tone: Brief qualitative tone label, e.g. "positive", "neutral", "negative".
        confidence: Model's self-reported confidence, 0.0 to 1.0.
    """

    input_id: str = Field(description="ID of the source row.")
    culture_type: CultureTypeLiteral = Field(description="Primary culture category.")
    tone: str = Field(description="Tone of the culture signal.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence, 0-1.")


class CultureBatch(BaseModel):
    """Container for a batch of CultureRow results returned by the LLM.

    The LLM is asked to return a JSON object with an 'all_results' key whose
    value is a list of CultureRow objects. Using a typed container ensures
    Pydantic validates the entire batch response.

    Attributes:
        all_results: List of culture extraction results, one per input row in the chunk.
    """

    all_results: list[CultureRow] = Field(
        default_factory=list,
        description="List of extraction results, one per input row.",
    )
