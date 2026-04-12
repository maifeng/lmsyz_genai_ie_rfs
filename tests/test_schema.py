"""Tests for schema.py: Pydantic validation roundtrips.

No API calls required. Validates that CultureRow and CultureBatch accept
valid data, reject invalid data, and serialise/deserialise correctly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from genai_batch_ie_rfs.schema import CultureBatch, CultureRow


class TestCultureRow:
    """Pydantic roundtrip tests for CultureRow."""

    def test_valid_row_constructs(self) -> None:
        """A fully valid dict should produce a CultureRow without errors."""
        row = CultureRow(
            input_id="doc_001",
            culture_type="innovation_adaptability",
            tone="positive",
            confidence=0.92,
        )
        assert row.input_id == "doc_001"
        assert row.culture_type == "innovation_adaptability"
        assert row.tone == "positive"
        assert row.confidence == pytest.approx(0.92)

    def test_all_culture_types_valid(self) -> None:
        """Every literal culture type should be accepted."""
        valid_types = [
            "collaboration_people",
            "customer_oriented",
            "innovation_adaptability",
            "integrity_risk",
            "performance_oriented",
            "miscellaneous",
        ]
        for ct in valid_types:
            row = CultureRow(input_id="x", culture_type=ct, tone="neutral", confidence=0.5)
            assert row.culture_type == ct

    def test_invalid_culture_type_raises(self) -> None:
        """An unrecognized culture type should raise ValidationError."""
        with pytest.raises(ValidationError):
            CultureRow(
                input_id="doc_001",
                culture_type="unknown_type",  # not in the Literal
                tone="neutral",
                confidence=0.5,
            )

    def test_confidence_below_zero_raises(self) -> None:
        """Confidence below 0.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            CultureRow(
                input_id="doc_001",
                culture_type="miscellaneous",
                tone="neutral",
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self) -> None:
        """Confidence above 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            CultureRow(
                input_id="doc_001",
                culture_type="miscellaneous",
                tone="neutral",
                confidence=1.01,
            )

    def test_model_dump_and_reconstruct(self) -> None:
        """model_dump() followed by CultureRow(**dict) should be a lossless roundtrip."""
        original = CultureRow(
            input_id="abc",
            culture_type="performance_oriented",
            tone="negative",
            confidence=0.75,
        )
        reconstructed = CultureRow(**original.model_dump())
        assert reconstructed == original

    def test_model_json_roundtrip(self) -> None:
        """JSON serialization and deserialization should be lossless."""
        original = CultureRow(
            input_id="json_test",
            culture_type="integrity_risk",
            tone="cautious",
            confidence=0.6,
        )
        json_str = original.model_dump_json()
        reconstructed = CultureRow.model_validate_json(json_str)
        assert reconstructed == original


class TestCultureBatch:
    """Pydantic roundtrip tests for CultureBatch."""

    def test_empty_batch(self) -> None:
        """CultureBatch with no results should have an empty list."""
        batch = CultureBatch(all_results=[])
        assert batch.all_results == []

    def test_batch_with_rows(self) -> None:
        """CultureBatch should contain and validate nested CultureRow instances."""
        rows = [
            {"input_id": "1", "culture_type": "collaboration_people", "tone": "warm", "confidence": 0.8},
            {"input_id": "2", "culture_type": "customer_oriented", "tone": "service", "confidence": 0.9},
        ]
        batch = CultureBatch(all_results=rows)
        assert len(batch.all_results) == 2
        assert batch.all_results[0].input_id == "1"

    def test_nested_validation_failure_propagates(self) -> None:
        """An invalid nested CultureRow should raise ValidationError on the batch."""
        with pytest.raises(ValidationError):
            CultureBatch(
                all_results=[
                    {"input_id": "1", "culture_type": "bad_type", "tone": "x", "confidence": 0.5}
                ]
            )
