"""Live-API integration tests hitting real OpenAI and Anthropic endpoints.

All artifacts (JSONL inputs, submission manifests, raw results, output CSVs,
SQLite caches) persist to ``test_artifacts/<test_name>/`` under the repo
root so you can inspect them after a run.

Run:
    pytest tests/test_live_api.py -m live -v                    # all live
    pytest tests/test_live_api.py -m "live and not slow" -v     # concurrent only
    pytest tests/test_live_api.py -m "live and slow" -v         # batch only

Fixtures in ``tests/data/``:
    culture_segments_20.csv       - 20 analyst-report-style rows
    culture_extraction_prompt.txt - 4-field culture classification prompt
    culture_relation_prompt.txt   - RFS 2026 round-1 relation prompt
    culture_batch_schema.json     - Optional JSON schema file
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from lmsyz_genai_ie_rfs import (
    AnthropicBatchExtractor,
    OpenAIBatchExtractor,
    extract_df,
)
from lmsyz_genai_ie_rfs.settings import settings

DATA_DIR = Path(__file__).parent / "data"
ARTIFACTS_DIR = Path(__file__).parent.parent / "test_artifacts"
CSV_PATH = DATA_DIR / "culture_segments_20.csv"
PROMPT_PATH = DATA_DIR / "culture_extraction_prompt.txt"
RELATION_PROMPT_PATH = DATA_DIR / "culture_relation_prompt.txt"
SCHEMA_PATH = DATA_DIR / "culture_batch_schema.json"

HAS_OPENAI_KEY = settings.openai_api_key is not None
HAS_ANTHROPIC_KEY = settings.anthropic_api_key is not None

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

ALLOWED_CULTURE_TYPES = {
    "collaboration_people", "customer_oriented", "innovation_adaptability",
    "integrity_risk", "performance_oriented", "miscellaneous",
}
ALLOWED_TONES = {"positive", "neutral", "negative"}
EXPECTED_IDS = {f"seg_{i:03d}" for i in range(1, 21)}


@pytest.fixture(scope="module")
def culture_df() -> pd.DataFrame:
    """Load the 20-row culture DataFrame."""
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 20
    return df


@pytest.fixture(scope="module")
def culture_prompt() -> str:
    """System prompt for the 4-field culture classification task."""
    return PROMPT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def relation_prompt() -> str:
    """System prompt for the RFS 2026 round-1 relation-extraction task."""
    return RELATION_PROMPT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def culture_schema_dict() -> dict:
    """OpenAI strict json_schema response_format for the 4-field task."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _artifact_dir(name: str, fresh: bool = True) -> Path:
    """Persistent per-test output directory under test_artifacts/."""
    d = ARTIFACTS_DIR / name
    if fresh and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _assert_culture_shape(df: pd.DataFrame) -> None:
    """Check the 4-field culture DataFrame shape (input_id, culture_type, tone, confidence)."""
    assert len(df) == 20
    assert {"input_id", "culture_type", "tone", "confidence"}.issubset(df.columns)
    assert set(df["culture_type"]).issubset(ALLOWED_CULTURE_TYPES)
    assert set(df["tone"]).issubset(ALLOWED_TONES)
    assert df["confidence"].between(0.0, 1.0).all()
    assert set(df["input_id"].astype(str)) == EXPECTED_IDS


def _assert_relation_shape(df: pd.DataFrame) -> None:
    """Smoke check on the relation-extraction DataFrame shape."""
    assert len(df) == 20
    expected = {
        "input_id", "identified_corporate_culture", "corporate_culture_type",
        "detailed_causal_analysis", "causes_of_culture", "outcomes_from_culture",
        "tone", "causal_graph_triples",
    }
    assert expected.issubset(df.columns)
    for col in ("causes_of_culture", "outcomes_from_culture", "causal_graph_triples"):
        assert df[col].apply(lambda x: isinstance(x, list)).all()
    assert set(df["input_id"].astype(str)) == EXPECTED_IDS


# ======================================================================
# Concurrent path: prompt only (no schema)
# ======================================================================


@pytest.mark.live
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_concurrent_no_schema(
    culture_df: pd.DataFrame, culture_prompt: str
) -> None:
    """OpenAI, prompt-only (schema=None), free-form JSON object."""
    out_dir = _artifact_dir("openai_concurrent_no_schema")
    out = extract_df(
        culture_df, prompt=culture_prompt, schema=None,
        backend="openai", model=OPENAI_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4,
        cache_path=out_dir / "results.sqlite",
    )
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)


@pytest.mark.live
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_concurrent_no_schema(
    culture_df: pd.DataFrame, culture_prompt: str
) -> None:
    """Anthropic, prompt-only (schema=None), free-form text parsed as JSON."""
    out_dir = _artifact_dir("anthropic_concurrent_no_schema")
    prompt_json_only = (
        culture_prompt
        + "\n\nRespond with ONLY a JSON object, no preamble, no markdown fences."
    )
    out = extract_df(
        culture_df, prompt=prompt_json_only, schema=None,
        backend="anthropic", model=ANTHROPIC_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4,
        cache_path=out_dir / "results.sqlite",
    )
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)


# ======================================================================
# Concurrent path: WITH JSON schema file (OpenAI json_schema / Anthropic tool_use)
# ======================================================================


@pytest.mark.live
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_concurrent_with_schema_file(
    culture_df: pd.DataFrame, culture_prompt: str
) -> None:
    """OpenAI, strict JSON schema loaded from a .json file (path passed directly)."""
    out_dir = _artifact_dir("openai_concurrent_with_schema_file")
    out = extract_df(
        culture_df, prompt=culture_prompt, schema=SCHEMA_PATH,
        backend="openai", model=OPENAI_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4,
        cache_path=out_dir / "results.sqlite",
    )
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)


@pytest.mark.live
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_concurrent_with_schema_file(
    culture_df: pd.DataFrame, culture_prompt: str
) -> None:
    """Anthropic, same JSON schema file, used as forced tool_use input_schema."""
    out_dir = _artifact_dir("anthropic_concurrent_with_schema_file")
    out = extract_df(
        culture_df, prompt=culture_prompt, schema=SCHEMA_PATH,
        backend="anthropic", model=ANTHROPIC_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4,
        cache_path=out_dir / "results.sqlite",
    )
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)


# ======================================================================
# Concurrent path: relation extraction (RFS 2026 round-1 prompt)
# Persists both output CSV, JSONL, and SQLite cache.
# ======================================================================


@pytest.mark.live
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_relation_extraction(
    culture_df: pd.DataFrame, relation_prompt: str
) -> None:
    """OpenAI relation extraction: entities, causes, outcomes, causal triples."""
    out_dir = _artifact_dir("openai_relation_extraction")
    cache = out_dir / "relation_cache.sqlite"
    out = extract_df(
        culture_df, prompt=relation_prompt, schema=None,
        backend="openai", model=OPENAI_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4, cache_path=cache,
    )
    out.to_csv(out_dir / "output.csv", index=False)
    out.to_json(out_dir / "output.jsonl", orient="records", lines=True)
    _assert_relation_shape(out)


@pytest.mark.live
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_relation_extraction(
    culture_df: pd.DataFrame, relation_prompt: str
) -> None:
    """Anthropic relation extraction (same prompt, no schema)."""
    out_dir = _artifact_dir("anthropic_relation_extraction")
    cache = out_dir / "relation_cache.sqlite"
    prompt_json_only = (
        relation_prompt
        + "\n\nRespond with ONLY a JSON object, no preamble, no markdown fences."
    )
    out = extract_df(
        culture_df, prompt=prompt_json_only, schema=None,
        backend="anthropic", model=ANTHROPIC_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4, cache_path=cache,
    )
    out.to_csv(out_dir / "output.csv", index=False)
    out.to_json(out_dir / "output.jsonl", orient="records", lines=True)
    _assert_relation_shape(out)


# ======================================================================
# Cache resume test
# ======================================================================


@pytest.mark.live
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_resume_via_sqlite_cache(
    culture_df: pd.DataFrame, culture_prompt: str
) -> None:
    """First run processes 20 rows; second run with same cache is a fast no-op."""
    out_dir = _artifact_dir("openai_resume_cache")
    cache = out_dir / "culture_cache.sqlite"

    t0 = time.monotonic()
    first = extract_df(
        culture_df, prompt=culture_prompt, schema=SCHEMA_PATH,
        backend="openai", model=OPENAI_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4, cache_path=cache,
    )
    t1 = time.monotonic() - t0
    first.to_csv(out_dir / "output_first.csv", index=False)
    assert len(first) == 20

    t0 = time.monotonic()
    second = extract_df(
        culture_df, prompt=culture_prompt, schema=SCHEMA_PATH,
        backend="openai", model=OPENAI_MODEL,
        id_col="segment_id", text_col="text",
        chunk_size=5, max_workers=4, cache_path=cache,
    )
    t2 = time.monotonic() - t0
    second.to_csv(out_dir / "output_second.csv", index=False)
    assert len(second) == 20
    assert t2 < t1 / 3, f"Cache hit should be fast. first={t1:.2f}s, second={t2:.2f}s"


# ======================================================================
# Batch path: OpenAI
# ======================================================================


def _poll_openai_batch(ext: OpenAIBatchExtractor, job_id: str, max_wait_s: int = 1800) -> None:
    """Block until all OpenAI batches for this job complete or timeout."""
    output_dir = ext.batch_root_dir / job_id / "batch_output"
    start = time.monotonic()
    while time.monotonic() - start < max_wait_s:
        ext.check_batch_status(job_id, continuous=False)
        subs = list(output_dir.glob("submission_*.json"))
        results = list(output_dir.glob("batch_result_*.jsonl"))
        errors = list(output_dir.glob("batch_error_*.txt"))
        if subs and len(results) + len(errors) == len(subs):
            return
        time.sleep(20)
    raise TimeoutError(f"OpenAI batch {job_id} not complete after {max_wait_s}s.")


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_batch_with_schema(
    culture_df: pd.DataFrame, culture_prompt: str, culture_schema_dict: dict,
) -> None:
    """OpenAI Batch API with strict json_schema."""
    out_dir = _artifact_dir("openai_batch_with_schema")
    job_id = f"live-openai-schema-{int(time.time())}"
    ext = OpenAIBatchExtractor(batch_root_dir=str(out_dir))
    ext.create_batch_jsonl(
        dataframe=culture_df, id_col="segment_id", text_col="text",
        prompt=culture_prompt, job_id=job_id,
        model_name=OPENAI_MODEL, chunk_size=5,
        schema_dict=culture_schema_dict,
    )
    ext.submit_batches(job_id)
    _poll_openai_batch(ext, job_id)
    out = ext.retrieve_results_as_dataframe(job_id)
    assert out is not None
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_batch_without_schema(
    culture_df: pd.DataFrame, culture_prompt: str,
) -> None:
    """OpenAI Batch API with free-form json_object."""
    out_dir = _artifact_dir("openai_batch_without_schema")
    job_id = f"live-openai-noschema-{int(time.time())}"
    ext = OpenAIBatchExtractor(batch_root_dir=str(out_dir))
    ext.create_batch_jsonl(
        dataframe=culture_df, id_col="segment_id", text_col="text",
        prompt=culture_prompt, job_id=job_id,
        model_name=OPENAI_MODEL, chunk_size=5,
        schema_dict=None,
    )
    ext.submit_batches(job_id)
    _poll_openai_batch(ext, job_id)
    out = ext.retrieve_results_as_dataframe(job_id)
    assert out is not None
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)


# ======================================================================
# Batch path: Anthropic
# ======================================================================


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_batch_with_schema(
    culture_df: pd.DataFrame, culture_prompt: str, culture_schema_dict: dict,
) -> None:
    """Anthropic Message Batches with tool_use (schema provided)."""
    out_dir = _artifact_dir("anthropic_batch_with_schema")
    job_id = f"live-anthropic-schema-{int(time.time())}"
    ext = AnthropicBatchExtractor(batch_root_dir=str(out_dir))
    ext.create_batch_requests(
        dataframe=culture_df, id_col="segment_id", text_col="text",
        prompt=culture_prompt, job_id=job_id,
        model_name=ANTHROPIC_MODEL, chunk_size=5,
        schema_dict=culture_schema_dict["json_schema"]["schema"],
    )
    ext.submit_batch(job_id)
    status = ext.check_batch_status(job_id, continuous=True, interval=20, timeout=1800)
    assert status == "ended"
    out = ext.retrieve_results_as_dataframe(job_id)
    assert out is not None
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_batch_without_schema(
    culture_df: pd.DataFrame, culture_prompt: str,
) -> None:
    """Anthropic Message Batches with free-form text (no tool)."""
    out_dir = _artifact_dir("anthropic_batch_without_schema")
    job_id = f"live-anthropic-noschema-{int(time.time())}"
    ext = AnthropicBatchExtractor(batch_root_dir=str(out_dir))
    prompt_json_only = (
        culture_prompt
        + "\n\nRespond with ONLY a JSON object, no preamble, no markdown fences."
    )
    ext.create_batch_requests(
        dataframe=culture_df, id_col="segment_id", text_col="text",
        prompt=prompt_json_only, job_id=job_id,
        model_name=ANTHROPIC_MODEL, chunk_size=5,
        schema_dict=None,
    )
    ext.submit_batch(job_id)
    status = ext.check_batch_status(job_id, continuous=True, interval=20, timeout=1800)
    assert status == "ended"
    out = ext.retrieve_results_as_dataframe(job_id)
    assert out is not None
    out.to_csv(out_dir / "output.csv", index=False)
    _assert_culture_shape(out)
