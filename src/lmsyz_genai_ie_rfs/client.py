"""Single-function concurrent LLM extraction over a pandas DataFrame.

Public entry point: ``extract_df``. It chunks the DataFrame, sends each
chunk to OpenAI or Anthropic in parallel, optionally persists results to
a SQLite cache, and returns a DataFrame.

The ``schema`` argument is optional. Pass ``None`` for free-form JSON;
pass a JSON schema (dict or path) for strict validation on both
providers. OpenAI uses ``response_format={"type": "json_schema", ...}``.
Anthropic uses the same schema as the ``input_schema`` of a forced tool.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from lmsyz_genai_ie_rfs.dataframe import DataFrameIterator, SqliteCache, compute_prompt_hash
from lmsyz_genai_ie_rfs.retry import retry_api_call
from lmsyz_genai_ie_rfs.settings import settings

log = logging.getLogger(__name__)

_TEMP_ONE_PREFIXES = ("o1", "o3")
_TEMP_ONE_SUBSTRINGS = ("gpt-5",)

SchemaInput = None | dict[str, Any] | str | Path


def _requires_temp_one(model_name: str) -> bool:
    """Return True for model families that only accept temperature=1 (o1, o3, gpt-5)."""
    lower = model_name.lower()
    return lower.startswith(_TEMP_ONE_PREFIXES) or any(
        s in lower for s in _TEMP_ONE_SUBSTRINGS
    )


def _load_schema(schema: SchemaInput) -> dict[str, Any] | None:
    """Load a schema from a path or return the dict (or None) unchanged.

    Args:
        schema: ``None``, a ``dict``, or a path to a JSON file.

    Returns:
        The schema as a dict, or None.
    """
    if schema is None:
        return None
    if isinstance(schema, (str, Path)):
        return json.loads(Path(schema).read_text())
    if isinstance(schema, dict):
        return schema
    raise TypeError(f"schema must be None, dict, str, or Path (got {type(schema)}).")


@retry_api_call
def _call_openai(
    client: Any,
    system_prompt: str,
    user_message: str,
    response_format: dict[str, Any] | None,
    model: str,
) -> list[dict[str, Any]]:
    """Send one OpenAI chat completion and return a list of row dicts.

    Args:
        client: An ``openai.OpenAI`` instance.
        system_prompt: Full system prompt text.
        user_message: JSON-serialised chunk of rows.
        response_format: Either a ``json_schema`` dict, a ``json_object`` dict,
            or ``None`` (interpreted as ``json_object``).
        model: OpenAI model identifier.

    Returns:
        List of row dicts parsed from ``all_results`` (or ``results``) in the
        response JSON.
    """
    temp = 1.0 if _requires_temp_one(model) else 0.0
    if response_format is None:
        rf: dict[str, Any] = {"type": "json_object"}
    elif response_format.get("type") in ("json_object", "json_schema"):
        rf = response_format
    else:
        # Caller passed the inner schema. Wrap it minimally so OpenAI accepts it.
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "strict": True, "schema": response_format},
        }
    resp = client.chat.completions.create(
        model=model, temperature=temp,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=rf,
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    rows = payload.get("all_results") or payload.get("results") or []
    return [rows] if isinstance(rows, dict) else list(rows)


@retry_api_call
def _call_anthropic(
    client: Any,
    system_prompt: str,
    user_message: str,
    input_schema: dict[str, Any] | None,
    model: str,
) -> list[dict[str, Any]]:
    """Send one Anthropic message and return a list of row dicts.

    If ``input_schema`` is provided, uses forced ``tool_use`` with that
    JSON schema as the tool's ``input_schema``. Otherwise returns free-form
    text and parses JSON tolerantly (strips ``` fences, finds outermost ``{}``).

    The system prompt always carries ``cache_control={"type": "ephemeral"}``
    so long reused prompts hit Anthropic's prompt cache.

    Args:
        client: An ``anthropic.Anthropic`` instance.
        system_prompt: Full system prompt text.
        user_message: JSON-serialised chunk of rows.
        input_schema: JSON schema dict for the response (must have
            ``all_results`` at the top level). ``None`` for free-form text.
        model: Anthropic model identifier.

    Returns:
        List of row dicts.
    """
    system_block = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
    ]
    if input_schema is None:
        resp = client.messages.create(
            model=model, max_tokens=32000, system=system_block,
            messages=[{"role": "user", "content": user_message}],
        )
        text_parts = [b.text for b in resp.content if b.type == "text"]
        if not text_parts:
            raise ValueError(f"No text block in Anthropic response: {resp.content!r}")
        raw = "".join(text_parts).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in Anthropic response: {raw[:500]}")
        payload = json.loads(raw[start : end + 1])
        rows = payload.get("all_results") or payload.get("results") or []
        return [rows] if isinstance(rows, dict) else list(rows)

    tool_name = "extract_results"
    resp = client.messages.create(
        model=model, max_tokens=32000, system=system_block,
        tools=[{
            "name": tool_name,
            "description": "Return structured extraction results for all input rows.",
            "input_schema": input_schema,
        }],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user_message}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            payload = block.input or {}
            rows = payload.get("all_results") or payload.get("results") or []
            return [rows] if isinstance(rows, dict) else list(rows)
    raise ValueError(f"No tool_use block in Anthropic response: {resp.content!r}")


def _make_client(backend: str, api_key: str | None, base_url: str | None) -> Any:
    """Build the provider SDK client (called once per ``extract_df`` invocation)."""
    if backend == "openai":
        import openai
        key = api_key or (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )
        return openai.OpenAI(api_key=key, base_url=base_url or settings.openai_base_url)
    if backend == "anthropic":
        import anthropic
        key = api_key or (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key else None
        )
        return anthropic.Anthropic(api_key=key)
    raise ValueError(f"Unknown backend {backend!r}. Use 'openai' or 'anthropic'.")


def extract_df(
    df: pd.DataFrame,
    *,
    prompt: str,
    cache_path: str | Path,
    model: str,
    schema: SchemaInput = None,
    backend: str = "openai",
    id_col: str = "id",
    text_col: str = "text",
    chunk_size: int = 5,
    max_workers: int = 20,
    fresh: bool = False,
    ignore_prompt_hash: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    client: Any = None,
) -> pd.DataFrame:
    """Run a prompt over each row of ``df`` concurrently and return a DataFrame.

    The ``schema`` argument is optional. When omitted, the prompt alone
    defines the output shape and the model returns free-form JSON. When
    supplied, it enforces structure on both providers: OpenAI via
    ``response_format={"type": "json_schema", ...}``, Anthropic via forced
    ``tool_use`` with the same schema.

    Args:
        df: Input DataFrame. Must contain ``id_col`` and ``text_col``.
        prompt: System prompt sent with every chunk.
        schema: Optional JSON schema. Accepts ``None``, a dict (full OpenAI
            wrapper, a response schema with ``all_results``, or a row schema
            that will be auto-wrapped), or a ``str``/``Path`` pointing to a
            JSON file containing any of the above.
        backend: ``"openai"`` or ``"anthropic"``.
        model: Model identifier (e.g., ``"gpt-4.1-mini"``).
        id_col: Column in ``df`` holding row identifiers.
        text_col: Column in ``df`` holding text content.
        chunk_size: Rows per LLM request.
        max_workers: ThreadPoolExecutor size.
        cache_path: Required path to a SQLite file. Every completed row is
            written to this file as it finishes, so an interrupted run
            resumes without re-spending tokens. Each row is stamped with a
            hash of the prompt that produced it; a later run with a
            different prompt re-executes those rows automatically (override
            with ``ignore_prompt_hash=True``). The file is never auto-deleted.
        fresh: If True, re-process every row regardless of cache contents.
        ignore_prompt_hash: If True, reuse cached rows even when the
            prompt has changed since they were written. Default False:
            changing the prompt invalidates the cache, which is usually
            what you want during iteration. Set True when resuming a run
            where you deliberately edited the prompt in a non-semantic
            way (e.g., typo fix).
        api_key: Override the API key from settings / env.
        base_url: Override the OpenAI base URL (for OpenRouter, Gemini compat).
        client: A pre-built SDK client. When given, ``backend``, ``api_key``,
            and ``base_url`` are only used to pick the call helper.

    Returns:
        DataFrame of result dicts, one row per input row. Rows whose chunks
        fail are logged and omitted; remaining chunks' results are returned.
    """
    if client is None:
        client = _make_client(backend, api_key, base_url)

    schema_dict = _load_schema(schema)
    if backend == "openai":
        call_fn = _call_openai
        call_args: dict[str, Any] = {"response_format": schema_dict}
    else:
        call_fn = _call_anthropic
        # Anthropic uses the inner JSON schema as a tool's input_schema. If the
        # user passed OpenAI's wrapper form, unwrap it; otherwise assume the
        # dict is already the inner schema.
        if schema_dict is None:
            input_schema = None
        elif schema_dict.get("type") == "json_schema" and "json_schema" in schema_dict:
            input_schema = schema_dict["json_schema"]["schema"]
        else:
            input_schema = schema_dict
        call_args = {"input_schema": input_schema}

    cache = SqliteCache(Path(cache_path))
    phash = compute_prompt_hash(prompt)
    # None means "any hash" (reuse cached rows even when prompt changed).
    lookup_hash = None if ignore_prompt_hash else phash

    working = df.copy()
    if not fresh:
        done = cache.all_ids(prompt_hash=lookup_hash)
        if done:
            before = len(working)
            working = working[~working[id_col].astype(str).isin(done)]
            log.info(
                "SqliteCache: skipping %d / %d rows (prompt_hash=%s).",
                before - len(working), before,
                "any" if ignore_prompt_hash else phash,
            )

    if working.empty:
        log.info("extract_df: all rows cached; returning from cache.")
        rows = [cache.get(str(rid), prompt_hash=lookup_hash) for rid in df[id_col].astype(str)]
        return pd.DataFrame([r for r in rows if r is not None])

    working = working.sample(frac=1, random_state=42).reset_index(drop=True)
    chunks = list(DataFrameIterator(working, id_col=id_col, text_col=text_col, chunk_size=chunk_size))

    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(call_fn, client, prompt, json.dumps(chunk), **call_args, model=model): chunk
            for chunk in chunks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extract_df"):
            try:
                rows = fut.result()
                all_results.extend(rows)
                for row in rows:
                    rid = str(row.get("input_id", ""))
                    if rid:
                        cache.put(rid, row, prompt_hash=phash)
            except Exception:
                log.exception("extract_df: chunk failed; results for this chunk skipped.")

    if not fresh:
        done_in_run = {str(r.get("input_id", "")) for r in all_results}
        for rid in df[id_col].astype(str):
            if rid not in done_in_run:
                cached = cache.get(rid, prompt_hash=lookup_hash)
                if cached is not None:
                    all_results.append(cached)

    return pd.DataFrame(all_results)
