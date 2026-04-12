"""LLM provider abstraction: OpenAI and Anthropic backends with a unified LLMClient.

Input: a pandas DataFrame with id and text columns.
Output: a pandas DataFrame of Pydantic-validated extraction results.

The Anthropic backend uses cache_control on the system prompt to reduce costs
when the same long system prompt is reused across thousands of chunks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd
from pydantic import BaseModel

from genai_batch_ie_rfs.settings import settings

if TYPE_CHECKING:
    import anthropic
    import openai

log = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=type[BaseModel])


def _requires_temp_one(model_name: str) -> bool:
    """Return True if the model only accepts temperature=1.

    Covers o1, o3, and gpt-5 model families. Note: gpt-4o accepts temperature=0
    and is intentionally excluded here.

    Args:
        model_name: The model identifier string.

    Returns:
        True for o1/o3/gpt-5 models, False otherwise.
    """
    lower = model_name.lower()
    return (
        lower.startswith("o1")
        or lower.startswith("o3")
        or "gpt-5" in lower
    )


class _Backend(ABC):
    """Abstract base class for LLM provider backends.

    Subclasses implement _call to handle provider-specific API details.
    All public behaviour is shared through this interface.
    """

    @abstractmethod
    def _call(
        self,
        system_prompt: str,
        user_message: str,
        schema: type[BaseModel],
        model: str,
    ) -> list[dict[str, Any]]:
        """Send one request and return a list of extracted row dicts.

        Args:
            system_prompt: Full system prompt text.
            user_message: JSON-serialised chunk of input rows.
            schema: Pydantic model class describing one output row.
            model: Model identifier.

        Returns:
            List of dicts, each representing one extracted row.

        Raises:
            NotImplementedError: Until subclass implements this method.
        """
        raise NotImplementedError


class OpenAIBackend(_Backend):
    """OpenAI backend using chat completions with Pydantic structured outputs.

    Uses client.chat.completions.parse to enforce schema validation on every
    response. Retries are applied via the retry_api_call decorator.

    Attributes:
        client: An openai.OpenAI client instance.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialise the OpenAI backend.

        Args:
            api_key: OpenAI API key. Falls back to settings.openai_api_key.
            base_url: Optional custom base URL (e.g., OpenRouter).
        """
        import openai  # local import so anthropic-only users don't need openai

        resolved_key = api_key or (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )
        resolved_base = base_url or settings.openai_base_url
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
        )

    def _call(
        self,
        system_prompt: str,
        user_message: str,
        schema: type[BaseModel],
        model: str,
    ) -> list[dict[str, Any]]:
        """Call the OpenAI chat completions API and return parsed row dicts.

        Args:
            system_prompt: Full system prompt text.
            user_message: JSON-serialised chunk of input rows.
            schema: Pydantic model class for one output row.
            model: OpenAI model identifier.

        Returns:
            List of dicts extracted from the structured response.

        Raises:
            NotImplementedError: Implementation pending.

        Note:
            Implementation should call client.chat.completions.parse with
            a container model (e.g., CultureBatch) wrapping list[schema],
            then return .all_results as a list of dicts via model.model_dump().
        """
        raise NotImplementedError(
            "OpenAIBackend._call is not yet implemented. "
            "Wire client.chat.completions.parse with a container Pydantic model."
        )


class AnthropicBackend(_Backend):
    """Anthropic backend using tool-use and system-prompt caching.

    Uses cache_control on the system prompt block so that a long prompt
    is only tokenized once per cache window (5 min default). This reduces
    cost significantly when thousands of chunks share the same prompt.

    Attributes:
        client: An anthropic.Anthropic client instance.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the Anthropic backend.

        Args:
            api_key: Anthropic API key. Falls back to settings.anthropic_api_key.
        """
        import anthropic  # local import

        resolved_key = api_key or (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        )
        self.client: anthropic.Anthropic = anthropic.Anthropic(api_key=resolved_key)

    def _call(
        self,
        system_prompt: str,
        user_message: str,
        schema: type[BaseModel],
        model: str,
    ) -> list[dict[str, Any]]:
        """Call the Anthropic messages API with prompt caching and tool-use.

        Args:
            system_prompt: Full system prompt text.
            user_message: JSON-serialised chunk of input rows.
            schema: Pydantic model class for one output row.
            model: Anthropic model identifier.

        Returns:
            List of dicts extracted from the tool-use response.

        Raises:
            NotImplementedError: Implementation pending.

        Note:
            Implementation should:
            1. Build a system list with cache_control={"type": "ephemeral"} on
               the system prompt block so repeated chunks reuse the cached tokens.
            2. Define a tool whose input_schema is derived from schema.model_json_schema().
            3. Call client.messages.create with tool_choice={"type": "tool", "name": ...}.
            4. Extract the tool_use content block and validate against schema.
        """
        raise NotImplementedError(
            "AnthropicBackend._call is not yet implemented. "
            "Wire client.messages.create with cache_control on the system block "
            "and tool_choice for structured output."
        )


class LLMClient:
    """Unified LLM client providing concurrent DataFrame classification.

    Dispatches to either OpenAIBackend or AnthropicBackend based on the
    backend argument. Chunking, concurrency, and progress reporting are
    handled by classify_df in dataframe.py; this class holds the backend
    and model configuration.

    Attributes:
        backend_name: Either "openai" or "anthropic".
        model: Model identifier string.
        _backend: The underlying provider backend instance.

    Example:
        client = LLMClient(backend="openai", model="gpt-4.1-mini")
        results = client.classify_df(
            df,
            schema=CultureRow,
            prompt="Extract culture type and tone.",
        )
    """

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        **backend_kwargs: Any,
    ) -> None:
        """Initialise the LLMClient with a backend and model.

        Args:
            backend: "openai" or "anthropic". Defaults to settings.default_backend.
            model: Model identifier. Defaults to settings.default_model.
            **backend_kwargs: Keyword arguments forwarded to the backend constructor
                (e.g., api_key, base_url for OpenAI).
        """
        self.backend_name = backend or settings.default_backend
        self.model = model or settings.default_model

        if self.backend_name == "openai":
            self._backend: _Backend = OpenAIBackend(**backend_kwargs)
        elif self.backend_name == "anthropic":
            self._backend = AnthropicBackend(**backend_kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {self.backend_name!r}. Choose 'openai' or 'anthropic'."
            )

    def classify_df(
        self,
        df: pd.DataFrame,
        schema: type[BaseModel],
        prompt: str,
        id_col: str = "id",
        text_col: str = "text",
        chunk_size: int | None = None,
        max_workers: int | None = None,
        fresh: bool = False,
    ) -> pd.DataFrame:
        """Classify every row in df using an LLM and return structured results.

        Delegates to dataframe.classify_df for the actual concurrent execution.
        Parameters documented here reflect the public API surface.

        Args:
            df: Input DataFrame. Must contain id_col and text_col.
            schema: Pydantic model class for one output row.
            prompt: System prompt text. A long shared prompt benefits from
                Anthropic prompt caching on the AnthropicBackend.
            id_col: Column name for row identifiers. Default "id".
            text_col: Column name for text content. Default "text".
            chunk_size: Rows per LLM request. Defaults to settings.chunk_size.
            max_workers: Thread-pool size. Defaults to settings.max_workers.
            fresh: If True, ignore any cached/prior results and reprocess all rows.
                   If False (default), skip rows already present in the cache.

        Returns:
            DataFrame of validated schema instances. One row per input row.

        Raises:
            NotImplementedError: Until dataframe.classify_df is implemented.
        """
        from genai_batch_ie_rfs.dataframe import classify_df as _classify_df

        return _classify_df(
            df=df,
            backend=self._backend,
            schema=schema,
            prompt=prompt,
            model=self.model,
            id_col=id_col,
            text_col=text_col,
            chunk_size=chunk_size or settings.chunk_size,
            max_workers=max_workers or settings.max_workers,
            fresh=fresh,
        )
