"""Draft an ``extract_df`` prompt from a plain-English goal.

``draft_prompt(goal=...)`` sends a one-shot meta-prompt to an LLM and
returns a candidate prompt string in the house style that ``extract_df``
expects: numbered step-by-step instructions, a strict
``Return a JSON object with this EXACT structure: {"all_results": [...]}``
block, and a closing "do not include any other fields" line.

The output is a *starting point*. Read it, edit field names, tighten
type constraints, then pass to ``extract_df(prompt=...)``.

Example:
    >>> from lmsyz_genai_ie_rfs import draft_prompt, extract_df
    >>> p = draft_prompt(
    ...     goal="extract the firm's stated values and a 1-5 risk-tone score",
    ...     api_key="sk-...",
    ... )
    >>> # ... edit p ...
    >>> out = extract_df(df, prompt=p, cache_path="run.sqlite", model="gpt-4.1-mini",
    ...                  api_key="sk-...")
"""

from __future__ import annotations

from typing import Any

EXEMPLAR_GOAL = (
    "extract named entities, causal relations, and overall sentiment "
    "from short news sentences"
)

EXEMPLAR_PROMPT = """\
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
      "causal_triples": [["cause_1", "relation_1", "effect_1"]],
      "sentiment": "positive"
    }
  ]
}

Do not include any fields besides input_id, entities, causal_triples, and sentiment.
"""

META_SYSTEM = """\
You write extraction prompts for the lmsyz_genai_ie_rfs Python library. Each prompt instructs an LLM to extract structured information from a row of a pandas DataFrame and return JSON.

Every prompt you write MUST follow this house style:

1. Open with one short sentence framing the model as an extraction assistant operating per row.
2. Give numbered, step-by-step instructions. The FIRST numbered step is always:
   "input_id: Copy the input_id from the row verbatim."
   Subsequent steps name each output field, describe its type, and constrain valid values where appropriate (closed enums, value ranges, list element shapes).
3. End with a fenced 'Return a JSON object with this EXACT structure:' block whose top-level shape is { "all_results": [ { ...one example row, fully populated... } ] }. Use realistic placeholder values, not "string" / "int".
4. Close with a single sentence: "Do not include any fields besides <field1>, <field2>, ... ."

Field names are snake_case. Keep prompts compact: every line should add a constraint or a field. No preamble, no apologies, no commentary outside the prompt itself.
"""

META_USER_TEMPLATE = """\
Here is one canonical example.

USER GOAL:
{exemplar_goal}

PROMPT:
{exemplar_prompt}
---

Now write a prompt for this user goal:

USER GOAL:
{user_goal}

Output only the prompt text. No markdown fence, no preamble.
"""


def _make_client(backend: str, api_key: str | None, base_url: str | None) -> Any:
    """Build an SDK client for the meta-call.

    Args:
        backend: ``"openai"`` or ``"anthropic"``.
        api_key: Provider API key. If None, the SDK reads its own env var.
        base_url: OpenAI base-URL override (OpenRouter, Gemini compat).

    Returns:
        An SDK client instance.

    Raises:
        ValueError: If ``backend`` is unrecognised.
    """
    if backend == "openai":
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
    if backend == "anthropic":
        from anthropic import Anthropic

        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        return Anthropic(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Use 'openai' or 'anthropic'.")


def draft_prompt(
    goal: str,
    *,
    backend: str = "openai",
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    print_prompt: bool = True,
) -> str:
    """Draft a candidate ``extract_df`` prompt from a plain-English goal.

    Sends a one-shot meta-prompt to ``model`` and returns the prompt text.
    The result is a starting point; read it and edit before running
    ``extract_df``.

    Args:
        goal: Plain-English description of what to extract or measure.
        backend: ``"openai"`` or ``"anthropic"``.
        model: Model identifier for the meta-call (defaults to
            ``gpt-4.1-mini``; any chat model works).
        api_key: Override the API key from environment.
        base_url: OpenAI base-URL override (for OpenRouter, Gemini compat).
        print_prompt: When True, print the result to stdout so a notebook
            user sees it without an extra ``print`` call.

    Returns:
        Prompt string ready to edit and pass to ``extract_df(prompt=...)``.
    """
    client = _make_client(backend, api_key, base_url)
    user_msg = META_USER_TEMPLATE.format(
        exemplar_goal=EXEMPLAR_GOAL,
        exemplar_prompt=EXEMPLAR_PROMPT,
        user_goal=goal,
    )
    if backend == "openai":
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": META_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        text = resp.choices[0].message.content or ""
    else:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            system=META_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if print_prompt:
        print(text)
    return text
