"""Test prompt drafting from fake SDK responses and temporary .env files.

Inputs are local configuration and controlled provider responses; output is pytest assertions.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmsyz_genai_ie_rfs.settings import Settings

drafting = importlib.import_module("lmsyz_genai_ie_rfs.draft_prompt")


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Load isolated .env settings into the singleton used by drafting.

    Args:
        monkeypatch: Scoped environment and module attribute patcher.
        tmp_path: Temporary directory for a configuration file.

    Returns:
        Settings loaded from the temporary .env file.
    """
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=test-openai\n"
        "ANTHROPIC_API_KEY=test-anthropic\n"
        "OPENAI_BASE_URL=https://example.invalid/v1\n"
    )
    config = Settings(_env_file=env_path)  # type: ignore[call-arg]
    monkeypatch.setattr(drafting, "settings", config)
    return config


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_dotenv_credentials_reach_sdk(
    backend: str, isolated_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify .env keys and the OpenAI base URL reach their SDK constructors."""
    constructor = Mock()
    monkeypatch.setattr(f"{backend}.{'OpenAI' if backend == 'openai' else 'Anthropic'}", constructor)
    drafting._make_client(backend, None, None)
    expected = {"api_key": f"test-{backend}"}
    if backend == "openai":
        expected["base_url"] = "https://example.invalid/v1"
    constructor.assert_called_once_with(**expected)


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_explicit_configuration_wins(
    backend: str, isolated_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep explicit key and endpoint overrides ahead of settings values."""
    constructor = Mock()
    monkeypatch.setattr(f"{backend}.{'OpenAI' if backend == 'openai' else 'Anthropic'}", constructor)
    drafting._make_client(backend, "explicit-key", "https://override.invalid/v1")
    expected = {"api_key": "explicit-key"}
    if backend == "openai":
        expected["base_url"] = "https://override.invalid/v1"
    constructor.assert_called_once_with(**expected)


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_unconfigured_settings_preserve_sdk_environment_fallback(
    backend: str, isolated_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow SDKs to read environment values set after settings initialization."""
    isolated_settings.openai_api_key = None
    isolated_settings.anthropic_api_key = None
    isolated_settings.openai_base_url = None
    constructor = Mock()
    monkeypatch.setattr(f"{backend}.{'OpenAI' if backend == 'openai' else 'Anthropic'}", constructor)
    drafting._make_client(backend, None, None)
    constructor.assert_called_once_with()


def _install_response(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    content: str | None,
    stop_reason: str = "stop",
) -> Mock:
    """Install a fake SDK client returning one controlled provider response.

    Args:
        monkeypatch: Scoped attribute patcher.
        backend: Provider response format to emulate.
        content: Text to return, or None for an absent text response.
        stop_reason: Provider completion termination reason.

    Returns:
        Mock SDK create method for request assertions.
    """
    client = Mock()
    create: Mock
    if backend == "openai":
        create = client.chat.completions.create
        create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason=stop_reason
                )
            ]
        )
    else:
        create = client.messages.create
        create.return_value = SimpleNamespace(
            content=[] if content is None else [SimpleNamespace(type="text", text=content)],
            stop_reason=stop_reason,
        )
    monkeypatch.setattr(drafting, "_make_client", Mock(return_value=client))
    return create


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_draft_request_and_printed_output(
    backend: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve model and goal, accept edited-style text, and print the cleaned draft."""
    create = _install_response(monkeypatch, backend, "```text\nDraft this goal.\n```")
    result = drafting.draft_prompt("extract values", backend=backend, model="requested-model")
    assert result == "Draft this goal."
    assert capsys.readouterr().out == "Draft this goal.\n"
    assert create.call_args.kwargs["model"] == "requested-model"
    assert "extract values" in create.call_args.kwargs["messages"][-1]["content"]
    assert "temperature" not in create.call_args.kwargs


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
@pytest.mark.parametrize("content", [None, "", "   ", "```text\n```"])
def test_empty_output_is_rejected(
    backend: str, content: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject missing, blank, and fence-only drafts with a clear error."""
    _install_response(monkeypatch, backend, content)
    with pytest.raises(ValueError, match="empty draft"):
        drafting.draft_prompt("extract values", backend=backend, print_prompt=False)


@pytest.mark.parametrize(
    ("backend", "stop_reason"), [("openai", "length"), ("anthropic", "max_tokens")]
)
def test_explicit_token_truncation_is_rejected(
    backend: str, stop_reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a partial draft when the provider explicitly reports a token limit."""
    _install_response(monkeypatch, backend, "A partial draft", stop_reason)
    with pytest.raises(ValueError, match="truncated"):
        drafting.draft_prompt("extract values", backend=backend, print_prompt=False)


def test_no_openai_choices_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handle a response with no choices without leaking an IndexError."""
    create = _install_response(monkeypatch, "openai", "unused")
    create.return_value.choices = []
    with pytest.raises(ValueError, match="no prompt choices"):
        drafting.draft_prompt("extract values", print_prompt=False)


def test_unknown_backend_is_rejected() -> None:
    """Fail clearly before attempting an unsupported provider call."""
    with pytest.raises(ValueError, match="Unknown backend"):
        drafting.draft_prompt("extract values", backend="typo")


def test_print_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Return the draft without writing to stdout when printing is disabled."""
    _install_response(monkeypatch, "openai", "Candidate")
    assert drafting.draft_prompt("extract values", print_prompt=False) == "Candidate"
    assert capsys.readouterr().out == ""
