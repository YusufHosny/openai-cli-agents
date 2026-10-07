# ABOUTME: Tests for the OpenAI-compatible server, driven through the official openai SDK against
# ABOUTME: an in-process app whose CLI backends are replaced by a scripted fake.
from collections.abc import Iterator

import openai
import pytest
from fastapi.testclient import TestClient

from openai_cli_agents.api import ServerConfig, create_app, resolve_model
from openai_cli_agents.api.backends import ModelRoute
from openai_cli_agents.core import CLIResult, LimitError, StreamEvent, TextDelta


def _res(text: str) -> CLIResult:
  return CLIResult(text=text, is_error=False,
                   usage={"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 90})


class _FakeBackends:
  def __init__(self) -> None:
    self.replies: list[str | Exception] = []
    self.calls: list[tuple[ModelRoute, str, str | None]] = []

  def _next(self) -> str:
    reply = self.replies.pop(0) if self.replies else "hello"
    if isinstance(reply, Exception):
      raise reply
    return reply

  def complete(self, route: ModelRoute, prompt: str, system: str | None) -> CLIResult:
    self.calls.append((route, prompt, system))
    return _res(self._next())

  def stream(self, route: ModelRoute, prompt: str, system: str | None) -> Iterator[StreamEvent]:
    self.calls.append((route, prompt, system))
    text = self._next()
    for word in text.split(" "):
      yield TextDelta(word + " ")
    yield _res(text)


@pytest.fixture
def fake() -> _FakeBackends:
  return _FakeBackends()


def _client(fake: _FakeBackends, **cfg) -> openai.OpenAI:
  app = create_app(ServerConfig(**cfg), backends=fake)
  return openai.OpenAI(base_url="http://testserver/v1", api_key=cfg.get("api_key") or "unused",
                       http_client=TestClient(app), max_retries=0)


def test_resolve_model_routes_by_name():
  assert resolve_model("claude/sonnet") == ModelRoute("claude", "sonnet")
  assert resolve_model("opus") == ModelRoute("claude", "opus")
  assert resolve_model("claude-opus-4-1") == ModelRoute("claude", "claude-opus-4-1")
  assert resolve_model("opencode/big-pickle") == ModelRoute("opencode", "opencode/big-pickle")
  assert resolve_model("openrouter/qwen/qwen3") == ModelRoute("opencode", "openrouter/qwen/qwen3")


def test_unknown_model_is_404(fake):
  with pytest.raises(openai.NotFoundError):
    _client(fake).chat.completions.create(model="gpt-4o", messages=[{"role": "user",
                                                                     "content": "hi"}])


def test_models_lists_configured(fake):
  client = _client(fake, models=("claude/haiku", "opencode/big-pickle"))
  assert [m.id for m in client.models.list()] == ["claude/haiku", "opencode/big-pickle"]
  assert client.models.retrieve("opencode/big-pickle").owned_by == "opencode-cli"


def test_completion_flattens_and_reports_usage(fake):
  out = _client(fake).chat.completions.create(
    model="claude/sonnet", temperature=0.2, max_tokens=50,
    messages=[{"role": "developer", "content": "be terse"},
              {"role": "user", "content": [{"type": "text", "text": "hi"}]},
              {"role": "assistant", "content": "hey"},
              {"role": "user", "content": "how are you?"}],
  )
  assert out.choices[0].message.content == "hello"
  assert out.choices[0].finish_reason == "stop"
  assert out.usage.prompt_tokens == 100 and out.usage.completion_tokens == 4
  assert out.usage.prompt_tokens_details.cached_tokens == 90
  route, prompt, system = fake.calls[0]
  assert route == ModelRoute("claude", "sonnet") and system == "be terse"
  assert prompt == "USER: hi\n\nASSISTANT: hey\n\nUSER: how are you?"


def test_no_system_message_means_no_system_prompt(fake):
  _client(fake).chat.completions.create(model="sonnet",
                                        messages=[{"role": "user", "content": "hi"}])
  assert fake.calls[0][1:] == ("hi", None)


def test_stream_emits_openai_chunks(fake):
  fake.replies.append("one two")
  chunks = list(_client(fake).chat.completions.create(
    model="sonnet", messages=[{"role": "user", "content": "hi"}], stream=True,
    stream_options={"include_usage": True}))
  assert chunks[0].choices[0].delta.role == "assistant"
  text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
  assert text == "one two "
  assert chunks[-2].choices[0].finish_reason == "stop"
  assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 104
  assert len({c.id for c in chunks}) == 1


def test_json_schema_is_prompt_enforced_and_retried(fake):
  fake.replies += ["sorry, no", 'Here: {"value": 42}']
  out = _client(fake).chat.completions.create(
    model="sonnet", messages=[{"role": "user", "content": "answer"}],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "a", "schema": {"type": "object"}}})
  assert out.choices[0].message.content == '{"value": 42}'
  assert "JSON Schema" in fake.calls[0][2]
  assert "not a valid JSON object" in fake.calls[1][1]


def test_tools_are_rejected(fake):
  with pytest.raises(openai.BadRequestError, match="tool"):
    _client(fake).chat.completions.create(
      model="sonnet", messages=[{"role": "user", "content": "hi"}],
      tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}])


def test_image_parts_are_rejected(fake):
  with pytest.raises(openai.BadRequestError, match="image_url"):
    _client(fake).chat.completions.create(
      model="sonnet", messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:,"}}]}])


def test_limit_is_429_with_retry_after(fake):
  fake.replies.append(LimitError("hit your session limit", reset_at=None))
  with pytest.raises(openai.RateLimitError):
    _client(fake).chat.completions.create(model="sonnet",
                                          messages=[{"role": "user", "content": "hi"}])


def test_limit_on_stream_start_is_429(fake):
  fake.replies.append(LimitError("hit your session limit", reset_at=None))
  with pytest.raises(openai.RateLimitError):
    list(_client(fake).chat.completions.create(
      model="sonnet", messages=[{"role": "user", "content": "hi"}], stream=True))


def test_api_key_is_enforced(fake):
  client = _client(fake, api_key="secret")
  assert client.models.list().data
  bad = openai.OpenAI(base_url="http://testserver/v1", api_key="wrong", max_retries=0,
                      http_client=TestClient(create_app(ServerConfig(api_key="secret"), fake)))
  with pytest.raises(openai.AuthenticationError):
    bad.models.list()
