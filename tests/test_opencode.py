# ABOUTME: Unit tests for the opencode harness: server config per system-prompt mode, message
# ABOUTME: body construction and response parsing / error mapping, with HTTP mocked out.
import pytest

from openai_cli_agents.core import CLIResult, HarnessError, LimitError, TextDelta, TransientError
from openai_cli_agents.opencode import OpenCodeCLI
from openai_cli_agents.opencode import cli as oc_cli
from openai_cli_agents.opencode.server import OpenCodeServerPool, parse_message, split_model


def _msg(text: str = "ok", **info) -> dict:
  return {"info": {"finish": "stop", "cost": 0.0,
                   "tokens": {"input": 3, "output": 2, "cache": {"read": 1, "write": 0}}, **info},
          "parts": [{"type": "text", "text": text}]}


class _FakeServer:
  def __init__(self, config: dict) -> None:
    self.config = config

  def url(self) -> str:
    return "http://fake"


class _FakePool(OpenCodeServerPool):
  def __init__(self) -> None:
    super().__init__()
    self.configs: list[dict] = []

  def get(self, config: dict):
    self.configs.append(config)
    return _FakeServer(config)


@pytest.fixture
def sent(monkeypatch):
  bodies: list[dict] = []
  replies: list = []

  def fake_send(base, body, timeout):
    bodies.append(body)
    reply = replies.pop(0) if replies else _msg()
    if isinstance(reply, Exception):
      raise reply
    return "sid", reply

  monkeypatch.setattr(oc_cli, "send_message", fake_send)
  return bodies, replies


def test_split_model_defaults_provider():
  assert split_model("big-pickle") == ("opencode", "big-pickle")
  assert split_model("openrouter/qwen/qwen3") == ("openrouter", "qwen/qwen3")


def test_append_mode_uses_default_agent(sent):
  bodies, _ = sent
  pool = _FakePool()
  OpenCodeCLI(pool=pool).invoke("q", system="sys", system_mode="append")
  assert pool.configs == [{}]
  assert bodies[0]["system"] == "sys" and "agent" not in bodies[0]


def test_replace_is_default_and_uses_placeholder_agent(sent):
  bodies, _ = sent
  pool = _FakePool()
  OpenCodeCLI(pool=pool).invoke("q", system="sys")
  agent_name = bodies[0]["agent"]
  agent_cfg = pool.configs[0]["agent"][agent_name]
  assert agent_cfg["prompt"].strip() == "" and agent_cfg["prompt"]
  assert agent_cfg["tools"] == {"*": False}
  assert bodies[0]["system"] == "sys"


def test_replace_mode_shares_one_config_across_prompts(sent):
  pool = _FakePool()
  cli = OpenCodeCLI(pool=pool)
  cli.invoke("q", system="a", system_mode="replace")
  cli.invoke("q", system="b", system_mode="replace")
  assert pool.configs[0] == pool.configs[1]


def test_transient_is_retried(sent, monkeypatch):
  _, replies = sent
  monkeypatch.setattr("time.sleep", lambda s: None)
  replies += [TransientError("503"), _msg("second")]
  assert OpenCodeCLI(pool=_FakePool()).invoke("q").text == "second"


def test_parse_message_maps_errors():
  assert parse_message("s", _msg("hi")).total_input_tokens == 4
  with pytest.raises(LimitError):
    parse_message("s", {"info": {"error": "429 Too Many Requests"}, "parts": []})
  with pytest.raises(TransientError):
    parse_message("s", {"info": {"finish": "stop"}, "parts": []})
  with pytest.raises(HarnessError):
    parse_message("s", {"info": {"finish": "content-filter"}, "parts": []})


def test_stream_is_one_delta_then_result(sent):
  events = list(OpenCodeCLI(pool=_FakePool()).stream("q", model="opencode/other"))
  assert events[0] == TextDelta("ok") and isinstance(events[1], CLIResult)
  assert sent[0][0]["modelID"] == "other"
