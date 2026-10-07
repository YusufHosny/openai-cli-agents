# ABOUTME: OpenCodeCLI — single-turn completions through a shared `opencode serve` HTTP server
# ABOUTME: (POST /session + /message), mirroring ClaudeCLI: concurrent, usage-tracked, auto-pausing.
from __future__ import annotations

from collections.abc import Iterator
from typing import assert_never

from openai_cli_agents.core import (
  DEFAULT_SYSTEM_MODE, CLIResult, PauseGate, StreamEvent, SystemMode, TextDelta,
)
from openai_cli_agents.opencode.server import (
  SHARED_POOL, OpenCodeServerPool, agent_config, message_body, parse_message, send_message,
  split_model,
)

_REPLACE_AGENT = "openai-cli-agents-chat"


# "append" keeps opencode's default agent untouched; "replace" routes through a
# builtins-off agent whose placeholder prompt suppresses the provider default
def _server_config(system_mode: SystemMode) -> tuple[dict, str | None]:
  match system_mode:
    case "append":
      return {}, None
    case "replace":
      cfg = agent_config("replace", tools={"*": False})
      return {"agent": {_REPLACE_AGENT: cfg}}, _REPLACE_AGENT
    case _:
      assert_never(system_mode)


class OpenCodeCLI:
  def __init__(
    self,
    model: str = "opencode/x-preview-f-free",
    timeout: int = 150,
    binary: str = "opencode",
    auto_pause: bool = True,
    backoff_seconds: float = 120.0,
    max_pause_retries: int = 1000,
    max_transient_retries: int = 8,
    variant: str | None = None,
    pool: OpenCodeServerPool | None = None,
  ) -> None:
    self.model = model
    self.provider_id, self.model_id = split_model(model)
    self.timeout = timeout
    self.auto_pause = auto_pause
    self.max_pause_retries = max_pause_retries
    self.max_transient_retries = max_transient_retries
    self.variant = variant
    self.pool = pool if pool is not None else (
      SHARED_POOL if binary == SHARED_POOL.binary else OpenCodeServerPool(binary))
    self.gate = PauseGate("opencode", backoff_seconds)

  def invoke(
    self,
    prompt: str,
    *,
    system: str | None = None,
    system_mode: SystemMode = DEFAULT_SYSTEM_MODE,
    model: str | None = None,
    timeout: int | None = None,
  ) -> CLIResult:
    provider_id, model_id = split_model(model) if model else (self.provider_id, self.model_id)
    config, agent = _server_config(system_mode)
    server = self.pool.get(config)
    body = message_body(provider_id, model_id, prompt, system=system, agent=agent,
                        variant=self.variant)
    call_timeout = timeout if timeout is not None else self.timeout

    def _once() -> CLIResult:
      sid, msg = send_message(server.url(), body, call_timeout)
      return parse_message(sid, msg)

    return self.gate.run(_once, auto_pause=self.auto_pause,
                         max_limit_retries=self.max_pause_retries,
                         max_transient_retries=self.max_transient_retries)

  # opencode returns the finished message only, so the "stream" is the whole text in one delta
  def stream(
    self,
    prompt: str,
    *,
    system: str | None = None,
    system_mode: SystemMode = DEFAULT_SYSTEM_MODE,
    model: str | None = None,
    timeout: int | None = None,
  ) -> Iterator[StreamEvent]:
    res = self.invoke(prompt, system=system, system_mode=system_mode, model=model, timeout=timeout)
    yield TextDelta(res.text)
    yield res
