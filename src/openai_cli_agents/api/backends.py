# ABOUTME: Model routing (OpenAI model name -> CLI backend + model id) and the Backends manager that
# ABOUTME: owns the ClaudeCLI / OpenCodeCLI instances and runs pure, tool-free completions on them.
from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, assert_never

from openai_cli_agents.api.schemas import APIError
from openai_cli_agents.claude import ClaudeCLI
from openai_cli_agents.core import CLIResult, StreamEvent
from openai_cli_agents.opencode import OpenCodeCLI

BackendKind = Literal["claude", "opencode"]
CLAUDE_ALIASES = ("sonnet", "opus", "haiku")


@dataclass(frozen=True)
class ModelRoute:
  kind: BackendKind
  model: str   # model id as the backend CLI expects it


# "claude/<m>" and bare claude names ("sonnet", "claude-opus-4-1") go to claude -p; any other
# "provider/model" is an opencode model ("opencode/big-pickle", "openrouter/qwen/qwen3")
def resolve_model(name: str) -> ModelRoute:
  match name.split("/", 1):
    case ["claude", model] if model:
      return ModelRoute("claude", model)
    case [model] if model in CLAUDE_ALIASES or model.startswith("claude-"):
      return ModelRoute("claude", model)
    case [provider, model] if provider and model:
      return ModelRoute("opencode", name)
    case _:
      raise APIError(404, f"The model {name!r} does not exist. Use 'claude/<model>', a claude "
                     "alias (sonnet, opus, haiku) or an opencode 'provider/model'.",
                     type="not_found_error", param="model", code="model_not_found")


class CompletionBackend(Protocol):
  def complete(self, route: ModelRoute, prompt: str, system: str | None) -> CLIResult: ...

  def stream(self, route: ModelRoute, prompt: str, system: str | None) -> Iterator[StreamEvent]: ...


# every call is a stateless, tool-free single turn under no harness system prompt; limits surface
# immediately (auto-pause off) so clients get a 429 instead of a request held open for hours
class Backends:
  def __init__(self, timeout: int = 600, claude_binary: str = "claude",
               opencode_binary: str = "opencode") -> None:
    self.timeout = timeout
    self.claude_binary = claude_binary
    self._lock = threading.Lock()
    self._claude: dict[str, ClaudeCLI] = {}
    self._opencode = OpenCodeCLI(timeout=timeout, binary=opencode_binary, auto_pause=False)

  def _claude_cli(self, model: str) -> ClaudeCLI:
    with self._lock:
      if model not in self._claude:
        self._claude[model] = ClaudeCLI(model=model, timeout=self.timeout,
                                        binary=self.claude_binary, auto_pause=False)
      return self._claude[model]

  def complete(self, route: ModelRoute, prompt: str, system: str | None) -> CLIResult:
    match route.kind:
      case "claude":
        return self._claude_cli(route.model).invoke(
          prompt, system=system, system_mode="replace", allowed_tools=[], builtin_tools=[],
          max_turns=1, persist_session=False, bypass_permissions=False,
        )
      case "opencode":
        return self._opencode.invoke(prompt, system=system, system_mode="replace",
                                     model=route.model)
      case _:
        assert_never(route.kind)

  def stream(self, route: ModelRoute, prompt: str, system: str | None) -> Iterator[StreamEvent]:
    match route.kind:
      case "claude":
        return self._claude_cli(route.model).stream(prompt, system=system, system_mode="replace")
      case "opencode":
        return self._opencode.stream(prompt, system=system, system_mode="replace",
                                     model=route.model)
      case _:
        assert_never(route.kind)
