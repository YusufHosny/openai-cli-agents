# ABOUTME: openai_cli_agents — drive headless coding-agent CLIs (`claude -p`, `opencode serve`) as plain
# ABOUTME: stateless completions, and serve them behind an OpenAI-compatible HTTP API ([server]).
from openai_cli_agents.core import (
  DEFAULT_SYSTEM_MODE, CLIResult, HarnessError, LimitError, StreamEvent, SystemMode, TextDelta,
  TransientError, Turn, extract_json, flatten_turns,
)

__all__ = [
  "DEFAULT_SYSTEM_MODE", "CLIResult", "HarnessError", "LimitError", "StreamEvent", "SystemMode",
  "TextDelta", "TransientError", "Turn", "extract_json", "flatten_turns",
]
