# ABOUTME: Thin wrapper around headless `claude -p` (JSON or stream-JSON output). Every invoke is a
# ABOUTME: fresh, stateless session with usage/cost tracking and subscription-limit auto-pause.
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Sequence
from typing import Literal, assert_never

from openai_cli_agents.core import (
  DEFAULT_SYSTEM_MODE, CLIResult, HarnessError, LimitError, PauseGate, StreamEvent, SystemMode,
  TextDelta, parse_epoch,
)
from openai_cli_agents.core import detect_limit as _detect_limit

LIMIT_PATTERNS = (
  "usage limit", "rate limit", "limit reached", "limit will reset",
  "resets at", "too many requests", "overloaded", "429", "quota",
  # claude code subscription notice, returned as a normal result on success
  "hit your session limit", "hit your usage limit", "session limit ·",
  "reached your usage", "5-hour limit",
  "hit your weekly limit", "weekly limit", "weekly limit ·",
)

OutputFormat = Literal["json", "stream-json"]

_empty_cwd_lock = threading.Lock()
_empty_cwd: str | None = None


# replace-mode calls run here so no project CLAUDE.md / git state leaks into the context
def empty_cwd() -> str:
  global _empty_cwd
  with _empty_cwd_lock:
    if _empty_cwd is None or not os.path.isdir(_empty_cwd):
      _empty_cwd = tempfile.mkdtemp(prefix="openai-cli-agents-claude-")
    return _empty_cwd


# epoch first, then a clock time like "resets 12:20am (...)" / "resets at 3pm"
def parse_reset(blob: str) -> float | None:
  if (epoch := parse_epoch(blob)) is not None:
    return epoch
  m = re.search(r"resets?(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)", blob)
  if not m:
    return None
  hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
  now = dt.datetime.now()
  target = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
  if target <= now:
    target += dt.timedelta(days=1)
  return target.timestamp()


def detect_limit(*texts: str) -> tuple[bool, float | None]:
  return _detect_limit(LIMIT_PATTERNS, parse_reset, *texts)


def _exit_error(returncode: int, stderr: str, stdout: str) -> HarnessError:
  is_limit, reset_at = detect_limit(stderr, stdout)
  msg = f"claude -p exited {returncode}: {stderr.strip()[:500]}"
  return LimitError(msg, reset_at) if is_limit else HarnessError(msg)


# the final result envelope, shared by the json output and the last stream-json line
def parse_result(data: dict) -> CLIResult:
  # the session-limit notice comes back as a normal (is_error=false) result,
  # so always scan the result text, not just the error envelope
  result_text = str(data.get("result", "") or "")
  is_limit, reset_at = detect_limit(result_text, str(data.get("subtype", "")))
  if is_limit:
    raise LimitError(result_text[:500], reset_at)
  return CLIResult(
    text=result_text.strip(),
    is_error=bool(data.get("is_error", False)),
    usage=data.get("usage", {}) or {},
    cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
    num_turns=int(data.get("num_turns", 0) or 0),
    session_id=data.get("session_id"),
    raw=data,
  )


def _text_delta(line: dict) -> str | None:
  if line.get("type") != "stream_event":
    return None
  event = line.get("event") or {}
  delta = event.get("delta") or {}
  if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
    return delta.get("text") or None
  return None


class ClaudeCLI:
  def __init__(
    self,
    model: str = "sonnet",
    timeout: int = 300,
    binary: str = "claude",
    extra_env: dict[str, str] | None = None,
    extra_args: Sequence[str] = (),
    auto_pause: bool = True,
    backoff_seconds: float = 600.0,
    max_pause_retries: int = 1000,
  ) -> None:
    self.model = model
    self.timeout = timeout
    self.binary = binary
    self.extra_env = extra_env or {}
    self.extra_args = list(extra_args)
    self.auto_pause = auto_pause
    self.max_pause_retries = max_pause_retries
    self.gate = PauseGate("claude", backoff_seconds)

  def build_cmd(
    self,
    *,
    system: str | None = None,
    system_mode: SystemMode = DEFAULT_SYSTEM_MODE,
    mcp_config: dict | str | None = None,
    allowed_tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    builtin_tools: Sequence[str] | None = None,
    max_turns: int | None = None,
    autocompact: str | int | None = None,
    add_dirs: Sequence[str] | None = None,
    bypass_permissions: bool = True,
    strict_mcp: bool = True,
    resume: str | None = None,
    persist_session: bool = True,
    output_format: OutputFormat = "json",
  ) -> list[str]:
    # extra_args first: variadic flags like --allowed-tools would swallow trailing positionals
    cmd = [self.binary, "-p", "--model", self.model, "--output-format", output_format,
           *self.extra_args]
    if output_format == "stream-json":
      cmd += ["--verbose", "--include-partial-messages"]
    match system_mode:
      case "replace":
        # always sent: no prompt must mean none, not claude code's default. "" is accepted
        # but whitespace-only is a 400 from the API
        cmd += ["--system-prompt", system or ""]
        # safe mode drops CLAUDE.md, skills, hooks and plugins but keeps subscription auth;
        # skipped with an MCP config since it may also switch off MCP servers
        if mcp_config is None:
          cmd += ["--safe-mode"]
      case "append":
        if system:
          cmd += ["--append-system-prompt", system]
      case _:
        assert_never(system_mode)
    if mcp_config is not None:
      cmd += ["--mcp-config", mcp_config if isinstance(mcp_config, str) else json.dumps(mcp_config)]
      if strict_mcp:
        cmd += ["--strict-mcp-config"]
    # [] permits no tools, None omits the flag
    if allowed_tools is not None:
      cmd += ["--allowed-tools", *allowed_tools] if allowed_tools else ["--allowed-tools", ""]
    if disallowed_tools:
      cmd += ["--disallowed-tools", *disallowed_tools]
    # --allowed-tools only gates permission; --tools controls which built-in tool
    # definitions exist at all ([] strips them, MCP tools are unaffected)
    if builtin_tools is not None:
      cmd += ["--tools", ",".join(builtin_tools)]
    if max_turns is not None:
      cmd += ["--max-turns", str(max_turns)]
    if autocompact is not None:
      cmd += ["--autocompact", str(autocompact)]
    for d in add_dirs or []:
      cmd += ["--add-dir", d]
    if bypass_permissions:
      cmd += ["--permission-mode", "bypassPermissions"]
    if resume:
      cmd += ["--resume", resume]
    if not persist_session:
      cmd += ["--no-session-persistence"]
    return cmd

  def invoke(
    self,
    prompt: str,
    *,
    system: str | None = None,
    system_mode: SystemMode = DEFAULT_SYSTEM_MODE,
    mcp_config: dict | str | None = None,
    allowed_tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    builtin_tools: Sequence[str] | None = None,
    max_turns: int | None = None,
    autocompact: str | int | None = None,
    add_dirs: Sequence[str] | None = None,
    bypass_permissions: bool = True,
    strict_mcp: bool = True,
    cwd: str | None = None,
    resume: str | None = None,
    persist_session: bool = True,
    timeout: int | None = None,
  ) -> CLIResult:
    cmd = self.build_cmd(
      system=system, system_mode=system_mode, mcp_config=mcp_config,
      allowed_tools=allowed_tools, disallowed_tools=disallowed_tools,
      builtin_tools=builtin_tools, max_turns=max_turns,
      autocompact=autocompact, add_dirs=add_dirs, bypass_permissions=bypass_permissions,
      strict_mcp=strict_mcp, resume=resume, persist_session=persist_session,
    )
    run_cwd = self._cwd(cwd, system_mode)
    env = {**os.environ, **self.extra_env}
    call_timeout = timeout if timeout is not None else self.timeout
    return self.gate.run(
      lambda: self._run_once(cmd, prompt, run_cwd, env, call_timeout),
      auto_pause=self.auto_pause, max_limit_retries=self.max_pause_retries,
    )

  # no tools and a single turn: a pure completion whose text arrives as it is generated.
  # Not retried or auto-paused; a limit surfaces as LimitError to the consumer.
  def stream(
    self,
    prompt: str,
    *,
    system: str | None = None,
    system_mode: SystemMode = DEFAULT_SYSTEM_MODE,
    cwd: str | None = None,
    persist_session: bool = False,
    timeout: int | None = None,
  ) -> Iterator[StreamEvent]:
    cmd = self.build_cmd(
      system=system, system_mode=system_mode, allowed_tools=[], builtin_tools=[], max_turns=1,
      bypass_permissions=False, persist_session=persist_session, output_format="stream-json",
    )
    call_timeout = timeout if timeout is not None else self.timeout
    self.gate.wait()
    proc = subprocess.Popen(
      cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
      cwd=self._cwd(cwd, system_mode), env={**os.environ, **self.extra_env},
    )
    timer = threading.Timer(call_timeout, proc.kill)
    timer.start()
    try:
      assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
      proc.stdin.write(prompt)
      proc.stdin.close()
      for raw in proc.stdout:
        try:
          line = json.loads(raw)
        except json.JSONDecodeError:
          continue
        if (text := _text_delta(line)) is not None:
          yield TextDelta(text)
        elif line.get("type") == "result":
          yield parse_result(line)
          return
      returncode = proc.wait()
      if not timer.is_alive():
        raise HarnessError(f"claude -p timed out after {call_timeout}s")
      raise _exit_error(returncode, proc.stderr.read(), "")
    finally:
      timer.cancel()
      # the consumer may stop early (client disconnect): never leave the process running
      if proc.poll() is None:
        proc.kill()
        proc.wait()

  @staticmethod
  def _cwd(cwd: str | None, system_mode: SystemMode) -> str | None:
    if cwd is not None:
      return cwd
    return empty_cwd() if system_mode == "replace" else None

  def _run_once(self, cmd: list[str], prompt: str, cwd: str | None, env: dict[str, str],
                call_timeout: int) -> CLIResult:
    # prompt via stdin to avoid ARG_MAX on large extraction prompts
    try:
      proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd,
                            env=env, timeout=call_timeout)
    except subprocess.TimeoutExpired as e:
      raise HarnessError(f"claude -p timed out after {call_timeout}s") from e

    if proc.returncode != 0 and not proc.stdout.strip():
      raise _exit_error(proc.returncode, proc.stderr, proc.stdout)

    try:
      data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
      is_limit, reset_at = detect_limit(proc.stdout, proc.stderr)
      if is_limit:
        raise LimitError(proc.stdout[:500], reset_at) from e
      raise HarnessError(f"could not parse claude -p output: {proc.stdout[:500]!r}") from e

    return parse_result(data)
