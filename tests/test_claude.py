# ABOUTME: Unit tests for the claude -p harness: flag building (system-prompt modes, isolation),
# ABOUTME: output parsing, streaming and limit detection, with the subprocess mocked out.
import io
import json
import subprocess

import pytest

from openai_cli_agents.claude import ClaudeCLI
from openai_cli_agents.claude import cli as claude_cli
from openai_cli_agents.claude.cli import detect_limit
from openai_cli_agents.core import CLIResult, HarnessError, LimitError, TextDelta


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
  return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _ok(result: str = "hi", **extra) -> str:
  return json.dumps({"result": result, "is_error": False, "num_turns": 1,
                     "total_cost_usd": 0.01, "session_id": "s1",
                     "usage": {"input_tokens": 10, "output_tokens": 5,
                               "cache_read_input_tokens": 100,
                               "cache_creation_input_tokens": 7}, **extra})


@pytest.fixture
def captured(monkeypatch):
  calls: list[dict] = []
  outputs: list[subprocess.CompletedProcess] = []

  def fake_run(cmd, **kwargs):
    calls.append({"cmd": cmd, **kwargs})
    return outputs.pop(0) if outputs else _completed(_ok())

  monkeypatch.setattr(subprocess, "run", fake_run)
  return calls, outputs


def _flag(cmd: list[str], flag: str) -> str:
  return cmd[cmd.index(flag) + 1]


def test_replace_is_default_and_means_no_system_prompt():
  cmd = ClaudeCLI().build_cmd()
  assert _flag(cmd, "--system-prompt") == ""
  assert "--safe-mode" in cmd
  assert "--append-system-prompt" not in cmd


def test_replace_uses_system_prompt_flag():
  cmd = ClaudeCLI().build_cmd(system="be terse")
  assert _flag(cmd, "--system-prompt") == "be terse"
  assert "--append-system-prompt" not in cmd


def test_replace_with_mcp_skips_safe_mode():
  cmd = ClaudeCLI().build_cmd(mcp_config={"mcpServers": {}})
  assert "--safe-mode" not in cmd and "--system-prompt" in cmd


def test_append_keeps_default_prompt():
  cmd = ClaudeCLI().build_cmd(system="be terse", system_mode="append")
  assert _flag(cmd, "--append-system-prompt") == "be terse"
  assert "--system-prompt" not in cmd and "--safe-mode" not in cmd
  assert "--append-system-prompt" not in ClaudeCLI().build_cmd(system_mode="append")


def test_builtin_tools_flag_only_when_set():
  assert "--tools" not in ClaudeCLI().build_cmd()
  assert _flag(ClaudeCLI().build_cmd(builtin_tools=[]), "--tools") == ""


def test_empty_allowed_tools_disables_all_tools():
  assert _flag(ClaudeCLI().build_cmd(allowed_tools=[]), "--allowed-tools") == ""


def test_extra_args_precede_variadic_flags():
  cmd = ClaudeCLI(extra_args=["--setting-sources", ""]).build_cmd(allowed_tools=["a", "b"])
  assert cmd.index("--setting-sources") < cmd.index("--allowed-tools")


def test_session_persistence_flag():
  assert "--no-session-persistence" not in ClaudeCLI().build_cmd()
  assert "--no-session-persistence" in ClaudeCLI().build_cmd(persist_session=False)


def test_invoke_parses_usage_and_sends_prompt_on_stdin(captured):
  calls, _ = captured
  res = ClaudeCLI().invoke("question?")
  assert res.text == "hi"
  assert res.total_input_tokens == 117
  assert res.cost_usd == pytest.approx(0.01)
  assert calls[0]["input"] == "question?"


def test_replace_runs_in_empty_cwd_append_inherits(captured, tmp_path):
  calls, _ = captured
  ClaudeCLI().invoke("q")
  ClaudeCLI().invoke("q", system_mode="append")
  ClaudeCLI().invoke("q", cwd=str(tmp_path))
  assert calls[0]["cwd"] == claude_cli.empty_cwd()
  assert calls[1]["cwd"] is None
  assert calls[2]["cwd"] == str(tmp_path)


def test_session_limit_notice_in_result_raises_limit():
  assert detect_limit("You've hit your session limit · resets 3pm")[0]
  assert not detect_limit("the answer is 42")[0]


def test_limit_pauses_then_retries(captured, monkeypatch):
  _, outputs = captured
  outputs += [_completed(_ok("You've hit your usage limit")), _completed(_ok("fine"))]
  cli = ClaudeCLI(backoff_seconds=0.0)
  monkeypatch.setattr(cli.gate, "trigger", lambda reset_at: None)
  assert cli.invoke("q").text == "fine"


def test_limit_raises_without_auto_pause(captured):
  _, outputs = captured
  outputs.append(_completed(_ok("You've hit your usage limit")))
  with pytest.raises(LimitError):
    ClaudeCLI(auto_pause=False).invoke("q")


def test_nonzero_exit_without_stdout_is_harness_error(captured):
  _, outputs = captured
  outputs.append(_completed("", returncode=1, stderr="boom"))
  with pytest.raises(HarnessError, match="boom"):
    ClaudeCLI().invoke("q")


class _KeptStringIO(io.StringIO):
  def close(self) -> None:
    self.closed_by_caller = True


class _FakePopen:
  def __init__(self, lines: list[dict], returncode: int = 0, stderr: str = "") -> None:
    self.stdin = _KeptStringIO()
    self.stdout = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
    self.stderr = io.StringIO(stderr)
    self.returncode = returncode
    self.killed = False

  def poll(self):
    return self.returncode

  def wait(self, timeout=None):
    return self.returncode

  def kill(self):
    self.killed = True


def _delta(text: str) -> dict:
  return {"type": "stream_event",
          "event": {"type": "content_block_delta", "index": 1,
                    "delta": {"type": "text_delta", "text": text}}}


@pytest.fixture
def popen(monkeypatch):
  made: list[tuple[list[str], _FakePopen]] = []
  scripts: list[_FakePopen] = []

  def fake(cmd, **kwargs):
    proc = scripts.pop(0)
    made.append((cmd, proc))
    return proc

  monkeypatch.setattr(subprocess, "Popen", fake)
  return made, scripts


def test_stream_yields_text_deltas_then_result(popen):
  made, scripts = popen
  thinking = {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "thinking_delta"}}}
  scripts.append(_FakePopen([{"type": "system", "subtype": "init"}, thinking,
                             _delta("Hel"), _delta("lo"), json.loads(_ok("Hello"))
                             | {"type": "result"}]))
  events = list(ClaudeCLI().stream("q", system="s"))
  assert events[:2] == [TextDelta("Hel"), TextDelta("lo")]
  assert isinstance(events[2], CLIResult) and events[2].total_input_tokens == 117
  cmd, proc = made[0]
  assert _flag(cmd, "--output-format") == "stream-json" and "--include-partial-messages" in cmd
  assert _flag(cmd, "--tools") == "" and "--no-session-persistence" in cmd
  assert proc.stdin.getvalue() == "q"


def test_stream_without_result_raises_with_stderr(popen):
  _, scripts = popen
  scripts.append(_FakePopen([], returncode=1, stderr="429 rate limit"))
  with pytest.raises(LimitError):
    list(ClaudeCLI().stream("q"))
