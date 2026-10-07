# ABOUTME: Lifecycle of `opencode serve` processes (one per distinct opencode.json config) plus the
# ABOUTME: small HTTP client and response parsing shared by OpenCodeCLI and OpenCodeAgent.
from __future__ import annotations

import atexit
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

from openai_cli_agents.core import (
  CLIResult, HarnessError, LimitError, SystemMode, TransientError, detect_limit, matches_any,
  parse_epoch,
)

LIMIT_PATTERNS = (
  "rate limit", "rate-limit", "rate limit exceeded", "too many requests", "429",
  "quota", "overloaded", "capacity", "usage limit", "limit reached",
  "try again later", "ai_apicallerror",
)

# transient provider blips (retryable, NOT the model's fault), e.g. 503 endpoint
# unavailable, provider network_error, upstream failures
TRANSIENT_PATTERNS = (
  "endpoint is unavailable", "service unavailable", "temporarily unavailable",
  "upstream request failed", "network_error", "bad gateway", "gateway timeout",
  "'isretryable': true", '"isretryable": true', "statuscode': 503", "statuscode': 502",
  "statuscode': 504", "statuscode': 500",
)

_OK_FINISH = ("stop", "end_turn", "length", "")

# opencode builds the system prompt as [agent.prompt or <provider default>, <env block +
# AGENTS.md>, request.system]. A non-empty placeholder agent prompt suppresses the provider
# default so request.system effectively replaces it without one server per distinct prompt.
_REPLACE_PLACEHOLDER = " "


def parse_reset(blob: str) -> float | None:
  if (epoch := parse_epoch(blob)) is not None:
    return epoch
  m = re.search(r"retry(?:-| )after[:\s]+(\d+)", blob, re.IGNORECASE)
  return time.time() + float(m.group(1)) if m else None


def detect_opencode_limit(*texts: str) -> tuple[bool, float | None]:
  return detect_limit(LIMIT_PATTERNS, parse_reset, *texts)


def detect_transient(*texts: str) -> bool:
  return matches_any(TRANSIENT_PATTERNS, *texts)


def agent_config(system_mode: SystemMode, tools: dict[str, bool] | None = None) -> dict:
  cfg: dict = {}
  if system_mode == "replace":
    cfg["prompt"] = _REPLACE_PLACEHOLDER
  if tools is not None:
    cfg["tools"] = tools
  return cfg


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def post_json(url: str, payload: dict, timeout: float) -> dict:
  req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                               headers={"content-type": "application/json"})
  with urllib.request.urlopen(req, timeout=timeout) as r:
    return json.loads(r.read().decode())


def get_json(url: str, timeout: float = 30) -> object:
  with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as r:
    return json.loads(r.read().decode())


class OpenCodeServer:
  def __init__(self, config: dict, binary: str = "opencode", startup_timeout: float = 60.0) -> None:
    self.config = config
    self.binary = binary
    self.startup_timeout = startup_timeout
    self._lock = threading.Lock()
    self._proc: subprocess.Popen | None = None
    self._url: str | None = None
    self._cwd: str | None = None

  @property
  def alive(self) -> bool:
    return self._proc is not None and self._proc.poll() is None

  def url(self) -> str:
    with self._lock:
      if self._url and self.alive:
        return self._url
      return self._start_locked()

  def _start_locked(self) -> str:
    self._cwd = self._cwd or tempfile.mkdtemp(prefix="opencode-srv-")
    with open(os.path.join(self._cwd, "opencode.json"), "w") as f:
      json.dump({"$schema": "https://opencode.ai/config.json", **self.config}, f)
    port = _free_port()
    self._proc = subprocess.Popen(
      [self.binary, "serve", "--port", str(port), "--hostname", "127.0.0.1"],
      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self._cwd,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + self.startup_timeout
    while time.time() < deadline:
      if self._proc.poll() is not None:
        raise HarnessError("opencode serve died on startup")
      try:
        urllib.request.urlopen(url, timeout=2)
      except urllib.error.HTTPError:
        pass   # any HTTP response means it is listening
      except Exception:
        time.sleep(0.5)
        continue
      self._url = url
      atexit.register(self.stop)
      return url
    raise HarnessError(f"opencode serve did not come up in {self.startup_timeout:.0f}s")

  def stop(self) -> None:
    with self._lock:
      if self.alive:
        assert self._proc is not None
        self._proc.terminate()
        try:
          self._proc.wait(timeout=5)
        except Exception:
          self._proc.kill()
      self._proc = None
      self._url = None


# one server per distinct config, shared by every caller of the pool
class OpenCodeServerPool:
  def __init__(self, binary: str = "opencode") -> None:
    self.binary = binary
    self._lock = threading.Lock()
    self._servers: dict[str, OpenCodeServer] = {}

  def get(self, config: dict) -> OpenCodeServer:
    key = json.dumps(config, sort_keys=True)
    with self._lock:
      if key not in self._servers:
        self._servers[key] = OpenCodeServer(config, binary=self.binary)
      return self._servers[key]

  def stop_all(self) -> None:
    with self._lock:
      servers, self._servers = list(self._servers.values()), {}
    for s in servers:
      s.stop()


# process-wide default so every OpenCodeCLI shares one `opencode serve` per config
SHARED_POOL = OpenCodeServerPool()


def split_model(model: str) -> tuple[str, str]:
  provider, _, model_id = model.partition("/")
  return (provider, model_id) if model_id else ("opencode", model)


def message_body(provider_id: str, model_id: str, prompt: str, *, system: str | None,
                 agent: str | None = None, variant: str | None = None) -> dict:
  body: dict = {"providerID": provider_id, "modelID": model_id,
                "parts": [{"type": "text", "text": prompt}]}
  if system:
    body["system"] = system
  if agent:
    body["agent"] = agent
  if variant:
    body["variant"] = variant
  return body


# one fresh session + one message; maps HTTP/provider failures onto the error hierarchy
def send_message(base: str, body: dict, timeout: float) -> tuple[str, dict]:
  try:
    sid = post_json(f"{base}/session", {"title": "openai-cli-agents"}, timeout=30).get("id")
    return sid, post_json(f"{base}/session/{sid}/message", body, timeout=timeout)
  except urllib.error.HTTPError as e:
    err = e.read().decode(errors="ignore") if hasattr(e, "read") else str(e)
    is_limit, reset_at = detect_opencode_limit(err, str(e))
    if is_limit:
      raise LimitError(err[:400], reset_at) from e
    if e.code in (500, 502, 503, 504) or detect_transient(err, str(e)):
      raise TransientError(f"HTTP {e.code}: {err[:300]}") from e
    raise HarnessError(f"HTTP {e.code}: {err[:400]}") from e
  except (urllib.error.URLError, TimeoutError) as e:
    raise HarnessError(f"opencode request failed: {e}") from e


def message_text(msg: dict) -> str:
  parts = msg.get("parts", []) or []
  return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()


def parse_message(sid: str, msg: dict) -> CLIResult:
  info = msg.get("info", {}) or {}
  parts = msg.get("parts", []) or []
  text = message_text(msg)
  finish = info.get("finish") or info.get("error")
  failed = bool(finish) and str(finish).lower() not in _OK_FINISH

  # a finished-with-error message (e.g. rate limit) carries no text
  if not text:
    blob = json.dumps(info) + json.dumps(parts)
    is_limit, reset_at = detect_opencode_limit(blob)
    if is_limit:
      raise LimitError(blob[:400], reset_at)
    if detect_transient(blob):
      raise TransientError(f"opencode transient: {blob[:300]}")
    if failed:
      raise HarnessError(f"opencode finish={finish}: {blob[:300]}")
    # an empty completion is a provider blip (the same prompt answers fine on retry)
    raise TransientError(f"empty response (finish={finish})")

  tokens = info.get("tokens", {}) or {}
  cache = tokens.get("cache", {}) or {}
  return CLIResult(
    text=text,
    is_error=failed,
    usage={"input_tokens": int(tokens.get("input", 0) or 0),
           "output_tokens": int(tokens.get("output", 0) or 0),
           "cache_read_input_tokens": int(cache.get("read", 0) or 0),
           "cache_creation_input_tokens": int(cache.get("write", 0) or 0)},
    cost_usd=float(info.get("cost", 0.0) or 0.0),
    num_turns=1,
    session_id=sid,
    raw=msg,
  )
