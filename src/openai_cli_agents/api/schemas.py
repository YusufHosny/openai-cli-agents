# ABOUTME: Pydantic wire models for the OpenAI chat-completions surface (requests, responses,
# ABOUTME: stream chunks, errors) plus the request-side rules for what a CLI backend can honour.
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from openai_cli_agents.core import JSON_OBJECT_FMT, STRUCTURED_FMT, CLIResult, Turn, flatten_turns

ErrorType = Literal[
  "invalid_request_error", "authentication_error", "not_found_error", "rate_limit_error",
  "api_error", "service_unavailable",
]


# raised anywhere in request handling; rendered as OpenAI's {"error": {...}} envelope
class APIError(Exception):
  def __init__(self, status: int, message: str, type: ErrorType = "invalid_request_error",
               param: str | None = None, code: str | None = None,
               headers: dict[str, str] | None = None) -> None:
    super().__init__(message)
    self.status = status
    self.message = message
    self.type = type
    self.param = param
    self.code = code
    self.headers = headers or {}

  def body(self) -> dict:
    return {"error": {"message": self.message, "type": self.type, "param": self.param,
                      "code": self.code}}


def unsupported(param: str, what: str) -> APIError:
  return APIError(400, f"{what} is not supported by openai-cli-agents (headless CLI backends).",
                  param=param, code="unsupported_parameter")


# ---- request ----

class ChatMessage(BaseModel):
  model_config = ConfigDict(extra="allow")

  role: Literal["system", "developer", "user", "assistant", "tool", "function"]
  content: str | list[dict[str, Any]] | None = None
  name: str | None = None
  tool_calls: list[dict[str, Any]] | None = None

  # text parts are concatenated; anything else (images, audio, files) has no CLI equivalent
  def text(self) -> str:
    match self.content:
      case None:
        return ""
      case str(s):
        return s
      case parts:
        for p in parts:
          if p.get("type") != "text":
            raise unsupported("messages", f"content part type {p.get('type')!r}")
        return "".join(str(p.get("text", "")) for p in parts)

  def turn(self) -> Turn:
    match self.role:
      case "tool" | "function":
        raise unsupported("messages", f"a {self.role!r} message (tool calling)")
      case "assistant" if self.tool_calls:
        raise unsupported("messages", "assistant tool_calls (tool calling)")
      case "system" | "developer":
        return Turn("system", self.text())
      case role:
        return Turn(role, self.text())


class JSONSchemaFormat(BaseModel):
  model_config = ConfigDict(populate_by_name=True)

  name: str = "response"
  description: str | None = None
  schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
  strict: bool | None = None


class ResponseFormat(BaseModel):
  type: Literal["text", "json_object", "json_schema"] = "text"
  json_schema: JSONSchemaFormat | None = None

  # prompt-enforced: the CLIs have no constrained decoding, so the format becomes an instruction
  def instruction(self) -> str | None:
    match self.type:
      case "text":
        return None
      case "json_object":
        return JSON_OBJECT_FMT
      case "json_schema":
        schema = self.json_schema.schema_ if self.json_schema else {}
        return STRUCTURED_FMT.format(schema=json.dumps(schema))


class StreamOptions(BaseModel):
  include_usage: bool = False


# sampling knobs (temperature, top_p, max_tokens, stop, seed, ...) are accepted and ignored:
# extra="allow" keeps clients that always send them working
class ChatCompletionRequest(BaseModel):
  model_config = ConfigDict(extra="allow")

  model: str
  messages: list[ChatMessage] = Field(min_length=1)
  stream: bool = False
  stream_options: StreamOptions | None = None
  n: int | None = None
  response_format: ResponseFormat | None = None
  tools: list[dict[str, Any]] | None = None
  functions: list[dict[str, Any]] | None = None
  tool_choice: Any = None

  @property
  def wants_json(self) -> bool:
    return self.response_format is not None and self.response_format.type != "text"

  @property
  def include_usage(self) -> bool:
    return self.stream_options is not None and self.stream_options.include_usage

  def check_supported(self) -> None:
    if self.tools or self.functions:
      raise unsupported("tools", "tool / function calling")
    if self.n not in (None, 1):
      raise unsupported("n", "n > 1")

  # (system or None, flattened transcript), with any response_format folded into the system
  def system_and_prompt(self) -> tuple[str | None, str]:
    system, prompt = flatten_turns([m.turn() for m in self.messages])
    if not prompt.strip():
      raise APIError(400, "messages must contain at least one non-empty user or assistant "
                     "message.", param="messages")
    if self.response_format and (instr := self.response_format.instruction()):
      system = f"{system}\n\n{instr}" if system else instr
    return system or None, prompt


# ---- response ----

class PromptTokensDetails(BaseModel):
  cached_tokens: int = 0


class Usage(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  prompt_tokens_details: PromptTokensDetails

  @classmethod
  def from_result(cls, res: CLIResult) -> Usage:
    prompt = res.total_input_tokens
    return cls(prompt_tokens=prompt, completion_tokens=res.output_tokens,
               total_tokens=prompt + res.output_tokens,
               prompt_tokens_details=PromptTokensDetails(cached_tokens=res.cache_read_tokens))


class AssistantMessage(BaseModel):
  role: Literal["assistant"] = "assistant"
  content: str


class Choice(BaseModel):
  index: int = 0
  message: AssistantMessage
  finish_reason: Literal["stop"] = "stop"


class Delta(BaseModel):
  role: Literal["assistant"] | None = None
  content: str | None = None


class ChunkChoice(BaseModel):
  index: int = 0
  delta: Delta
  finish_reason: Literal["stop"] | None = None


# id/created/model shared by a completion and every chunk of its stream
class CompletionMeta(BaseModel):
  id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
  created: int = Field(default_factory=lambda: int(time.time()))
  model: str


class ChatCompletion(CompletionMeta):
  object: Literal["chat.completion"] = "chat.completion"
  choices: list[Choice]
  usage: Usage

  @classmethod
  def from_result(cls, meta: CompletionMeta, res: CLIResult) -> ChatCompletion:
    return cls(**meta.model_dump(), choices=[Choice(message=AssistantMessage(content=res.text))],
               usage=Usage.from_result(res))


class ChatCompletionChunk(CompletionMeta):
  object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
  choices: list[ChunkChoice]
  usage: Usage | None = None

  @classmethod
  def of(cls, meta: CompletionMeta, delta: Delta, finish: bool = False) -> ChatCompletionChunk:
    return cls(**meta.model_dump(),
               choices=[ChunkChoice(delta=delta, finish_reason="stop" if finish else None)])

  # the trailing stream_options.include_usage chunk: no choices, only usage
  @classmethod
  def usage_only(cls, meta: CompletionMeta, res: CLIResult) -> ChatCompletionChunk:
    return cls(**meta.model_dump(), choices=[], usage=Usage.from_result(res))


class Model(BaseModel):
  id: str
  object: Literal["model"] = "model"
  created: int = 0
  owned_by: str


class ModelList(BaseModel):
  object: Literal["list"] = "list"
  data: list[Model]
