# ABOUTME: FastAPI app exposing the CLI backends as an OpenAI-compatible API: /v1/models and
# ABOUTME: /v1/chat/completions (plain, SSE streaming, prompt-enforced response_format).
from __future__ import annotations

import dataclasses
import itertools
import json
import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from openai_cli_agents.api.backends import Backends, CompletionBackend, ModelRoute, resolve_model
from openai_cli_agents.api.schemas import (
  APIError, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, CompletionMeta, Delta,
  Model, ModelList,
)
from openai_cli_agents.core import (
  CLIResult, HarnessError, LimitError, StreamEvent, TextDelta, TransientError, extract_json,
)

DEFAULT_MODELS = ("claude/sonnet", "claude/opus", "claude/haiku")
_JSON_RETRY = ("\n\nYour previous output was not a valid JSON object. "
               "Return the JSON object only.")


@dataclass(frozen=True)
class ServerConfig:
  models: tuple[str, ...] = DEFAULT_MODELS   # advertised by /v1/models; any routable name works
  api_key: str | None = None                 # when set, required as `Authorization: Bearer <key>`
  structured_retries: int = 3


def _harness_error(e: HarnessError) -> APIError:
  match e:
    case LimitError():
      headers = ({"retry-after": str(max(1, int(e.reset_at - time.time())))}
                 if e.reset_at else {})
      return APIError(429, str(e), type="rate_limit_error", code="rate_limit_exceeded",
                      headers=headers)
    case TransientError():
      return APIError(503, str(e), type="service_unavailable")
    case _:
      return APIError(502, str(e), type="api_error")


def _sse(payload: BaseModel | dict) -> str:
  data = (payload.model_dump_json(exclude_none=True) if isinstance(payload, BaseModel)
          else json.dumps(payload))
  return f"data: {data}\n\n"


# OpenAI chunk sequence: role chunk, content chunks, finish chunk, optional usage chunk, [DONE]
def _sse_chunks(meta: CompletionMeta, first: StreamEvent, events: Iterator[StreamEvent],
                include_usage: bool) -> Iterator[str]:
  yield _sse(ChatCompletionChunk.of(meta, Delta(role="assistant", content="")))
  try:
    for ev in itertools.chain([first], events):
      match ev:
        case TextDelta(text=text):
          yield _sse(ChatCompletionChunk.of(meta, Delta(content=text)))
        case CLIResult() as res:
          yield _sse(ChatCompletionChunk.of(meta, Delta(), finish=True))
          if include_usage:
            yield _sse(ChatCompletionChunk.usage_only(meta, res))
  except HarnessError as e:
    # headers are already sent, so a mid-stream failure becomes an error event
    yield _sse(_harness_error(e).body())
  finally:
    # a client disconnect stops this generator early; closing the backend's kills its subprocess
    if (close := getattr(events, "close", None)) is not None:
      close()
  yield "data: [DONE]\n\n"


def create_app(config: ServerConfig | None = None,
               backends: CompletionBackend | None = None) -> FastAPI:
  cfg = config or ServerConfig()
  backend: CompletionBackend = backends if backends is not None else Backends()
  app = FastAPI(title="openai-cli-agents")

  @app.exception_handler(APIError)
  def _api_error(_: Request, e: APIError) -> JSONResponse:
    return JSONResponse(e.body(), status_code=e.status, headers=e.headers)

  @app.exception_handler(HarnessError)
  def _harness(_: Request, e: HarnessError) -> JSONResponse:
    err = _harness_error(e)
    return JSONResponse(err.body(), status_code=err.status, headers=err.headers)

  @app.exception_handler(RequestValidationError)
  def _invalid(_: Request, e: RequestValidationError) -> JSONResponse:
    first = e.errors()[0] if e.errors() else {}
    param = ".".join(str(p) for p in first.get("loc", ())[1:]) or None
    err = APIError(400, f"Invalid request: {first.get('msg', 'validation failed')}", param=param)
    return JSONResponse(err.body(), status_code=400)

  def _auth(request: Request) -> None:
    if cfg.api_key is None:
      return
    given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(given, cfg.api_key):
      raise APIError(401, "Incorrect API key provided.", type="authentication_error",
                     code="invalid_api_key")

  def _model(name: str) -> Model:
    route = resolve_model(name)
    return Model(id=name, owned_by=f"{route.kind}-cli")

  @app.get("/v1/models", dependencies=[Depends(_auth)])
  def list_models() -> ModelList:
    return ModelList(data=[_model(m) for m in cfg.models])

  @app.get("/v1/models/{model_id:path}", dependencies=[Depends(_auth)])
  def get_model(model_id: str) -> Model:
    return _model(model_id)

  def _complete_json(route: ModelRoute, prompt: str, system: str | None) -> CLIResult:
    res = backend.complete(route, prompt, system)
    for _ in range(cfg.structured_retries):
      if (obj := extract_json(res.text)) is not None:
        return dataclasses.replace(res, text=json.dumps(obj))
      res = backend.complete(route, prompt + _JSON_RETRY, system)
    return res

  # sync handler: FastAPI runs it (and the stream generator) in its threadpool, so concurrent
  # requests each block on their own CLI subprocess
  @app.post("/v1/chat/completions", dependencies=[Depends(_auth)], response_model=None)
  def chat_completions(req: ChatCompletionRequest) -> ChatCompletion | StreamingResponse:
    req.check_supported()
    route = resolve_model(req.model)
    system, prompt = req.system_and_prompt()
    meta = CompletionMeta(model=req.model)

    if not req.stream:
      res = (_complete_json(route, prompt, system) if req.wants_json
             else backend.complete(route, prompt, system))
      return ChatCompletion.from_result(meta, res)

    # JSON must be validated whole, so it is completed first and replayed as one delta
    if req.wants_json:
      res = _complete_json(route, prompt, system)
      events: Iterator[StreamEvent] = iter([TextDelta(res.text), res])
    else:
      events = backend.stream(route, prompt, system)
    # pull the first event before responding so startup failures (limits, bad model) get a
    # proper HTTP status instead of a 200 stream carrying an error
    first = next(events)
    return StreamingResponse(_sse_chunks(meta, first, events, req.include_usage),
                             media_type="text/event-stream")

  return app
