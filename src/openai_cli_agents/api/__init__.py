# ABOUTME: OpenAI-compatible HTTP API over the CLI harnesses: /v1/models and /v1/chat/completions
# ABOUTME: (plain and SSE streaming). Needs the [server] extra (fastapi, uvicorn, typer).
from openai_cli_agents.api.app import ServerConfig, create_app
from openai_cli_agents.api.backends import Backends, ModelRoute, resolve_model

__all__ = ["Backends", "ModelRoute", "ServerConfig", "create_app", "resolve_model"]
