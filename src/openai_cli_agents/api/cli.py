# ABOUTME: `openai-cli-agents serve` — run the OpenAI-compatible server over the local claude / opencode
# ABOUTME: CLIs, so any OpenAI client can point its base_url at a subscription-backed model.
from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


# a callback keeps `serve` an explicit subcommand even while it is the only one
@app.callback()
def main() -> None:
  pass


@app.command(help="Serve /v1/models and /v1/chat/completions over the local CLIs.")
def serve(
  host: str = typer.Option("127.0.0.1", help="Interface to bind."),
  port: int = typer.Option(8000, help="Port to bind."),
  model: list[str] = typer.Option(
    None, "--model", "-m",
    help="Model advertised by /v1/models (repeatable). Defaults to claude/sonnet, claude/opus, "
         "claude/haiku; any routable name is accepted regardless."),
  api_key: str | None = typer.Option(
    None, envvar="CLI_OPENAI_API_KEY",
    help="Require `Authorization: Bearer <key>`. Unset means no auth (bind to localhost)."),
  timeout: int = typer.Option(600, help="Per-request CLI timeout in seconds."),
  claude_binary: str = typer.Option("claude", help="Path to the claude CLI."),
  opencode_binary: str = typer.Option("opencode", help="Path to the opencode CLI."),
) -> None:
  import uvicorn

  from openai_cli_agents.api.app import DEFAULT_MODELS, ServerConfig, create_app
  from openai_cli_agents.api.backends import Backends

  config = ServerConfig(models=tuple(model) if model else DEFAULT_MODELS, api_key=api_key)
  backends = Backends(timeout=timeout, claude_binary=claude_binary,
                      opencode_binary=opencode_binary)
  uvicorn.run(create_app(config, backends), host=host, port=port)
