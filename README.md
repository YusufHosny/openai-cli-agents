# openai-cli-agents

Use coding-agent CLIs (`claude -p`, `opencode`) as plain LLM completions, and serve them behind an
OpenAI-compatible API so any OpenAI client can run on a Claude subscription or a free opencode model.

| layer | what it is | install |
|---|---|---|
| `ClaudeCLI`, `OpenCodeCLI` | one stateless call → `CLIResult` (text, usage, cost), or a stream of `TextDelta`s | `openai-cli-agents` (stdlib only) |
| `openai-cli-agents serve` | `/v1/models` + `/v1/chat/completions` (plain and SSE streaming) | `openai-cli-agents[server]` |

The LangChain chat models and MCP tool-calling agents built on this live in
[`langchain-cli-agents`](https://github.com/YusufHosny/langchain-cli-agents).

## Server

```bash
uv tool install "openai-cli-agents[server] @ git+https://github.com/YusufHosny/openai-cli-agents"
openai-cli-agents serve --port 8000                   # binds 127.0.0.1, no auth
openai-cli-agents serve --api-key "$KEY" -m claude/sonnet -m opencode/big-pickle
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="claude/sonnet", messages=[{"role": "user", "content": "hi"}])
```

Model names: `claude/<model>` or a bare claude name (`sonnet`, `opus`, `haiku`, `claude-…`) go to
`claude -p`; any other `provider/model` (`opencode/big-pickle`, `openrouter/qwen/qwen3`) goes to
opencode. `/v1/models` lists the `-m` models; any routable name is accepted.

| feature | behaviour |
|---|---|
| messages | flattened into one transcript (`USER: …\n\nASSISTANT: …`); `system`/`developer` become the system prompt |
| `stream` | claude: real token deltas from `--output-format stream-json`; opencode: the full text in one chunk |
| `stream_options.include_usage` | trailing usage-only chunk |
| usage | `prompt_tokens` includes cached tokens, `prompt_tokens_details.cached_tokens` = cache reads |
| `response_format` (`json_object`, `json_schema`) | prompt-enforced, parsed and retried; content is the extracted JSON |
| `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, … | accepted and ignored (the CLIs do not expose them) |
| `tools` / `functions`, `n > 1`, image/audio parts, `tool` messages | 400 `unsupported_parameter` |
| usage limit / provider rate limit | 429 `rate_limit_exceeded` with `Retry-After` when the reset time is known |
| transient provider failure / other CLI failure | 503 / 502 |

Every request is a fresh, tool-free, single-turn CLI session; nothing is persisted.

## Library

```python
from openai_cli_agents.claude import ClaudeCLI
from openai_cli_agents.opencode import OpenCodeCLI

ClaudeCLI(model="sonnet").invoke("Capital of France?", system="Answer in one word.").text
for ev in ClaudeCLI(model="haiku").stream("Count to 5"):
  ...   # TextDelta(text=...) as generated, then the final CLIResult
OpenCodeCLI(model="opencode/big-pickle").invoke("hi")
```

Library calls auto-pause on subscription limits by default (`auto_pause=True`), sleeping until the
reset time; the server turns that off and returns 429s instead.
