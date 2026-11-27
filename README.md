# Uzda

**A small AI agent harness runtime, configured in YAML and built as a Rust binary.**

Uzda runs a model in a harness: a set of instructions, tools, skills, stages, and limits. It connects to a streaming OpenAI-compatible endpoint, executes configured tools, records a trace, and streams events to clients. [Sedlo](assets/sedlo.html) is the included web interface.

Uzda is an early-stage agent constructor and runtime—not a model, a sandbox, or a hosted service. It is designed to keep agent configuration inspectable and execution understandable without requiring a database or orchestration platform.

## Why a harness?

An agent needs more than a prompt. It needs a way to use tools, carry context, work through stages, and show what happened. Uzda puts those parts in a declarative harness instead of hiding them in application code.

- **Configuration over scaffolding.** Define prompts, tools, skills, stages, and limits in YAML; keep instructions in ordinary files.
- **A small runtime.** Run a single Rust executable. A model endpoint and any configured external tools remain separate dependencies.
- **Visible execution.** Stream responses and tool events over SSE, inspect YAML traces, and resume a conversation by trace ID.
- **Composable tools.** Use existing command-line programs rather than building every capability into the runtime.
- **Practical boundaries.** Keep the core focused. Isolation, access controls, and deployment infrastructure belong to the environment where they are needed.

These are design goals, not guarantees of model accuracy, tool safety, or production readiness. See [Limitations](#limitations-and-security).

## Quick start

You need Rust with edition 2024 support and an OpenAI-compatible Chat Completions endpoint that supports streaming and tool calls. Uzda does not include a model. `HOME` must be set for trace storage.

1. Set `model.url` and `model.name` in [`examples/chat.yaml`](examples/chat.yaml). The example defaults to `http://localhost:9931/v1` and `default`.
2. Start the harness:

   ```sh
   cargo run -- serve examples/chat.yaml
   ```

3. Open [http://localhost:8080/sedlo.html](http://localhost:8080/sedlo.html).

Or call the API directly:

```sh
curl -N http://localhost:8080/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"Hello!"}'
```

Build a release binary with `cargo build --release`; run `uzda --help` for CLI options. The chat example does not execute external commands. See [`examples/README.md`](examples/README.md) for tool and multi-stage harnesses.

## A harness in YAML

```yaml
model:
  url: http://localhost:9931/v1
  name: default

prompts:
  - name: base
    path: prompts/base.md

tools:
  - name: read
    run: cat
    desc: Read a file.

stages:
  - name: investigate
    desc: Gather information.
    prompt: [base]
    tools: [read]
    next: [review]
  - name: review
    desc: Review and answer.
    prompt: [base]

limits:
  turns: 30
  time: 120
```

File paths in a manifest are resolved relative to its directory. If `stages` is omitted, Uzda creates a default stage containing the configured prompts, tools, and skills. Transitions listed in `next` are offered as `transfer_to_<stage>` tool calls; a model can still choose not to call them.

| Section | Purpose |
| --- | --- |
| `model` | Base URL and name for the model endpoint. |
| `prompts` | Named instruction files; `pin` is the supported policy. |
| `tools` | External commands, with optional timeout, argument schema, shell mode, UID and GID. |
| `skills` | Named instruction files invoked by the model as tool calls. |
| `stages` | Per-stage prompts, tools, skills, history, and allowed transitions. |
| `messages` / `history` | Named history files and stage history selection. |
| `limits` | Maximum model turns and optional session time in seconds. |

Tool schemas are separate YAML files with `params` and optional `argv` templates. See [`examples/read-document.yaml`](examples/read-document.yaml). Without a schema, tools accept an argument string. `shell: true` enables shell interpretation and cannot be combined with a schema.

A skill is loaded when called and added to the next planning request. The resulting plan is kept in the trace; the runtime does not add the skill instruction to subsequent requests. A failed or interrupted session does not currently restore a pending skill on resume.

## HTTP API

`POST /run` and `POST /run/{id}` return server-sent events. `session_start` contains the trace ID; events include `delta`, `answer`, `call`, `result`, `stage_enter`, `error`, and `done`.

| Endpoint | Description |
| --- | --- |
| `POST /run` | Start a session with JSON `query`; optional `harness` and `hitch`. |
| `POST /run/{id}` | Resume a trace with a new `query`. |
| `POST /say` | Queue input for an active session: `trace_id`, `text`, optional `from`. |
| `POST /stop/{id}` | Request a session stop. |
| `GET /events/{id}?since=<seq>` | Follow events for an active session. |
| `GET /trace/{id}?tail=<n>` | Read a trace, optionally limited to its last entries. |
| `GET /traces` | List recent traces. |
| `GET /sessions` | List active sessions. |
| `GET /health` | Check server availability. |
| `GET /sedlo.html` | Open the bundled web interface. |

Event replay is bounded and available only while the session is active. A message accepted by `/say` may not be processed before the session ends. Alternative harness paths must be inside the default manifest directory.

## Configuration

| Environment variable | Purpose |
| --- | --- |
| `UZDA_ADDR` | Listen address; default `127.0.0.1:8080`. |
| `UZDA_API_KEY` | Bearer token sent to the model endpoint. |
| `UZDA_TOKEN` | Require a bearer token for API requests. |
| `UZDA_CORS_ORIGIN` | Additional allowed web origin or `*`. |
| `HOME` | Base directory for `~/.uzda/traces`. |

The HTTP host guard accepts localhost names and loopback addresses. Local origins are allowed by default. Sedlo does not currently provide a field for `UZDA_TOKEN`; use an authenticated client or an appropriate proxy when a token is configured.

## Traces and observability

Sessions append YAML records to `~/.uzda/traces/<id>.yaml`. Records include user and model messages, tool calls and results, and stage transitions. Runtime lifecycle events are logged to the server terminal; SSE provides live events. Traces are not encrypted or automatically redacted: they and the model endpoint may receive sensitive inputs and tool output.

## Limitations and security

- **Tools are not sandboxed.** Commands run with the runtime process's permissions; `shell: true` permits arbitrary shell commands. Use least privilege and external isolation where appropriate.
- Model output, tool results, and documents can be wrong or contain adversarial instructions. Prompts are not an authorization boundary.
- Stop requests and session deadlines do not guarantee immediate termination of subprocesses or their descendants.
- Tool output and trace sizes are bounded; output may be truncated. Resume and event replay have limitations, including no coordination across server processes.
- The document examples handle text files. Office extraction, OCR, and writing office formats require additional tools and are not built in.
- Sedlo accepts text attachments up to 64 KiB; it does not extract binary documents. Its bundled MathJax resources do not require a CDN.
- Model providers differ. Uzda expects streaming Chat Completions responses with a `[DONE]` marker; compatibility is not universal.

## Development

```sh
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

## License

[MIT](LICENSE)
