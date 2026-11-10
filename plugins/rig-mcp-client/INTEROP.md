# MCP and ACP interop

Real servers and clients rig was run against, by hand, with the version and
date of the last run. Run the matrix again at each release and update the
rows. The automated check of the streamable HTTP transport is the loopback
test `a_streamable_http_server_connects_lists_answers_and_changes_its_tools`
in `src/tests.rs`.

Each run used a disposable home (`HOME`, `XDG_*`, `RIG_HOME`) with
`rig_mcp_client::McpPlugin` and `rig_acp::AcpPlugin` on, and a scripted
local model (`ollama/qwen3` at `OLLAMA_API_BASE_URL`) that calls the listed
tools in order. `rig -p --json` ran the MCP rows; the terminal view also
connected the filesystem server over stdio and server-everything over HTTP
and listed both with `/mcp`.

## MCP servers

Date 2026-11-09, rig commit `b1321662b`, rmcp 3.9.0, Node 26.13.0.

| Server | Version | Transport | Result | Notes |
| --- | --- | --- | --- | --- |
| `@modelcontextprotocol/server-filesystem` | 2.2.0 | stdio | works | 16 tools; `read_text_file` and `list_directory` answered. |
| `@modelcontextprotocol/server-filesystem` | 2.2.0 through supergateway 4.2.1 | streamable HTTP | works | The package serves only stdio; supergateway (`--outputTransport streamableHttp --stateful`) bridged it. Same calls and answers. |
| `@modelcontextprotocol/server-everything` | 2.2.1 | stdio | works | `get-sum`, `get-structured-content`, `get-tiny-image` (image content), `trigger-long-running-operation` (progress), `trigger-list-changed`, `get-header-params`, `echo`. |
| `@modelcontextprotocol/server-everything` | 2.2.1 | streamable HTTP | works | The same seven calls, with a custom header in `headers`. |
| `@playwright/mcp` with Chromium 156 | 0.0.85 | stdio, `--headless --isolated` | works | 25 tools; `browser_navigate` to a local page, `browser_snapshot`, `browser_close`. |
| GitHub MCP, `https://api.githubcopilot.com/mcp/` | `github-mcp-server/remote-b104ac9bf784` | streamable HTTP, `Authorization` and `X-MCP-Readonly` headers | works | Run with #3227, not `b1321662b`; see below. |

`get-header-params` answered `"regionHeader": null`. rmcp 3.9 copies a tool
argument marked `x-mcp-header` into an `Mcp-Param-*` header only on a session
that negotiated `2026-07-28`, and this session did not carry it. At
`b1321662b` the conformance suite's `http-custom-headers` passed 18/18 on
such a session (below); since #3227 rig offers `2025-11-25` and never
negotiates `2026-07-28`, so that no longer applies.

GitHub MCP connects with #3227 (2026-11-09): `rig -p --json`
said `MCP server gh connected: 29 tools.` and the scripted model's call to
`gh__get_me` answered. At `b1321662b` it did not connect (`Transport channel
closed, when send initialized notification.`): rig offered the draft
`2026-07-28`, and the server answered `initialize` with `2025-11-25` and a
session id, then refused that session with `400 invalid session`. With curl
the same 400 comes when the client offers `2026-07-28`, and 202 when it
offers `2025-11-25` or `2025-06-18`. Rig now offers `2025-11-25`.

## ACP clients

Date 2026-11-09, rig commit `b1321662b`.

| Client | Version | Transport | Result | Notes |
| --- | --- | --- | --- | --- |
| Scripted client on `@agentclientprotocol/sdk` (TypeScript) | 2.5.0 | stdio, ACP v1 | works | `initialize`, `session/new` with an editor's MCP server (stdio, then streamable HTTP with a header), `session/set_config_option` for the model, `session/prompt` with a built-in tool and an MCP tool (`tool_call` and `tool_call_update` updates, `end_turn`), `session/list`, `session/close`. No permission request came up: no approval plugin was on. |
| Zed | | | not run | A GUI editor; not run here. |
| JetBrains | | | not run | A GUI editor; not run here. |
| Neovim with CodeCompanion | | | not run | Not tried in this run. |

## Conformance suites

Run once by hand on 2026-11-09 against rig commit `b1321662b`.

- MCP conformance suite (`modelcontextprotocol/conformance` at
  `c37eec888e1c`), rig as the client: 11 of 48 client scenarios pass. Every
  failure is unsupported OAuth (33) or elicitation (1), or a `2026-07-28`
  draft item that comes from rmcp 3.9's stateful `initialize` lifecycle (3).
- ACP TCK (`agentclientprotocol/acp-tck` at `b7f40fb3c003`), `rig acp`, ACP
  v1: all 31 mandatory requirements pass. The 5 failures (`ACP-LOAD-001` to
  `003`, `ACP-RESUME-001` and `002`) come from rig refusing to load or resume
  a session the same connection already has open, which it does on purpose.
