# rig-agent

`rig-agent` contains Rig's classic agent runtime: builders, the serializable
sans-I/O run state, blocking and streaming drivers, typed hooks, contextual
tools, extraction, and runtime integrations.

Most applications should use the root `rig` facade, where this runtime remains
enabled by default. Low-level provider and backend contracts live in
`rig-core`.

Direct users import construction and prompting explicitly:

```rust,ignore
use rig_agent::prelude::*;
use rig_core::{client::ProviderClient, providers::openai};

let client = openai::Client::from_env()?;
let agent = client.agent(openai::GPT_5_2).build();
let answer = agent.prompt("Explain ownership briefly.").await?;
```

Portable tools implement `rig_core::tool::PortableTool` and work in both runtimes.
Classic tools that need mutable per-call state implement
`rig_agent::tool::Tool` and receive `&mut ToolContext`.

## Target support

| Tier | Target | Status |
| --- | --- | --- |
| 1 | native (linux / macOS / windows, `x86_64` and `aarch64`) | Full support, all features including `rmcp` |
| 2 | `wasm32-unknown-unknown` (browser) | Supported, with no feature flags to set; the `rmcp` feature is **not** available |
| — | `wasm32-wasip1` / `wasm32-wasip2` (WASI) | **Not supported** |
| — | `wasm32-unknown-emscripten` | Not supported |

**Building for `wasm32-unknown-unknown` is the entire opt-in** — there are no
wasm feature flags anywhere in the workspace. `rig-core` relaxes its
`WasmCompat*` bounds from the target alone.

Wasm gates name a `target_os` (`all(target_arch = "wasm32", target_os =
"unknown")`) rather than a bare `target_arch = "wasm32"`, because the latter
also matches WASI, which has no JS host. WASI itself does not build: `rig-core`
depends unconditionally on `reqwest`, which pulls `hyper`/`socket2` and a tokio
feature set WASI rejects. Supporting it would mean making `reqwest` optional and
adding a `wasi:http` client behind `rig_core::http_client` — a project, not a
`cfg` fix.

**`rmcp` is native-only** — because of the transport, not the handler. The old
reason (rmcp's `ClientHandler` demanding `Send + Sync`, which this crate's wasm
tool registry cannot supply) was fixed upstream in rmcp 3.0, which gates that
bound behind its `local` feature; rmcp core with `client` + `local` (default
features off) compiles for `wasm32-unknown-unknown`. What that leaves is no
usable transport: the streamable-HTTP client calls `reqwest::redirect` and
`pool_max_idle_per_host`, neither of which exists in reqwest's wasm backend;
`Transport` itself is unconditionally `Send` with `Send` futures (`local`
relaxes handlers, never transports), which no browser-`JsFuture`-backed
implementation can satisfy; and the remaining concrete transports are
child-process, unix-socket, and stdio. Reaching an MCP server from the browser
therefore needs upstream `Send`-bound relaxations, a hand-written `Transport`
over fetch/EventSource, and a spawn shim (rmcp's `local` spawns onto a tokio
`LocalSet`, not a browser event loop) — a feature project, not a `cfg` fix.
Enabling `rmcp` on a wasm target fails with a single explanatory
`compile_error!` rather than a wall of trait errors.
