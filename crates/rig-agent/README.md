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
| 2 | `wasm32-unknown-unknown` (browser) | Supported. `rig-core/wasm` is enabled automatically; the `rmcp` feature is **not** available |
| — | `wasm32-wasip1` / `wasm32-wasip2` (WASI) | **Not supported** |
| — | `wasm32-unknown-emscripten` | Not supported |

Browser wasm is the only supported wasm target, so target-gated dependencies use
`cfg(all(target_arch = "wasm32", target_os = "unknown"))` rather than a bare
`target_arch = "wasm32"`. The `Send`-relaxed stream aliases in
`agent::prompt_request::streaming` use the same predicate, because that is
exactly where `rig-core`'s `WasmCompatSend`/`WasmCompatSync` markers go no-op.

**WASI is unsupported** because the dependency graph does not build for it, not
merely because it is untested: `rig-core` depends unconditionally on `reqwest`,
which pulls `hyper`/`socket2` and a tokio feature set that WASI rejects
(`Socket2 doesn't support the compile target`; `Only features
sync,macros,io-util,rt,time are supported on wasm`). Supporting WASI would mean
making `reqwest` optional and supplying a `wasi:http` client behind
`rig_core::http_client` — a deliberate project, not a `cfg` fix.

**`rmcp` is native-only.** rmcp's `ClientHandler` is declared
`Sized + Send + Sync + 'static` unconditionally — its `local` feature relaxes
the future bounds but not the handler itself — while this crate's handler owns a
tool registry whose `Arc<dyn ErasedTool>` is deliberately neither `Send` nor
`Sync` on wasm. Enabling `rmcp` on a wasm target fails with a single explanatory
`compile_error!` rather than a wall of trait errors.
