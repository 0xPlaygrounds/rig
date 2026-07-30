# rig-agent

`rig-agent` contains Rig's agent runtime: builders, the serializable sans-I/O
run state, blocking and streaming session drivers, record-based hooks, tool
execution, extraction, and runtime integrations.

Most applications should use the root `rig` facade, where this runtime remains
enabled by default. Low-level provider and backend contracts live in
`rig-core`.

An agent is built from a `ProviderConfig` — plain serde configuration, not a
model object:

```rust,ignore
use rig_agent::prelude::*;
use rig_agent::{AgentBuilder, ProviderConfig};
use rig_core::providers::openai;

let cfg = openai::functions::Config::from_env(openai::GPT_5_2)?;
let agent = AgentBuilder::new(ProviderConfig::OpenAi(cfg))
    .preamble("Explain things briefly.")
    .build();
let answer = agent.prompt("Explain ownership briefly.").await?;
```

Tools implement `rig_core::tool::PortableTool`, the single portable record
contract. There is no separate contextual tool trait and no `ToolContext`:
a tool that needs per-call state owns it in the implementing struct, and
collections are assembled with `rig::executor::ToolExecutor`.

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

**`rmcp` is native-only.** rmcp's `ClientHandler` is declared
`Sized + Send + Sync + 'static` unconditionally — its `local` feature relaxes
the future bounds but not the handler itself — while this crate's handler owns a
tool registry whose `Arc<dyn ErasedTool>` is deliberately neither `Send` nor
`Sync` on wasm. Enabling `rmcp` on a wasm target fails with a single explanatory
`compile_error!` rather than a wall of trait errors.
