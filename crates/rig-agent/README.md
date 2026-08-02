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

`ProviderConfig` and `EmbedderConfig` are generated from one capability
registry. Their closed enum shapes do not change with Cargo feature
unification: Bedrock and Gemini gRPC config variants are always serializable
and exhaustively matchable, while attempting fulfillment without the matching
transport feature returns a boundary error. The embedding vocabulary includes
Llamafile, Mistral, OpenRouter, and Together. FastEmbed remains outside the
enum because loaded local weights are runtime state rather than honest serde
configuration.

Tools implement `rig_core::tool::PortableTool`, the single portable record
contract. There is no separate contextual tool trait and no `ToolContext`:
a tool that needs per-call state owns it in the implementing struct, and
collections are assembled with `rig::executor::ToolExecutor`.

Runtime-authored tools and hooks accept ordinary async closures; callers do
not box or pin their futures:

```rust,ignore
let tool = PortableDynamicTool::new(name, description, schema, |args| async move {
    Ok(ToolOutput::json(args))
});

let hook = HookEntry::with_state("audit", state, |state, event| async move {
    state.inspect(event).await;
    HookDecision::Continue
});
```

The records stay concrete and non-generic. Only their callback fields are
erased privately behind `Arc<dyn Fn + Send + Sync>`; each invocation is boxed
inside Rig. Stateful hooks receive an owned `Arc<S>`, so the returned future
can safely use the shared state across awaits. A sub-agent can be converted
directly into the same concrete tool record with
`agent.into_tool(name, description)`.

**Stored configuration is shareable; execution may remain worker-local.** Hook
callbacks and `HookEntry::with_state` values are `Send + Sync` on every target,
while the future returned by a hook uses `WasmCompatSend` and may be non-`Send`
on browser wasm. Per-invocation `Rc`, `RefCell`, network futures, Promises, and
JavaScript handles can live inside that future. Keep persistent
JavaScript-affine state in `thread_local!`; use `Arc<Mutex<_>>` or
`Arc<RwLock<_>>` for ordinary shared Rust state, and never add an unsafe
`Send`/`Sync` implementation for a JavaScript handle. See the checked,
copy-pasteable [`wasm_hooks` example](../../examples/wasm_hooks/README.md).

## Target support

| Tier | Target | Status |
| --- | --- | --- |
| 1 | native (linux / macOS / windows, `x86_64` and `aarch64`) | Full support, all features including `rmcp` |
| 2 | `wasm32-unknown-unknown` (browser) | Supported, with no feature flags to set; the `rmcp` feature is **not** available |
| — | `wasm32-wasip1` / `wasm32-wasip2` (WASI) | **Not supported** |
| — | `wasm32-unknown-emscripten` | Not supported |

**Building for `wasm32-unknown-unknown` is the entire opt-in** — there are no
wasm feature flags anywhere in the workspace. `rig-core` relaxes future,
stream, and invocation-local argument/output/error `WasmCompat*` bounds from
the target alone; values retained by an agent, including hook callbacks/state
and tools, remain `Send + Sync` on every target.

Wasm gates name a `target_os` (`all(target_arch = "wasm32", target_os =
"unknown")`) rather than a bare `target_arch = "wasm32"`, because the latter
also matches WASI, which has no JS host. WASI itself does not build: `rig-core`
depends unconditionally on `reqwest`, which pulls `hyper`/`socket2` and a tokio
feature set WASI rejects. Supporting it would mean making `reqwest` optional and
adding a `wasi:http` client behind `rig_core::http_client` — a project, not a
`cfg` fix.

**`rmcp` is native-only.** The companion integration's rmcp/tokio client
machinery and cancellation dispatch remain target-gated, so `rig-rmcp`
compiles to an empty library on wasm. This is an integration boundary, not a
limitation of the agent's tool registry, which is `Send + Sync` on every target.
