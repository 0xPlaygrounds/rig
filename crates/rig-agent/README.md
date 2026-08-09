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

## Runtime model routing

High-level agents are concrete values: the provider model is erased once into
an opaque, cloneable `ModelHandle`, whose `ProviderCapabilities` snapshot is
captured by value at erasure. Provider authors still implement the typed
`CompletionModel` trait, and direct `completion` or `stream` calls (plus each
provider's `raw_*` escape hatches) retain their provider-specific behavior.

Replace the default on one agent value with `set_model` or
`set_model_handle`, or change one run's default candidate with `using_model`.
Implement `AgentHook::on_model_select` to route before every model call. Model
selection hooks chain in registration order, with the last selection winning
and a stop terminating the run. Per model-call boundary, completion-call hooks
resolve first and their merged `RequestPatch` is passed to the selection event
(`ModelSelection::request_patch`); only then does request preparation run
against the selected model's captured capabilities and issue the attempt. A
selected handle cannot change while its future or stream is in flight. Retries
and calls after tool execution are new boundaries and may select another
handle.

Hooks attached through `AgentBuilder::add_hook` apply to every later runner
from that agent; hooks appended to a prompt or runner apply only to that run.
`using_model` changes the run's initial candidate but does not suppress routing
hooks. To force a model, omit routing hooks or append a final hook that always
selects it. Blocking and streaming prompts share this lifecycle: `Stop`
cancels before request preparation and provider execution, and dropping an
in-flight attempt still cancels it by dropping its retained future or stream.

Extractors support the same run-local choice through
`extractor.using_model(handle).extract(...)` or `using_model_value(model)`.
That handle is the default candidate for each extraction retry, routing hooks
may replace it, and the extractor's default is unchanged for later calls.

```rust,ignore
#[derive(Clone)]
struct RouteModels {
    fast: ModelHandle,
    strong: ModelHandle,
}

impl AgentHook for RouteModels {
    fn on_model_select(
        &self,
        context: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if context.turn() == 1 {
            ModelSelectionAction::select(self.fast.clone())
        } else {
            ModelSelectionAction::select(self.strong.clone())
        }
    }
}

let fast = ModelHandle::named("fast", fast_model);
let strong = ModelHandle::named("strong", strong_model);
let agent = AgentBuilder::from_model_handle(fast.clone())
    .tool(search_tool)
    .build();

let answer = agent
    .prompt("Research, then synthesize")
    .max_turns(3)
    .add_hook(RouteModels {
        fast: fast.clone(),
        strong: strong.clone(),
    })
    .await?;
```

Handles and routing hooks contain live clients, callbacks, and policy state, so
handles, hook stacks, and hook actions are deliberately not serializable;
persist an application model identifier and resolve it to a handle at runtime.
Clones share the retained model safely, while replacing an agent clone has
ordinary value semantics. Concurrent runs keep independent default and
hook-stack snapshots; explicitly synchronized state captured by a hook follows
that hook's own clone semantics. See the credential-free
`runtime_model_routing` example for a complete two-model tool round trip.

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

**`rmcp` is native-only.** rmcp's `ClientHandler` is declared
`Sized + Send + Sync + 'static` unconditionally — its `local` feature relaxes
the future bounds but not the handler itself — while this crate's handler owns a
tool registry whose `Arc<dyn ErasedTool>` is deliberately neither `Send` nor
`Sync` on wasm. Enabling `rmcp` on a wasm target fails with a single explanatory
`compile_error!` rather than a wall of trait errors.
