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
use rig_core::providers::openai::{self, OpenAI};
use rig_reqwest::prelude::*;

let agent = OpenAI::from_env()?
    .bound()?
    .agent(openai::GPT_5_2)
    .build();
let answer = agent.prompt("Explain ownership briefly.").await?;
```

## Recording and replay

The runtime accepts `rig_core::serve::Recorder`, not a concrete log type.
Enable the separate `rig-cassette` crate's `agent` feature for logs and replay
adapters; neither crate configuration adds ECS/Bevy or the native HTTP engine.

Retain a `rig_cassette::effect_log::EffectLogRecorder` and attach a clone with
`AgentBuilder::record_to`. Choose `keeping_stream_events()` when the recording
must preserve stream items. Import `rig_cassette::agent::AgentReplayExt` and
call `agent.stamp(recorder.take())` after the run, or stamp `recorder.log()` for
a snapshot. The extension trait also owns `run_spec_hash` and
`check_replayable`. A host-owned bus attaches its recorder through
`BusDriver::record_to`; agents over that bus cannot replace it.

Register recorded handlers through `rig_cassette::agent::replay::register_all`
or `register_all_checking`, then drive the ordinary bus. The runtime has no
normal cassette dependency, even with all runtime features enabled.
See [the cassette README](../rig-cassette/README.md) for feature isolation,
checkpoint formats and migration from implicit recording/log getters.

## Bus ownership and lifecycle

`Bus::channel` returns a dispatcher, registrar, and driver. The driver alone
owns handlers and must be polled or spawned by its owner; the bus supplies no
ambient executor. Dispatchers retain descriptors and are `Send + Sync` on all
targets. Registrars carry handlers and share their thread affinity; drivers
are `Send` natively but may be `!Send` on browser WASM. A frame-ticked executor
therefore imposes frame latency on sequential dispatches; batching belongs to
the host.

Registration publishes descriptors synchronously and refuses family changes
under a live key. Handler installation occurs before serving commands posted
after registration. Typed keys carry a family proof; binding either a typed or
untyped key checks the current descriptor's family. The driver owner controls registration, so
registrars are not obtained from dispatchers or hooks.

Dropping the driver closes the bus permanently and fails unfinished and future
dispatches with `BusClosed`. Missing handlers instead report `HandlerUnavailable`.
Dropping a pending call or stream cancels its execution and descendants,
including descendants of completed intermediate calls. Queued descendants do
not execute or produce records. Completed ancestors no longer occupy serial
slots; retained ancestry is reclaimed with its last owner. Pausing stream
consumption applies bounded backpressure.

Nested work must use the dispatcher in `DispatchScope`. It carries parent IDs
for cancellation and rejects serial dispatch back onto an active ancestor's
key rather than deadlocking. An unrelated dispatcher has no such ancestry and
can deadlock if used to reenter the same serial handler. Scoped handler
dispatchers do not keep the command channel open. Stable program scopes are
inherited by nested calls and recorded independently of runtime handles.

Handler layers intercept requests and outcomes, but recording observes the
innermost handler. A denial produces no handler record; an outcome replacement
leaves the original handler answer in the log. Suspended approval retains the
in-flight dispatch and serial slot. Recorder-provided adapter contexts propagate
to completion adapters unless the request supplies one explicitly; contexts
must distinguish concurrent logical calls, and adapter facts are not effect-log
entries.

## The run protocol

`rig_agent::run::AgentRun` is a steppable, serializable state machine: it owns
every *decision* the agent loop makes — turn budget, tool-call validation and
recovery, history threading, structured-output policy, usage accounting, final
response — and performs no IO. A *driver* calls `next_step()` and acts on the
returned `AgentRunStep` (`CallModel`, `CallTools`, `Done`), feeding results
back with `model_response` / `tool_results`. This crate's futures loop is the
driver; the machine itself never awaits (a source-level guard keeps it so) and
is `Serialize + Deserialize`, so a run can be suspended between steps and
resumed in another process. Resume with the same Rig version; the envelope format
is checked but cross-version state compatibility is not guaranteed. Serialized
state includes conversation content and every recorded provider response body,
so its sensitivity and size grow with the run. Hosts must apply their persistence
and redaction policy before storing it. Constructing a run directly does not
execute an agent's configured hooks, tools, retrieval, or memory.

Everything an agent loop *is* sits beside it in `rig_agent::run` — the
`RunSpec` it is configured by, `prepare_request` (the pure `(RunSpec,
capabilities, history, tools, patch) → PreparedRequest` step, so every driver
sends the same bytes), the output policy, the per-turn `RequestPatch`, the
run's response and error types, the invalid-call decisions, the streamed-turn
assembler and the loop-side transcript helpers — all sans-IO and serializable.
rig-core keeps only the message-model invariants (`validate_canonical`, the
tool-result constructors). A host that drives runs itself (an ECS schedule, a
job system) depends on this crate with default features off: that graph
carries no async runtime, transport or MCP client (a dependency guard pins it),
and `tests/fixtures/agent_run_stepper` is that host in miniature.

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

Extractors support the same run-local choice: `extract(...)` returns a
`TypedRun`, so `extractor.extract(text).using_model(handle)` or
`.using_model_value(model)` sets that run's default candidate. That handle is
the default candidate for each extraction retry, routing hooks may replace it,
and the extractor's default is unchanged for later calls.

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

Hook stacks preserve registration order, including nested stacks. Request patches
merge; dispatch patches and outcome replacements feed later hooks. Denials,
stops, and model-turn retries short-circuit later hooks, so observers that must
see an event belong before steering hooks. Dispatch interest is a gate: internal
families require opting in through `observes`. Replaced tool output reaches both
the model and result-content telemetry; handler recording retains the original
answer. Streaming deltas are provisional, and `ModelTurnRetried` tells consumers
to discard a rejected turn.

Keep hook-private state in the hook, keyed by run ID when shared across runs.
Use `Scratchpad` for transient cross-hook state and run entries for state that
must serialize or follow forks. Full snapshots read through `last_entry` avoid
custom replay folds; event-shaped state can instead fold `entries`. Entries
appended during settlement do not enter the finished run. Context-bound handles
use delegation rather than `Deref` so hooks cannot clone an owned dispatch
capability out of their borrowed view.

Retries of tool-free turns consume the ordinary model-call budget. Hooks may
impose narrower limits in run-local state. `ModelTurnFinished` exposes the
attempt's terminal reason and effective output cap for portable truncation
policies; the credential-free `retry_on_truncation` example demonstrates cap
growth on both surfaces.

Portable tools implement `rig_core::tool::PortableTool` and work in both runtimes.
Classic tools that need mutable per-call state implement
`rig_agent::tool::Tool` and receive `&mut ToolContext`.

Tool output remains typed through dispatch; providers and telemetry render it
at their boundaries. Ordinary serializable results become model output, while
`ToolOutput` and `ToolResultContent` preserve explicit structured or multimodal
content. `Tool::map_error` preserves arbitrary source errors for operators and
exposes safe kind-level feedback to the model by default. Explicit
`ToolExecutionError` constructors expose their detailed messages; use them or
`with_model_output` only for deliberately model-visible feedback. Inbound
context and host-only result metadata belong in `ToolContext`, not model output.

Tool registries publish changes to every attached bus. Generated tool keys use
`<owner>/tool:<name>#<generation>` so a request catalog can keep the implementation
it advertised even after replacement. Retired generations remain registered
until their final snapshot lease drops; if cleanup cannot acquire the registry
lock, a later registry access sweeps them. Explicit keys are preserved verbatim:
reusing one replaces its bus binding, even across tool names, and only the latest
binding's lease can remove it. Standalone registry execution runs inline rather
than through an agent bus.

## Target support

| Tier | Target | Status |
| --- | --- | --- |
| 1 | native (linux / macOS / windows, `x86_64` and `aarch64`) | Full support, all features; MCP tools via the companion `rig-rmcp` crate |
| 2 | `wasm32-unknown-unknown` (browser) | Supported, with no feature flags to set; `rig-rmcp` (MCP tools) is **not** available |
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

**MCP (`rig-rmcp`) is native-only.** rmcp's `ClientHandler` is declared
`Sized + Send + Sync + 'static` unconditionally — its `local` feature relaxes
the future bounds but not the handler itself — while this crate's handler owns a
tool registry whose `Arc<dyn ErasedTool>` is deliberately neither `Send` nor
`Sync` on wasm. Enabling `rmcp` on a wasm target fails with a single explanatory
`compile_error!` rather than a wall of trait errors.
