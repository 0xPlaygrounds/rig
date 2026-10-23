# rig-agent

`rig-agent` contains Rig's classic agent runtime: builders, the serializable
sans-I/O run state, blocking and streaming drivers, typed hooks, contextual
tools, extraction, and runtime integrations.

Most applications should use the root `rig` facade, where this runtime is
enabled by default. Low-level provider and backend contracts live in
`rig-core`.

Direct users import construction and prompting explicitly:

```rust,ignore
use rig_agent::prelude::*;
use rig_core::{Model, providers::openai::{self, OpenAI}};

let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
let agent = AgentBuilder::new(model).build();
let answer = agent.prompt("Explain ownership briefly.").await?;
```

## Conversation memory

`AgentBuilder::memory` keeps an agent's conversation in a rig-memory
`AsyncStore`: a local `MemoryStore` or `DirStore`, or a store behind a network
or a JavaScript promise. A run loads the conversation (`conversation(log)`
names it) before its first model call and saves its messages with one append
once it finished. A failed load or save fails the run with
`PromptError::Memory`; a run that fails or is cancelled saves nothing. A save
appends only if the log is as the save read it, the store checking and
appending in one step: of two runs that loaded the same log, the second to
save fails with `SessionError::Changed` and adds nothing to the log, even when
the saves overlap, in one process or several sharing the store. A save reads
only its own log, so one store can hold many conversations, a log each. See
`examples/persistent_chat.rs`.

## Recording and replay

The runtime accepts `rig_core::serve::Recorder`, not a concrete log type.
Enable the separate `rig-cassette` crate's `agent` feature for logs and replay
adapters; neither crate configuration adds the native HTTP engine.

Retain a `rig_cassette::effect_log::EffectLogRecorder` and attach a clone with
`AgentBuilder::record_to`. Choose `keeping_stream_events()` when the recording
must preserve stream items. Import `rig_cassette::agent::AgentReplayExt` and
call `agent.stamp(recorder.take())` after the run, or stamp `recorder.log()` for
a snapshot. The extension trait also owns `run_spec_hash` and
`check_replayable`. A host that owns the handler table attaches its recorder
through `Handlers::record_to`.

Register recorded handlers through `rig_cassette::agent::replay::register_all`
or `register_all_checking`, then run the agent as usual. The runtime has no
normal cassette dependency, even with all runtime features enabled.
See [the cassette README](../rig-cassette/README.md) for feature isolation,
checkpoint formats and migration from implicit recording/log getters.

## Handlers and dispatch

An agent dispatches every effect through a `Handlers` table: its own, or a
host's given to `AgentBuilder::over_handlers`. A dispatch looks its handler up
and runs it inside the caller's future or stream, so no executor or driver is
involved, and dropping the future or stream cancels the dispatch and everything
it dispatched. Missing handlers report `HandlerUnavailable`.

Registration refuses a family change under a live key. Typed keys carry a family
proof; binding either a typed or untyped key checks the current descriptor's
family. A handler dispatches nested work through `Handlers::serving(&dispatch)`,
whose dispatches are recorded as children of the one it serves.

Handler layers intercept requests and outcomes, but recording observes the
innermost handler. A denial produces no handler record; an outcome replacement
leaves the original handler answer in the log. Suspended approval retains the
in-flight dispatch. Recorder-provided adapter contexts propagate
to completion adapters unless the request supplies one explicitly; contexts
must distinguish concurrent logical calls, and adapter facts are not effect-log
entries.

## The run protocol

`rig_agent::run::AgentRun` is a steppable, serializable state machine: it owns
every *decision* the agent loop makes — turn budget, tool-call validation and
recovery, history threading, structured-output policy, usage accounting, final
response — and performs no IO. A *driver* calls `next_step()` and acts on the
returned `AgentRunStep` (`CallModel`, `CallTools`, `Done`), feeding results
back with `model_response` / `answer`. This crate's futures loop is the
driver; the machine itself never awaits (a source-level guard keeps it so) and
is `Serialize + Deserialize`, so a run can be suspended between steps and
resumed in another process. Resume with the same Rig version; the envelope format
is checked but cross-version state compatibility is not guaranteed. Serialized
state includes conversation content and every recorded provider response body,
so its sensitivity and size grow with the run. Hosts must apply their persistence
and redaction policy before storing it. Constructing a run directly does not
execute an agent's configured hooks, tools, or retrieval.

Everything an agent loop *is* sits beside it in `rig_agent::run` — the
`RunSpec` it is configured by, `prepare_request` (the pure `(RunSpec,
capabilities, history, tools, patch) → PreparedRequest` step, so every driver
sends the same bytes), the output policy, the per-turn `RequestPatch`, the
run's response and error types, the invalid-call decisions, the streamed-turn
assembler and the loop-side transcript helpers — all sans-IO and serializable.
rig-core keeps only the message-model invariants (`conversation::pairing`, the
`tool::feedback` constructors). A host that drives runs itself (an ECS schedule, a
job system) depends on this crate with default features off: that graph
carries no async runtime, transport or MCP client (a dependency guard pins it),
and `tests/fixtures/agent_run_stepper` is that host in miniature.

## Runtime model routing

High-level agents are concrete values: models are registered in the agent's
handler table under a `ModelRef` label, and each model's `ProviderCapabilities` snapshot
is captured by value at registration. `AgentBuilder::new` registers its model
as `default`; `AgentBuilder::named_model` and `model_route` register models
under labels of your choice. The typed `Model<W, T>` and its erased twin
`DynModel<Completion>` stay available for direct `call` and `stream`, and a
backend is a wire plus a transport rather than a trait implementation.

Replace the default on one agent value with `set_model` (a model value) or
`set_model_label` (a registered label), or change one run's default candidate
with `using_model(label)` or `using_model_value(model)`. Implement
`AgentHook::on_model_select` to route before every model call. Model
selection hooks chain in registration order, with the last selection winning
and a stop terminating the run. Per model-call boundary, completion-call hooks
resolve first and their merged `RequestPatch` is passed to the selection event
(`ModelSelection::request_patch`); only then does request preparation run
against the selected model's captured capabilities and issue the attempt. A
selected model cannot change while its future or stream is in flight. Retries
and calls after tool execution are new boundaries and may select another
model.

Hooks attached through `AgentBuilder::add_hook` apply to every later runner
from that agent; hooks appended to a prompt or runner apply only to that run.
`using_model` changes the run's initial candidate but does not suppress routing
hooks. To force a model, omit routing hooks or append a final hook that always
selects it. Blocking and streaming prompts share this lifecycle: `Stop`
cancels before request preparation and provider execution, and dropping an
in-flight attempt still cancels it by dropping its retained future or stream.

Extractors support the same run-local choice: `extract(...)` returns a
`TypedRun`, so `extractor.extract(text).using_model(label)` or
`.using_model_value(model)` sets that run's default candidate. That model is
the default candidate for each extraction retry, routing hooks may replace it,
and the extractor's default is unchanged for later calls.

```rust,ignore
#[derive(Clone)]
struct RouteModels;

impl AgentHook for RouteModels {
    fn on_model_select(
        &self,
        context: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if context.turn() == 1 {
            ModelSelectionAction::select("fast")
        } else {
            ModelSelectionAction::select("strong")
        }
    }
}

let agent = AgentBuilder::named_model("fast", fast_model)
    .model_route("strong", strong_model)
    .tool(search_tool)
    .build();

let answer = agent
    .prompt("Research, then synthesize")
    .max_turns(3)
    .add_hook(RouteModels)
    .await?;
```

Registered models and routing hooks contain live clients, callbacks, and policy
state, so hook stacks and hook actions are deliberately not serializable;
persist the model's label and register the model again at runtime.
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
Use `Scratchpad` for transient cross-hook state and run entries (rig-core
`conversation::Aside`s) for state that must serialize or follow forks. Full snapshots read through `last_entry` avoid
custom replay folds; event-shaped state can instead fold `entries`. Entries
appended during settlement do not enter the finished run. Context-bound handles
use delegation rather than `Deref` so hooks cannot clone an owned dispatch
capability out of their borrowed view.

Retries of tool-free turns consume the ordinary model-call budget. Hooks may
impose narrower limits in run-local state. `ModelTurnFinished` exposes the
attempt's terminal reason and effective output cap for portable truncation
policies; the credential-free `retry_on_truncation` example demonstrates cap
growth on both surfaces. The `run_budget` example caps a run's total tokens
and tool calls the same way, with counts kept in the scratchpad.

With `test-utils`, `test_utils::HookTestHarness` gives a hook stack the run
context a runner would, so a hook's actions and scratchpad state can be tested
without running an agent.

Tools implement `rig_core::tool::Tool`, work in both runtimes, and receive
each call's `&mut ToolContext`.

`AgentBuilder::provider_tool` adds a provider-hosted tool, such as web search,
to every request after the agent's function tools. The provider runs it; the
agent never dispatches it.

Tool output remains typed through dispatch; providers and telemetry render it
at their boundaries. Ordinary serializable results become model output, while
`ToolOutput` and `ToolResultContent` preserve explicit structured or multimodal
content. `Tool::map_error` preserves arbitrary source errors for operators and
exposes safe kind-level feedback to the model by default. Explicit
`ToolExecutionError` constructors expose their detailed messages; use them or
`with_model_output` only for deliberately model-visible feedback. Inbound
context and host-only result metadata belong in `ToolContext`, not model output.

Tool registries publish changes to every attached handler table. Generated tool
keys use `<owner>/tool:<name>#<generation>`, and a request catalog runs the
registration it advertised even after the registry replaced it. Explicit keys
are preserved verbatim: reusing one replaces its binding, even across tool
names. Standalone registry execution runs inline rather than through an agent's
handlers.

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
tool registry whose erased handlers are deliberately neither `Send` nor
`Sync` on wasm. Enabling `rmcp` on a wasm target fails with a single explanatory
`compile_error!` rather than a wall of trait errors.
