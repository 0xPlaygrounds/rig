# Runtime split migration guide

Rig now separates portable model/backend contracts from two independent agent
runtimes. The classic runtime remains the default. The Bevy ECS runtime is an
experimental, native-only opt-in.

**This split is a breaking (semver-major) release.** With default features, the
main mechanical source change is:

- `rig::tool::Tool` (with `ToolEmbedding` and `DynamicTool`) now names the
  portable, context-free trait. Classic contextual tool implementations must
  import `rig::agent::tool::Tool` (and `ToolSet`, `ToolContext`, server and
  rmcp items) instead.

Client construction is **not** a source change through the facade:
`use rig::client::CompletionClient; client.agent(model)` (and
`client.extractor(model)` / `client.completion_model(model)`) work exactly as
before. `rig::client::CompletionClient` is now the classic runtime's client
trait; it forwards `completion_model` to the portable provider trait and adds
`agent`/`extractor`, so the single import keeps its full pre-split surface.
Model-side construction uses `rig::client::AgentModelExt` the same way.

Direct `rig-core` dependents are the exception: `rig-core`'s
`CompletionClient` no longer provides `agent()`/`extractor()` — those moved to
`rig-agent`. Code depending on `rig-core` alone that constructed agents must
depend on `rig-agent` (or the facade) and import
`rig_agent::client::CompletionClient`.

Direct `rig-core` dependents: the `rmcp` and `discord-bot` features and the
`tool_macro` re-export moved to `rig-agent`; portable tools and the derive's
context-free output continue to work against `rig-core` alone.

## Choose a dependency surface

| Need | Dependency and imports |
| --- | --- |
| Provider calls, messages, embeddings, stores, memory, or portable tools only | Depend directly on `rig-core` |
| Existing agents, prompting, streaming, extraction, hooks, or contextual tools | Use root `rig` defaults or depend on `rig-agent` + `rig-core` |
| ECS-owned topology, effects, policy, persistence, and hosted/local driving | Enable root `ecs` and import `rig::ecs`, or depend on `rig-ecs` + `rig-core` |
| Both runtimes | Enable `agent,ecs`; use their distinct namespaces and construction extensions |

Disabling root defaults selects neither runtime:

```toml
rig = { version = "0.40", default-features = false }
```

Classic-only and ECS-only selections are explicit:

```toml
rig = { version = "0.40", default-features = false, features = ["agent", "rustls"] }
# or
rig = { version = "0.40", default-features = false, features = ["ecs", "rustls"] }
```

The default `rig::prelude` keeps the portable core names and adds non-colliding
classic conveniences. In particular, `rig::prelude::Tool` is always the
portable trait. Bevy is never added to this prelude; use
`rig::ecs::prelude::*` deliberately.

## Owner changes

| Previous core-owned surface | New owner |
| --- | --- |
| `Agent`, `AgentBuilder`, `AgentRunner`, `AgentRun`, prompt and streaming request APIs | `rig-agent` |
| Typed hooks, `HookStack`, `Scratchpad`, request patches, retry actions | `rig-agent::agent` |
| `Prompt`, `Chat`, `TypedPrompt`, `StreamingPrompt`, runtime prompt errors | `rig-agent::completion` / `rig-agent::streaming` |
| Extractors and classic runtime integrations | `rig-agent` |
| Mutable `ToolContext`, contextual tools, registries, tool servers and snapshots | `rig-agent::tool` |
| Canonical messages, completion/provider contracts, raw provider responses, portable tools, memory/store traits | `rig-core` |
| ECS entities, schedules, policies, effects, local/hosted handles and protected domain snapshots | `rig-ecs` |

The root facade keeps portable contracts at stable paths regardless of feature
unification. `rig::tool` is always portable; enabling `agent` adds the classic
runtime under `rig::agent`, including contextual tools at
`rig::agent::tool`. Direct `rig-core` users import only portable contracts;
core no longer re-exports agent progression.

## `rig-agent` root re-exports

`rig-agent` no longer re-exports the whole of `rig-core` at its crate root. It
previously carried `pub use rig_core::*;`, which made `rig-agent` an implicit
second facade and re-exported every present and future `rig-core` root item
without review. The crate root now exports only runtime-owned items (plus the
runtime-facing `rig_tool` / `tool_macro` macros).

This only affects code that depends on `rig-agent` **directly** and reached a
portable `rig-core` item through the `rig-agent` crate root. The root `rig`
facade is unchanged: `rig::…` and `rig::prelude::*` still expose portable
contracts exactly as before.

Portable contracts remain reachable from `rig-agent` through the explicit
`rig_agent::core` namespace (or a direct `rig-core` dependency):

```rust,ignore
// Before (relied on `pub use rig_core::*` at the rig-agent root)
use rig_agent::OneOrMany;
use rig_agent::Embed;
use rig_agent::message::Message;

// After — reach portable contracts through the explicit namespace
use rig_agent::core::OneOrMany;
use rig_agent::core::Embed;
use rig_agent::core::message::Message;
// ...or depend on rig-core directly and use `rig_core::…`.
```

Contextual, runtime-facing items keep their `rig-agent` paths unchanged:
`rig_agent::tool::Tool`, `rig_agent::Agent`, `rig_agent::rig_tool`, and the
rest of the runtime surface are not affected.

## Runtime construction

Classic client and model construction moved to extension traits:

```rust,ignore
use rig_agent::{client::{CompletionClient, AgentModelExt}, completion::Prompt};
use rig_core::{client::ProviderClient, providers::openai};

let client = openai::Client::from_env()?;
let agent = client.agent(openai::GPT_5_2).build();
let answer = agent.prompt("Hello").await?;
```

The Bevy spelling is intentionally distinct, so both extensions can be in
scope without two `agent()` candidates:

```rust,ignore
use rig_ecs::{EcsClientExt, LocalRuntime};
use rig_core::{client::ProviderClient, providers::openai};

let client = openai::Client::from_env()?;
let definition = client.ecs_agent(openai::GPT_5_2).build();
let mut runtime = LocalRuntime::new()?;
let agent = runtime.spawn_agent(definition)?;
let result = runtime.run(agent, "Hello").await?;
```

## Tools and `#[rig_tool]`

Portable tools implement `rig_core::tool::Tool` (also named `PortableTool`).
They receive owned typed arguments and have no access to classic mutable
context or an ECS world. The root facade exposes the same trait as
`rig::tool::Tool` and `rig::prelude::Tool` in every feature combination. Both
runtimes can execute it.

Classic contextual tools implement `rig_agent::tool::Tool` and receive
`&mut rig_agent::tool::ToolContext`. They can use the classic per-run type map
and host-only result metadata, but cannot be installed in `rig-ecs`. Through
the root facade, use `rig::agent::tool::{Tool, ToolContext}`.

`#[rig_tool]` chooses the boundary from the function signature:

- no explicitly typed `ToolContext` parameter produces a portable core tool;
- an explicitly typed classic `ToolContext` parameter produces a classic
  contextual tool;
- unrelated user types named `ToolContext` are not treated as runtime context.

The derive crate resolves renamed `rig`, `rig-core`, and `rig-agent`
dependencies and emits a migration error for invalid context signatures.

## Bevy runtime behavior

`rig-ecs` does not wrap `AgentRun` and has no hooks. ECS components and ordered
systems own topology and progression. Model, portable-tool, and memory futures
receive owned effect inputs; only validated ingress can mutate authoritative
state. Every effect carries stable runtime/run/operation identity, generation,
correlation, tenant, and—where applicable—capability/grant/revision facts.

The runtime provides:

- immutable per-turn advertised capability snapshots and exact implementation
  version dispatch;
- bounded global/model/tool execution, pending-effect, ingress, event, and
  rejection queues with explicit overflow or subscriber-lag behavior;
- provisional streaming observations with accepted-only canonical commit;
- deterministic tool-result commit in model call order;
- typed fail/retry/repair/skip/stop policy, response retry, structured output,
  cancellation, retention, retirement, and cleanup state;
- memory load before the first request and append of newly committed messages
  only;
- the same schedules behind `LocalRuntime` and `HostedRuntime`;
- redacted explanation views by default.

The crate deliberately fails to compile for WebAssembly targets. This does not
change `rig-core` or `rig-agent` WASM support.

## Raw provider finals

| Surface | Contract |
| --- | --- |
| Direct core blocking | `CompletionResponse<T>::raw_response` remains concrete |
| Direct core streaming | `StreamingCompletionResponse<R>` retains its concrete final when the provider emits one |
| Classic blocking | Canonical final; no new raw-final promise |
| Classic streaming | Concrete typed provider-final event |
| Bevy local blocking | `LocalRunResult::raw_final::<T>()` |
| Bevy local streaming | live `LocalStreamingRun::next_event`; `StreamingRunEvent::ProviderFinal` carries concrete `R` when supplied |
| Bevy hosted/erased | Redacted, non-persisted `HostedProviderDiagnostic`; no concrete-type claim |

Provider finals are side-channel data. They are never inserted into canonical
transcripts or protected snapshots, and a later stream error suppresses false
success.

## Persistence and restoration

`LocalRuntime::protected_snapshot` requires a caller-supplied
`SnapshotProtector` that encrypts/authenticates the stable domain payload and
uses `SnapshotContentPolicy::MetadataOnly` by default. Prompts, transcripts,
memory conversation keys, recovery feedback, structured values, and provider
parameters are omitted under that default. Because a memory-enabled agent cannot
be restored faithfully without its conversation key, metadata-only snapshot
creation returns `SnapshotError::MetadataOnlyMemory` for that topology. It also
returns `SnapshotError::MetadataOnlyPreamble` instead of restoring a runnable
agent after silently removing its system policy. Call
`protected_snapshot_with_policy(..., SnapshotContentPolicy::CanonicalRunState)`
only when persisting canonical run content is an explicit application decision.
Provider `additional_params` are never persisted under either policy because
they may contain credentials or other arbitrary secrets; snapshot creation
returns `SnapshotError::NonPersistableProviderParameters` when they are present.
Raw provider finals, tasks, channels, clients, and implementation pointers are
never persisted.

Models and memories are registered under a tenant. Any implementation that may
be snapshotted must also use a caller-defined `BindingIdentity` representing
both behavior and configuration revision. Portable tools have an equivalent
`install_persistable_tool` path. Ephemeral registrations continue to work for
live execution, but snapshot creation returns
`SnapshotError::MissingBindingIdentity` instead of making a false restart
compatibility claim.

Restore creates a new runtime identity, increments restored run generations,
validates relationships and integrity, and requires exact model/tool/memory
bindings through `RebindRegistry`. Missing, stable-identity-mismatched, or
cross-tenant bindings return typed `SnapshotError` variants. Rust type names
are diagnostic details and are not treated as compatibility identities.

## Intentional runtime differences

- Classic hooks are registration-ordered callbacks; Bevy policy is ECS data
  interpreted by ordered systems.
- Classic `AgentRun` is serializable sans-I/O state. Bevy persistence is a
  protected, explicit domain snapshot taken only at a safe quiescent point.
- Classic has no externally addressable world/generation/tenant ingress. Bevy
  validates those security boundaries on every effect completion.
- Classic remains the supported/default runtime. Bevy is experimental and
  native-only while it accumulates operational history.

Both runtimes nevertheless pass the same 15 observable conformance scenarios.
OpenAI, Anthropic, and Gemini cassette suites additionally run blocking,
streaming, portable-tool, structured-output, and typed-final acceptance through
the ECS runtime without changing recorded provider fixtures.
