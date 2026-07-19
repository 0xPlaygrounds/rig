# rig-bevy

`rig-bevy` is Rig's experimental, native-only Bevy ECS agent runtime. It uses
the portable model, message, memory, and tool contracts from `rig-core`, while
keeping topology, policy, progression, effects, cancellation, and persistence
inside an authoritative ECS world.

The classic `rig-agent` runtime remains Rig's default. Select this crate
directly or enable the root `rig` crate's `bevy` feature and import
`rig::bevy::prelude::*`.

This runtime does not wrap `rig-agent`, `AgentRun`, or hooks. Async providers,
tools, vector-store retrieval adapters, and memories receive owned requests and
return through bounded, correlated ingress. `install_vector_store` records a
distinct store capability while executing the portable vector-index tool
contract through the same immutable grant and effect path. Snapshots contain
stable domain records only and require tenant-owned bindings with explicit,
caller-defined implementation/configuration identities before resumable state
can be persisted or rebound.

Conversation memories must return canonical histories from `load`: every
assistant tool call paired with its results in the next message, no consecutive
assistant messages, and no orphaned tool results. Truncating or summarizing
memories must therefore cut on turn boundaries. A non-canonical loaded history
fails the run immediately (code `memory_history`) before any model call.

Run lifecycles, model/tool/memory effects, canonical tool commits, and rejected
ingress emit content-redacted `tracing` spans and events. Model effects use the
core-owned completion-parent marker so provider GenAI metadata enriches one
span instead of creating a duplicate model span.

WebAssembly is intentionally unsupported by this experimental crate. That
native target policy does not change `rig-core` or `rig-agent` WASM support.

```rust,ignore
use rig_bevy::{BevyClientExt, LocalRuntime};
use rig_core::{client::ProviderClient, providers::openai};

let client = openai::Client::from_env()?;
let definition = client
    .bevy_agent(openai::GPT_5_2)
    .preamble("Answer concisely.")
    .build();
let mut runtime = LocalRuntime::new()?;
let agent = runtime.spawn_agent(definition)?;
let result = runtime.run(agent, "Why use an ECS runtime?").await?;
```

`LocalRuntime` exposes concrete provider finals through typed local blocking
and live streaming APIs when the provider supplies one; `start_streaming`
yields provisional events before that optional typed final and reports subscriber
lag or a concrete-type mismatch explicitly.
`HostedRuntime` deliberately exposes only a redacted, non-persisted diagnostic
envelope. Protected snapshots require a caller-supplied authenticated
`SnapshotProtector`. The default snapshot policy is metadata-only; canonical
run content requires explicit `SnapshotContentPolicy::CanonicalRunState`
selection. Restore rejects duplicate, missing, identity-mismatched, or
cross-tenant topology before reconstructing ECS entities.
