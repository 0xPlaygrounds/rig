# rig-bevy

`rig-bevy` is Rig's experimental, native-only ECS runtime. It stores run state,
topology, policy, effect correlation, and terminal outcomes as Bevy ECS data.
It is opt-in through the root `rig` crate's `bevy` feature and does not depend
on the classic `rig-agent` runtime.

The crate intentionally uses `bevy_ecs 0.18.1`: it supports Rust 1.89, so Rig's
Rust 1.94 workspace toolchain remains sufficient. The runtime is not currently
published while its public surface is experimental.

The runtime accepts portable `rig_core::tool::Tool` implementations only. Model,
tool, and memory futures receive owned effect inputs; stable correlation,
generation, tenant, capability, and world identities are validated at ingress.
Blocking and streaming handles share canonical transcript, usage, tool-loop,
structured-output, terminal, and memory commit behavior. Snapshots persist
versioned stable domain records and require explicit implementation rebinding.
Common model settings (`temperature` and `max_tokens`) and provider-specific
`additional_params` are authoritative agent-policy data and apply to every
model effect. Debug output redacts provider-specific values, and snapshots never
persist `additional_params`; provide required values again with
`BindingManifest::bind_additional_params`. Snapshot content is omitted by
default; a drivable restore requires an explicit plaintext-content snapshot
plus concrete model, tool-revision, and memory bindings. Concrete models are
rebound by exact persisted agent ID, so two agents with the same model type or
diagnostic binding name can retain different host configurations. A repaired tool-result continuation uses
the prompt-free `resume_tool_turn` API so a newly supplied prompt cannot be
silently discarded. Raw provider finals and executor state are never persisted.
`PendingRun` owns a cancellation lease: dropping it, an unpolled driver future,
or an in-flight driver future cancels the nonterminal run. Cancellation prevents
queued sibling tools from starting; already-running tool futures may be dropped,
and their late completions cannot commit through ingress.

```rust,ignore
use rig::bevy::prelude::*;

let runtime = BevyRuntime::default();
let agent = runtime.spawn_agent(
    AgentSpec::new(model)
        .max_calls(8)
        .temperature(0.2)
        .max_tokens(1024),
);
let outcome = agent.prompt("Hello").await?;
```

See the repository [runtime migration guide](../../docs/runtime-migration.md)
for facade features and construction examples.
