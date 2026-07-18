# Runtime split migration guide

Rig now separates portable contracts from orchestration:

- `rig-core` owns provider clients/models, completion requests and responses,
  canonical messages, usage, portable tools, memory/store traits, telemetry,
  and WASM helpers.
- `rig-agent` owns the existing classic agent runtime, including prompt/chat
  traits, extractors, hooks, contextual tools, and multi-turn streaming.
- `rig-bevy` owns the experimental ECS runtime and is opt-in.

The root `rig` crate keeps classic behavior enabled by default. Existing root
imports normally need no change. Direct `rig-core` consumers that used runtime
APIs should depend on `rig-agent` and update imports such as:

```rust,ignore
use rig_agent::prelude::{AgentClientExt, Prompt};
use rig_core::providers::openai;
```

Portable tools implement `rig_core::tool::Tool::call(&self, args)`. Tools that
need the classic mutable per-call context implement
`rig_agent::tool::ContextualTool`. The `#[rig_tool]` macro selects the portable
trait for functions without a context parameter and the contextual trait when a
`ToolContext` parameter is declared.

For a core-only graph, disable root defaults. Runtime selections compose
without pulling their sibling:

```toml
rig = { version = "*", default-features = false }                # core facade
rig = { version = "*", default-features = false, features = ["agent"] }
rig = { version = "*", default-features = false, features = ["bevy"] }
rig = { version = "*", default-features = false, features = ["agent", "bevy"] }
```

`rig-core` and `rig-agent` preserve the workspace MSRV and WASM policy.
`rig-bevy` currently uses Bevy ECS 0.18 on the same workspace Rust 1.94 MSRV and
is explicitly native-only. The classic runtime remains supported and default;
the Bevy runtime is experimental pending operational history.
