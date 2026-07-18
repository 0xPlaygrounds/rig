# Runtime split migration guide

Rig now separates portable contracts from runtime orchestration. Classic agents
remain the default through the root `rig` facade; core-only users can depend on
`rig-core` without pulling in a runtime, and the experimental Bevy runtime is
opt-in.

| Previous owner | Final owner |
| --- | --- |
| provider models, canonical messages, direct completion/streaming responses | `rig-core` |
| portable `Tool`, memory and vector-store contracts | `rig-core` |
| agents, hooks, runners, extraction, prompt traits, contextual tools | `rig-agent` |
| ECS topology, effects, policies, persistence, handles | `rig-bevy` |
| default compatibility paths and companion integrations | `rig` |

## Existing facade applications

No dependency change is required. The root crate enables `agent` by default:

```rust,ignore
use rig::prelude::*;
use rig::providers::openai;

let client = openai::Client::from_env()?;
let agent = client.agent(openai::GPT_5_2).build();
let answer = agent.prompt("Hello").await?;
```

Code depending directly on `rig-core` must import classic APIs from
`rig-agent`. In particular:

```rust,ignore
use rig_agent::prelude::{AgentClientExt, Prompt};
use rig_core::{client::ProviderClient, providers::openai};
```

`CompletionClient` in `rig-core` now constructs models only. Classic `.agent()`
and `.extractor()` are supplied by `rig_agent::client::AgentClientExt`, and
model `.into_agent_builder()` is supplied by `AgentModelExt`.

## Tool implementations

A portable tool has no mutable runtime context and implements
`rig_core::tool::Tool`:

```rust,ignore
impl rig_core::tool::Tool for Lookup {
    const NAME: &'static str = "lookup";
    type Args = LookupArgs;
    type Output = String;
    type Error = LookupError;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.lookup(args).await
    }
}
```

Classic-only tools that need the mutable type map implement
`rig_agent::tool::ContextualTool`. `#[rig_tool]` selects the portable trait for
functions without a context parameter and the contextual trait only for an
explicit `&mut rig_agent::tool::ToolContext` parameter. Both portable and
contextual tools can be registered on a classic agent; `rig-bevy` accepts only
portable tools.

## ECS-native construction

Enable the opt-in feature and use the distinct constructor:

```toml
[dependencies]
rig = { version = "0.40", default-features = false, features = ["bevy"] }
```

```rust,ignore
use rig::bevy::prelude::*;
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openai;

let client = openai::Client::from_env()?;
let runtime = BevyRuntime::default();
let handle = runtime.spawn_agent(
    client
        .bevy_agent(openai::GPT_5_2)
        .name("researcher")
        .temperature(0.2)
        .max_tokens(1024)
        .max_calls(8),
);
let outcome = handle.prompt("Summarize the evidence").await?;
```

The ECS runtime is native-only and experimental. Classic remains the supported
default while operational evidence is gathered. Snapshots contain explicit
stable domain records, not a serialized Bevy `World`; model, tool, and memory
implementations must be rebound explicitly during restoration. Model bindings
hold the concrete implementation supplied to `BindingManifest::bind_model` for
each exact persisted agent ID, so same-typed agents retain distinct host model
configurations and a typed restored handle cannot substitute an unrelated model. Resume an
ordinary host-retained prompt with `resume_run`; when restoration repaired an
interrupted tool turn, use the prompt-free `resume_tool_turn` continuation.
An ECS `PendingRun` is also its driver's cancellation lease: dropping the value,
an unpolled driver future, or an in-flight driver future cancels the run. Once
cancellation wins, queued sibling tools are not started and late completions
from any already-running work are rejected by correlated ingress.
`AgentSpec::additional_params` carries provider-specific JSON when a provider
surface requires it. The value is redacted from debug output and is never
persisted in snapshots, so a restored run must receive it again through an
explicit `BindingManifest::bind_additional_params` host binding. Snapshot
content is omitted by default; persisting plaintext messages and preambles
requires an explicit opt-in.

## Feature selection

| Configuration | Result |
| --- | --- |
| `rig` defaults | portable core + classic runtime |
| `rig --no-default-features` | facade and portable core only |
| `features = ["agent"]` | classic runtime only |
| `features = ["bevy"]` | experimental Bevy runtime only |
| `features = ["agent", "bevy"]` | both, with distinct constructors/preludes |

Provider, vector-store, memory, loader, derive, and transport features do not
select a runtime merely because they are enabled. Direct provider calls retain
their concrete blocking and streaming raw-response types; classic streaming and
local Bevy handles expose typed provider finals, while hosted Bevy execution
uses a non-persisted diagnostics envelope without claiming a concrete type.
