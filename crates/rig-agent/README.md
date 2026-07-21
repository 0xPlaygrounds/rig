# rig-agent

`rig-agent` contains Rig's classic agent runtime: builders, the serializable
sans-I/O run state, blocking and streaming drivers, typed hooks, contextual
tools, extraction, and runtime integrations.

Most applications should use the root `rig` facade, where this runtime remains
enabled by default. Low-level provider and backend contracts live in
`rig-core`.

Direct users import construction and prompting explicitly:

```rust,ignore
use rig_agent::{client::CompletionClient, completion::Prompt};
use rig_core::{client::ProviderClient, providers::openai};

let client = openai::Client::from_env()?;
let agent = client.agent(openai::GPT_5_2).build();
let answer = agent.prompt("Explain ownership briefly.").await?;
```

Portable tools implement `rig_core::tool::PortableTool` and work in both runtimes.
Classic tools that need mutable per-call state implement
`rig_agent::tool::Tool` and receive `&mut ToolContext`.
