# Migrating to the runtime split

This release splits Rig's monolithic core into two crates behind the `rig`
facade:

- **`rig-core`** — portable, runtime-independent contracts: provider/model
  clients, canonical messages and completion values, streaming values, the
  portable tool contracts, memory and vector-store traits, and WASM support.
- **`rig-agent`** — the classic agent runtime: the builder, run state machine,
  typed hooks, contextual tools, memory orchestration, extraction, and the
  blocking/streaming driver.
- **`rig`** — the facade you depend on. It re-exports both at their familiar
  `rig::…` paths.

## If you depend on the `rig` facade (most users)

**Almost nothing changes.** These all keep working:

```rust
use rig::prelude::*;
use rig::tool::{Tool, ToolContext};   // classic contextual tool trait
use rig::client::CompletionClient;    // provides completion_model + agent + extractor
use rig::completion::Prompt;
```

`rig::tool::Tool` is still the classic *contextual* trait (the one whose
`call` takes `&mut ToolContext`). The only two things to know:

1. **Constructing agents/extractors needs the client trait in scope.** Provider
   clients no longer have inherent `.agent()` / `.extractor()` methods, so:

   ```rust
   use rig::client::CompletionClient;   // or `use rig::prelude::*;`
   let agent = client.agent(model).build();
   let extractor = client.extractor::<T>(model).build();
   let m = client.completion_model(model);   // same trait, one import
   ```

2. **The portable, context-free tool contract is `PortableTool`.** If you were
   using a runtime-independent tool, it is now `rig::tool::PortableTool`
   (`rig_core::tool::Tool` no longer exists). Classic contextual tools are
   unchanged. A `PortableTool` still registers with the classic runtime — it
   blanket-implements the contextual `Tool`.

   ```rust
   use rig::tool::PortableTool;              // or rig_core::tool::PortableTool
   // The whole portable surface also lives under rig::tool::portable::*
   ```

## If you depend on `rig-core` directly

`rig-core` is now portable-only. It no longer provides agent construction:
`CompletionClient::agent` / `extractor`, `AgentBuilder`, `ExtractorBuilder`,
contextual tools, hooks, and the run loop moved to `rig-agent`. Depend on
`rig-agent` (or the `rig` facade) for those, and import
`rig_agent::client::CompletionClient` for `.agent()` / `.extractor()`.

Portable tools implement `rig_core::tool::PortableTool`. The `#[rig_tool]` macro
for context-free functions produces a `PortableTool`; for functions that take
`#[rig(context)] &mut ToolContext` it produces the classic contextual `Tool`
(available via `rig-agent` or the facade).

## If you depend on `rig-agent` directly

Import portable `rig-core` contracts through the explicit `rig_agent::core`
namespace (e.g. `rig_agent::core::OneOrMany`) — the `rig-agent` crate root
exposes only runtime-owned items, not a blanket re-export of `rig-core`.

- The `impl From<rmcp::model::Tool> for ToolDefinition` and its `&`-borrow
  variant were removed: with `ToolDefinition` now in `rig-core` and
  `rmcp::model::Tool` foreign, the orphan rule forbids the impl in `rig-agent`.
  Normal use is unchanged — MCP tools are still turned into definitions through
  the `ToolSet` / `ErasedTool` path. If you relied on the direct
  `rmcp_tool.into()` / `ToolDefinition::from(&rmcp_tool)` conversion, build the
  `ToolDefinition` explicitly from the tool's `name`, `description`, and
  `schema_as_json_value()`.
