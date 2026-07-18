# rig-agent

`rig-agent` contains Rig's classic agent runtime: builders, runners, hooks,
prompt conveniences, structured extraction, and runtime tool orchestration.

Most applications should depend on the `rig` facade, whose default `agent`
feature re-exports this crate. Depend directly on `rig-core` when only portable
provider and data contracts are required.

Portable tools implement `rig_core::tool::Tool`. Classic-only tools that need a
mutable invocation type map implement `rig_agent::tool::ContextualTool`; the
classic builder accepts both. Client `.agent()` and `.extractor()` methods are
provided by `rig_agent::client::AgentClientExt`.

See the repository [runtime migration guide](../../docs/runtime-migration.md)
for dependency, import, macro, and feature changes.
