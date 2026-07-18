# rig-agent

The classic Rig agent runtime. It owns agents, extractors, prompt/chat traits,
hooks, contextual tool dispatch, streaming orchestration, and memory lifecycle.
Portable provider and data contracts live in `rig-core`.

Most applications should use the root `rig` crate, whose default `agent`
feature re-exports this runtime. Direct users import extension traits from
`rig_agent::prelude`.

The classic runtime retains Rig's WASM-compatible bounds and is the supported,
default runtime.
