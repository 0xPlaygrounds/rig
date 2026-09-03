# `ToolContext` census (phase A2 of the effect bus)

Baseline: `d9ed455cf`. Every in-tree insert or read of `ToolContext`'s two maps
(inbound, result), with the port each entry takes when the storage becomes
serde. Semantics-only calls (`for_dispatch`, `accept_dispatch_result`,
`clear_dispatch_result`) carry no value and keep their call pattern.

## Non-test sites

| Site | Map | Value type | Key (`ContextValue::KEY`) | Port |
| --- | --- | --- | --- | --- |
| `crates/rig-rmcp/src/native.rs` (request `_meta`) | inbound read | `McpMeta(rmcp::model::Meta)` | `rmcp.meta` | newtype: the context stores values under declared keys and `rmcp::model::Meta` is not rig-rmcp's to implement `ContextValue` for |
| `crates/rig-rmcp/src/native.rs` (`preserve_mcp_result`) | result write | `McpStructuredContent(serde_json::Value)` | `rmcp.structured_content` | newtype |
| ″ | result write | `McpResponseMeta(rmcp::model::Meta)` | `rmcp.response_meta` | newtype |
| ″ | result write | `McpCallToolResult(rmcp::model::CallToolResult)` | `rmcp.call_tool_result` | newtype |
| `crates/rig-agent/src/test_utils/tools.rs` | inbound read | `SessionId` | `test.session_id` | manual impl |
| ″ | result write | `MockRequestId` | `test.mock_request_id` | manual impl |
| `examples/tool_result_outcomes/src/main.rs` | result write/read | `FailureSite` | `example.failure_site` | `#[derive(ContextValue)]` |
| `crates/rig-core/src/tool/contextual.rs:584,601,798` | — | — | semantics only (dispatch snapshot / publish / clear) |
| `crates/rig-core/src/tool/catalog.rs:81` | — | — | semantics only |
| `crates/rig-agent/src/tool/server.rs:340` | — | — | semantics only |
| `crates/rig-agent/src/agent/engine.rs:1645` | — | — | semantics only |
| `crates/rig-agent/src/agent/tool.rs:48` | — | — | semantics only (sub-agent inherits the inbound map) |
| `crates/rig-core/src/tool/portable.rs:214` | — | — | constructs an empty context; unchanged |
| `crates/rig-agent/src/agent/runner.rs:94,149`, `agent/typed.rs:111` | — | — | pass-through; unchanged |
| `examples/rmcp/src/main.rs:188` | — | — | **not** a `ToolContext` (rmcp's own request extensions); no entry |

Doc-comment examples (`crates/rig-core/src/tool/contextual.rs:68`,
`crates/rig-agent/src/tool/mod.rs:61`, `crates/rig-rmcp/src/lib.rs:57,67`)
are rewritten to the serde bounds.

## Pass-through files

84 files name `ToolContext`; the ones not listed above only move the value
between signatures and are unaffected by the storage change.

## Test sites and the semantic shift

| Site | Value type | Port |
| --- | --- | --- |
| `crates/rig-core/src/tool/context/{tests,migrated_tests}.rs` | `u32`, `String`, `Vec<u8>`, local newtypes | derive serde; `get` returns owned values; `get_mut` tests become insert-replace |
| `…/migrated_tests.rs` `clone_preserves_intentionally_shared_value_state` | `Arc<Mutex<u32>>` | **deleted** — a shared referent cannot cross a serde boundary; shared state belongs in tool construction state |
| `…/migrated_tests.rs` `empty_context_is_default_and_allocation_free` | — | rewritten against the serde maps |
| `crates/rig-core/src/tool/contextual/tests.rs` `CloneTracked(Arc<AtomicUsize>)` | clone counter | replaced by an isolation assertion: mutation inside a dispatch does not reach the caller's map |
| `crates/rig-agent/src/tool/server/tests.rs` `CloneTrackedContext` | clone counter | same replacement |
| `crates/rig-agent/src/agent/runner/tests.rs` `SnapshotValue { clones: Arc<AtomicUsize> }` | clone counter | same replacement |
| `crates/rig-agent/src/agent/{hook,streaming,tool}/tests.rs`, `runner/prompt_tests.rs` | `String`, `SessionId` | derive serde |
| `crates/rig-derive/tests/tool_context.rs` | `Offset`, `Prefix`, `Invocation(&'static str)` | derive serde; `Invocation` holds a `String` |

## Semantic shift, stated once

`ToolContext` values are data under **declared keys**: a value type implements
`ContextValue { const KEY }` (derive it, or one line), inserting a value
serializes it under that key, reading one deserializes it. The key is the
value's identity on the wire — stable across a rename, a module move, a
persisted `EffectLog`, or a different toolchain, where `std::any::type_name`
(the first serde port's key) was not. `get`/`result` return
`Result<Option<T>, ToolContextError>`: an empty slot is `Ok(None)`, a slot
holding something that is not a `T` is `Err(Decode)` — absence and a shape
mismatch are different facts. `insert` never fails after a successful write:
a displaced value that does not decode as `T` is simply replaced. Values with interior sharing (`Arc<Mutex<_>>`, atomics,
channels) no longer travel through dispatch context; each such use moves to
the tool instance, which is what tool instances are for. Clone-per-dispatch
and dispatch-returns-result-without-replacing-inbound are preserved exactly.
`get_mut` is gone (there is no in-place reference into a serde value);
`MissingToolContext` is replaced by `ToolContextError`, which also reports
encode/decode failures.

## Accessor bounds after A2

- `insert<T: Serialize + DeserializeOwned + 'static>(value) -> Result<Option<T>, ToolContextError>`
- `get<T: DeserializeOwned + 'static>() -> Option<T>`
- `require<T: DeserializeOwned + 'static>() -> Result<T, ToolContextError>`
- `remove<T>() -> Option<T>`, `contains<T>() -> bool`
- `insert_result` / `result` / `require_result` mirror the inbound trio

Keys are derived from `std::any::type_name::<T>()`; no two values of one type
were in flight anywhere in the census, so no explicit-key variants exist.
