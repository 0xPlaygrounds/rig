# `rig-agent` root re-export inventory

Before/after inventory of the `rig-agent` crate-root public surface for the
removal of the blanket `pub use rig_core::*;` re-export.

## Summary

`rig-agent` previously declared, at its crate root:

```rust
pub use rig_core::*;

pub mod core {
    pub use rig_core::*;
}
```

The first line made every `rig-core` root item — present and future — appear at
the `rig-agent` root, turning `rig-agent` into an implicit second facade. It has
been removed. The explicit `rig_agent::core` namespace is retained as the
supported path to portable contracts, and remains the macro expansion root for
crates that depend on `rig-agent` without a direct `rig-core` dependency.

## Names removed from the `rig-agent` root

These came *only* from `pub use rig_core::*` and are no longer at the
`rig-agent` root. Each remains available at `rig_agent::core::<name>` (and at
`rig_core::<name>`).

| Removed root name | Kind | Reachable at |
| --- | --- | --- |
| `audio_generation`, `client`*, `completion`*, `embeddings`, `http_client`, `id`, `image_generation`, `loaders`, `markers`, `memory`, `model`, `one_or_many`, `providers`, `rerank`, `streaming`*, `tool`*, `transcription`, `vector_store`, `wasm_compat`, `telemetry` | portable modules | `rig_agent::core::…` |
| `message` | portable module (`completion::message`) | `rig_agent::core::message` |
| `Embed` (trait + derive) | portable derive/trait | `rig_agent::core::Embed` |
| `OneOrMany`, `EmptyListError` | portable values | `rig_agent::core::…` |
| `ProviderResponseError` | portable error | `rig_agent::core::ProviderResponseError` |
| `schemars` | portable re-export | `rig_agent::core::schemars` |

\* `client`, `completion`, `streaming`, and `tool` also exist as *runtime-owned*
modules declared directly in `rig-agent`. Those runtime modules are unchanged
and still live at the `rig-agent` root; only the portable `rig-core` versions of
those module paths stopped being injected by the glob. Within `rig-agent`,
`crate::tool` / `crate::completion` / etc. continue to name the runtime modules.

## Names retained at the `rig-agent` root

All are runtime-owned or deliberate runtime-facing entry points. None is a bare
re-export of a portable `rig-core` root item.

| Retained root name | Origin | Why it stays |
| --- | --- | --- |
| `agent`, `client`, `completion`, `extractor`, `integrations`, `streaming`, `tool`, `test_utils` | runtime modules | classic-runtime owned |
| `core` | `pub mod core { pub use rig_core::* }` | explicit portable namespace + macro expansion root |
| `Agent`, `AgentBuilder`, `AgentRun`, `AgentRunner` | `agent::…` | headline runtime types |
| `ExtractionResponse` | `extractor::…` | runtime type |
| `rig_tool`, `tool_macro` | `rig_derive` (feature `derive`) | runtime-facing tool macros; contextual tools are a runtime concept |

The two macro re-exports are the only crate-root re-exports from the portable
crates (`rig-core` / `rig-derive`) and are pinned in
`ci/rig-agent-root-reexport-allowlist.txt`.

## Selectively retained core re-exports

None. No portable `rig-core` type is re-exported at the `rig-agent` root. The
runtime's internals import portable types explicitly from `rig_core`, and
external consumers reach them through `rig_agent::core` or a direct `rig-core`
dependency. This keeps the crate root strictly runtime-owned and avoids
recreating a partial mirror of `rig-core`.

## Guards

- **Glob reintroduction** — the paired boundary-sentinel doctests on
  `rig_agent::core` (backed by `rig_core::__RigCoreBoundarySentinel`) fail if
  the glob returns: a reintroduced glob would make the sentinel resolve at the
  `rig-agent` root, breaking the `compile_fail` half.
- **Surface creep** — `ci/check-runtime-dependency-graph.sh` rejects a bare
  root glob outright and diffs the enumerated root re-exports against
  `ci/rig-agent-root-reexport-allowlist.txt`, so any new root re-export from
  the portable crates requires a reviewed allowlist change.
