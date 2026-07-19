# Runtime split implementation decisions and phase evidence

This ledger records implementation choices and reproducible phase-gate evidence
for the unique runtime-split implementation pull request. The newer user request
for a distinct implementation PR supersedes the research prompt's original
instruction to implement inside documentation PR #2186. The implementation
branch retains the complete research package so the final PR remains
self-contained.

## Refreshed baseline

| Item | Value |
| --- | --- |
| Implementation branch | `agent/implement-runtime-split` |
| Intended base | `origin/main` |
| Base and merge base at phase 0 | `87f3f5b77a3caeffa10d60225c41e386753bf05e` |
| Starting implementation head | `6c328d05aab31c4fbd294faaa5dc1a036614c6f2` |
| Workspace packages | 83 total: 21 library/proc-macro and 62 example packages |
| Root facade features | 41 |
| Runtime-coupled external Rust files | 326: 70 examples and 256 tests/fixtures |
| Cassette tree hash | `520a25d9ed89902ad478bcabbbc5f3ef941d1c2db08123db7f8a62fc1e281357` |
| Portable/classic compiler baseline | Rust and Cargo 1.94.0 |
| Installed targets used locally | `aarch64-apple-darwin`, `wasm32-unknown-unknown` |

The five research-time untracked executables were removed only after a separate
explicit user request. The implementation branch began with a clean worktree.

Phase-zero local commands passed before production edits:

```text
cargo fmt --all -- --check
cargo test -p rig-core --all-features --lib --no-run
cargo check --package rig-core --features wasm --target wasm32-unknown-unknown
```

The documentation head also passed all six repository CI jobs: formatting,
WASM core checking, clippy, tests, doctests, and documentation.

## Resolved implementation choices

### Tool boundary

`rig_core::tool::Tool` is the context-free portable authoring contract (also
exported under the explicit `PortableTool` name).
The complete mutable `ToolContext`, registry, server, snapshot, dispatch, and
concurrency implementation moves to `rig-agent`. `rig-agent` exposes a separate
contextual authoring trait and adapts both portable and contextual tools into
the classic registry. `rig-bevy` executes only portable tools through owned
effects and ECS capability/grant facts.

The derive macro selects portable expansion for functions without an explicitly
typed classic `ToolContext` parameter and contextual expansion for functions
with one. Dependency resolution and diagnostics must cover renamed `rig`,
`rig-core`, and `rig-agent` dependencies.

### Bevy version, MSRV, and targets

`rig-bevy` uses `bevy_ecs 0.18.1` with default features disabled and only the
features required by the implementation. Crates.io package metadata reports
Rust 1.89.0 for 0.18.1, which fits Rig's Rust 1.94.0 workspace baseline. The
newer `bevy_ecs 0.19.0` reports Rust 1.95.0 and is therefore rejected for this
migration because it would require an isolated newer compiler lane without
providing a required runtime invariant unavailable in 0.18.1.

Sources:

- <https://crates.io/crates/bevy_ecs/0.18.1>
- <https://crates.io/crates/bevy_ecs/0.19.0>

`rig-bevy` is initially experimental and native-only. That policy is enforced
inside its crate and facade feature; it does not alter `rig-core` or `rig-agent`
WASM bounds or their 1.94.0 baseline.

### Facade and construction

The root facade adds optional `agent` and `bevy` dependencies/features. Default
features select `agent`; `bevy` is opt-in and namespaced. The default prelude
keeps core contract identities and adds non-colliding classic extensions.
`rig::tool` and `rig::prelude::Tool` therefore remain portable when `agent` is
enabled; contextual tools live at `rig::agent::tool`. `rig::bevy::prelude`
exposes the ECS runtime. Classic clients use `AgentClientExt::agent`; Bevy
clients use the deliberately distinct `BevyClientExt::bevy_agent` spelling.

### Provider finals

Direct core blocking and streaming APIs retain concrete typed provider finals.
Classic streaming retains its typed provider-final event; classic blocking
requires the canonical final but no new raw-final API. Local Bevy blocking and
streaming paths expose typed finals without persisting them. Hosted/erased Bevy
paths expose only redacted, non-persisted diagnostics and make no concrete-type
claim.

### Shared conformance

`rig-runtime-conformance` is an unpublished workspace test-support crate. It may
publish fixture and scenario items within the workspace, but no production
crate or facade re-exports it. Both runtimes consume it only as a dev dependency
and implement their own test adapters; it never contains production
orchestration or a public cross-runtime trait.

## Dependency guardrail

`ci/check-runtime-dependency-graph.sh` checks Cargo metadata for all direct
normal, build, development, optional, and target-specific dependencies. It
rejects the forbidden core/runtime/Bevy and sibling-runtime edges and rejects
production runtime dependencies from provider, vector-store, and memory
companions. Once each runtime crate exists, it also requires its direct
dependency on `rig-core`.

The guard runs in CI before formatting and remains applicable throughout the
migration rather than checking only the final manifests.

## Implemented runtime and acceptance evidence

The ECS runtime owns progression through an explicitly ordered schedule with
deferred-command boundaries. Its bounded owned effects carry runtime, run,
operation, generation, correlation, tenant, capability, grant, and revision
identity. Ingress validates these facts before exactly-once accounting or
canonical commit. Tool batches commit in provider call order even when bodies
finish out of order. Vector indexes register as distinct store capabilities and
execute through that same immutable grant/effect path. Cancellation is processed
before ingress, terminal state remains observable until retention cleanup, and
late completions are diagnostic only. Protected restore validates duplicate,
missing, mismatched, and cross-tenant relationships before creating entities.

The shared conformance ledger contains all 15 researched scenarios. Both
runtime adapters produce exact observation reports checked by
`verify_report`; the support crate contains no production runtime trait or
orchestration engine.

Committed cassette replay plus scripted HTTP responses exercised through the
real provider request/response converters supply the provider acceptance matrix
without live credentials or cassette churn:

| Provider | Blocking | Streaming + typed final | Portable tools | Structured output |
| --- | --- | --- | --- | --- |
| OpenAI Responses | classic + ECS | classic + ECS | classic + ECS | explicit native + tool mapping in both runtimes |
| Anthropic Messages | classic + ECS | classic + ECS | classic + ECS | explicit tool + prompted mapping in both runtimes |
| Gemini Generate Content | classic + ECS | classic + ECS | classic + ECS | explicit native mapping + invalid-output recovery in both runtimes |

Direct core typed-final coverage remains in provider suites. Each provider's
classic streaming smoke test explicitly binds the final event to its concrete
provider streaming-response type; classic blocking intentionally promises only
the canonical final. ECS local blocking and streaming tests type the raw final
through the concrete model's associated response type. Hosted diagnostics do
not claim that concrete type.
