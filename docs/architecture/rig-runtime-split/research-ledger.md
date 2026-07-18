# Research ledger

> **Historical research snapshot.** Facts and questions in this ledger describe
> baseline `87f3f5b77a3caeffa10d60225c41e386753bf05e`. The selected migration was
> implemented in `940483b4fb30aa77d83dab52a1599652cb9e0c2a`; baseline source
> evidence is commit-pinned.

This ledger prevents recommendations from being reported as repository facts.
Facts are reproducible at source revision
`87f3f5b77a3caeffa10d60225c41e386753bf05e` or the exact reference PR
revisions. Inferences explain evidence without claiming implementation intent.

## Verified facts

| ID | Fact | Evidence |
| --- | --- | --- |
| F-01 | Research source and `origin/main` are `87f3f5b77a3caeffa10d60225c41e386753bf05e`; the research branch merge base with `origin/main` is the same commit. | `git rev-parse`, `git merge-base` |
| F-02 | PR #2182 base/head/merge are `d6d2dfa0...`, `f5737b34...`, and `87f3f5b7...`; it merged on 2026-07-18. | GitHub PR metadata |
| F-03 | ECS PR #6 base/head are `6f3df71c...` and `8f1d72dd...`; it is open and unmerged. | GitHub PR metadata and fetched Git objects |
| F-04 | ECS PR #6 and current Rig merge at `1f1ee4d9...`; PR base/current Rig are 1/12 commits from that merge base, and PR head/current Rig are 40/12 commits from it. | `git merge-base`, `git rev-list --left-right --count` |
| F-05 | At the start there were no staged or unstaged tracked changes and five pre-existing untracked Mach-O executables: `rtn`, `rtn2`, `rtnalias`, `rtnalias2`, `unstable`. | `git status -sb`, `git diff`, `file` |
| F-06 | Workspace metadata contains 83 packages: 21 library/proc-macro packages and 62 example packages. | `cargo metadata --no-deps --format-version 1` |
| F-07 | `rig-core` declares 25 public top-level modules at this revision (including feature-gated modules and `telemetry`). | [`rig-core/src/lib.rs:152-198`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/lib.rs#L152) |
| F-08 | The root `rig` facade glob re-exports `rig_core` and exposes 18 companion modules behind features (memory is always a core module with optional companion additions). | [`src/lib.rs:30-158`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/src/lib.rs#L30), [`Cargo.toml:254-311`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/Cargo.toml#L254) |
| F-09 | Every current non-example companion library has a normal dependency on `rig-core`; `rig-derive` has only a dev dependency on it. | companion manifests and `cargo metadata` |
| F-10 | `CompletionClient` imports classic `AgentBuilder` and `ExtractorBuilder`; its `agent()` and `extractor()` methods return those builders. | [`client/completion.rs:1-60`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/client/completion.rs#L1) |
| F-11 | Current main defines `Prompt`, `Chat`, and `TypedPrompt`, but no public `Completion` facade trait; `CompletionModel` is the low-level provider trait. | [`completion/request.rs:357-631`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L357), repository symbol search |
| F-12 | `PromptError` contains completion, memory, max-turn, cancellation, and unknown-tool variants. | [`completion/request.rs:140-184`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L140) |
| F-13 | `CompletionResponse<T>` stores canonical choice/usage and a typed raw provider response. | [`completion/request.rs:448-458`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L448) |
| F-14 | `streaming.rs` contains both raw provider streaming types and high-level traits returning classic runtime requests. | [`streaming.rs:67-261`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/streaming.rs#L67), [`streaming.rs:565-626`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/streaming.rs#L565) |
| F-15 | `AgentRun` is serializable sans-I/O state; `drive_agent` is shared by blocking and streaming surfaces. | [`agent/run/mod.rs:277-317`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L277), [`prompt_request/streaming.rs:471-489`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/prompt_request/streaming.rs#L471) |
| F-16 | `AgentHook` has event-specific action types. `HookStack` merges completion patches, chains tool rewrites/results, and short-circuits terminal actions. | [`agent/hook.rs:726-1031`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L726), [`agent/hook.rs:1251-1377`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L1251) |
| F-17 | PR #2182 adds model-turn retry/stop actions; retry is documented as tool-free and within the total call budget. | [`agent/hook.rs:427-478`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L427), [`agent/hook.rs:940-949`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L940) |
| F-18 | Current `Tool` uses `WasmCompatSend`/`WasmCompatSync`, accepts `&mut ToolContext`, and is distinct from `ToolSet` and tool server types in the same module tree. | [`tool/mod.rs:133-180`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/tool/mod.rs#L133), [`tool/mod.rs:511`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/tool/mod.rs#L511), [`tool/server.rs:126-230`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/tool/server.rs#L126) |
| F-19 | `ConversationMemory` is a WASM-compatible backend contract over canonical messages. | [`memory.rs:85-117`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/memory.rs#L85) |
| F-20 | `Extractor` stores an `Agent`, uses its runner/output tool mode, and exposes hook registration. | [`extractor.rs:37-79`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/extractor.rs#L37), [`extractor.rs:199-335`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/extractor.rs#L199) |
| F-21 | `CompletionSpanBuilder` checks for the classic target `rig::agent_chat`; `ProviderResponseExt` is otherwise provider metadata. | [`telemetry/mod.rs:66-156`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/telemetry/mod.rs#L66), [`telemetry/mod.rs:426-461`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/telemetry/mod.rs#L426) |
| F-22 | `rig-derive` resolves `rig-core` then `rig` and emits `rig_core::tool::Tool` plus `ToolContext` paths for `rig_tool`. | [`rig-derive/src/lib.rs:22-34`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-derive/src/lib.rs#L22), [`rig-derive/src/lib.rs:691-735`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-derive/src/lib.rs#L691) |
| F-23 | ECS PR #6 changes 674 files, adding 48,572 and deleting 52,033 lines; `rig-core` accounts for 103 files, +33,205/-30,699. | `git diff --shortstat`, `git diff --numstat` at exact PR revisions |
| F-24 | PR #6 deletes ten classic agent implementation files and adds twelve `runtime/*.rs` files. | exact PR name-status diff |
| F-25 | PR #6 makes `bevy_ecs 0.19` unconditional in `rig-core`, enables `std`/`multi_threaded`, raises Rust to 1.95, and publicly re-exports `bevy_ecs`. | exact PR manifest/lib diff |
| F-26 | PR #6 deletes `wasm_compat` and replaces its marker bounds with raw `Send + Sync` across many core/provider APIs. | exact PR source diff |
| F-27 | PR #6 changes 36 provider source files by +92/-205; most inspected changes are bound/import propagation. It disables Copilot's WASM token exchange with a runtime-architecture error. | exact PR provider diff |
| F-28 | PR #6 changes 315 cassette files (+8,688/-9,745); 254 have equal additions/deletions and 187 are at most +2/-2. Representative smoke requests are identical while response text/usage changes. | exact PR cassette numstat and representative diffs |
| F-29 | The target dependency graph documented in this package is acyclic when only `rig`, `rig-agent`, `rig-bevy`, integrations, `rig-core`, derive, and the test-only conformance package are considered. | graph edge set and acyclicity validator |
| F-30 | The root facade declares 41 features. Twenty feature names select 18 optional companion crates (`rig-fastembed` has three entry features); the remainder provide core forwarding and cross-package TLS composition. | `cargo metadata` and the [complete feature map](runtime-import-inventory.md#complete-root-facade-feature-map) |
| F-31 | A conservative public-reference scan finds 326 external Rust files that directly import or fully qualify a current runtime-bearing surface: 70 examples, 234 provider tests, 12 other root tests/fixtures, and 10 derive tests. | [runtime-import inventory](runtime-import-inventory.md#runtime-import-scan) and source scan at the exact revision |
| F-32 | OpenAI defines an inherent `GenericCompletionModel::into_agent_builder()` that returns classic `AgentBuilder` directly from provider code. | [`openai/completion/mod.rs:1898-1901`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/providers/openai/completion/mod.rs#L1898) |

## Architectural inferences

| ID | Inference | Basis and confidence |
| --- | --- | --- |
| I-01 | `rig-core` currently combines at least three ownership layers: portable contracts, classic runtime, and provider/integration implementation. | Direct module/symbol dependencies; high confidence. |
| I-02 | The lowest-risk classic extraction moves `AgentRun` and `drive_agent` together before refactoring either. | They share progression invariants and current parity tests; high confidence. |
| I-03 | Hook types cannot stay in a provider-neutral core without either preserving classic lifecycle assumptions or becoming vague callbacks. | Event/action shapes and HookStack semantics; high confidence. |
| I-04 | A Bevy runtime driven by `AgentRun` would make ECS secondary rather than authoritative. | `AgentRunStep` and internal state own progression; PR #6 demonstrates richer operation topology; high confidence. |
| I-05 | Most PR #6 provider/cassette churn is migration collateral rather than provider protocol necessity. | Small provider source diff dominated by bounds; identical cassette requests with rerecorded responses; high confidence for the sampled/general pattern, not a claim that every cassette is semantically identical. |
| I-06 | Mandatory Bevy in core would impose compile/MSRV/target costs on consumers that do not use agents. | All companions depend on core and PR #6 makes Bevy unconditional; high confidence. |
| I-07 | A shared conformance harness can constrain observable behavior without a public common runtime trait. | Current scripted model tests already assert canonical outcomes; medium-high confidence pending prototype. |
| I-08 | Provider-native final responses should be side-channel observations, not persisted canonical progression. | Current generic raw response type plus heterogeneous ECS/persistence constraints; high confidence. |
| I-09 | Built-in provider extraction is desirable for the ideal core but should be sequenced separately. | Provider implementations are not provider-neutral, but their movement is orthogonal and broad; high confidence. |

## Recommendations

| ID | Recommendation | Reason |
| --- | --- | --- |
| R-01 | Combine three-crate topology with narrow-contract/independent-runtime discipline. | Best isolation without sacrificing native extension models. |
| R-02 | Keep `rig-core` free of `rig-agent`, `rig-bevy`, and `bevy_ecs`. | Enforces direction mechanically. |
| R-03 | Move current classic runtime without semantic refactors before beginning Bevy implementation. | Preserves current PR #2182 and parity behavior. |
| R-04 | Put client `agent()`/`extractor()` and model `into_agent_builder()` on classic extension traits re-exported by the default facade prelude. | Retains client and OpenAI-model ergonomics without provider/runtime coupling. |
| R-05 | Give Bevy a distinct constructor method/namespace and no default-prelude glob. | Avoids extension-trait collisions and hides no ECS semantics. |
| R-06 | Split tools into portable authoring/canonical values and runtime-specific execution infrastructure. | Both runtimes need the former; their contexts/registries/grants differ. |
| R-07 | Preserve core/agent WASM contracts; apply stricter native bounds at Bevy adapters. | Avoids ECS constraint leakage. |
| R-08 | Use test-only shared scenarios, not a production shared state machine. | Detects drift while retaining ECS authority. |
| R-09 | Keep provider cassettes unchanged during runtime movement. | Provider wire behavior should not change; fixture churn obscures review. |
| R-10 | Build Bevy in vertical slices with correlation, cancellation, stale-result, and persistence tests from the first effect slice. | These are architectural invariants, not polish. |

## Research questions and implementation resolutions

| ID | Research question | Implementation resolution |
| --- | --- | --- |
| Q-01 | Is core `Tool` context-free, given a narrow portable invocation context, or paired with a classic contextual trait? | Core `Tool` is context-free and pairs with classic `ContextualTool`. |
| Q-02 | How does `#[rig_tool]` select portable versus classic contextual expansion? | The macro selects contextual expansion only for an explicit mutable `ToolContext` parameter and has renamed-dependency coverage. |
| Q-03 | Which legacy classic module paths, if any, does root `rig` re-export temporarily? | Root `rig` deliberately re-exports classic surfaces; `rig-core` has no reverse runtime dependency. |
| Q-04 | Is `rig-bevy` native-only initially? | Yes. It emits an explicit WASM compile error and remains experimental. |
| Q-05 | How are provider-native finals retained/exposed by hosted Bevy runs? | Local APIs expose typed finals; hosted paths expose redacted diagnostics; neither form is persisted. |
| Q-06 | When do built-in providers leave `rig-core`, and into how many crates? | Provider decomposition is deferred to a separate packaging decision. |
| Q-07 | Is conformance support unpublished workspace code or a public helper crate? | It is the unpublished workspace-only `rig-runtime-conformance` crate. |
| Q-08 | What exact evidence moves `rig-bevy` from experimental to supported/default-eligible? | The gate retains the listed conformance and operational criteria; this PR intentionally leaves the runtime experimental pending operational history. |

## Rejected approaches

| ID | Approach | Reason rejected |
| --- | --- | --- |
| X-01 | Make Bevy mandatory inside `rig-core`. | Transitive dependency/MSRV/WASM leakage; PR #6 demonstrates the blast radius. |
| X-02 | Leave classic runtime in core and add Bevy beside it. | Core remains runtime-specific; providers/stores still pull classic behavior. |
| X-03 | Extract hooks only. | State machine, facade traits, tools, memory, extraction, telemetry, and constructors remain coupled. |
| X-04 | Shared production agent state machine with hook and ECS adapters. | Prevents authoritative ECS topology/progression and turns systems into adapters. |
| X-05 | Recreate HookStack as an ECS callback component. | Not ECS-native; hidden ordering and state outside systems. |
| X-06 | Put both runtimes' `agent()` extension methods in one prelude. | Rust method ambiguity and unclear runtime selection. |
| X-07 | Treat arbitrary provider responses as canonical serialized ECS state. | Provider-specific, possibly non-serializable, unstable, and unnecessary for progression. |
| X-08 | Copy PR #6 wholesale. | Based on an older runtime, replaces classic behavior, leaks constraints, and mixes unrelated provider/cassette changes. |
| X-09 | Rerecord all cassettes for runtime work. | Hides provider changes, creates nondeterministic churn, and expands security review. |
| X-10 | Introduce a broad public `AgentRuntime` now. | No demonstrated stable consumer contract; would erase or bias runtime-native features. |
| X-11 | Re-export moved `rig-agent` symbols from `rig-core` as compatibility shims. | Necessarily creates the prohibited `rig-core -> rig-agent` reverse dependency and a package cycle. Compatibility belongs only in root `rig` or `rig-agent` itself. |
