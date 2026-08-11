# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- *(agent)* [**breaking**] persisted histories and `AgentRun`/`PromptResponse` JSON carry rig-core's tagged assistant content (`{"type": "text", ...}`); the untagged shape does not load — see rig-core's entry and MIGRATING. The flatten `Some({})` round-trip artifact is gone, so `is_empty_assistant_turn`'s classification is identical before and after a persist/restore with no special-casing

- *(agent)* [**behavior**] the streamed assembler counts the stream items it excludes from assembly that carry assistant content — replayed tagged assistant blocks (the tagged `AssistantContent` serialization is not a stream-item shape) and text items whose `additional_params` is malformed — and logs a single warning per turn, on every termination path, instead of one per stream item; `StreamedTurnAssembler::excluded_assistant_content` exposes the count, and the full decode-outcome contract is pinned by an enum-driven matrix test. A stream item whose text block carries stray sibling keys decodes as stream *text* — the text is assembled and only the stray keys drop

- *(agent)* [**breaking**] `PromptResponse::content` returns `&[AssistantContent]` instead of `&Vec<AssistantContent>`, matching its slice-returning siblings

- *(agent)* [**breaking**] `PromptResponse` JSON serialized before the `content` field existed no longer deserializes — the missing-`content` reconstruction (and the serde shadow repr that carried it) is deleted and `content` is a required field; the JSON wire shape is unchanged

- *(agent)* [**breaking**] the serialized `AgentRun` carries a `$schemaVersion` tag (`rig_agent::agent::run::RUN_SCHEMA_VERSION`, currently `"1.0"`) and a build reads only the version it writes. A payload from another version — including one written before versioning existed, which has no tag — fails deserialization with a named error instead of being silently reinterpreted. Runs suspended by an earlier build cannot be resumed by this one; drain them before upgrading

- *(agent)* [**breaking**] message content is a plain `Vec<T>`, following rig-core's removal of the non-empty container — every content list the agent constructs, inspects, or hands to hooks is a `Vec`, and the `OneOrMany` re-export is gone from this crate

- *(agent)* [**behavior**] a genuinely empty assistant turn no longer cancels the run and is no longer padded with a fabricated empty-text part — with the non-empty container gone the turn is honestly representable as an empty list, and `is_empty_assistant_turn` neutralizes it instead of the agent inventing content or failing a run that used to succeed

- *(agent)* every model call validates request content before the round trip: both turn drivers issue their request through `CompletionRequestBuilder::send`/`stream`, which run `CompletionRequest::validate_message_content`, so an empty history or a content-less message fails locally with a named index instead of a remote 400

- *(agent)* [**breaking**] the `ToolCallDelta` hook payload loses `tool_call_id` (provider ids arrive on the completed call; `internal_call_id` is the correlator); the streamed assembler keys delta state by `internal_call_id` and takes the assembled reasoning block's durable id only from the provider-issued `provider_id` — a rig correlator can no longer enter history. `MultiTurnStreamItem` delta ids are rig-generated correlators, stable per part and unique per run turn — correlate across a run by `internal_call_id`, never by delta id

- *(agent)* [**breaking**] tool-call identity follows rig-core's typed model: every call a hook or consumer observes carries a unique, non-empty `ToolCallId` (the provider's id when issued, minted otherwise, with provider absence recorded on `ToolCall::provider`). Hook contexts (`ToolCallEvent`/`ToolResultEvent`/`InvalidToolCallContext`) surface that durable id — never `Some("")` — and the invalid-call retry transcript correlates the invalid call and every validated peer by their own ids, so id-less wires no longer collapse peers onto shared feedback or replay empty `tool_call_id`s. `StreamedResolution::TurnAbandoned::skipped_tool_result` is boxed

### Added

- *(agent)* add `AgentHook::on_reasoning_delta` with the Rig stream correlator, optional provider reasoning id, current fragment, and per-part aggregate; reasoning hooks share the existing observation-interest and stop-before-yield semantics used by other streaming deltas

- *(agent)* add `Agent::prepare_turn`, returning the new public `PreparedTurn`/`TurnTools`: the agent's baseline configuration resolved into one turn's completion request plus the turn's executable and allowed tool-name sets, the synthetic output-tool name, and tool dispatch pinned to the turn's registry snapshot — so hand-driven `AgentRun` loops (custom provider transport, durable suspend/resume) reuse the configured `Agent` instead of restating its preamble and tools. Impossible `tool_choice`/tool-set combinations fail at prepare time with no provider round-trip. A prepared turn is a configuration read: hooks, memory, retrieval policy, and telemetry still run only under `AgentRunner`

- *(agent)* add `Agent::drive` / `Agent::drive_run`, returning the new public `AgentDriver` (with `DriveStep` and `TurnTools`): hand-drive the sans-IO `AgentRun` machine with the agent's own configuration while owning every side effect. The driver seeds the run from the agent (`default_max_turns`, `tool_choice`, output schema), yields each turn's fully configured completion request for the caller to send (or hand to a custom transport), pairs every pending tool batch with the exact registry snapshot that advertised it, assembles the model turn internally, and maintains the run's committed structured-output tool across turns exactly as the runner does. Impossible `tool_choice`/tool-set combinations fail at prepare time with no provider round-trip. Resuming a serialized run in a fresh process re-derives a fresh dispatch snapshot from the rebuilt agent, and surfaces missing pending tools as an error (opt out with `allow_missing_resumed_tools`). The driver performs no provider IO and no dispatch of its own: hooks, memory, retrieval policy, and telemetry still run only under `AgentRunner`
- *(agent)* `AgentDriver` is resumable at **every** step boundary, not just while tool calls are pending. The turn's advertised tool names are recorded on the run (new public `TurnToolNames`, the serializable half of `TurnTools`, reachable via `AgentRun::advertised_tools()`), so a run serialized after `DriveStep::SendRequest` — the natural suspension point for a queued or long-running provider call — can be resumed in another process and fed its reply, validated against the set the request actually carried rather than against whatever the resuming registry holds. The driver now keeps no state of its own beyond a registry-snapshot cache
- *(agent)* a failed model-turn preparation costs nothing: the run advances only once a request exists, so an unreachable tool server or an impossible `tool_choice` leaves the run byte-identical — same state, same turn budget — and the step can be retried in place. Previously the turn was consumed and the run was left unresumable
- *(agent)* add `AgentRun::rollback_model_call` / `AgentDriver::rollback_model_call` for the other half of that story: a request that could not be sent, or whose reply is known to be lost, hands its turn back and returns the run to preparing, so the next step yields a **freshly prepared** request rather than replaying one whose tool snapshot has gone stale. Refunds the turn, preserves usage the provider already billed (including a streamed turn that reports usage after the failure), and counts attempts via `AgentRun::model_call_rollbacks`. Deliberately at-least-once: only the caller can tell "never arrived" from "arrived, reply lost", so bounding attempts is the caller's job
- *(agent)* `AgentRun`'s peek/commit halves are public — `peek_model_call`, `commit_model_call`, `advance`, `is_preparing_request`, with `Advance` and `ModelCallInputs` — so any hand-driver whose request preparation can fail gets the same "a failure costs no turn" guarantee the driver has, not just rig's own. Their preconditions are `PromptError` protocol violations rather than debug assertions
- *(agent)* [**behavior**] `TurnTools::execute` now refuses any name the turn did not advertise, instead of trusting its snapshot to be narrowed. The two agreed by construction in-process; on a resumed turn the names come from the run and the snapshot is rebuilt locally, so a tool registered after suspension could previously be dispatched through a turn that never advertised it
- *(agent)* [**behavior**] a resumed run's dispatch snapshot resolves the turn's advertised names explicitly rather than relying on retrieval to rank them again, so a registered *dynamic* tool that the re-derived query does not return is no longer misreported as "not registered in this process" — advice that could not be followed — and no longer feeds the model a not-found result for a tool that exists
- *(agent)* add `AgentDriver::request_patch` / `set_request_patch`, giving a hand-driven run the per-turn configuration the runner gets from its `CompletionCall` hooks — per-turn preamble, sampling parameters, `tool_choice`, `active_tools` narrowing, extra context, substituted history. Because the caller owns the loop, per-turn variation needs no callback
- *(agent)* add `AgentRun::tool_choice()`. A custom run's own `tool_choice` now reaches the provider when the run is hand-driven, instead of only governing the run's internal decisions while the request silently carried the agent's baseline
- *(tool)* add `ToolErrorKind::NotExecutable` (`ToolExecutionError::not_executable`): the tool is advertised to the model but not executable by the dispatcher — produced when dispatching the synthetic structured-output tool, whose call carries the final structured answer
- *(agent)* add opaque, cloneable `ModelHandle` values with by-value `ProviderCapabilities` snapshots, plus default replacement, per-run default override (`using_model`), and hook-driven per-call selection via `AgentHook::on_model_select`
- *(agent)* add run-local extractor default-model overrides used across retries

### Changed

- *(agent)* [**breaking**] remove concrete model parameters from long-lived classic runtime types (`Agent`, `AgentBuilder` after `new()`, `AgentRunner`, prompt/stream requests, `Extractor`) — the typed model is erased once at construction; direct provider-model completion and streaming APIs remain typed
- *(agent)* completion-call hooks now resolve before model selection: the merged `RequestPatch` is exposed on `ModelSelection::request_patch`, request preparation runs against the selected model's captured capabilities, and `ModelSelection::previous_model` reflects issued attempts only

- *(completion)* [**breaking**] normalize completion responses at the provider boundary — `CompletionResponse` and `StreamingCompletionResponse` are concrete, carry normalized `finish_reason`/`provider`/`model`/`message_id`, and every provider model exposes typed `raw_completion`/`raw_stream` escape hatches
- *(completion)* add public `ProviderCapabilities`, replacing `CompletionModel::composes_native_output_with_tools`

### Removed

- *(completion)* [**breaking**] remove `CompletionModel::{Response, StreamingResponse, Client, make}`; model construction moves to the required `CompletionClient::completion_model`
- *(completion)* [**breaking**] remove the `GetTokenUsage` trait — read `StreamFinal::usage`
- *(completion)* [**breaking**] remove `CompletionResponse::raw_response` — use a provider model's `raw_completion`/`raw_stream`

## [0.41.0](https://github.com/0xPlaygrounds/rig/compare/rig-agent-v0.0.0...rig-agent-v0.41.0) - 2026-07-28

### Added

- *(agent)* restore dynamic context helper ([#2219](https://github.com/0xPlaygrounds/rig/pull/2219)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] split rig-core and rig-agent behind the rig facade ([#2197](https://github.com/0xPlaygrounds/rig/pull/2197)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2197

### Other

- *(core,agent)* [**breaking**] make the WASM support matrix explicit and true ([#2213](https://github.com/0xPlaygrounds/rig/pull/2213)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(telemetry)* single declarative completion-parent contract ([#2208](https://github.com/0xPlaygrounds/rig/pull/2208)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] remove premature runtime-conformance crate, backfill gaps ([#2206](https://github.com/0xPlaygrounds/rig/pull/2206)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(client)* [**breaking**] single canonical CompletionClient + AgentClientExt ([#2205](https://github.com/0xPlaygrounds/rig/pull/2205)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
