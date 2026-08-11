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

- *(agent)* [**breaking**] `ModelTurnOutcome` is `#[must_use]`. It carries `NeedsResolution`, which must be answered via `resolve_invalid_tool_call` before the run may advance; dropping it lets a hallucinated tool name surface two steps later as an unrelated "next_step called while an invalid tool-call resolution is pending". Code writing `run.model_response(turn)?;` now warns

- *(agent)* [**breaking**] the serialized `AgentRun` carries a `$schemaVersion` tag (`rig_agent::agent::run::RUN_SCHEMA_VERSION`, currently `"1.0"`) and a build reads only the version it writes. A payload from another version — including one written before versioning existed, which has no tag — fails deserialization with a named error instead of being silently reinterpreted. Runs suspended by an earlier build cannot be resumed by this one; drain them before upgrading

- *(agent)* [**breaking**] message content is a plain `Vec<T>`, following rig-core's removal of the non-empty container — every content list the agent constructs, inspects, or hands to hooks is a `Vec`, and the `OneOrMany` re-export is gone from this crate

- *(agent)* [**behavior**] a genuinely empty assistant turn no longer cancels the run and is no longer padded with a fabricated empty-text part — with the non-empty container gone the turn is honestly representable as an empty list, and `is_empty_assistant_turn` neutralizes it instead of the agent inventing content or failing a run that used to succeed

- *(agent)* every model call validates request content before the round trip: both turn drivers issue their request through `CompletionRequestBuilder::send`/`stream`, which run `CompletionRequest::validate_message_content`, so an empty history or a content-less message fails locally with a named index instead of a remote 400

- *(agent)* [**breaking**] the `ToolCallDelta` hook payload loses `tool_call_id` (provider ids arrive on the completed call; `internal_call_id` is the correlator); the streamed assembler keys delta state by `internal_call_id` and takes the assembled reasoning block's durable id only from the provider-issued `provider_id` — a rig correlator can no longer enter history. `MultiTurnStreamItem` delta ids are rig-generated correlators, stable per part and unique per run turn — correlate across a run by `internal_call_id`, never by delta id

- *(agent)* [**breaking**] tool-call identity follows rig-core's typed model: every call a hook or consumer observes carries a unique, non-empty `ToolCallId` (the provider's id when issued, minted otherwise, with provider absence recorded on `ToolCall::provider`). Hook contexts (`ToolCallEvent`/`ToolResultEvent`/`InvalidToolCallContext`) surface that durable id — never `Some("")` — and the invalid-call retry transcript correlates the invalid call and every validated peer by their own ids, so id-less wires no longer collapse peers onto shared feedback or replay empty `tool_call_id`s. `StreamedResolution::TurnAbandoned::skipped_tool_result` is boxed

### Added

- *(agent)* add `AgentHook::on_reasoning_delta` with the Rig stream correlator, optional provider reasoning id, current fragment, and per-part aggregate; reasoning hooks share the existing observation-interest and stop-before-yield semantics used by other streaming deltas

- *(agent)* add `Agent::drive` / `Agent::drive_run`, returning the new public `AgentDriver` (with `DriveStep`, `TurnTools` and `TurnToolNames`): hand-drive the sans-IO `AgentRun` machine with the agent's own configuration while owning every side effect. The driver seeds the run from the agent (`default_max_turns`, `tool_choice`, output schema), yields each turn's fully configured completion request for the caller to send (or hand to a custom transport), pairs every pending tool batch with the exact registry snapshot that advertised it, assembles the model turn internally, and maintains the run's committed structured-output tool across turns exactly as the runner does. Dispatch is gated on the turn's advertised names, so a tool registered after a turn was prepared — or after a run was suspended — cannot be dispatched through a turn that never advertised it. Impossible `tool_choice`/tool-set combinations fail at prepare time with no provider round-trip, and such a failure costs nothing: the run advances only once a request exists, so it keeps its state and its turn budget and the step can be retried in place. The driver performs no provider IO and no dispatch of its own: hooks, memory, retrieval policy, and telemetry still run only under `AgentRunner`

- *(agent)* a driven run is resumable at **every** step boundary, not only while tool calls are pending. The turn's advertised tool names travel with the run — `TurnToolNames`, the serializable half of `TurnTools`, readable via `AgentRun::advertised_tools()` and constructible with `TurnToolNames::new` — so a run serialized after `DriveStep::SendRequest`, the natural suspension point for a queued or long-running provider call, resumes in another process and accepts its reply, validated against the set the request actually carried rather than whatever the resuming registry holds. The names describe the turn in flight and are cleared on every route back to preparing, so a run parked for a fresh call never reports a turn that is over. Resuming rebuilds the dispatch target by resolving those names against the registry directly, without re-running retrieval: a registered *dynamic* tool the turn used is found whether or not a fresh query would rank it, absence means the tool is genuinely gone (surfaced as an error, opt out with `allow_missing_resumed_tools`), and resume does not fail when a vector index is unavailable

- *(agent)* a committed model call records what the turn *resolved to*, not just what it advertised: `PreparedTurnMetadata` (tool names, the tool choice the request actually carried, the synthetic output tool) is committed atomically by `AgentRun::commit_model_call` and readable via `AgentRun::prepared_turn()`. Invalid-tool-call context and the `Skip` rejection now answer from the turn rather than the run's baseline, so a per-turn `RequestPatch` that overrides `tool_choice` — from a `CompletionCall` hook or from a hand-driven turn — can no longer leave the state machine disagreeing with the request that went out. `AgentRun::advertised_tools()` remains as shorthand for the names

- *(agent)* `AgentRun`'s peek/commit halves are public — `peek_model_call`, `commit_model_call`, `advance`, `is_preparing_request`, with `Advance` and `ModelCallInputs` — so any hand-driver whose request preparation can fail gets the same "a failure costs no turn" guarantee the driver has, not just rig's own. Preconditions are `PromptError` protocol violations rather than debug assertions, and the model-call budget is enforced at `commit_model_call`, where the turn is actually spent, as well as at `peek_model_call`

- *(agent)* add `AgentRun::rollback_model_call` / `AgentDriver::rollback_model_call`: a request that could not be sent, or whose reply is known to be lost, hands its turn back and returns the run to preparing, so the next step yields a **freshly prepared** request rather than replaying one whose tool snapshot has gone stale. Refunds the turn, preserves usage the provider already billed (including a streamed turn that reports usage after the failure), and counts attempts via `AgentRun::model_call_rollbacks`. Deciding to use it takes two answers, not one: `CompletionError::is_retryable` says whether a retry *could succeed*, and only the caller can say whether one is *safe* — a stream that died after the request was written is retryable and not replay-safe, and retrying it bills a second completion. Bounding attempts is the caller's job

- *(agent)* add `AgentDriver::next_step_with`, taking a per-turn preparation callback (`TurnPreparation`, `TurnPreparationContext`) that supplies the `RequestPatch` and optionally a model for the turn about to be prepared. This is how a hand-driven run gets what the runner gets from its `CompletionCall` and model-selection hooks — per-turn preamble, sampling parameters, `tool_choice`, `active_tools` narrowing, extra context, substituted history — and the callback runs *before* anything advances, so a decision that fails costs no turn. Per-turn configuration is an input, never driver state: a driver holding it would resume a serialized run with configuration the suspending process never recorded, and nothing on the run would say so. `AgentRun::tool_choice()` is public and a custom run's own choice reaches the provider, rather than governing only the run's internal decisions while the request carries the agent's baseline

- *(agent)* add driver-owned streamed ingress — `AgentDriver::record_stream_usage`, `accept_streamed_turn`, `resolve_streamed_invalid_tool_call` — so a streamed turn enters through the same object that prepared it and stays paired with its dispatch snapshot. Driving a custom streaming transport is a headline use for this type, and it previously required reaching around the driver into the run; the alternative, rebuilding the driver from `into_run()`, discards the per-turn snapshot cache and makes the driver treat a turn prepared in the same process as a resume, drift check included

- *(agent)* add `AgentDriver::max_invalid_tool_call_retries`, mirroring the runner's. `Agent::drive` seeds the budget at zero, so answering a `NeedsResolution` with `InvalidToolCallAction::Retry` needs this raised first
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
