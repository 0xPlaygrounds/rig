# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- *(agent)* [**breaking**] [**behavior**] the terminal finish reason survives to the caller, and an empty **truncated** turn is no longer a successful empty answer (#2322). The streamed assembler read `usage` and "did I see text" off the provider's terminal record and discarded the rest, so `FinishReason::Length` — which the provider layer carried correctly on `StreamFinal` — never reached the agent surface: a turn cut short at the output-token limit was *undetectable* through `prompt`/`stream_prompt`, and finalized as a successful `""`. Combined with rig-core's hardcoded Gemini `maxOutputTokens: 4096`, that is how a truncated response reached users as an unexplained blank answer with nothing to inspect. `CompletionCall` gains `finish_reason` (per call, not per run — a multi-turn run has N reasons and the truncated one is the diagnostic), populated on **both** surfaces; `ModelTurn` and `StreamedTurn` gain `finish_reason` with `with_finish_reason` builders; `StreamedTurnEvent::Completed` gains a `finish_reason` field. A turn that delivered **no answer** — no tool call and no non-empty text — **and** reports `Length` or `ContentFilter` now fails with a `ResponseError` naming the reason, matching what the blocking Gemini path already did for a content-less candidate. Reasoning does not count as an answer, and that is load-bearing rather than incidental: Gemini counts thinking tokens against `maxOutputTokens`, so a truncated thinking turn *typically* carries reasoning and no text, making reasoning-only the common shape of this failure. Deliberately narrow: partial output followed by truncation stays a valid answer (with the reason preserved on the call), a turn reporting `Stop`/`ToolCalls` is unchanged whatever its shape, and `FinishReason::Other` is **not** treated as truncation because it carries a provider's own wire spelling with no normalized meaning. The turn is still recorded to history before the error, so partial reasoning stays available for debugging. This also narrows `OutputMode::Tool` recovery: an answerless turn reporting a truncating reason now fails immediately instead of consuming an output-retry to re-prompt, because the budget or the filter — not the phrasing — is what stopped it, so the retry would truncate again and fail less specifically; re-prompting is unchanged for answerless turns with any other reason. [**breaking**] `AgentRun::record_streamed_completion_call` takes the finish reason as a third argument

### Added

- *(agent)* response identity metadata reaches every completed model call's observers (#2265), keyed on the shared `rig_core::completion::ResponseIdentity` carrier: the `CompletionResponse`, `StreamResponseFinish`, and `ModelTurnFinished` hook events carry `identity: &ResponseIdentity` — `ModelTurnFinished` fires for every accepted turn on both surfaces (text, tool-only, reasoning-only, multi-turn), so an observer of that one event records identity for every completed call, and on a retry each event carries the retried attempt's own ids, never a previous attempt's. `PromptResponse.completion_calls` entries gain `message_id`, `response_id`, and `provider_request_id` (serde-defaulted; pre-identity run JSON still loads). `StreamResponseFinish` remains text-turn-scoped and both response events remain suppressed for invalid-tool-recovered turns — intentional, documented, and now covered by tests. [**breaking**] `CompletionCall` is no longer `Copy` (owned identity strings); `AgentRun::record_streamed_completion_call` takes the identity as a second argument; `ModelTurn` gains `response_id`/`provider_request_id` with a `with_identity` builder; the three hook events have a new `identity` field

### Changed

- *(agent)* `AgentBuilder`, `Agent`, and `AgentRunner` now share one private `AgentConfig`; `AgentRunner::from_agent` clones it as a single unit instead of copying 15 settings field by field, so a new agent setting can no longer compile while silently failing to propagate to execution (#2326). Per-run overrides mutate only the runner's cloned config, never the source agent. Internal renames within the private config: `default_max_turns` is now a resolved `max_turns: usize` (default `1`, preserving the one-call budget) and `default_conversation_id` is now `conversation_id`. No public API change

- *(agent)* [**breaking**] persisted histories and `AgentRun`/`PromptResponse` JSON carry rig-core's tagged assistant content (`{"type": "text", ...}`); the untagged shape does not load — see rig-core's entry and MIGRATING. The flatten `Some({})` round-trip artifact is gone, so `is_empty_assistant_turn`'s classification is identical before and after a persist/restore with no special-casing

- *(agent)* [**behavior**] the streamed assembler counts the stream items it excludes from assembly that carry assistant content — replayed tagged assistant blocks (the tagged `AssistantContent` serialization is not a stream-item shape) and text items whose `additional_params` is malformed — and logs a single warning per turn, on every termination path, instead of one per stream item; `StreamedTurnAssembler::excluded_assistant_content` exposes the count, and the full decode-outcome contract is pinned by an enum-driven matrix test. A stream item whose text block carries stray sibling keys decodes as stream *text* — the text is assembled and only the stray keys drop

- *(agent)* [**breaking**] `PromptResponse::content` returns `&[AssistantContent]` instead of `&Vec<AssistantContent>`, matching its slice-returning siblings

- *(agent)* [**breaking**] `PromptResponse` JSON serialized before the `content` field existed no longer deserializes — the missing-`content` reconstruction (and the serde shadow repr that carried it) is deleted and `content` is a required field; the JSON wire shape is unchanged

- *(agent)* [**breaking**] message content is a plain `Vec<T>`, following rig-core's removal of the non-empty container — every content list the agent constructs, inspects, or hands to hooks is a `Vec`, and the `OneOrMany` re-export is gone from this crate

- *(agent)* [**behavior**] a genuinely empty assistant turn no longer cancels the run and is no longer padded with a fabricated empty-text part — with the non-empty container gone the turn is honestly representable as an empty list, and `is_empty_assistant_turn` neutralizes it instead of the agent inventing content or failing a run that used to succeed

- *(agent)* every model call validates request content before the round trip: both turn drivers issue their request through `CompletionRequestBuilder::send`/`stream`, which run `CompletionRequest::validate_message_content`, so an empty history or a content-less message fails locally with a named index instead of a remote 400

- *(agent)* [**breaking**] the `ToolCallDelta` hook payload loses `tool_call_id` (provider ids arrive on the completed call; `internal_call_id` is the correlator); the streamed assembler keys delta state by `internal_call_id` and takes the assembled reasoning block's durable id only from the provider-issued `provider_id` — a rig correlator can no longer enter history. `MultiTurnStreamItem` delta ids are rig-generated correlators, stable per part and unique per run turn — correlate across a run by `internal_call_id`, never by delta id

- *(agent)* [**breaking**] tool-call identity follows rig-core's typed model: every call a hook or consumer observes carries a unique, non-empty `ToolCallId` (the provider's id when issued, minted otherwise, with provider absence recorded on `ToolCall::provider`). Hook contexts (`ToolCallEvent`/`ToolResultEvent`/`InvalidToolCallContext`) surface that durable id — never `Some("")` — and the invalid-call retry transcript correlates the invalid call and every validated peer by their own ids, so id-less wires no longer collapse peers onto shared feedback or replay empty `tool_call_id`s. `StreamedResolution::TurnAbandoned::skipped_tool_result` is boxed

### Added

- *(agent)* add `AgentHook::on_reasoning_delta` with the Rig stream correlator, optional provider reasoning id, current fragment, and per-part aggregate; reasoning hooks share the existing observation-interest and stop-before-yield semantics used by other streaming deltas

- *(agent)* add opaque, cloneable `ModelHandle` values with by-value `ProviderCapabilities` snapshots, plus default replacement, per-run default override (`using_model`), and hook-driven per-call selection via `AgentHook::on_model_select`
- *(agent)* add run-local extractor default-model overrides used across retries

### Changed

- *(agent)* [**breaking**] remove concrete model parameters from long-lived classic runtime types (`Agent`, `AgentBuilder` after `new()`, `AgentRunner`, prompt/stream requests, `Extractor`) — the typed model is erased once at construction; direct provider-model completion and streaming APIs remain typed
- *(agent)* completion-call hooks now resolve before model selection: the merged `RequestPatch` is exposed on `ModelSelection::request_patch`, request preparation runs against the selected model's captured capabilities, and `ModelSelection::previous_model` reflects issued attempts only

- *(agent)* internal consolidation with no API or semantic change: `AgentBuilder`'s typestate transitions and all three `build()`s thread their fields through one shared core (a new field can no longer silently miss one of the five copies); the erased hook dispatch and the first-non-`Continue` observation loops are generated from one event list, with the rewrite-salvage `tool_call` frame, first-`Some` `invalid_tool_call`, and sync `model_select` staying hand-written; the blocking and streaming surfaces share the agent-span/memory-resolution prologue (explicit history still bypasses memory entirely, and a streamed memory-load failure still surfaces under the agent span)

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
