# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- [**breaking**] `ProviderConfig` adds one `External` serde arm for out-of-tree completion providers. `ExternalCompletionProvider` is a typed, non-object authoring contract whose ordinary futures are erased by `ExternalCompletionProviderEntry::from_provider` into private callbacks stored in a frozen, host-owned `Runtime` registry. Serialized external config contains only an exact-version ID, model, and redacted-debug JSON settings; handler-owned capabilities, unary/streaming behavior, and closure state are re-registered after restore. `ProviderConfig::descriptor` now takes a runtime and returns a borrowed descriptor view so external capability data is never duplicated into config.
- [**breaking**] `ProviderConfig` and `EmbedderConfig` are closed, feature-stable serde vocabularies generated from one capability registry. Their deterministic `Mock` variants are always present and production-visible, so exhaustive downstream matches must handle them. Hosts accepting serialized provider configuration across an untrusted boundary must validate or allowlist provider variants and fields before fulfillment; mock completion scripts can deliberately remain pending until cancellation.
- [**breaking**] `StreamedTurnAssembler::partial_turn` now borrows an `Option<&str>` and owns the ID only when it creates the rollback snapshot; `finish` now consumes its final `OneOrMany<AssistantContent>`. Drivers should inspect live streams by borrow, then convert an exhausted `CompletionStream` into `CompletionResponse` and move its `message_id` and `choice` into `finish`.
- [**breaking**] `ModelTurnOutcome::Continue` carries the shared `AcceptedModelTurn` record and the enum is now `#[must_use]`. Its committed attempt context supplies the real prompt, provider-operation identity, and effective Tool-mode name/schema. `AgentRunStep::CallModel` exposes an opaque `ModelAttemptId`; hand-driven callers pass it to `prepare_request`, direct `ModelTurn`/`StreamedTurnAssembler` construction, and streamed completion recording. Stale or replayed unary turns, streamed partial/final turns, and completion records are rejected without poisoning the current reissue. `AgentRun::with_output_validation` takes `Option<schemars::Schema>`, `prepare_request` accepts `AgentRun::inherited_output_contract()` for corrective calls, and the non-`Clone` `PreparedModelAttempt` is consumed by `into_model_turn` or `into_streamed_turn_assembler`. The streaming assembler carries that exact contract through partial invalid-call Retry/Skip and final assembly, eliminating private post-recovery repair. `AgentRun` serializes pending-verdict and ready-to-advance turns separately, so callers must exhaustively handle the outcome and call `continue_model_turn` only for `Continue`.
- [**breaking**] `max_turns` now counts every emitted model-call attempt, including local preparation failures, failed/cancelled provider operations, and their reissues. Transaction rollback restores the request patch and logical turn position but never refunds the attempt counter; reissues retain the logical turn number, receive a fresh attempt identity, and terminate with `MaxTurnsError` at the configured boundary on both session drivers. Streaming uses one post-`Final` guard for normal and recovery ingestion, and no-observer runs allocate no cumulative text aggregation.
- [**breaking**] Tool-result submissions and execution records carry `ToolInvocationDisposition`. Submissions join by Rig identity but commit in the pending calls' model-emitted order, so manual out-of-order `provide_tool_results` calls now produce source-ordered wire results. `ToolExecutionCommitted` means a local tool body ran or the host explicitly supplied external execution; policy skips, invalid-recovery skips, and unknown tools still commit model-visible results without an execution observation. Unknown tools also open no `execute_tool` span and do not advance `last_span_id`. Live execution/policy records carry authoritative `raw_result`; reconstructed host/serialized pre-resolved content is a lossy success view, while disposition remains execution provenance.

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
