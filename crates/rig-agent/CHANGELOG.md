# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- [**breaking**] `StreamedTurnAssembler::partial_turn` now borrows an `Option<&str>` and owns the ID only when it creates the rollback snapshot; `finish` now consumes its final `OneOrMany<AssistantContent>`. Drivers should inspect live streams by borrow, then convert an exhausted `CompletionStream` into `CompletionResponse` and move its `message_id` and `choice` into `finish`.
- [**breaking**] `ModelTurnOutcome::Continue` now carries an `AcceptedModelTurn`, the shared canonical post-resolution record used by both drivers. Its committed attempt context supplies the real prompt, a per-provider-operation identity, and the effective Tool-mode name/schema; `PreparedRequest::output_tool_contract` replaces the former name-only `output_tool_name` field, and its opaque `PreparedModelAttempt` receipt binds that exact attempt plus the validation name sets to hand-driven unary or streamed responses after provider I/O. Response observations no longer depend on `BeforeModelCall` surfacing; Tool-mode validation follows the advertised attempt schema, corrective retries inherit it unless explicitly replaced, and streamed text aggregation resets for every reissued provider operation. `AgentRun` now serializes pending-verdict and ready-to-advance turns as distinct variants, so `continue_model_turn` is required before `next_step`.
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
