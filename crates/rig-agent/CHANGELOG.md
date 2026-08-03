# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- [**breaking**] `StreamedTurnAssembler::partial_turn` now borrows an `Option<&str>` and owns the ID only when it creates the rollback snapshot; `finish` now consumes its final `OneOrMany<AssistantContent>`. Drivers should inspect live streams by borrow, then convert an exhausted `CompletionStream` into `CompletionResponse` and move its `message_id` and `choice` into `finish`.
- [**breaking**] `ModelTurnOutcome::Continue` now carries an `AcceptedModelTurn`, the shared canonical post-resolution record used by both drivers. Repaired invalid-call turns cross `ModelTurnFinished` exactly once in blocking and streaming runs before tool preflight; provider-response observations retain their intentionally medium-specific suppression behavior.
- [**breaking**] Tool-result submissions and execution records carry `ToolInvocationDisposition`. `ToolExecutionCommitted` now means a local tool body ran or the host explicitly supplied an externally executed result; policy skips, invalid-recovery skips, unknown tools, and other pre-resolved outcomes still commit model-visible results without emitting an execution observation.

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
