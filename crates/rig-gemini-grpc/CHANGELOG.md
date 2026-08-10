# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- *(streaming)* [**behavior**] streamed function calls carry a single-wire identity: the wire's one id travels as the part id only, so `provider` is `{call_id, item_id: None}` — filling both slots fabricated a dual identity the wire never issued (mirrors the rig-core gemini fix)

- *(completion)* message and tool-result content conversions follow rig-core's message-content change from `OneOrMany<T>` to `Vec<T>`; wire payloads are unchanged

### Fixed

- *(gemini)* [**behavior**] the gRPC surface now reports `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL` and `TOO_MANY_TOOL_CALLS` as errors and stops the stream, matching REST — previously an aborted turn was reported as a completed one, and the wire's `finish_message` was never read
- *(gemini)* a `thought_signature` carried on a trailing non-thought part is no longer dropped — it attaches to the reasoning block it signs via the shared `ReasoningSignature` lifecycle event
- *(gemini)* [**behavior**] the tool name is no longer used as the tool-call id when the wire omits one; id-less calls carry the absent id and stay distinct

### Changed

- *(streaming)* the gRPC stream routes through the shared `WireAdapter` driver; an unrecognized part kind (`part.data` oneof decoding to `None`) is warn-skipped instead of silently dropped; `streaming::stream_from_events` is the events-first conformance seam; the generated `proto` module is public to support it

## [0.2.6](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.5...rig-gemini-grpc-v0.2.6) - 2026-05-13

### Fixed

- *(gemini)* Token usage correctness for posthog llm analytics ([#1761](https://github.com/0xPlaygrounds/rig/pull/1761)) (by @mateobelanger)

### Other

- bump dependencies ([#1728](https://github.com/0xPlaygrounds/rig/pull/1728)) (by @gold-silver-copper) - #1728
- AGENTS.MD, CONTRIBUTING.MD, and docs ([#1714](https://github.com/0xPlaygrounds/rig/pull/1714)) (by @gold-silver-copper) - #1714
- improve project organization and create rig crate ([#1699](https://github.com/0xPlaygrounds/rig/pull/1699)) (by @gold-silver-copper) - #1699

### Contributors

* @mateobelanger
* @gold-silver-copper## [0.38.1](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.7...rig-gemini-grpc-v0.38.1) - 2026-06-02

### Other

- unify workspace crate versions ([#1853](https://github.com/0xPlaygrounds/rig/pull/1853)) (by @gold-silver-copper) - #1853

### Contributors

* @gold-silver-copper## [0.39.0](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.38.2...rig-gemini-grpc-v0.39.0) - 2026-06-19

### Added

- *(agent)* [**breaking**] sans-IO AgentRun state machine; both agent loops become thin drivers ([#1899](https://github.com/0xPlaygrounds/rig/pull/1899)) (by @gold-silver-copper)

### Contributors

* @gold-silver-copper## [0.41.0](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.40.0...rig-gemini-grpc-v0.41.0) - 2026-07-28

### Added

- [**breaking**] split rig-core and rig-agent behind the rig facade ([#2197](https://github.com/0xPlaygrounds/rig/pull/2197)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2197
- *(telemetry)* make sensitive span content opt-in ([#2151](https://github.com/0xPlaygrounds/rig/pull/2151)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- Simplify tool execution and hook APIs ([#2132](https://github.com/0xPlaygrounds/rig/pull/2132)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2132

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)

## [0.40.0](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.39.0...rig-gemini-grpc-v0.40.0) - 2026-07-10

### Added

- *(rig-core)* [**breaking**] broaden provider error-response inspection workspace-wide ([#1944](https://github.com/0xPlaygrounds/rig/pull/1944)) (by @gold-silver-copper)

### Contributors

* @gold-silver-copper

## [0.38.2](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.38.1...rig-gemini-grpc-v0.38.2) - 2026-06-09

### Other

- update Cargo.toml dependencies

## [0.2.7](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.6...rig-gemini-grpc-v0.2.7) - 2026-06-02

### Added

- *(gemini)* expose streaming response metadata ([#1790](https://github.com/0xPlaygrounds/rig/pull/1790)) (by @mateobelanger)
- *(anthropic)* support document citations ([#1778](https://github.com/0xPlaygrounds/rig/pull/1778)) (by @temrjan)

### Fixed

- *(rig-gemini-grpc)* populate FunctionDeclaration.parameters from ToolDefinition ([#1763](https://github.com/0xPlaygrounds/rig/pull/1763)) (by @abhicris)

### Contributors

* @abhicris
* @mateobelanger
* @temrjan

## [0.2.5](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.4...rig-gemini-grpc-v0.2.5) - 2026-04-28

### Other

- Add clippy no panic lints ([#1663](https://github.com/0xPlaygrounds/rig/pull/1663)) (by @gold-silver-copper) - #1663

### Contributors

* @gold-silver-copper

## [0.2.4](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.3...rig-gemini-grpc-v0.2.4) - 2026-04-12

### Other

- updated the following local packages: rig-core

## [0.2.3](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.2...rig-gemini-grpc-v0.2.3) - 2026-03-29

### Other

- OTel GenAI semconv fix +  anthropic automatic prompt caching  ([#1572](https://github.com/0xPlaygrounds/rig/pull/1572))

## [0.2.2](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.1...rig-gemini-grpc-v0.2.2) - 2026-03-17

### Other

- Change preamble to system message internally ([#1527](https://github.com/0xPlaygrounds/rig/pull/1527))

## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-gemini-grpc-v0.2.0...rig-gemini-grpc-v0.2.1) - 2026-03-05

### Other

- updated the following local packages: rig-core

## [0.1.0] - 2026-01-14

### Added
- Initial release of rig-gemini-grpc
- Gemini gRPC completion support
- Gemini gRPC embedding support
- Streaming completions
- Tool calling support
- Reasoning support with thought signatures
- Image input support
- Migration guide from rig-core's gemini_grpc module
