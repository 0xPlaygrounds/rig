# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(bedrock)* `CompletionModel::with_guardrail` attaches a [Bedrock guardrail](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) (identifier, version, trace mode) to every Converse request the model issues. Requests previously had no way to carry `guardrailConfig` at all, which also made the response-side trace unreachable
- *(bedrock)* `types::converse_output` and `types::assistant_content` are public modules. `raw_completion` returns `AwsConverseOutput`, which wraps `InternalConverseOutput`; both lived behind `pub(crate)` modules, so a caller could not name the type the escape hatch hands back

### Fixed

- *(bedrock)* `raw_completion` no longer discards provider-only response data. The conversion from the SDK's `ConverseOutput` matched five fields and swallowed the rest, so the guardrail `trace`, `performance_config`, `service_tier` and the AWS `request_id` never reached the escape hatch whose contract is that nothing the provider sent was dropped. The trace is the costly one: a blocked turn normalizes to a content-filter finish reason and nothing else, so the assessment naming the policy that fired was unrecoverable. All four are now carried — the three SDK-typed ones as the SDK's own types, `#[serde(skip)]` because they are not `Serialize`, so an in-process caller reads them in full while a serialized response omits them

- *(bedrock)* [**breaking**] the model constants are the identifiers Bedrock can actually invoke. 39 of the 72 shipped constants could only fail: 29 identifiers (every `anthropic.claude-*` constant among them, plus the Titan text/image generators, Claude 2/Instant, Llama 3.2, Jamba Instruct, the older Stability ids and the Cohere `command-text` pair) are absent from `ListFoundationModels` in us-east-1, us-west-2, eu-central-1 and ap-northeast-1 and answer `ResourceNotFoundException` ("This model version has reached the end of its life"); 10 more exist but are servable only through a cross-region inference profile, so the bare identifier answers `ValidationException` ("Invocation of model ID … with on-demand throughput isn't supported"). Retired identifiers are removed; the profile-only families (`DEEPSEEK_R1`, `META_LLAMA_3_3_70B_INSTRUCT`, `META_LLAMA_4_*`, `MISTRAL_PIXTRAL_LARGE_2502`, `WRITER_PALMYRA_X4`/`X5`) now carry their `us.` profile identifier, and Anthropic — which had no working constant at all — is represented by `ANTHROPIC_CLAUDE_HAIKU_4_5`, `ANTHROPIC_CLAUDE_SONNET_4_5`, `ANTHROPIC_CLAUDE_OPUS_4_5`, `ANTHROPIC_CLAUDE_SONNET_4_6`, `ANTHROPIC_CLAUDE_SONNET_5` and `ANTHROPIC_CLAUDE_OPUS_5`. A `us.` prefix names a region family: callers outside the US substitute `eu.`/`apac.` (or `global.` where offered). Every retained identifier was invoked with `Converse` in us-east-1

- *(bedrock)* a provider error this SDK version cannot classify keeps the service's own body. `SdkError::into_service_error` funnels unmodeled exceptions — and any response whose `x-amzn-errortype` the transport dropped — into `Unhandled`, whose `meta()` is empty and whose message hides in its source, so the conversion's catch-all reported Bedrock's end-of-life notice as `ProviderError("An unexpected error occurred. Verify Internet connection or AWS keys")` with `provider_response_body() == None`. The raw HTTP body is now the fallback on all four conversions (converse, converse-stream, invoke-model → image and embedding); a classified exception's own message still wins

### Changed

- *(completion)* [**behavior**] an assistant message that converts to zero content blocks is rejected with rig-core's shared empty-response wording (via `message::require_non_empty_response`) — previously "Bedrock returned an assistant message with no content"

- *(completion)* message and tool-result content conversions follow rig-core's message-content change from `OneOrMany<T>` to `Vec<T>`; wire payloads are unchanged

- *(streaming)* the Converse stream routes through the shared `WireAdapter` driver: frame triage (unknown-variant warn-skip) lives in the one policy site, and `streaming::stream_from_events` is the events-first conformance seam driving already-typed SDK events through the full pipeline

### Fixed

- *(bedrock)* the assistant `toolUse` echo derives its `toolUseId` exactly like the result leg (provider-issued call id when one exists, else rig's minted handle) — a history whose handle and provider id diverge no longer replays an unmatched `toolUseId` pair that Converse rejects
- *(bedrock)* foreign encrypted reasoning (`ReasoningContent::Encrypted` — OpenAI Responses `encrypted_content`, OpenRouter `reasoning.encrypted`, Anthropic) is never shipped as Bedrock's own `redactedContent` and never fails the request: it degrades away with a warning, in every position including the all-encrypted block. Only Bedrock-native `Redacted` blobs (base64 applied by this crate's inbound legs) decode back onto the wire, and one that no longer decodes also degrades instead of erroring the request
- *(bedrock)* redacted reasoning survives all three legs — streaming no longer drops `RedactedContent`, the non-streaming path no longer fails the whole response, and it is replayed as `redactedContent` instead of being flattened into unsigned plaintext
- *(bedrock)* an unmodeled `ContentBlockStart` variant warns and skips instead of failing the stream with "Stream is empty", matching its sibling arms and the classify layer's Unknown policy
- *(bedrock)* a reasoning block mixing text with opaque (redacted/encrypted) content — the exact shape OpenAI Responses histories carry when `encrypted_content` is requested — degrades by dropping the un-representable opaque part instead of failing the whole request locally
- *(streaming)* the `MessageStop` straggler flush is gated on a `ToolUse` stop reason: a tool block truncated by `MaxTokens` is dropped with a warning instead of fabricating a `{}`-args call or a spurious error
- *(streaming)* emit every parallel tool call (in-flight state is keyed by `content_block_index` and flushed per `ContentBlockStop`); text after a closed tool block is no longer dropped; malformed tool-call JSON surfaces an `Err` item instead of silently dropping the call under a `ToolCalls` terminal
## [0.41.0](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.40.0...rig-bedrock-v0.41.0) - 2026-07-28

### Added

- *(agent)* restore dynamic context helper ([#2219](https://github.com/0xPlaygrounds/rig/pull/2219)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] split rig-core and rig-agent behind the rig facade ([#2197](https://github.com/0xPlaygrounds/rig/pull/2197)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2197
- *(telemetry)* make sensitive span content opt-in ([#2151](https://github.com/0xPlaygrounds/rig/pull/2151)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(aws)* remove legacy rustls connector ([#2152](https://github.com/0xPlaygrounds/rig/pull/2152)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(derive)* [**breaking**] single resolution authority, coherent required semantics, dependency hygiene ([#2207](https://github.com/0xPlaygrounds/rig/pull/2207)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(client)* [**breaking**] single canonical CompletionClient + AgentClientExt ([#2205](https://github.com/0xPlaygrounds/rig/pull/2205)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Make managed agent hooks provider-independent ([#2176](https://github.com/0xPlaygrounds/rig/pull/2176)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2176
- Remove built-in agent dynamic context ([#2174](https://github.com/0xPlaygrounds/rig/pull/2174)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2174
- Simplify tool execution and hook APIs ([#2132](https://github.com/0xPlaygrounds/rig/pull/2132)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2132
- *(telemetry)* centralize completion span lifecycle ([#2115](https://github.com/0xPlaygrounds/rig/pull/2115)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.40.0](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.39.0...rig-bedrock-v0.40.0) - 2026-07-10

### Added

- *(rig-core)* [**breaking**] broaden provider error-response inspection workspace-wide ([#1944](https://github.com/0xPlaygrounds/rig/pull/1944)) (by @gold-silver-copper)

### Other

- add Bedrock cassette coverage ([#2084](https://github.com/0xPlaygrounds/rig/pull/2084)) (by @gold-silver-copper) - #2084
- Flatten Tool metadata API ([#2029](https://github.com/0xPlaygrounds/rig/pull/2029)) (by @gold-silver-copper) - #2029
- *(rig-core)* replace nanoid with fastrand for internal IDs ([#1938](https://github.com/0xPlaygrounds/rig/pull/1938)) (by @gold-silver-copper)

### Contributors

* @gold-silver-copper
## [0.39.0](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.38.2...rig-bedrock-v0.39.0) - 2026-06-19

### Added

- *(agent)* [**breaking**] sans-IO AgentRun state machine; both agent loops become thin drivers ([#1899](https://github.com/0xPlaygrounds/rig/pull/1899)) (by @gold-silver-copper)

### Contributors

* @gold-silver-copper
## [0.38.1](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.7...rig-bedrock-v0.38.1) - 2026-06-02

### Other

- unify workspace crate versions ([#1853](https://github.com/0xPlaygrounds/rig/pull/1853)) (by @gold-silver-copper) - #1853

### Contributors

* @gold-silver-copper
## [0.4.7](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.6...rig-bedrock-v0.4.7) - 2026-06-02

### Added

- *(gemini)* expose streaming response metadata ([#1790](https://github.com/0xPlaygrounds/rig/pull/1790)) (by @mateobelanger)
- *(anthropic)* support document citations ([#1778](https://github.com/0xPlaygrounds/rig/pull/1778)) (by @temrjan)

### Contributors

* @mateobelanger
* @temrjan
## [0.4.6](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.5...rig-bedrock-v0.4.6) - 2026-05-13

### Added

- *(rig-bedrock)* add structured output support via Converse API ([#1667](https://github.com/0xPlaygrounds/rig/pull/1667)) (by @jdwil)

### Fixed

- *(gemini)* Token usage correctness for posthog llm analytics ([#1761](https://github.com/0xPlaygrounds/rig/pull/1761)) (by @mateobelanger)

### Other

- fix "a ancient" grammar in glarb-glarb sample text ([#1755](https://github.com/0xPlaygrounds/rig/pull/1755)) (by @abhicris) - #1755
- AGENTS.MD, CONTRIBUTING.MD, and docs ([#1714](https://github.com/0xPlaygrounds/rig/pull/1714)) (by @gold-silver-copper) - #1714
- improve project organization and create rig crate ([#1699](https://github.com/0xPlaygrounds/rig/pull/1699)) (by @gold-silver-copper) - #1699

### Contributors

* @mateobelanger
* @abhicris
* @jdwil
* @gold-silver-copper
## [0.4.5](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.4...rig-bedrock-v0.4.5) - 2026-04-28

### Fixed

- *(bedrock)* preserve adaptive-thinking signatures in streaming reasoning ([#1683](https://github.com/0xPlaygrounds/rig/pull/1683)) (by @byQuexo)
- *(bedrock)* handle adaptive-thinking interactions for prompt caching and reasoning conversion ([#1675](https://github.com/0xPlaygrounds/rig/pull/1675)) (by @byQuexo)
- *(bedrock)* preserve all parallel tool calls in completion response ([#1626](https://github.com/0xPlaygrounds/rig/pull/1626)) (by @aleksmeshr)

### Other

- Add clippy no panic lints ([#1663](https://github.com/0xPlaygrounds/rig/pull/1663)) (by @gold-silver-copper) - #1663
- remove deprecated code ([#1633](https://github.com/0xPlaygrounds/rig/pull/1633)) (by @gold-silver-copper) - #1633

### Contributors

* @byQuexo
* @gold-silver-copper
* @aleksmeshr
## [0.4.4](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.3...rig-bedrock-v0.4.4) - 2026-04-12

### Added

- *(rig-bedrock)* add OpenTelemetry tracing to completion model ([#1567](https://github.com/0xPlaygrounds/rig/pull/1567)) (by @sachin-punyani)

### Other

- Add support for prompt caching in rig-bedrock ([#1584](https://github.com/0xPlaygrounds/rig/pull/1584)) (by @marcbrooker) - #1584

### Contributors

* @sachin-punyani
* @marcbrooker

## [0.4.3](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.2...rig-bedrock-v0.4.3) - 2026-03-29

### Other

- OTel GenAI semconv fix +  anthropic automatic prompt caching  ([#1572](https://github.com/0xPlaygrounds/rig/pull/1572))

## [0.4.2](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.1...rig-bedrock-v0.4.2) - 2026-03-17

### Other

- Change preamble to system message internally ([#1527](https://github.com/0xPlaygrounds/rig/pull/1527))


## [0.4.1](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.4.0...rig-bedrock-v0.4.1) - 2026-03-05

### Other

- updated the following local packages: rig-derive, rig-core

## [0.3.13](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.12...rig-bedrock-v0.3.13) - 2026-02-17

### Added

- cross-provider reasoning trace roundtrip ([#1396](https://github.com/0xPlaygrounds/rig/pull/1396))
- *(rig-1189)* structured outputs ([#1382](https://github.com/0xPlaygrounds/rig/pull/1382))
- *(rig-core)* add optional model override to CompletionRequest ([#1374](https://github.com/0xPlaygrounds/rig/pull/1374))

### Other

- Disable default features on aws-bedrock-runtime ([#1363](https://github.com/0xPlaygrounds/rig/pull/1363))
- typed reasoning content model ([#1395](https://github.com/0xPlaygrounds/rig/pull/1395))

## [0.3.12](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.11...rig-bedrock-v0.3.12) - 2026-02-03

### Other

- updated the following local packages: rig-core

## [0.3.11](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.10...rig-bedrock-v0.3.11) - 2026-01-20

### Other

- updated the following local packages: rig-core

## [0.3.10](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.9...rig-bedrock-v0.3.10) - 2026-01-06

### Other

- add tool name to tool call delta streaming events ([#1222](https://github.com/0xPlaygrounds/rig/pull/1222))

## [0.3.9](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.8...rig-bedrock-v0.3.9) - 2025-12-15

### Other

- ToolCall Signature and additional parameters ([#1154](https://github.com/0xPlaygrounds/rig/pull/1154))
- *(rig-1090)* crate re-org ([#1145](https://github.com/0xPlaygrounds/rig/pull/1145))

## [0.3.8](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.7...rig-bedrock-v0.3.8) - 2025-12-04

### Other

- updated the following local packages: rig-core

## [0.3.7](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.6...rig-bedrock-v0.3.7) - 2025-12-01

### Added

- Gemini Assistant Image Responses ([#1048](https://github.com/0xPlaygrounds/rig/pull/1048))
- *(rig-985)* Consolidate provider clients ([#1050](https://github.com/0xPlaygrounds/rig/pull/1050))

### Fixed

- *(rig-1050)* Inconsistent model/agent initialisation methods ([#1069](https://github.com/0xPlaygrounds/rig/pull/1069))

### Other

- Deprecate `DynClientBuilder` ([#1105](https://github.com/0xPlaygrounds/rig/pull/1105))

## [0.3.6](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.5...rig-bedrock-v0.3.6) - 2025-11-10

### Added

- *(providers)* Emit tool call deltas ([#1020](https://github.com/0xPlaygrounds/rig/pull/1020))

## [0.3.5](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.4...rig-bedrock-v0.3.5) - 2025-10-28

### Other

- updated the following local packages: rig-core

## [0.3.4](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.3...rig-bedrock-v0.3.4) - 2025-10-27

### Added

- *(bedrock)* Support streaming thinking ([#946](https://github.com/0xPlaygrounds/rig/pull/946))
- *(bedrock)* Implement usage ([#934](https://github.com/0xPlaygrounds/rig/pull/934))

### Other

- Fix bedrock tool calls with zero arguments ([#989](https://github.com/0xPlaygrounds/rig/pull/989))
- Dependent packages no longer force unnecessary features on rig-core ([#964](https://github.com/0xPlaygrounds/rig/pull/964))

## [0.3.3](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.2...rig-bedrock-v0.3.3) - 2025-10-14

### Added

- *(rig-973)* DocumentSourceKind::String ([#882](https://github.com/0xPlaygrounds/rig/pull/882))

### Other

- provider SDK has issue with DocumentBlock ([#892](https://github.com/0xPlaygrounds/rig/pull/892))

## [0.3.2](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.1...rig-bedrock-v0.3.2) - 2025-09-29

### Added

- *(rig-795)* support file URLs for audio, video, documents ([#823](https://github.com/0xPlaygrounds/rig/pull/823))

### Other

- *(rig-963)* fix feature regression in AWS bedrock ([#863](https://github.com/0xPlaygrounds/rig/pull/863))

## [0.3.1](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.3.0...rig-bedrock-v0.3.1) - 2025-09-15

### Added

- *(rig-931)* support file input for images on Gemini ([#790](https://github.com/0xPlaygrounds/rig/pull/790))

## [0.3.0](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.9...rig-bedrock-v0.3.0) - 2025-09-02

### Added

- VerifyClient trait ([#724](https://github.com/0xPlaygrounds/rig/pull/724))

### Other

- added AWS Bedrock client creation using from_env ([#710](https://github.com/0xPlaygrounds/rig/pull/710))

## [0.2.9](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.8...rig-bedrock-v0.2.9) - 2025-08-20

### Other

- updated the following local packages: rig-core

## [0.2.8](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.7...rig-bedrock-v0.2.8) - 2025-08-19

### Other

- updated the following local packages: rig-core

## [0.2.7](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.6...rig-bedrock-v0.2.7) - 2025-08-19

### Added

- *(rig-865)* multi turn streaming ([#712](https://github.com/0xPlaygrounds/rig/pull/712))
- video input for gemini ([#690](https://github.com/0xPlaygrounds/rig/pull/690))

## [0.2.6](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.5...rig-bedrock-v0.2.6) - 2025-08-05

### Other

- updated the following local packages: rig-core

## [0.2.5](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.4...rig-bedrock-v0.2.5) - 2025-08-05

### Other

- updated the following local packages: rig-core

## [0.2.4](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.3...rig-bedrock-v0.2.4) - 2025-07-30

### Added

- *(rig-812)* yield final response with total usage metrics from streaming completion response in stream impl ([#584](https://github.com/0xPlaygrounds/rig/pull/584))
- *(rig-784)* thinking/reasoning ([#557](https://github.com/0xPlaygrounds/rig/pull/557))

## [0.2.3](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.2...rig-bedrock-v0.2.3) - 2025-07-16

### Other

- updated the following local packages: rig-core

## [0.2.2](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.1...rig-bedrock-v0.2.2) - 2025-07-14

### Added

- *(rig-801)* DynClientBuilder::from_values ([#556](https://github.com/0xPlaygrounds/rig/pull/556))
- add `.extended_details` to `PromptRequest` ([#555](https://github.com/0xPlaygrounds/rig/pull/555))

## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.2.0...rig-bedrock-v0.2.1) - 2025-07-07

### Added

- *(rig-780)* integrate openAI responses API ([#508](https://github.com/0xPlaygrounds/rig/pull/508))

### Other

- Migrate all crates to Rust 2024 ([#539](https://github.com/0xPlaygrounds/rig/pull/539))
- Declare shared dependencies in workspace ([#538](https://github.com/0xPlaygrounds/rig/pull/538))
- Make clippy happy on all targets ([#542](https://github.com/0xPlaygrounds/rig/pull/542))

## [0.2.0](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.1.3...rig-bedrock-v0.2.0) - 2025-06-09

### Added

- Improve Streaming API ([#388](https://github.com/0xPlaygrounds/rig/pull/388))

### Other

- Introduce Client Traits and Testing ([#440](https://github.com/0xPlaygrounds/rig/pull/440))

## [0.1.3](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.1.2...rig-bedrock-v0.1.3) - 2025-04-30

### Fixed

- fixed bug with base64 encoding on AWS Bedrock ([#432](https://github.com/0xPlaygrounds/rig/pull/432))

## [0.1.2](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.1.1...rig-bedrock-v0.1.2) - 2025-04-29

### Added

- multi-turn / reasoning loops + parallel tool calling ([#370](https://github.com/0xPlaygrounds/rig/pull/370))
- support custom clients for bedrock ([#403](https://github.com/0xPlaygrounds/rig/pull/403))

## [0.1.1](https://github.com/0xPlaygrounds/rig/compare/rig-bedrock-v0.1.0...rig-bedrock-v0.1.1) - 2025-04-12

### Other

- updated the following local packages: rig-derive
