# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
