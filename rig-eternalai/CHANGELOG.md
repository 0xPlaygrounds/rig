# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.9](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.8...rig-eternalai-v0.3.9) - 2025-10-14

### Added

- *(rig-951)* generic HTTP client ([#875](https://github.com/0xPlaygrounds/rig/pull/875))

## [0.3.8](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.7...rig-eternalai-v0.3.8) - 2025-09-29

### Other

- updated the following local packages: rig-core

## [0.3.7](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.6...rig-eternalai-v0.3.7) - 2025-09-15

### Other

- updated the following local packages: rig-core

## [0.3.6](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.5...rig-eternalai-v0.3.6) - 2025-09-02

### Other

- *(rig-907)* use where clause for trait bounds ([#749](https://github.com/0xPlaygrounds/rig/pull/749))

## [0.3.5](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.4...rig-eternalai-v0.3.5) - 2025-08-20

### Other

- updated the following local packages: rig-core

## [0.3.4](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.3...rig-eternalai-v0.3.4) - 2025-08-19

### Other

- updated the following local packages: rig-core

## [0.3.3](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.2...rig-eternalai-v0.3.3) - 2025-08-19

### Added

- *(rig-865)* multi turn streaming ([#712](https://github.com/0xPlaygrounds/rig/pull/712))

## [0.3.2](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.1...rig-eternalai-v0.3.2) - 2025-08-05

### Other

- updated the following local packages: rig-core

## [0.3.1](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.3.0...rig-eternalai-v0.3.1) - 2025-08-05

### Other

- updated the following local packages: rig-core

## [0.3.0](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.10...rig-eternalai-v0.3.0) - 2025-07-30

### Added

- *(rig-812)* yield final response with total usage metrics from streaming completion response in stream impl ([#584](https://github.com/0xPlaygrounds/rig/pull/584))
- *(rig-784)* thinking/reasoning ([#557](https://github.com/0xPlaygrounds/rig/pull/557))

### Other

- Refactor clients with builder pattern ([#615](https://github.com/0xPlaygrounds/rig/pull/615))

### Migration

- If you are using `Client::from_url()` to add in your own base URL, you will now need to use the `Client::builder()` method and add in the base URL.

## [0.2.10](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.9...rig-eternalai-v0.2.10) - 2025-07-16

### Other

- updated the following local packages: rig-core

## [0.2.9](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.8...rig-eternalai-v0.2.9) - 2025-07-14

### Added

- add `.extended_details` to `PromptRequest` ([#555](https://github.com/0xPlaygrounds/rig/pull/555))

## [0.2.8](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.7...rig-eternalai-v0.2.8) - 2025-07-07

### Added

- *(rig-780)* integrate openAI responses API ([#508](https://github.com/0xPlaygrounds/rig/pull/508))

### Other

- Migrate all crates to Rust 2024 ([#539](https://github.com/0xPlaygrounds/rig/pull/539))
- Declare shared dependencies in workspace ([#538](https://github.com/0xPlaygrounds/rig/pull/538))
- Make clippy happy on all targets ([#542](https://github.com/0xPlaygrounds/rig/pull/542))

## [0.2.7](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.6...rig-eternalai-v0.2.7) - 2025-06-09

### Other

- Introduce Client Traits and Testing ([#440](https://github.com/0xPlaygrounds/rig/pull/440))

## [0.2.6](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.5...rig-eternalai-v0.2.6) - 2025-04-29

### Added

- multi-turn / reasoning loops + parallel tool calling ([#370](https://github.com/0xPlaygrounds/rig/pull/370))

### Fixed

- function call conversion typo ([#415](https://github.com/0xPlaygrounds/rig/pull/415))

## [0.2.5](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.4...rig-eternalai-v0.2.5) - 2025-04-12

### Other

- updated the following local packages: rig-core

## [0.2.4](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.3...rig-eternalai-v0.2.4) - 2025-03-31

### Other

- updated the following local packages: rig-core

## [0.2.3](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.2...rig-eternalai-v0.2.3) - 2025-03-17

### Added

- add reqwest/rustls-tls support ([#339](https://github.com/0xPlaygrounds/rig/pull/339))

## [0.2.2](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.1...rig-eternalai-v0.2.2) - 2025-03-03

### Other

- updated the following local packages: rig-core

## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.2.0...rig-eternalai-v0.2.1) - 2025-02-17

### Other

- updated the following local packages: rig-core

## [0.2.0](https://github.com/0xPlaygrounds/rig/compare/rig-eternalai-v0.1.0...rig-eternalai-v0.2.0) - 2025-02-10

### Added

- *(core)* overhaul message API (#199)

### Other

- fix typos (#242)

## [0.1.0](https://github.com/0xPlaygrounds/rig/releases/tag/rig-eternalai-v0.1.0) - 2025-01-27

### Added

- *(rig-eternalai)* add support for EternalAI onchain toolset (#205)

### Fixed

- Use of deprecated `prelude` module (#241)

### Other

- *(rig-eternalai)* Add missing manifest fields + basic README (#240)
