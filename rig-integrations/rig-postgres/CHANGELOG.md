# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.29](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.28...rig-postgres-v0.1.29) - 2026-01-06

### Other

- updated the following local packages: rig-core

## [0.1.28](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.27...rig-postgres-v0.1.28) - 2025-12-15

### Other

- *(rig-1090)* crate re-org ([#1145](https://github.com/0xPlaygrounds/rig/pull/1145))

## [0.1.27](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.26...rig-postgres-v0.1.27) - 2025-12-04

### Other

- updated the following local packages: rig-core

## [0.1.26](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.25...rig-postgres-v0.1.26) - 2025-12-01

### Added

- *(rig-985)* Consolidate provider clients ([#1050](https://github.com/0xPlaygrounds/rig/pull/1050))

### Fixed

- *(rig-1050)* Inconsistent model/agent initialisation methods ([#1069](https://github.com/0xPlaygrounds/rig/pull/1069))

## [0.1.25](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.24...rig-postgres-v0.1.25) - 2025-11-10

### Added

- *(rig-1014)* add backend specific vector search filters ([#1032](https://github.com/0xPlaygrounds/rig/pull/1032))

## [0.1.24](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.23...rig-postgres-v0.1.24) - 2025-10-28

### Other

- updated the following local packages: rig-core

## [0.1.23](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.22...rig-postgres-v0.1.23) - 2025-10-27

### Added

- *(rig-976)* support filters for `VectorSearchRequest` ([#952](https://github.com/0xPlaygrounds/rig/pull/952))
- *(rig-996)* generic streaming ([#955](https://github.com/0xPlaygrounds/rig/pull/955))

### Fixed

- *(rig-1006)* text-embedding-ada-002 doesn't support custom dimensions ([#967](https://github.com/0xPlaygrounds/rig/pull/967))

### Other

- Dependent packages no longer force unnecessary features on rig-core ([#964](https://github.com/0xPlaygrounds/rig/pull/964))

## [0.1.22](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.21...rig-postgres-v0.1.22) - 2025-10-14

### Added

- *(rig-951)* generic HTTP client ([#875](https://github.com/0xPlaygrounds/rig/pull/875))

### Fixed

- trying to fix test regressions part 2 ([#913](https://github.com/0xPlaygrounds/rig/pull/913))

## [0.1.21](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.20...rig-postgres-v0.1.21) - 2025-09-29

### Other

- updated the following local packages: rig-core

## [0.1.20](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.19...rig-postgres-v0.1.20) - 2025-09-15

### Other

- updated the following local packages: rig-core

## [0.1.19](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.18...rig-postgres-v0.1.19) - 2025-09-02

### Other

- *(rig-907)* use where clause for trait bounds ([#749](https://github.com/0xPlaygrounds/rig/pull/749))

## [0.1.18](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.17...rig-postgres-v0.1.18) - 2025-08-20

### Other

- updated the following local packages: rig-core

## [0.1.17](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.16...rig-postgres-v0.1.17) - 2025-08-19

### Other

- updated the following local packages: rig-core

## [0.1.16](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.15...rig-postgres-v0.1.16) - 2025-08-19

### Other

- updated the following local packages: rig-core

## [0.1.15](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.14...rig-postgres-v0.1.15) - 2025-08-05

### Other

- updated the following local packages: rig-core

## [0.1.14](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.13...rig-postgres-v0.1.14) - 2025-08-05

### Added

- *(rig-845)* cosine similarity for vector search ([#664](https://github.com/0xPlaygrounds/rig/pull/664))

## [0.1.13](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.12...rig-postgres-v0.1.13) - 2025-07-30

### Added

- *(rig-819)* vector store index request struct ([#623](https://github.com/0xPlaygrounds/rig/pull/623))

### Other

- Refactor clients with builder pattern ([#615](https://github.com/0xPlaygrounds/rig/pull/615))

## [0.1.12](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.11...rig-postgres-v0.1.12) - 2025-07-16

### Other

- updated the following local packages: rig-core

## [0.1.11](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.10...rig-postgres-v0.1.11) - 2025-07-14

### Other

- updated the following local packages: rig-core

## [0.1.10](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.9...rig-postgres-v0.1.10) - 2025-07-07

### Added

- support inserting documents as a trait ([#563](https://github.com/0xPlaygrounds/rig/pull/563))

### Other

- Migrate all crates to Rust 2024 ([#539](https://github.com/0xPlaygrounds/rig/pull/539))
- Declare shared dependencies in workspace ([#538](https://github.com/0xPlaygrounds/rig/pull/538))
- Make clippy happy on all targets ([#542](https://github.com/0xPlaygrounds/rig/pull/542))

## [0.1.9](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.8...rig-postgres-v0.1.9) - 2025-06-09

### Other

- Introduce Client Traits and Testing ([#440](https://github.com/0xPlaygrounds/rig/pull/440))

## [0.1.8](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.7...rig-postgres-v0.1.8) - 2025-04-29

### Other

- updated the following local packages: rig-core

## [0.1.7](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.6...rig-postgres-v0.1.7) - 2025-04-12

### Other

- updated the following local packages: rig-core

## [0.1.6](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.5...rig-postgres-v0.1.6) - 2025-03-31

### Other

- updated the following local packages: rig-core

## [0.1.5](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.4...rig-postgres-v0.1.5) - 2025-03-17

### Other

- updated the following local packages: rig-core

## [0.1.4](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.3...rig-postgres-v0.1.4) - 2025-03-03

### Other

- updated the following local packages: rig-core

## [0.1.3](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.2...rig-postgres-v0.1.3) - 2025-02-17

### Other

- updated the following local packages: rig-core

## [0.1.2](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.1...rig-postgres-v0.1.2) - 2025-02-10

### Other

- fix spelling errors in `Makefile` and `message.rs` (#284)

## [0.1.1](https://github.com/0xPlaygrounds/rig/compare/rig-postgres-v0.1.0...rig-postgres-v0.1.1) - 2025-01-27

### Other

- release (#203)

## [0.1.0](https://github.com/0xPlaygrounds/rig/releases/tag/rig-postgres-v0.1.0) - 2025-01-27

### Added

- *(rig-postgres)* postgres vector store integration (#231)
