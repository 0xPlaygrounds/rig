# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.42.0...rig-s3vectors-v0.43.0) - 2026-08-22

### Added

- [**breaking**] rig-reqwest — cut the bundled transport into its own crate; rig-core has no default transport and no reqwest/tokio ([#2397](https://github.com/0xPlaygrounds/rig/pull/2397)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2397
- [**breaking**] finish the type-erasure sweep — normalize transcription/image/audio responses, move construction off every model trait, erase the embedding model in vector stores ([#2385](https://github.com/0xPlaygrounds/rig/pull/2385)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2385

### Other

- [**breaking**] ownership audit round 2 — borrow-shaped telemetry getters, slice-shaped embed seams, Copy usage types, dead Default/Debug transport bounds ([#2392](https://github.com/0xPlaygrounds/rig/pull/2392)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2392
- [**breaking**] ownership audit — borrow-shaped signatures, dead clones, clone_from in accumulators, minimal bounds ([#2391](https://github.com/0xPlaygrounds/rig/pull/2391)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2391

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.42.0](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.41.0...rig-s3vectors-v0.42.0) - 2026-08-16

### Other

- reconcile the changelogs and the migration guide with what actually merged ([#2353](https://github.com/0xPlaygrounds/rig/pull/2353)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2353
- workspace-wide LOC consolidation pass 8 (net −1,353 production lines) ([#2320](https://github.com/0xPlaygrounds/rig/pull/2320)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2320
- *(rig-core)* consolidate provider boilerplate ([#2317](https://github.com/0xPlaygrounds/rig/pull/2317)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- workspace-wide LOC consolidation pass 7 (net −366 production lines) ([#2310](https://github.com/0xPlaygrounds/rig/pull/2310)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2310
- workspace-wide LOC consolidation pass 6 (net −3,424 lines) ([#2308](https://github.com/0xPlaygrounds/rig/pull/2308)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2308
- [**breaking**] `OneOrMany<T>` becomes `Vec<T>` — the fake is deleted, the enforcement moves ([#2273](https://github.com/0xPlaygrounds/rig/pull/2273)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2273

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)

### Removed

- *(filter)* [**breaking**] the exported `document!` macro. It was `#[macro_export]`ed from the crate root to spell `aws_smithy_types::Document` filter literals; `S3SearchFilter`'s constructors now build those values through the private `document_object`/`document_comparison` helpers, and there is no public replacement

### Changed

- *(vector-store)* [**breaking**] `InsertDocuments::insert_documents` takes `Vec<(Doc, Vec<Embedding>)>` instead of `Vec<(Doc, OneOrMany<Embedding>)>`, following rig-core's removal of the non-empty container — a source-only signature change; serialized embeddings are unchanged

## [0.41.0](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.40.0...rig-s3vectors-v0.41.0) - 2026-07-28

### Fixed

- *(aws)* remove legacy rustls connector ([#2152](https://github.com/0xPlaygrounds/rig/pull/2152)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.38.1](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.7...rig-s3vectors-v0.38.1) - 2026-06-02

### Other

- unify workspace crate versions ([#1853](https://github.com/0xPlaygrounds/rig/pull/1853)) (by @gold-silver-copper) - #1853

### Contributors

* @gold-silver-copper
## [0.2.7](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.6...rig-s3vectors-v0.2.7) - 2026-06-02

### Other

- update Cargo.toml dependencies
## [0.2.6](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.5...rig-s3vectors-v0.2.6) - 2026-05-13

### Other

- fix "a ancient" grammar in glarb-glarb sample text ([#1755](https://github.com/0xPlaygrounds/rig/pull/1755)) (by @abhicris) - #1755
- AGENTS.MD, CONTRIBUTING.MD, and docs ([#1714](https://github.com/0xPlaygrounds/rig/pull/1714)) (by @gold-silver-copper) - #1714
- improve project organization and create rig crate ([#1699](https://github.com/0xPlaygrounds/rig/pull/1699)) (by @gold-silver-copper) - #1699

### Contributors

* @abhicris
* @gold-silver-copper
## [0.2.5](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.4...rig-s3vectors-v0.2.5) - 2026-04-28

### Other

- Add clippy no panic lints ([#1663](https://github.com/0xPlaygrounds/rig/pull/1663)) (by @gold-silver-copper) - #1663
- standardize required fields handling across builders ([#1611](https://github.com/0xPlaygrounds/rig/pull/1611)) (by @isSerge) - #1611

### Contributors

* @gold-silver-copper
* @isSerge
## [0.2.4](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.3...rig-s3vectors-v0.2.4) - 2026-04-12

### Other

- updated the following local packages: rig-core

## [0.2.3](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.2...rig-s3vectors-v0.2.3) - 2026-03-29

### Other

- updated the following local packages: rig-core

## [0.2.2](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.1...rig-s3vectors-v0.2.2) - 2026-03-17

### Other

- updated the following local packages: rig-core


## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.2.0...rig-s3vectors-v0.2.1) - 2026-03-05

### Other

- updated the following local packages: rig-core

## [0.1.20](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.19...rig-s3vectors-v0.1.20) - 2026-02-17

### Other

- update Cargo.toml dependencies

## [0.1.19](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.18...rig-s3vectors-v0.1.19) - 2026-02-03

### Other

- updated the following local packages: rig-core

## [0.1.18](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.17...rig-s3vectors-v0.1.18) - 2026-01-20

### Added

- improve vector store documentation and filter ergonomics (breaking) ([#1258](https://github.com/0xPlaygrounds/rig/pull/1258))
- make integration filters available to be used as rig agent rag store ([#1249](https://github.com/0xPlaygrounds/rig/pull/1249))

## [0.1.17](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.16...rig-s3vectors-v0.1.17) - 2026-01-06

### Other

- updated the following local packages: rig-core

## [0.1.16](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.15...rig-s3vectors-v0.1.16) - 2025-12-15

### Other

- *(rig-1090)* crate re-org ([#1145](https://github.com/0xPlaygrounds/rig/pull/1145))

## [0.1.15](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.14...rig-s3vectors-v0.1.15) - 2025-12-04

### Other

- updated the following local packages: rig-core

## [0.1.14](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.13...rig-s3vectors-v0.1.14) - 2025-12-01

### Added

- *(rig-985)* Consolidate provider clients ([#1050](https://github.com/0xPlaygrounds/rig/pull/1050))

### Fixed

- *(rig-1050)* Inconsistent model/agent initialisation methods ([#1069](https://github.com/0xPlaygrounds/rig/pull/1069))

## [0.1.13](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.12...rig-s3vectors-v0.1.13) - 2025-11-10

### Added

- *(rig-1014)* add backend specific vector search filters ([#1032](https://github.com/0xPlaygrounds/rig/pull/1032))

## [0.1.12](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.11...rig-s3vectors-v0.1.12) - 2025-10-28

### Other

- updated the following local packages: rig-core

## [0.1.11](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.10...rig-s3vectors-v0.1.11) - 2025-10-27

### Added

- *(rig-976)* support filters for `VectorSearchRequest` ([#952](https://github.com/0xPlaygrounds/rig/pull/952))

### Other

- Dependent packages no longer force unnecessary features on rig-core ([#964](https://github.com/0xPlaygrounds/rig/pull/964))

## [0.1.10](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.9...rig-s3vectors-v0.1.10) - 2025-10-14

### Other

- updated the following local packages: rig-core

## [0.1.9](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.8...rig-s3vectors-v0.1.9) - 2025-09-29

### Other

- updated the following local packages: rig-core

## [0.1.8](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.7...rig-s3vectors-v0.1.8) - 2025-09-15

### Other

- updated the following local packages: rig-core

## [0.1.7](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.6...rig-s3vectors-v0.1.7) - 2025-09-02

### Other

- update Cargo.toml dependencies

## [0.1.6](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.5...rig-s3vectors-v0.1.6) - 2025-08-20

### Other

- updated the following local packages: rig-core

## [0.1.5](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.4...rig-s3vectors-v0.1.5) - 2025-08-19

### Other

- updated the following local packages: rig-core

## [0.1.4](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.3...rig-s3vectors-v0.1.4) - 2025-08-19

### Other

- updated the following local packages: rig-core

## [0.1.3](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.2...rig-s3vectors-v0.1.3) - 2025-08-05

### Other

- updated the following local packages: rig-core

## [0.1.2](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.1...rig-s3vectors-v0.1.2) - 2025-08-05

### Added

- *(rig-845)* cosine similarity for vector search ([#664](https://github.com/0xPlaygrounds/rig/pull/664))

## [0.1.1](https://github.com/0xPlaygrounds/rig/compare/rig-s3vectors-v0.1.0...rig-s3vectors-v0.1.1) - 2025-07-30

### Added

- *(rig-819)* vector store index request struct ([#623](https://github.com/0xPlaygrounds/rig/pull/623))
