# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.42.0...rig-vectorize-v0.43.0) - 2026-08-22

### Added

- [**breaking**] rig-reqwest — cut the bundled transport into its own crate; rig-core has no default transport and no reqwest/tokio ([#2397](https://github.com/0xPlaygrounds/rig/pull/2397)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2397
- [**breaking**] finish the type-erasure sweep — normalize transcription/image/audio responses, move construction off every model trait, erase the embedding model in vector stores ([#2385](https://github.com/0xPlaygrounds/rig/pull/2385)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2385

### Other

- [**breaking**] ownership audit — borrow-shaped signatures, dead clones, clone_from in accumulators, minimal bounds ([#2391](https://github.com/0xPlaygrounds/rig/pull/2391)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2391

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.42.0](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.41.0...rig-vectorize-v0.42.0) - 2026-08-16

### Other

- workspace-wide LOC consolidation pass 7 (net −366 production lines) ([#2310](https://github.com/0xPlaygrounds/rig/pull/2310)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2310
- workspace-wide LOC consolidation pass 6 (net −3,424 lines) ([#2308](https://github.com/0xPlaygrounds/rig/pull/2308)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2308
- [**breaking**] `OneOrMany<T>` becomes `Vec<T>` — the fake is deleted, the enforcement moves ([#2273](https://github.com/0xPlaygrounds/rig/pull/2273)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2273

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)

### Changed

- *(vector-store)* [**breaking**] `InsertDocuments::insert_documents` takes `Vec<(Doc, Vec<Embedding>)>` instead of `Vec<(Doc, OneOrMany<Embedding>)>`, following rig-core's removal of the non-empty container — a source-only signature change; serialized embeddings are unchanged

## [0.38.1](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.7...rig-vectorize-v0.38.1) - 2026-06-02

### Other

- unify workspace crate versions ([#1853](https://github.com/0xPlaygrounds/rig/pull/1853)) (by @gold-silver-copper) - #1853

### Contributors

* @gold-silver-copper
## [0.2.7](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.6...rig-vectorize-v0.2.7) - 2026-06-02

### Other

- update Cargo.toml dependencies
## [0.2.6](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.5...rig-vectorize-v0.2.6) - 2026-05-13

### Other

- AGENTS.MD, CONTRIBUTING.MD, and docs ([#1714](https://github.com/0xPlaygrounds/rig/pull/1714)) (by @gold-silver-copper) - #1714
- improve project organization and create rig crate ([#1699](https://github.com/0xPlaygrounds/rig/pull/1699)) (by @gold-silver-copper) - #1699

### Contributors

* @gold-silver-copper
## [0.2.5](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.4...rig-vectorize-v0.2.5) - 2026-04-28

### Other

- Add clippy no panic lints ([#1663](https://github.com/0xPlaygrounds/rig/pull/1663)) (by @gold-silver-copper) - #1663
- standardize required fields handling across builders ([#1611](https://github.com/0xPlaygrounds/rig/pull/1611)) (by @isSerge) - #1611

### Contributors

* @gold-silver-copper
* @isSerge
## [0.2.4](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.3...rig-vectorize-v0.2.4) - 2026-04-12

### Other

- updated the following local packages: rig-core

## [0.2.3](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.2...rig-vectorize-v0.2.3) - 2026-03-29

### Other

- updated the following local packages: rig-core

## [0.2.2](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.1...rig-vectorize-v0.2.2) - 2026-03-17

### Other

- updated the following local packages: rig-core


## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-vectorize-v0.2.0...rig-vectorize-v0.2.1) - 2026-03-05

### Other

- updated the following local packages: rig-core
