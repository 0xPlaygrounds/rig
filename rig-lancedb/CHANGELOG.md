# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.9](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.8...rig-lancedb-v0.2.9) - 2025-04-12

### Other

- *(lancedb)* docs, make examples idempotent ([#387](https://github.com/0xPlaygrounds/rig/pull/387))

## [0.2.8](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.7...rig-lancedb-v0.2.8) - 2025-03-31

### Other

- added mcp_tool + Example ([#335](https://github.com/0xPlaygrounds/rig/pull/335))

## [0.2.7](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.6...rig-lancedb-v0.2.7) - 2025-03-17

### Other

- updated the following local packages: rig-core

## [0.2.6](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.5...rig-lancedb-v0.2.6) - 2025-03-03

### Other

- updated the following local packages: rig-core

## [0.2.5](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.4...rig-lancedb-v0.2.5) - 2025-02-17

### Other

- updated the following local packages: rig-core

## [0.2.4](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.3...rig-lancedb-v0.2.4) - 2025-02-10

### Other

- updated the following local packages: rig-core

## [0.2.3](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.2...rig-lancedb-v0.2.3) - 2025-01-27

### Other

- update Cargo.lock dependencies

## [0.2.2](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.1...rig-lancedb-v0.2.2) - 2025-01-13

### Other

- Mock provider API in vector store integration tests (#186)

## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.2.0...rig-lancedb-v0.2.1) - 2024-12-19

### Other

- *(integration-tests)* LanceDB (#136)

## [0.2.0](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.1.2...rig-lancedb-v0.2.0) - 2024-12-03

### Added

- embeddings API overhaul ([#120](https://github.com/0xPlaygrounds/rig/pull/120))

### Fixed

- *(rig-lancedb)* rag embedding filtering ([#104](https://github.com/0xPlaygrounds/rig/pull/104))

## [0.1.2](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.1.1...rig-lancedb-v0.1.2) - 2024-11-13

### Other

- update Cargo.lock dependencies

## [0.1.1](https://github.com/0xPlaygrounds/rig/compare/rig-lancedb-v0.1.0...rig-lancedb-v0.1.1) - 2024-11-07

### Fixed

- wrong reference to companion crate
- missing qdrant readme reference

### Other

- update deps version
- add coloured logos for integrations
- *(readme)* test new logo coloration

## [0.1.0](https://github.com/0xPlaygrounds/rig/releases/tag/rig-lancedb-v0.1.0) - 2024-10-24

### Added

- update examples to use new version of VectorStoreIndex trait
- replace document embeddings with serde json value
- merge all arrow columns into JSON document in deserializer
- finish implementing deserialiser for record batch
- implement deserialization for any recordbatch returned from lanceDB
- add indexes and tables for simple search
- create enum for embedding models
- add vector_search_s3_ann example
- implement ANN search example
- start implementing top_n_from_query for trait VectorStoreIndex
- implement get_document method of VectorStore trait
- implement search by id for VectorStore trait
- implement add_documents on VectorStore trait
- start implementing VectorStore trait for lancedb

### Fixed

- update lancedb examples test data
- make PR changes Pt II
- make PR changes pt I
- *(lancedb)* replace VectorStoreIndexDyn with VectorStoreIndex in examples
- mongodb vector search - use num_candidates from search params
- fix bug in deserializing type run end
- make PR requested changes
- reduce opanai generated content in ANN examples

### Other

- cargo fmt
- lance db examples
- add example docstring
- add doc strings
- update rig core version on lancedb crate, remove implementation of VectorStore trait
- remove print statement
- use constants instead of enum for model names
- remove associated type on VectorStoreIndex trait
- cargo fmt
- conversions from arrow types to primitive types
- Add doc strings to utility methods
- add doc string to mongodb search params struct
- Merge branch 'main' into feat(vector-store)/lancedb
- create wrapper for vec<DocumentEmbeddings> for from/tryfrom traits
