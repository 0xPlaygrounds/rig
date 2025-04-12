# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.11.0...rig-core-v0.11.1) - 2025-04-12

### Added

- trait for embedding images ([#396](https://github.com/0xPlaygrounds/rig/pull/396))
- Add `rig_tool` macro ([#353](https://github.com/0xPlaygrounds/rig/pull/353))
- impl From<mcp_core::types::Tool> for ToolDefinition ([#385](https://github.com/0xPlaygrounds/rig/pull/385))
- AWS Bedrock provider ([#318](https://github.com/0xPlaygrounds/rig/pull/318))

### Fixed

- gemini embeddings does not work for multiple documents ([#386](https://github.com/0xPlaygrounds/rig/pull/386))
- deserialization error due to serde rename of tool result ([#374](https://github.com/0xPlaygrounds/rig/pull/374))

### Other

- Updated broken link xaiAPI in `completion.rs` ([#384](https://github.com/0xPlaygrounds/rig/pull/384))
- Fix Clippy warnings for doc indentation and Error::other usage ([#364](https://github.com/0xPlaygrounds/rig/pull/364))

## [0.11.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.10.0...rig-core-v0.11.0) - 2025-03-31

### Added

- Add audio generation to all providers ([#359](https://github.com/0xPlaygrounds/rig/pull/359))
- Add image generation to all providers that support it ([#357](https://github.com/0xPlaygrounds/rig/pull/357))
- *(provider)* cohere-v2 ([#350](https://github.com/0xPlaygrounds/rig/pull/350))

### Fixed

- no params tools definition for Gemini ([#363](https://github.com/0xPlaygrounds/rig/pull/363))
- *(openai)* serde rename for image_url UserContent ([#355](https://github.com/0xPlaygrounds/rig/pull/355))

### Other

- New model provider: Anthropic Claude 3.7 Addition ([#341](https://github.com/0xPlaygrounds/rig/pull/341))
- added mcp_tool + Example ([#335](https://github.com/0xPlaygrounds/rig/pull/335))

## [0.10.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.9.1...rig-core-v0.10.0) - 2025-03-17

### Added

- Add streaming to all model providers ([#347](https://github.com/0xPlaygrounds/rig/pull/347))
- OpenRouter support ([#344](https://github.com/0xPlaygrounds/rig/pull/344))
- add reqwest/rustls-tls support ([#339](https://github.com/0xPlaygrounds/rig/pull/339))
- add transcription to all providers that support it ([#336](https://github.com/0xPlaygrounds/rig/pull/336))
- Azure OpenAI Token Authentication ([#329](https://github.com/0xPlaygrounds/rig/pull/329))
- SSE/JSONL decoders ported from Anthropic TS SDK ([#332](https://github.com/0xPlaygrounds/rig/pull/332))
- mira integration ([#282](https://github.com/0xPlaygrounds/rig/pull/282))
- Huggingface provider integration ([#321](https://github.com/0xPlaygrounds/rig/pull/321))

### Fixed

- unnecessary `unwrap`, skip serializing empty vec ([#343](https://github.com/0xPlaygrounds/rig/pull/343))
- fix error handling for Qwen's responses when using tools ([#351](https://github.com/0xPlaygrounds/rig/pull/351))
- reqwest can not use SOCKS proxy ([#311](https://github.com/0xPlaygrounds/rig/pull/311))
- fix wrong debug message ([#342](https://github.com/0xPlaygrounds/rig/pull/342))

### Other

- Update openai.rs ([#340](https://github.com/0xPlaygrounds/rig/pull/340))
- support svg ([#333](https://github.com/0xPlaygrounds/rig/pull/333))

## [0.9.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.9.0...rig-core-v0.9.1) - 2025-03-03

### Added

- Transcription Model support ([#322](https://github.com/0xPlaygrounds/rig/pull/322))
- Add EpubFileLoader for EPUB file processing ([#192](https://github.com/0xPlaygrounds/rig/pull/192))
- add ollama client ([#285](https://github.com/0xPlaygrounds/rig/pull/285))
- *(openai)* add updated OpenAI model constants ([#314](https://github.com/0xPlaygrounds/rig/pull/314))
- support together AI ([#230](https://github.com/0xPlaygrounds/rig/pull/230))

### Fixed

- *(openai)* skip serializing empty tool_calls vector ([#327](https://github.com/0xPlaygrounds/rig/pull/327))
- *(openai)* correct some fields for tools ([#286](https://github.com/0xPlaygrounds/rig/pull/286))
- *(loaders)* bump lodpf to allow more PDFs to parse correctly ([#307](https://github.com/0xPlaygrounds/rig/pull/307))

### Other

- rename DeepSeek_R1.pdf to deepseek_r1.pdf ([#316](https://github.com/0xPlaygrounds/rig/pull/316))

## [0.9.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.8.0...rig-core-v0.9.0) - 2025-02-17

### Added

- *(streaming)* add `Send` to `StreamingResult` inner Stream (#302)
- groq integration (#263)

### Fixed

- xai agent prompt provider error (#305) (#306)
- enhance tracing messages (#287)
- *(gemini)* fixed tool calling + tool extractor demo (#297)
- o3-mini doesn't support temperature (#266)

### Other

- EchoChambers Example Integration ([#244](https://github.com/0xPlaygrounds/rig/pull/244))
- deepseek message to remove dependencies with openai (#283)

## [0.8.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.7.0...rig-core-v0.8.0) - 2025-02-10

### Added

- fastembed integration (#268)
- *(core)* overhaul message API (#199)
- Add support for Azure OpenAI (#234)
- support moonshot language model (#223)
- galadriel api integration (redux) (#265)
- add Galadriel API integration (#188)
- support extractor for deepseek (#255)
- support tools for DeepSeek provider (#251)
- streaming API implementation for Anthropic provider (#232)

### Fixed

- deepseek client auth (#279)
- *(galadriel)* missed fixes from messages pr (#270)

### Other

- fix spelling errors in `Makefile` and `message.rs` (#284)
- Correct `tracing::debug` message. ([#275](https://github.com/0xPlaygrounds/rig/pull/275))
- agent recipes (#215)
- Revert "feat: add Galadriel API integration ([#188](https://github.com/0xPlaygrounds/rig/pull/188))" ([#264](https://github.com/0xPlaygrounds/rig/pull/264))
- *(example)* fix grammar mistake (#260)
- Fix typos  "substract" → "subtract" ([#256](https://github.com/0xPlaygrounds/rig/pull/256))
- fix typos (#242)
- add more provider notes (#237)

## [0.7.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.6.1...rig-core-v0.7.0) - 2025-01-27

### Added

- Add hyperbolic inference API integration (#238)
- *(rig-eternalai)* add support for EternalAI onchain toolset (#205)
- *(pipeline)* Add conditional op (#200)
- Add support for DeepSeek (#220)

### Fixed

- *(providers)* provider wasm support (#245)
- Use of deprecated `prelude` module (#241)
- anthropic tool use (#168)

### Other

- Fix typos (#233)
- *(README)* add SQLite as a supported vector store (#201)

## [0.6.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.6.0...rig-core-v0.6.1) - 2025-01-13

### Added

- Add `from_url` method to Gemini client (#194)
- Feature flag for CF worker compatibility (#176) (#175)
- *(eternal-ai)* Eternal-AI provider for rig (#171)
- Add gpt-4o-mini to openai model list (#187)

### Fixed

- *(example)* ollama example uses wrong url

### Other

- Add additional check for empty tool_calls ([#166](https://github.com/0xPlaygrounds/rig/pull/166))
- Mock provider API in vector store integration tests (#186)
- fix comment (#182)
- fix various typos

## [0.6.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.5.0...rig-core-v0.6.0) - 2024-12-19

### Added

- agent pipelines (#131)
- *(rig-anthropic)* Add default `max_tokens` for standard models (#151)

### Fixed

- *(openai)* Make integration more general (#156)

### Other

- *(ollama-example)* implement example showcasing ollama (#148)
- *(embeddings)* add embedding distance calculator module (#142)

## [0.5.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.4.1...rig-core-v0.5.0) - 2024-12-03

### Added

- Improve `InMemoryVectorStore` API ([#130](https://github.com/0xPlaygrounds/rig/pull/130))
- embeddings API overhaul ([#120](https://github.com/0xPlaygrounds/rig/pull/120))
- *(provider)* xAI (grok) integration ([#106](https://github.com/0xPlaygrounds/rig/pull/106))

### Fixed

- *(rig-lancedb)* rag embedding filtering ([#104](https://github.com/0xPlaygrounds/rig/pull/104))

## [0.4.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.4.0...rig-core-v0.4.1) - 2024-11-13

### Other

- Inefficient context documents serialization ([#100](https://github.com/0xPlaygrounds/rig/pull/100))

## [0.4.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.3.0...rig-core-v0.4.0) - 2024-11-07

### Added

- *(gemini)* move system prompt to correct request field
- *(provider-gemini)* add support for gemini specific completion parameters
- *(provider-gemini)* add agent support in client
- *(provider-gemini)* add gemini embedding support
- *(provider-gemini)* add gemini support for basic completion
- *(provider-gemini)* add gemini API client

### Fixed

- *(gemini)* issue when additionnal param is empty
- docs imports and refs
- *(gemini)* missing param to be marked as optional in completion res

### Other

- Cargo fmt
- Add module level docs for the `tool` module
- Fix loaders module docs references
- Add docstrings to loaders module
- Improve main lib docs
- Add `all` feature flag to rig-core
- *(gemini)* add utility config docstring
- *(gemini)* remove try_from and use serde deserialization
- Merge branch 'main' into feat/model-provider/16-add-gemini-completion-embedding-models
- *(gemini)* separate gemini api types module, fix pr comments
- add debug trait to embedding struct
- *(gemini)* add addtionnal types from the official documentation, add embeddings example
- *(provider-gemini)* test pre-commits
- *(provider-gemini)* Update readme entries, add gemini agent example

## [0.3.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.2.1...rig-core-v0.3.0) - 2024-10-24

### Added

- Generalize `EmbeddingModel::embed_documents` with `IntoIterator`
- Add `from_env` constructor to Cohere and Anthropic clients
- Small optimization to serde_json object merging
- Add better error handling for provider clients

### Fixed

- Bad Anthropic request/response handling
- *(vector-index)* In memory vector store index incorrect search

### Other

- Made internal `json_utils` module private
- Update lib docs
- Made CompletionRequest helper method private to crate
- lint + fmt
- Simplify `agent_with_tools` example
- Fix docstring links
- Add nextest test runner to CI
- Merge pull request [#42](https://github.com/0xPlaygrounds/rig/pull/42) from 0xPlaygrounds/refactor(vector-store)/update-vector-store-index-trait

## [0.2.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.2.0...rig-core-v0.2.1) - 2024-10-01

### Fixed

- *(docs)* Docs still referring to old types

### Other

- Merge pull request [#45](https://github.com/0xPlaygrounds/rig/pull/45) from 0xPlaygrounds/fix/docs

## [0.2.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.1.0...rig-core-v0.2.0) - 2024-10-01

### Added

- anthropic models

### Fixed

- *(context)* displaying documents should be deterministic (sorted by alpha)
- *(context)* spin out helper method + add tests
- move context documents to user prompt message
- adjust version const naming
- implement review suggestions + renaming
- add `completion_request.documents` to `chat_history`
- adjust API to be cleaner + add docstrings

### Other

- Merge pull request [#43](https://github.com/0xPlaygrounds/rig/pull/43) from 0xPlaygrounds/fix/context-documents
- Merge pull request [#27](https://github.com/0xPlaygrounds/rig/pull/27) from 0xPlaygrounds/feat/anthropic
- Fix docstrings
- Deprecate RagAgent and Model in favor of versatile Agent
- Make RagAgent VectorStoreIndex dynamic trait objects

## [0.1.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.0.7...rig-core-v0.1.0) - 2024-09-16

### Added

- add o1-preview and o1-mini

### Fixed

- *(perplexity)* fix preamble and context in completion request
- clippy warnings

### Other

- Merge pull request [#18](https://github.com/0xPlaygrounds/rig/pull/18) from 0xPlaygrounds/feat/perplexity-support
- Add logging of http errors
- fmt code
