# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.17.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.17.0...rig-core-v0.17.1) - 2025-08-05

### Other

- remove unnecessary warning traces ([#672](https://github.com/0xPlaygrounds/rig/pull/672))
- *(rig-851)* update provider integrations list ([#651](https://github.com/0xPlaygrounds/rig/pull/651))

## [0.17.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.16.0...rig-core-v0.17.0) - 2025-08-05

### Added

- *(rig-845)* cosine similarity for vector search ([#664](https://github.com/0xPlaygrounds/rig/pull/664))
- add `delete_tool` method to `Toolset` ([#663](https://github.com/0xPlaygrounds/rig/pull/663))
- Read the OPENAI_BASE_URL env variable when constructing an OpenAI client from_env ([#659](https://github.com/0xPlaygrounds/rig/pull/659))
- add agent name ([#633](https://github.com/0xPlaygrounds/rig/pull/633))

### Fixed

- *(rig-853)* gemini streaming impl ignores reasoning chunks ([#654](https://github.com/0xPlaygrounds/rig/pull/654))
- Ollama provider handling of canonical URLs ([#656](https://github.com/0xPlaygrounds/rig/pull/656))
- *(rig-852)* dynamic context does not work correctly with ollama ([#660](https://github.com/0xPlaygrounds/rig/pull/660))

### Other

- *(rig-861)* make Agent<M> non-exhaustive ([#670](https://github.com/0xPlaygrounds/rig/pull/670))

## [0.16.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.15.1...rig-core-v0.16.0) - 2025-07-30

### Added

- *(rig-798)* `rig-wasm` ([#611](https://github.com/0xPlaygrounds/rig/pull/611))
- *(rig-819)* vector store index request struct ([#623](https://github.com/0xPlaygrounds/rig/pull/623))
- *(rig-830)* map documents to text for OpenAI Response API ([#622](https://github.com/0xPlaygrounds/rig/pull/622))
- Add GROK_4 model constant to xAI provider ([#614](https://github.com/0xPlaygrounds/rig/pull/614))
- *(rig-812)* yield final response with total usage metrics from streaming completion response in stream impl ([#584](https://github.com/0xPlaygrounds/rig/pull/584))
- *(rig-799)* add support for official rust sdk for mcp ([#553](https://github.com/0xPlaygrounds/rig/pull/553))
- *(rig-823)* impl size hint for OneOrMany types ([#606](https://github.com/0xPlaygrounds/rig/pull/606))
- *(rig-784)* thinking/reasoning ([#557](https://github.com/0xPlaygrounds/rig/pull/557))
- *(rig-821)* add tracing when submit tool is never called in extractor ([#603](https://github.com/0xPlaygrounds/rig/pull/603))
- make PromptResponse public ([#593](https://github.com/0xPlaygrounds/rig/pull/593))

### Fixed

- *(rig-824)* ToolResultContent should be serde-tagged ([#621](https://github.com/0xPlaygrounds/rig/pull/621))
- *(rig-828)* support done message on openai streaming completions api ([#619](https://github.com/0xPlaygrounds/rig/pull/619))
- *(rig-827)* openai responses streaming api placeholder panic ([#620](https://github.com/0xPlaygrounds/rig/pull/620))
- *(rig-834)* erroeneous tracing log level ([#626](https://github.com/0xPlaygrounds/rig/pull/626))
- *(rig-820)* ensure call ID is properly propagated ([#601](https://github.com/0xPlaygrounds/rig/pull/601))

### Other

- Add new claude models and default max tokens ([#634](https://github.com/0xPlaygrounds/rig/pull/634))
- *(rig-836)* deprecate mcp-core integration ([#631](https://github.com/0xPlaygrounds/rig/pull/631))
- Refactor clients with builder pattern ([#615](https://github.com/0xPlaygrounds/rig/pull/615))
- change log level to debug for input/output ([#627](https://github.com/0xPlaygrounds/rig/pull/627))
- fix spelling issue  ([#607](https://github.com/0xPlaygrounds/rig/pull/607))

### Migration
- If you are using `Client::from_url()`, you will now need to use `Client::builder()` and add it in from there. Otherwise if you don't care about changing your inner HTTP client or changing the base URL, you can still use `Client::new(<api_key_here>)` or `Client::from_env()` to achieve the same result as you normally would.
- `VectorStoreIndex` and `VectorStoreIndexDyn` now take a `rig::vector_search::VectorSearchRequest`, instead of a query and max result size. This has been done to enable much more ergonomic requesting in the future. Please see any of the `vector_search` examples for practical usage.
- The final response of a completion stream now yields the completion usage from the stream itself. You may wish to adjust your code to account for this.
- The `mcp-core` integration is now officially deprecated because the official Rust MCP SDK is now supported as it has feature parity. You will need to ensure you have moved to the `rmcp` integration (`rmcp` feature flag) by Rig 0.18.0 at the earliest.
- ToolResultContent is now `#[serde(tag = "type")]`. If you're storing the serialized Rig structs anywhere as JSON, you may need to account for this and write a script to backfill your stored JSON.

## [0.15.1](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.15.0...rig-core-v0.15.1) - 2025-07-16

### Fixed

- *(rig-815)* gemini completion fails when used with no tools ([#589](https://github.com/0xPlaygrounds/rig/pull/589))

## [0.15.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.14.0...rig-core-v0.15.0) - 2025-07-14

### Added

- *(rig-801)* DynClientBuilder::from_values ([#556](https://github.com/0xPlaygrounds/rig/pull/556))
- add `.extended_details` to `PromptRequest` ([#555](https://github.com/0xPlaygrounds/rig/pull/555))

### Fixed

- *(rig-811)* ollama fails to return results from multiple tools ([#581](https://github.com/0xPlaygrounds/rig/pull/581))
- *(rig-810)* prompting OpenAI reponses with message history fails ([#578](https://github.com/0xPlaygrounds/rig/pull/578))
- *(rig-809)* gemini function declarations should not be OneOrMany ([#576](https://github.com/0xPlaygrounds/rig/pull/576))

## [0.14.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.13.0...rig-core-v0.14.0) - 2025-07-07

### Added

- support inserting documents as a trait ([#563](https://github.com/0xPlaygrounds/rig/pull/563))
- Add max_tokens method to ExtractorBuilder ([#560](https://github.com/0xPlaygrounds/rig/pull/560))
- *(rig-780)* integrate openAI responses API ([#508](https://github.com/0xPlaygrounds/rig/pull/508))
- Stream cancellation using AbortHandle ([#525](https://github.com/0xPlaygrounds/rig/pull/525))
- *(rig-779)* allow extractor to be turned into inner agent ([#502](https://github.com/0xPlaygrounds/rig/pull/502))
- *(ollama)* add support for OLLAMA_API_BASE_URL environment var ([#541](https://github.com/0xPlaygrounds/rig/pull/541))
- *(rig-766)* add support for Voyage AI ([#493](https://github.com/0xPlaygrounds/rig/pull/493))
- *(rig-789)* add support for loading in pdfs/files as Vec<u8> ([#523](https://github.com/0xPlaygrounds/rig/pull/523))
- multi turn streaming example ([#413](https://github.com/0xPlaygrounds/rig/pull/413))
- *(rig-754)* support custom client configurations ([#511](https://github.com/0xPlaygrounds/rig/pull/511))

### Fixed

- Retain multi-turn tool call results in case of response error ([#526](https://github.com/0xPlaygrounds/rig/pull/526))
- *(rig-794)* parse openAI SSE response error ([#545](https://github.com/0xPlaygrounds/rig/pull/545))
- *(rig-796)* OpenRouter extractor fails ([#544](https://github.com/0xPlaygrounds/rig/pull/544))
- *(rig-792)* inconsistent implementations of with_custom_client ([#530](https://github.com/0xPlaygrounds/rig/pull/530))
- *(rig-783)* tool call example doesn't work with Gemini and OpenRouter ([#515](https://github.com/0xPlaygrounds/rig/pull/515))
- *(rig-773)* xAI embeddings endpoint is wrong ([#492](https://github.com/0xPlaygrounds/rig/pull/492))

### Other

- *(rig-803)* improve documentation for multi-turn ([#562](https://github.com/0xPlaygrounds/rig/pull/562))
- Migrate all crates to Rust 2024 ([#539](https://github.com/0xPlaygrounds/rig/pull/539))
- update deps ([#543](https://github.com/0xPlaygrounds/rig/pull/543))
- Declare shared dependencies in workspace ([#538](https://github.com/0xPlaygrounds/rig/pull/538))
- error fixes for clarity
- Make clippy happy on all targets ([#542](https://github.com/0xPlaygrounds/rig/pull/542))
- *(rig-791)* documents not consistently added to DeepSeek prompts ([#528](https://github.com/0xPlaygrounds/rig/pull/528))
- Fix `ToolResult` serialization in ollama provider ([#504](https://github.com/0xPlaygrounds/rig/pull/504))

## [0.13.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.12.0...rig-core-v0.13.0) - 2025-06-09

### Added

- add additional Gemini completion models ([#498](https://github.com/0xPlaygrounds/rig/pull/498))
- *(rig-758)* the extractor can pass additional params to be passed to the model ([#473](https://github.com/0xPlaygrounds/rig/pull/473))
- *(rig-744)* Add support for Milvus vector store ([#463](https://github.com/0xPlaygrounds/rig/pull/463))
- Improve Streaming API ([#388](https://github.com/0xPlaygrounds/rig/pull/388))

### Fixed

- OpenAI provider streaming tool call response for local LLM ([#442](https://github.com/0xPlaygrounds/rig/pull/442))
- *(rig-761)* ollama drops tool call results ([#478](https://github.com/0xPlaygrounds/rig/pull/478))
- Update of xAI model list ([#486](https://github.com/0xPlaygrounds/rig/pull/486))
- *(rig-757)* CI fails because of new clippy lints ([#470](https://github.com/0xPlaygrounds/rig/pull/470))
- *(extractor)* correct typo in extractor prompt ([#460](https://github.com/0xPlaygrounds/rig/pull/460))
- *(message)* correct ToolCall to Message conversion ([#461](https://github.com/0xPlaygrounds/rig/pull/461))
- Fix `dims` value for gemini's `EMBEDDING_004` ([#452](https://github.com/0xPlaygrounds/rig/pull/452)) ([#453](https://github.com/0xPlaygrounds/rig/pull/453))
- bump mcp-core to latest version and fixed breaking changes ([#443](https://github.com/0xPlaygrounds/rig/pull/443))

### Other

- Fix typo in AudioGenerationModel field name ([#487](https://github.com/0xPlaygrounds/rig/pull/487))
- Introduce Client Traits and Testing ([#440](https://github.com/0xPlaygrounds/rig/pull/440))
- Only PDF docs are supported by their API ([#465](https://github.com/0xPlaygrounds/rig/pull/465))
- Add mistral provider ([#437](https://github.com/0xPlaygrounds/rig/pull/437))
- `impl {Debug,Clone} for CompletionRequest` ([#457](https://github.com/0xPlaygrounds/rig/pull/457))
- fix some typos in comment ([#445](https://github.com/0xPlaygrounds/rig/pull/445))

## [0.12.0](https://github.com/0xPlaygrounds/rig/compare/rig-core-v0.11.1...rig-core-v0.12.0) - 2025-04-29

### Added

- add gpt-image-1 ([#418](https://github.com/0xPlaygrounds/rig/pull/418))
- multi-turn / reasoning loops + parallel tool calling ([#370](https://github.com/0xPlaygrounds/rig/pull/370))

### Fixed

- system and developer messages for openai ([#430](https://github.com/0xPlaygrounds/rig/pull/430))
- o-series models + constants ([#426](https://github.com/0xPlaygrounds/rig/pull/426))
- dynamically pull rag text from chat history ([#425](https://github.com/0xPlaygrounds/rig/pull/425))
- rig tool macro struct not public ([#409](https://github.com/0xPlaygrounds/rig/pull/409))
- function call conversion typo ([#415](https://github.com/0xPlaygrounds/rig/pull/415))
- deepseek function call conversion typo ([#414](https://github.com/0xPlaygrounds/rig/pull/414))

### Other

- Donot use async closure + Bump mcp-core ([#428](https://github.com/0xPlaygrounds/rig/pull/428))
- Remove broken xAI reference link in embedding.rs ([#427](https://github.com/0xPlaygrounds/rig/pull/427))
- Style/trace gemini embedding ([#411](https://github.com/0xPlaygrounds/rig/pull/411))
- Update agent_with_huggingface.rs ([#401](https://github.com/0xPlaygrounds/rig/pull/401))

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
