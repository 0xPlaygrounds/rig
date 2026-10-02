# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.44.0](https://github.com/0xPlaygrounds/rig/compare/rig-cassette-v0.43.0...rig-cassette-v0.44.0) - 2026-10-02

### Added

- *(rig-cassette)* add non-panicking try_start_at and try_finish ([#2701](https://github.com/0xPlaygrounds/rig/pull/2701)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(agent)* [**breaking**] derive PromptResponse output from content, tighten extractor ([#2690](https://github.com/0xPlaygrounds/rig/pull/2690)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(vector_store)* [**breaking**] named VectorStoreIndex search results ([#2688](https://github.com/0xPlaygrounds/rig/pull/2688)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] type tool definition and tool choice names as ToolName ([#2689](https://github.com/0xPlaygrounds/rig/pull/2689)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2689
- *(ecs)* [**breaking**] typed checkpoint save/restore errors ([#2692](https://github.com/0xPlaygrounds/rig/pull/2692)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(vector_store)* [**breaking**] tighten the VectorSearchRequest surface ([#2683](https://github.com/0xPlaygrounds/rig/pull/2683)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] consistent media content constructors ([#2685](https://github.com/0xPlaygrounds/rig/pull/2685)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2685
- *(agent)* [**breaking**] plain run results and streamed tool results ([#2686](https://github.com/0xPlaygrounds/rig/pull/2686)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- leftovers sweep (bedrock text helper, derive trybuild, one-impl traits) ([#2680](https://github.com/0xPlaygrounds/rig/pull/2680)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2680
- *(rig-cassette)* one scripted long-loop suite instead of five ([#2678](https://github.com/0xPlaygrounds/rig/pull/2678)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] one request-assembly pass and no hand-kept mirrors in rig-ecs ([#2672](https://github.com/0xPlaygrounds/rig/pull/2672)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cassette)* one scripted ECS fault suite instead of six ([#2671](https://github.com/0xPlaygrounds/rig/pull/2671)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(openrouter)* [**breaking**] read OpenRouter replies through OpenAI's chat types ([#2664](https://github.com/0xPlaygrounds/rig/pull/2664)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] read DeepSeek and Mistral replies through OpenAI's chat types ([#2661](https://github.com/0xPlaygrounds/rig/pull/2661)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2661
- [**breaking**] share structured-output policy between rig-agent and rig-ecs ([#2660](https://github.com/0xPlaygrounds/rig/pull/2660)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2660
- *(test-utils)* one tracing capture layer for span and event assertions ([#2659](https://github.com/0xPlaygrounds/rig/pull/2659)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] drop mirror types kept in step by hand-written conversions ([#2656](https://github.com/0xPlaygrounds/rig/pull/2656)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2656
- *(rig-agent)* [**breaking**] one agent-run error type that keeps what it knows ([#2644](https://github.com/0xPlaygrounds/rig/pull/2644)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-cassette-v0.0.1...rig-cassette-v0.43.0) - 2026-09-30

### Added

- *(providers)* base support for the new Anthropic and OpenAI models ([#2632](https://github.com/0xPlaygrounds/rig/pull/2632)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(gemini)* automatic explicit caching for long runs ([#2628](https://github.com/0xPlaygrounds/rig/pull/2628)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* [**breaking**] refuse account failures and stored state, keep failed attempts, and move recording tooling into xtask ([#2587](https://github.com/0xPlaygrounds/rig/pull/2587)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* [**breaking**] signature slots, upstream reasoning provenance and the legacy fixture corpus ([#2580](https://github.com/0xPlaygrounds/rig/pull/2580)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* [**breaking**] stateful handle round-trips and reasoning provenance ([#2578](https://github.com/0xPlaygrounds/rig/pull/2578)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* [**breaking**] record provider-opaque fields verbatim ([#2577](https://github.com/0xPlaygrounds/rig/pull/2577)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* skip non-recordable tests during live re-recording ([#2556](https://github.com/0xPlaygrounds/rig/pull/2556)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* add durable tool-turn checkpoint boundaries ([#2514](https://github.com/0xPlaygrounds/rig/pull/2514)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443

### Fixed

- *(openai)* preserve assistant message phase across streamed and multi-message turns ([#2639](https://github.com/0xPlaygrounds/rig/pull/2639)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- pre-0.43 release fixes (docs.rs, opt-in rig::cassette, rust-version, READMEs) ([#2638](https://github.com/0xPlaygrounds/rig/pull/2638)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2638
- *(openai)* preserve reasoning and citations across response decoding ([#2636](https://github.com/0xPlaygrounds/rig/pull/2636)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(usage)* [**breaking**] one meaning for Usage on every provider ([#2631](https://github.com/0xPlaygrounds/rig/pull/2631)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- repair eleven downstream consumer conformance defects ([#2561](https://github.com/0xPlaygrounds/rig/pull/2561)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2561
- *(rig-cassette)* restore start_at and checkpoint_recording for consumers that stage candidates ([#2498](https://github.com/0xPlaygrounds/rig/pull/2498)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- [**breaking**] replace NonEmpty<T> with Vec<T>, check empty turns at the request boundary ([#2640](https://github.com/0xPlaygrounds/rig/pull/2640)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2640
- *(caching)* long-run cache coverage for realistic workloads on GPT-6 and Claude Opus 5 ([#2633](https://github.com/0xPlaygrounds/rig/pull/2633)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* [**breaking**] the driver alone writes provider, request id and raw ([#2626](https://github.com/0xPlaygrounds/rig/pull/2626)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cache)* long-run cache proof and one cache cost report across providers ([#2629](https://github.com/0xPlaygrounds/rig/pull/2629)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] archimpro — one typed decoder and one fold per wire ([#2617](https://github.com/0xPlaygrounds/rig/pull/2617)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2617
- *(rig-core)* [**breaking**] a smaller model layer (Operation, Fold, Wire, Decoder, Transport) ([#2613](https://github.com/0xPlaygrounds/rig/pull/2613)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] provider clients own their transport; conversations use plain values ([#2611](https://github.com/0xPlaygrounds/rig/pull/2611)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2611
- *(rig-core)* [**breaking**] classify every request-building failure as a request error ([#2586](https://github.com/0xPlaygrounds/rig/pull/2586)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* [**breaking**] standardize modality request builders and unify GenAI spans ([#2583](https://github.com/0xPlaygrounds/rig/pull/2583)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* [**breaking**] unify provider operation errors ([#2582](https://github.com/0xPlaygrounds/rig/pull/2582)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* guard history round-trips and fix Groq and Ollama replay ([#2576](https://github.com/0xPlaygrounds/rig/pull/2576)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* cover long native ECS tasks and cache accounting ([#2573](https://github.com/0xPlaygrounds/rig/pull/2573)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* pin every native cell to its own world golden ([#2572](https://github.com/0xPlaygrounds/rig/pull/2572)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tests)* stop comparing effect logs across runtimes in the ECS parity harness ([#2571](https://github.com/0xPlaygrounds/rig/pull/2571)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- simplify production comments and preserve contract rationale ([#2568](https://github.com/0xPlaygrounds/rig/pull/2568)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2568
- [**breaking**] establish host-owned assembly and execution lifetimes ([#2567](https://github.com/0xPlaygrounds/rig/pull/2567)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2567
- [**breaking**] remove backwards-compatibility shims and rename aliases ([#2557](https://github.com/0xPlaygrounds/rig/pull/2557)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2557
- [**breaking**] consolidate record/replay in rig-cassette ([#2552](https://github.com/0xPlaygrounds/rig/pull/2552)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2552
- *(tests)* one cassette response-header reader, one raw-capture harness ([#2549](https://github.com/0xPlaygrounds/rig/pull/2549)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Reduce test-suite compilation and duplicated fixtures ([#2518](https://github.com/0xPlaygrounds/rig/pull/2518)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2518
- *(ecs)* the ECS contract under faults on six wires — setup, retryable statuses, truncated and error-bearing streams, refusals, failing tools, stops, scenes ([#2503](https://github.com/0xPlaygrounds/rig/pull/2503)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* make rig-cassette publishable ([#2497](https://github.com/0xPlaygrounds/rig/pull/2497)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
