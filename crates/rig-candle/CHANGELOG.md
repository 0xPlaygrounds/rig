# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-candle-v0.44.0...rig-candle-v0.45.0) - 2026-10-08

### Fixed

- *(catalog)* [**breaking**] overrides reach every wire, one lookup and reference rule, connect from the environment ([#2769](https://github.com/0xPlaygrounds/rig/pull/2769)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* [**breaking**] one meaning for a document's text on every wire (DocumentSourceKind::String becomes DocumentData::Text) ([#2754](https://github.com/0xPlaygrounds/rig/pull/2754)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.44.0](https://github.com/0xPlaygrounds/rig/compare/rig-candle-v0.43.0...rig-candle-v0.44.0) - 2026-10-07

### Added

- typed generation options, model catalog, provider extensions and normalized replies ([#2750](https://github.com/0xPlaygrounds/rig/pull/2750)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2750

### Fixed

- [**breaking**] unknown finish reasons are one outcome, a caller's choice, and never silent ([#2726](https://github.com/0xPlaygrounds/rig/pull/2726)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2726

### Other

- shape-matched cassettes, encoder snapshots, and a coverage gate ([#2720](https://github.com/0xPlaygrounds/rig/pull/2720)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2720
- *(message)* [**breaking**] item-shaped assistant history with provenance ([#2713](https://github.com/0xPlaygrounds/rig/pull/2713)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] derive PromptResponse output from content, tighten extractor ([#2690](https://github.com/0xPlaygrounds/rig/pull/2690)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] type tool definition and tool choice names as ToolName ([#2689](https://github.com/0xPlaygrounds/rig/pull/2689)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2689
- [**breaking**] consistent media content constructors ([#2685](https://github.com/0xPlaygrounds/rig/pull/2685)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2685
- *(streaming)* stop hand-rolling the open text part in every stream decoder ([#2653](https://github.com/0xPlaygrounds/rig/pull/2653)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-candle-v0.42.0...rig-candle-v0.43.0) - 2026-09-30

### Added

- *(providers)* base support for the new Anthropic and OpenAI models ([#2632](https://github.com/0xPlaygrounds/rig/pull/2632)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-candle)* local YOLOv8 pose estimation ([#2615](https://github.com/0xPlaygrounds/rig/pull/2615)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- feat!(core): Usage counters are Option<u64> — an absent counter is representable ([#2535](https://github.com/0xPlaygrounds/rig/pull/2535)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2535
- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443

### Fixed

- pre-0.43 release fixes (docs.rs, opt-in rig::cassette, rust-version, READMEs) ([#2638](https://github.com/0xPlaygrounds/rig/pull/2638)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2638
- [**breaking**] the merge review's defects on main, each pinned by a matrix ([#2499](https://github.com/0xPlaygrounds/rig/pull/2499)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2499

### Other

- [**breaking**] replace NonEmpty<T> with Vec<T>, check empty turns at the request boundary ([#2640](https://github.com/0xPlaygrounds/rig/pull/2640)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2640
- *(rig-core)* [**breaking**] the driver alone writes provider, request id and raw ([#2626](https://github.com/0xPlaygrounds/rig/pull/2626)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] archimpro — one typed decoder and one fold per wire ([#2617](https://github.com/0xPlaygrounds/rig/pull/2617)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2617
- *(rig-core)* [**breaking**] a smaller model layer (Operation, Fold, Wire, Decoder, Transport) ([#2613](https://github.com/0xPlaygrounds/rig/pull/2613)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] provider clients own their transport; conversations use plain values ([#2611](https://github.com/0xPlaygrounds/rig/pull/2611)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2611
- *(rig-core)* [**breaking**] unify provider operation errors ([#2582](https://github.com/0xPlaygrounds/rig/pull/2582)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- finish the comment cleanup across the workspace ([#2575](https://github.com/0xPlaygrounds/rig/pull/2575)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2575
- [**breaking**] remove backwards-compatibility shims and rename aliases ([#2557](https://github.com/0xPlaygrounds/rig/pull/2557)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2557
- Unify every provider onto one wire model ([#2538](https://github.com/0xPlaygrounds/rig/pull/2538)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2538
- [**breaking**] one run type in rig-agent ([#2438](https://github.com/0xPlaygrounds/rig/pull/2438)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2438
- move every inline test module to a sibling file ([#2433](https://github.com/0xPlaygrounds/rig/pull/2433)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2433
- [**breaking**] remove every backwards-compatibility shim ([#2429](https://github.com/0xPlaygrounds/rig/pull/2429)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2429
- ownership sweep round 4 — avoidable clones, dead public items, is_false dedup ([#2416](https://github.com/0xPlaygrounds/rig/pull/2416)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2416
- idiomatic Rust sweep, round 2 ([#2410](https://github.com/0xPlaygrounds/rig/pull/2410)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2410
- idiomatic Rust sweep across the workspace ([#2409](https://github.com/0xPlaygrounds/rig/pull/2409)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2409
- [**breaking**] ownership audit — borrow-shaped signatures, dead clones, clone_from in accumulators, minimal bounds ([#2391](https://github.com/0xPlaygrounds/rig/pull/2391)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2391

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.42.0](https://github.com/0xPlaygrounds/rig/compare/rig-candle-v0.41.0...rig-candle-v0.42.0) - 2026-08-16

### Other

- reconcile the changelogs and the migration guide with what actually merged ([#2353](https://github.com/0xPlaygrounds/rig/pull/2353)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2353
- remove #[non_exhaustive] from the workspace ([#2335](https://github.com/0xPlaygrounds/rig/pull/2335)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2335
- workspace-wide LOC consolidation pass 7 (net −366 production lines) ([#2310](https://github.com/0xPlaygrounds/rig/pull/2310)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2310
- workspace-wide LOC consolidation pass 6 (net −3,424 lines) ([#2308](https://github.com/0xPlaygrounds/rig/pull/2308)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2308
- [**breaking**] `OneOrMany<T>` becomes `Vec<T>` — the fake is deleted, the enforcement moves ([#2273](https://github.com/0xPlaygrounds/rig/pull/2273)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2273
- Stream parts become entities: lifecycle grammar, opaque keys, and tool names as data (the 84a43e9e C→B→A program) ([#2262](https://github.com/0xPlaygrounds/rig/pull/2262)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2262
- Canonical stream grammar: mandatory identity, one accumulator, decode-then-validate, and a wire-conformance corpus ([#2258](https://github.com/0xPlaygrounds/rig/pull/2258)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2258
- Normalize completion responses at the provider boundary and erase the model type at agent construction ([#2257](https://github.com/0xPlaygrounds/rig/pull/2257)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2257

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)

### Changed

- *(model)* [**breaking**] seven dead public items are removed: the `LlamaModelBuilder<'a>` alias and its crate-root re-export, `CandleModel::from_artifacts`, `CandleModel::from_artifacts_async`, `CandleModel::from_gguf_async`, `CandleModel::from_gguf_bytes_async`, `CandleModel::model_family()` and `CandleModelBuilder::model_family()`. Each was an alias over an API that stays — `CandleModelBuilder`, `builder_from_artifacts(..).build()`/`.build_async()`, `builder_from_gguf_bytes(..).build_async()`, and `conversation_protocol` on both types — and the `ModelFamily` alias for `ConversationProtocol` is untouched, so every call site has a one-line replacement

- *(protocol)* [**behavior**] a generation that produces no assistant content stays empty instead of being padded with a fabricated empty-text part, and the emptiness check that padding made unreachable is removed rather than made live — a model that emits EOS immediately, or only whitespace the parser trims, keeps succeeding with genuinely empty content instead of failing with `CandleError::Inference`

- *(streaming)* generation events route through the shared `WireAdapter` driver (this family never produces `Unknown`); `stream_from_events` is the events-first conformance seam driving typed events through the full pipeline with no model load

## [0.41.0](https://github.com/0xPlaygrounds/rig/compare/rig-candle-v0.1.0...rig-candle-v0.41.0) - 2026-07-28

### Added

- [**breaking**] split rig-core and rig-agent behind the rig facade ([#2197](https://github.com/0xPlaygrounds/rig/pull/2197)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2197

### Other

- *(candle)* harden local model runtime ([#2214](https://github.com/0xPlaygrounds/rig/pull/2214)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core,agent)* [**breaking**] make the WASM support matrix explicit and true ([#2213](https://github.com/0xPlaygrounds/rig/pull/2213)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Add rig-candle local inference and WASM chat ([#2155](https://github.com/0xPlaygrounds/rig/pull/2155)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2155

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
