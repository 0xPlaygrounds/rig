# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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
