# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
