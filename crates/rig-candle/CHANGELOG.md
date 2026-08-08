# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.42.0](https://github.com/0xPlaygrounds/rig/compare/rig-candle-v0.41.0...rig-candle-v0.42.0) - 2026-08-08

### Other

- Stream parts become entities: lifecycle grammar, opaque keys, and tool names as data (the 84a43e9e C→B→A program) ([#2262](https://github.com/0xPlaygrounds/rig/pull/2262)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2262
- Canonical stream grammar: mandatory identity, one accumulator, decode-then-validate, and a wire-conformance corpus ([#2258](https://github.com/0xPlaygrounds/rig/pull/2258)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2258
- Normalize completion responses at the provider boundary and erase the model type at agent construction ([#2257](https://github.com/0xPlaygrounds/rig/pull/2257)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2257

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)

### Changed

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
