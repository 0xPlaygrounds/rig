# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.44.0](https://github.com/0xPlaygrounds/rig/compare/rig-typesafeai-v0.43.0...rig-typesafeai-v0.44.0) - 2026-10-07

### Added

- typed generation options, model catalog, provider extensions and normalized replies ([#2750](https://github.com/0xPlaygrounds/rig/pull/2750)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2750

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-typesafeai-v0.0.0...rig-typesafeai-v0.43.0) - 2026-09-30

### Added

- initial typesafeai provider (jev) ([#2550](https://github.com/0xPlaygrounds/rig/pull/2550)) (by [0xMochan](https://github.com/0xMochan)) - #2550

### Fixed

- pre-0.43 release fixes (docs.rs, opt-in rig::cassette, rust-version, READMEs) ([#2638](https://github.com/0xPlaygrounds/rig/pull/2638)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2638

### Other

- [**breaking**] archimpro — one typed decoder and one fold per wire ([#2617](https://github.com/0xPlaygrounds/rig/pull/2617)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2617
- *(rig-core)* [**breaking**] a smaller model layer (Operation, Fold, Wire, Decoder, Transport) ([#2613](https://github.com/0xPlaygrounds/rig/pull/2613)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] provider clients own their transport; conversations use plain values ([#2611](https://github.com/0xPlaygrounds/rig/pull/2611)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2611
- *(rig-core)* [**breaking**] classify every request-building failure as a request error ([#2586](https://github.com/0xPlaygrounds/rig/pull/2586)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* [**breaking**] unify provider operation errors ([#2582](https://github.com/0xPlaygrounds/rig/pull/2582)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
* [0xMochan](https://github.com/0xMochan)
