# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-tungstenite-v0.0.0...rig-tungstenite-v0.43.0) - 2026-09-26

### Added

- feat!(core): Usage counters are Option<u64> — an absent counter is representable ([#2535](https://github.com/0xPlaygrounds/rig/pull/2535)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2535
- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443

### Other

- *(rig-core)* [**breaking**] unify provider operation errors ([#2582](https://github.com/0xPlaygrounds/rig/pull/2582)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- simplify production comments and preserve contract rationale ([#2568](https://github.com/0xPlaygrounds/rig/pull/2568)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2568
- [**breaking**] establish host-owned assembly and execution lifetimes ([#2567](https://github.com/0xPlaygrounds/rig/pull/2567)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2567
- Unify every provider onto one wire model ([#2538](https://github.com/0xPlaygrounds/rig/pull/2538)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2538
- delete the reflect and replay features and the source-shape guards ([#2523](https://github.com/0xPlaygrounds/rig/pull/2523)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2523
- forbid inline test modules via cargo xtask check-test-layout ([#2434](https://github.com/0xPlaygrounds/rig/pull/2434)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2434
- move every inline test module to a sibling file ([#2433](https://github.com/0xPlaygrounds/rig/pull/2433)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2433
- [**breaking**] make websockets transport-agnostic — protocol to rig-core, socket to rig-tungstenite ([#2426](https://github.com/0xPlaygrounds/rig/pull/2426)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2426

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
