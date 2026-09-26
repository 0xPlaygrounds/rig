# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-reqwest-v0.0.0...rig-reqwest-v0.43.0) - 2026-09-26

### Added

- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443
- [**breaking**] BoxedHttpClient — an erased HTTP transport; Client<Ext> defaults to it ([#2401](https://github.com/0xPlaygrounds/rig/pull/2401)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2401
- *(agent)* run_channel/RunEvents, static Send+Sync pins, bevy_tasks example, dependency-graph guard ([#2399](https://github.com/0xPlaygrounds/rig/pull/2399)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] rig-reqwest — cut the bundled transport into its own crate; rig-core has no default transport and no reqwest/tokio ([#2397](https://github.com/0xPlaygrounds/rig/pull/2397)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2397

### Fixed

- publish the facade, not the repository — and stop asking for dependencies nothing uses ([#2563](https://github.com/0xPlaygrounds/rig/pull/2563)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2563
- repair eleven downstream consumer conformance defects ([#2561](https://github.com/0xPlaygrounds/rig/pull/2561)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2561
- *(rig-reqwest)* default-transport constructors return an error instead of panicking when the client cannot be built ([#2471](https://github.com/0xPlaygrounds/rig/pull/2471)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- simplify production comments and preserve contract rationale ([#2568](https://github.com/0xPlaygrounds/rig/pull/2568)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2568
- [**breaking**] establish host-owned assembly and execution lifetimes ([#2567](https://github.com/0xPlaygrounds/rig/pull/2567)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2567
- Unify every provider onto one wire model ([#2538](https://github.com/0xPlaygrounds/rig/pull/2538)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2538
- delete the reflect and replay features and the source-shape guards ([#2523](https://github.com/0xPlaygrounds/rig/pull/2523)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2523
- [**breaking**] collapse the client machinery to Provider + Has* ([#2441](https://github.com/0xPlaygrounds/rig/pull/2441)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2441
- [**breaking**] provider types default to the erased transport; delete the alias tree ([#2440](https://github.com/0xPlaygrounds/rig/pull/2440)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2440
- move every inline test module to a sibling file ([#2433](https://github.com/0xPlaygrounds/rig/pull/2433)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2433
- [**breaking**] rig-reqwest API and hygiene cleanups ([#2428](https://github.com/0xPlaygrounds/rig/pull/2428)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2428
- generate the provider alias tree from rig-core's rustdoc output ([#2427](https://github.com/0xPlaygrounds/rig/pull/2427)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2427
- [**breaking**] make websockets transport-agnostic — protocol to rig-core, socket to rig-tungstenite ([#2426](https://github.com/0xPlaygrounds/rig/pull/2426)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2426
- idiomatic Rust sweep, round 2 ([#2410](https://github.com/0xPlaygrounds/rig/pull/2410)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2410
- idiomatic Rust sweep across the workspace ([#2409](https://github.com/0xPlaygrounds/rig/pull/2409)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2409

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
