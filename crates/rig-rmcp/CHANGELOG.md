# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.44.0](https://github.com/0xPlaygrounds/rig/compare/rig-rmcp-v0.43.0...rig-rmcp-v0.44.0) - 2026-10-02

### Other

- [**breaking**] type tool definition and tool choice names as ToolName ([#2689](https://github.com/0xPlaygrounds/rig/pull/2689)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2689
- [**breaking**] stop flattening tool and MCP failures into strings ([#2687](https://github.com/0xPlaygrounds/rig/pull/2687)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2687

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-rmcp-v0.0.0...rig-rmcp-v0.43.0) - 2026-09-30

### Added

- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443
- [**breaking**] lower ModelHandle and the erased tool set into rig-core; pure rig_run::prepare_request ([#2405](https://github.com/0xPlaygrounds/rig/pull/2405)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2405
- *(agent)* run_channel/RunEvents, static Send+Sync pins, bevy_tasks example, dependency-graph guard ([#2399](https://github.com/0xPlaygrounds/rig/pull/2399)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] rig-rmcp — move MCP tool support out of rig-agent into its own crate ([#2398](https://github.com/0xPlaygrounds/rig/pull/2398)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2398

### Fixed

- pre-0.43 release fixes (docs.rs, opt-in rig::cassette, rust-version, READMEs) ([#2638](https://github.com/0xPlaygrounds/rig/pull/2638)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2638

### Other

- [**breaking**] remove duplicated tool and error surface from rig-core ([#2589](https://github.com/0xPlaygrounds/rig/pull/2589)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2589
- finish the comment cleanup across the workspace ([#2575](https://github.com/0xPlaygrounds/rig/pull/2575)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2575
- [**breaking**] one run type in rig-agent ([#2438](https://github.com/0xPlaygrounds/rig/pull/2438)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2438
- move every inline test module to a sibling file ([#2433](https://github.com/0xPlaygrounds/rig/pull/2433)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2433
- idiomatic Rust sweep, round 2 ([#2410](https://github.com/0xPlaygrounds/rig/pull/2410)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2410
- idiomatic Rust sweep across the workspace ([#2409](https://github.com/0xPlaygrounds/rig/pull/2409)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2409

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
