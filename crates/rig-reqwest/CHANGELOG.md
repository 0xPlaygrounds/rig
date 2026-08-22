# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-reqwest-v0.0.0...rig-reqwest-v0.43.0) - 2026-08-22

### Added

- [**breaking**] BoxedHttpClient — an erased HTTP transport; Client<Ext> defaults to it ([#2401](https://github.com/0xPlaygrounds/rig/pull/2401)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2401
- *(agent)* run_channel/RunEvents, static Send+Sync pins, bevy_tasks example, dependency-graph guard ([#2399](https://github.com/0xPlaygrounds/rig/pull/2399)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] rig-reqwest — cut the bundled transport into its own crate; rig-core has no default transport and no reqwest/tokio ([#2397](https://github.com/0xPlaygrounds/rig/pull/2397)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2397

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
