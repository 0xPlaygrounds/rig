# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-claude-cloak-v0.46.0...rig-claude-cloak-v0.47.0) - 2026-11-01

### Fixed

- *(models)* [**breaking**] /model lists only models you can use now; a model that doesn't connect is refused and the current one stays ([#2983](https://github.com/0xPlaygrounds/rig/pull/2983)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] only an explicit choice becomes the default; /model offers only models that answer ([#2973](https://github.com/0xPlaygrounds/rig/pull/2973)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(http)* [**breaking**] an error keeps its retry class from where it was made ([#2969](https://github.com/0xPlaygrounds/rig/pull/2969)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(rig-claude-cloak)* publish claude-cloak with every release ([#2891](https://github.com/0xPlaygrounds/rig/pull/2891)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* add claude-cloak and harden Anthropic handling ([#2875](https://github.com/0xPlaygrounds/rig/pull/2875)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(core)* [**breaking**] stop forcing serde_json features ([#2934](https://github.com/0xPlaygrounds/rig/pull/2934)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(agent)* [**breaking**] extension points as ECS data and systems ([#2900](https://github.com/0xPlaygrounds/rig/pull/2900)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- provider knowledge out of generic code, typed lookups ([#2897](https://github.com/0xPlaygrounds/rig/pull/2897)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2897
- *(core)* each wire owns where its reasoning blob lives ([#2896](https://github.com/0xPlaygrounds/rig/pull/2896)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
