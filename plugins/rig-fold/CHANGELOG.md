# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-fold-v0.46.0...rig-fold-v0.47.0) - 2026-11-20

### Added

- *(rig-basics, rig-tui, rig-harness, rig-harness-protocol, rig-cli, rig-steel, rig-ecs)* [**breaking**] plugin tree, forks off by default, one first-model rule, and release fixes ([#3425](https://github.com/0xPlaygrounds/rig/pull/3425)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* draw tool output whole by default ([#3395](https://github.com/0xPlaygrounds/rig/pull/3395)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(fold)* [**breaking**] rig-fold, an optional plugin that folds finished tool calls in the terminal view ([#3223](https://github.com/0xPlaygrounds/rig/pull/3223)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-tui, rig-fold)* a refused key binding is shown once in the terminal view ([#3397](https://github.com/0xPlaygrounds/rig/pull/3397)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-fold)* draw a result whole unless folding it saves two rows ([#3251](https://github.com/0xPlaygrounds/rig/pull/3251)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-tui)* [**breaking**] one type per render input; TranscriptSlot loses its unused Ord ([#3345](https://github.com/0xPlaygrounds/rig/pull/3345)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
