# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-cli-v0.46.0...rig-cli-v0.47.0) - 2026-11-07

### Added

- *(harness,cli,mcp-client)* [**breaking**] one settings file, plugins from the home list, full trust in projects, and the launcher in its own crate ([#3005](https://github.com/0xPlaygrounds/rig/pull/3005)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(tui)* the wheel scrolls the transcript and a drag copies the text under it ([#3104](https://github.com/0xPlaygrounds/rig/pull/3104)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* rig -p refuses a plugin that does not compile, and plugin disable takes out an unknown plugin ([#3094](https://github.com/0xPlaygrounds/rig/pull/3094)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* an unknown plugin in settings refuses the start instead of running the previous build ([#3057](https://github.com/0xPlaygrounds/rig/pull/3057)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
