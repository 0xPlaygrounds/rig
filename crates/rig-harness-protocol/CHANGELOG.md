# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-harness-protocol-v0.46.0...rig-harness-protocol-v0.47.0) - 2026-11-01

### Fixed

- *(io)* [**breaking**] input and output are bounded; a stop is never blocked by output ([#2974](https://github.com/0xPlaygrounds/rig/pull/2974)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(session)* [**breaking**] resume draws what live drew; an empty session is not kept ([#2964](https://github.com/0xPlaygrounds/rig/pull/2964)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a stopped turn keeps what streamed, and a stop is never an error ([#2966](https://github.com/0xPlaygrounds/rig/pull/2966)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(acp)* rig acp, an ACP v1 front with a rig process per session, on rig-mcp-client ([#2948](https://github.com/0xPlaygrounds/rig/pull/2948)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(agent)* [**breaking**] headless runs hear SIGTERM in any setup, stop when nobody reads, and leave no tool processes behind ([#2942](https://github.com/0xPlaygrounds/rig/pull/2942)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui, harness)* one Ctrl+D quits; a session with no messages isn't saved ([#2909](https://github.com/0xPlaygrounds/rig/pull/2909)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-harness-protocol-v0.44.0...rig-harness-protocol-v0.45.0) - 2026-10-19

### Added

- *(rig-basics)* [**breaking**] rig over its pipes: --json, -p as the same front, and the helper tool ([#2854](https://github.com/0xPlaygrounds/rig/pull/2854)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 2: deletions, forks are agents, one recovery path, Esc stops subagents ([#2841](https://github.com/0xPlaygrounds/rig/pull/2841)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- [**breaking**] everything the pre-0.45 API review found ([#2848](https://github.com/0xPlaygrounds/rig/pull/2848)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2848

### Other

- *(harness)* [**breaking**] beat Pi, part 5: a smaller core, more plugins, fewer crates ([#2844](https://github.com/0xPlaygrounds/rig/pull/2844)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
