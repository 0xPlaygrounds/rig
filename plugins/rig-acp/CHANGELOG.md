# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-acp-v0.46.0...rig-acp-v0.47.0) - 2026-11-01

### Fixed

- *(print,acp)* [**breaking**] --json names a call by its place (protocol v 2); ACP shows a result the same live and loaded ([#2988](https://github.com/0xPlaygrounds/rig/pull/2988)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a call is known by its reply and place; every stop goes through one owner ([#2985](https://github.com/0xPlaygrounds/rig/pull/2985)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] /model lists only models you can use now; a model that doesn't connect is refused and the current one stays ([#2983](https://github.com/0xPlaygrounds/rig/pull/2983)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(io)* [**breaking**] input and output are bounded; a stop is never blocked by output ([#2974](https://github.com/0xPlaygrounds/rig/pull/2974)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] only an explicit choice becomes the default; /model offers only models that answer ([#2973](https://github.com/0xPlaygrounds/rig/pull/2973)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one run contract for every front; unknown input is refused out loud ([#2972](https://github.com/0xPlaygrounds/rig/pull/2972)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one current agent per session; a fork says what it copies ([#2967](https://github.com/0xPlaygrounds/rig/pull/2967)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a stopped turn keeps what streamed, and a stop is never an error ([#2966](https://github.com/0xPlaygrounds/rig/pull/2966)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(acp)* rig acp, an ACP v1 front with a rig process per session, on rig-mcp-client ([#2948](https://github.com/0xPlaygrounds/rig/pull/2948)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
