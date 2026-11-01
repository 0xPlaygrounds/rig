# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-steel-v0.46.0...rig-steel-v0.47.0) - 2026-11-01

### Fixed

- *(ecs)* [**breaking**] a call is known by its reply and place; every stop goes through one owner ([#2985](https://github.com/0xPlaygrounds/rig/pull/2985)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] --tools limits only the run it is given to ([#2984](https://github.com/0xPlaygrounds/rig/pull/2984)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one run contract for every front; unknown input is refused out loud ([#2972](https://github.com/0xPlaygrounds/rig/pull/2972)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-steel-v0.1.0...rig-steel-v0.45.0) - 2026-10-19

### Added

- *(harness)* [**breaking**] beat Pi, part 1: fail closed, one fixed system prompt, keys redacted, /plugins, kernel tests ([#2838](https://github.com/0xPlaygrounds/rig/pull/2838)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] a thin coding agent over rig libraries, with rig-ecs and rig-tools ([#2771](https://github.com/0xPlaygrounds/rig/pull/2771)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- [**breaking**] everything the pre-0.45 API review found ([#2848](https://github.com/0xPlaygrounds/rig/pull/2848)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2848

### Other

- a README that is a front door: the agent first, then the library; the agent's reference in docs/agent.md ([#2862](https://github.com/0xPlaygrounds/rig/pull/2862)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2862
- *(rig-harness)* publish rig-ecs, rig-harness, rig-tools and the plugin crates (onto main) ([#2836](https://github.com/0xPlaygrounds/rig/pull/2836)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
