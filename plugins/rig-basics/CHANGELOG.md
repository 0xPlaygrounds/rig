# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-basics-v0.44.0...rig-basics-v0.45.0) - 2026-10-19

### Added

- *(rig-basics)* the default plugins.toml no longer lists SessionPickerPlugin ([#2863](https://github.com/0xPlaygrounds/rig/pull/2863)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* a welcome at the top of the transcript: rig's mark, version, model and directory ([#2861](https://github.com/0xPlaygrounds/rig/pull/2861)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tui)* [**breaking**] a flat look: no box drawing, call rows tinted by state, the status line in the input ([#2859](https://github.com/0xPlaygrounds/rig/pull/2859)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* [**breaking**] rig over its pipes: --json, -p as the same front, and the helper tool ([#2854](https://github.com/0xPlaygrounds/rig/pull/2854)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 4: fix what the benchmark found ([#2843](https://github.com/0xPlaygrounds/rig/pull/2843)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 1: fail closed, one fixed system prompt, keys redacted, /plugins, kernel tests ([#2838](https://github.com/0xPlaygrounds/rig/pull/2838)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* smaller defaults, whole color-coded transcript, syntax highlighting ([#2832](https://github.com/0xPlaygrounds/rig/pull/2832)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] a thin coding agent over rig libraries, with rig-ecs and rig-tools ([#2771](https://github.com/0xPlaygrounds/rig/pull/2771)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-tools)* [**breaking**] triobench round 8: a read of part of a small remote file gets it whole so an edit after it is one call, an edit's reply shows the changed lines and a few around them ([#2860](https://github.com/0xPlaygrounds/rig/pull/2860)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* triobench round 6: shell output drops terminal escape codes, a cut keeps the first and last errors, a remote write cut off mid-way says whether it happened, an edit after a write is one call ([#2858](https://github.com/0xPlaygrounds/rig/pull/2858)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* triobench round 4: edits apply in order, an edit of a file seen whole is one transport call, a stopped command says how it was stopped ([#2857](https://github.com/0xPlaygrounds/rig/pull/2857)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* [**breaking**] triobench round 2: one transport call per remote file operation, a per-host cap and read retries, SIGTERM before SIGKILL, quiet calls, readable cuts, edit misses, long-line reads ([#2852](https://github.com/0xPlaygrounds/rig/pull/2852)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig)* [**breaking**] triobench round 1: subagent messages and reports, remote HOST:PATH paths, shell streams and cuts, search paths, OSC 133 turn marks ([#2849](https://github.com/0xPlaygrounds/rig/pull/2849)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] everything the pre-0.45 API review found ([#2848](https://github.com/0xPlaygrounds/rig/pull/2848)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2848
- *(compaction)* a stop during a held summary cancels it ([#2846](https://github.com/0xPlaygrounds/rig/pull/2846)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-basics)* [**breaking**] fold the headless runner into the print front ([#2856](https://github.com/0xPlaygrounds/rig/pull/2856)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- make the test suite deterministic ([#2853](https://github.com/0xPlaygrounds/rig/pull/2853)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2853
- *(rig-tools)* [**breaking**] the koh transport is koh exec HOST --sh, as koh/8 takes argv after -- ([#2850](https://github.com/0xPlaygrounds/rig/pull/2850)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 5: a smaller core, more plugins, fewer crates ([#2844](https://github.com/0xPlaygrounds/rig/pull/2844)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* publish rig-ecs, rig-harness, rig-tools and the plugin crates (onto main) ([#2836](https://github.com/0xPlaygrounds/rig/pull/2836)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
