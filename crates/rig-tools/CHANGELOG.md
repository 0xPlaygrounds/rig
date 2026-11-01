# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-tools-v0.46.0...rig-tools-v0.47.0) - 2026-11-01

### Fixed

- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one run contract for every front; unknown input is refused out loud ([#2972](https://github.com/0xPlaygrounds/rig/pull/2972)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] an @ reference attaches exactly the file that was picked ([#2962](https://github.com/0xPlaygrounds/rig/pull/2962)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] tool results show what they found ([#2965](https://github.com/0xPlaygrounds/rig/pull/2965)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(acp)* rig acp, an ACP v1 front with a rig process per session, on rig-mcp-client ([#2948](https://github.com/0xPlaygrounds/rig/pull/2948)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* [**breaking**] shell commands run to their end, and a remote command stops with its call ([#2874](https://github.com/0xPlaygrounds/rig/pull/2874)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(agent)* [**breaking**] headless runs hear SIGTERM in any setup, stop when nobody reads, and leave no tool processes behind ([#2942](https://github.com/0xPlaygrounds/rig/pull/2942)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* [**breaking**] a tool path starting with ~ is in the home directory ([#2940](https://github.com/0xPlaygrounds/rig/pull/2940)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] read returns images, long lines can be read on, a clearer timeout hint ([#2913](https://github.com/0xPlaygrounds/rig/pull/2913)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* [**breaking**] prompt history per project, the shown conversation's prompts first ([#2916](https://github.com/0xPlaygrounds/rig/pull/2916)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] a failed message goes back into the input ([#2895](https://github.com/0xPlaygrounds/rig/pull/2895)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(corpus, tools, xtask)* shared policies, typed lookups ([#2898](https://github.com/0xPlaygrounds/rig/pull/2898)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-tools-v0.44.0...rig-tools-v0.45.0) - 2026-10-19

### Added

- *(harness)* [**breaking**] beat Pi, part 4: fix what the benchmark found ([#2843](https://github.com/0xPlaygrounds/rig/pull/2843)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 1: fail closed, one fixed system prompt, keys redacted, /plugins, kernel tests ([#2838](https://github.com/0xPlaygrounds/rig/pull/2838)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* smaller defaults, whole color-coded transcript, syntax highlighting ([#2832](https://github.com/0xPlaygrounds/rig/pull/2832)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] a thin coding agent over rig libraries, with rig-ecs and rig-tools ([#2771](https://github.com/0xPlaygrounds/rig/pull/2771)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-tools)* [**breaking**] triobench round 8: a read of part of a small remote file gets it whole so an edit after it is one call, an edit's reply shows the changed lines and a few around them ([#2860](https://github.com/0xPlaygrounds/rig/pull/2860)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* triobench round 6: shell output drops terminal escape codes, a cut keeps the first and last errors, a remote write cut off mid-way says whether it happened, an edit after a write is one call ([#2858](https://github.com/0xPlaygrounds/rig/pull/2858)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* triobench round 4: edits apply in order, an edit of a file seen whole is one transport call, a stopped command says how it was stopped ([#2857](https://github.com/0xPlaygrounds/rig/pull/2857)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* triobench round 3: one transport call per remote write, remote reads sent only their lines, killed commands name the signal ([#2855](https://github.com/0xPlaygrounds/rig/pull/2855)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* [**breaking**] triobench round 2: one transport call per remote file operation, a per-host cap and read retries, SIGTERM before SIGKILL, quiet calls, readable cuts, edit misses, long-line reads ([#2852](https://github.com/0xPlaygrounds/rig/pull/2852)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig)* [**breaking**] triobench round 1: subagent messages and reports, remote HOST:PATH paths, shell streams and cuts, search paths, OSC 133 turn marks ([#2849](https://github.com/0xPlaygrounds/rig/pull/2849)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] everything the pre-0.45 API review found ([#2848](https://github.com/0xPlaygrounds/rig/pull/2848)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2848

### Other

- *(rig-tools)* [**breaking**] the koh transport is koh exec HOST --sh, as koh/8 takes argv after -- ([#2850](https://github.com/0xPlaygrounds/rig/pull/2850)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* publish rig-ecs, rig-harness, rig-tools and the plugin crates (onto main) ([#2836](https://github.com/0xPlaygrounds/rig/pull/2836)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- faster, simpler, smaller rig (one PR) ([#2835](https://github.com/0xPlaygrounds/rig/pull/2835)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2835
- *(rig)* [**breaking**] measured performance pass on rig's libraries, with benchmarks ([#2796](https://github.com/0xPlaygrounds/rig/pull/2796)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
