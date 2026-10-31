# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(rig-basics)* [**breaking**] AGENTS.md and skills as plugins, /login with API keys ([#2873](https://github.com/0xPlaygrounds/rig/pull/2873)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(tui)* [**breaking**] Up never loses what you typed, and a recalled command opens no list ([#2947](https://github.com/0xPlaygrounds/rig/pull/2947)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] headless runs hear SIGTERM in any setup, stop when nobody reads, and leave no tool processes behind ([#2942](https://github.com/0xPlaygrounds/rig/pull/2942)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* a diff in tool output ends its hunk at the first line that cannot be in it ([#2941](https://github.com/0xPlaygrounds/rig/pull/2941)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* [**breaking**] fix picker layout, status text, edit diffs and paging ([#2917](https://github.com/0xPlaygrounds/rig/pull/2917)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(corpus, tui)* [**breaking**] editor commands run through the shell, and their failures show ([#2914](https://github.com/0xPlaygrounds/rig/pull/2914)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(chatgpt, tui)* ask ChatGPT for a reasoning summary, draw reasoning as markdown ([#2912](https://github.com/0xPlaygrounds/rig/pull/2912)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] Esc keeps the partial reply, /compact keeps history, accurate summaries, ctx ? after compaction ([#2918](https://github.com/0xPlaygrounds/rig/pull/2918)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* [**breaking**] prompt history per project, the shown conversation's prompts first ([#2916](https://github.com/0xPlaygrounds/rig/pull/2916)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui, harness)* one Ctrl+D quits; a session with no messages isn't saved ([#2909](https://github.com/0xPlaygrounds/rig/pull/2909)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] a failed message goes back into the input ([#2895](https://github.com/0xPlaygrounds/rig/pull/2895)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(tui)* [**breaking**] resume and big streamed tool calls are fast: code is colored only on screen ([#2952](https://github.com/0xPlaygrounds/rig/pull/2952)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] extension points as ECS data and systems ([#2900](https://github.com/0xPlaygrounds/rig/pull/2900)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- provider knowledge out of generic code, typed lookups ([#2897](https://github.com/0xPlaygrounds/rig/pull/2897)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2897

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-tui-v0.44.0...rig-tui-v0.45.0) - 2026-10-19

### Added

- *(rig-basics)* a welcome at the top of the transcript: rig's mark, version, model and directory ([#2861](https://github.com/0xPlaygrounds/rig/pull/2861)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tui)* [**breaking**] a flat look: no box drawing, call rows tinted by state, the status line in the input ([#2859](https://github.com/0xPlaygrounds/rig/pull/2859)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 4: fix what the benchmark found ([#2843](https://github.com/0xPlaygrounds/rig/pull/2843)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 2: deletions, forks are agents, one recovery path, Esc stops subagents ([#2841](https://github.com/0xPlaygrounds/rig/pull/2841)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 1: fail closed, one fixed system prompt, keys redacted, /plugins, kernel tests ([#2838](https://github.com/0xPlaygrounds/rig/pull/2838)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* smaller defaults, whole color-coded transcript, syntax highlighting ([#2832](https://github.com/0xPlaygrounds/rig/pull/2832)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] a thin coding agent over rig libraries, with rig-ecs and rig-tools ([#2771](https://github.com/0xPlaygrounds/rig/pull/2771)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig)* [**breaking**] triobench round 1: subagent messages and reports, remote HOST:PATH paths, shell streams and cuts, search paths, OSC 133 turn marks ([#2849](https://github.com/0xPlaygrounds/rig/pull/2849)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] everything the pre-0.45 API review found ([#2848](https://github.com/0xPlaygrounds/rig/pull/2848)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2848

### Other

- *(harness)* [**breaking**] beat Pi, part 5: a smaller core, more plugins, fewer crates ([#2844](https://github.com/0xPlaygrounds/rig/pull/2844)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* publish rig-ecs, rig-harness, rig-tools and the plugin crates (onto main) ([#2836](https://github.com/0xPlaygrounds/rig/pull/2836)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- faster, simpler, smaller rig (one PR) ([#2835](https://github.com/0xPlaygrounds/rig/pull/2835)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2835
- *(rig)* [**breaking**] measured performance pass on rig's libraries, with benchmarks ([#2796](https://github.com/0xPlaygrounds/rig/pull/2796)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
