# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-basics-v0.46.0...rig-basics-v0.47.0) - 2026-11-01

### Fixed

- *(print,acp)* [**breaking**] --json names a call by its place (protocol v 2); ACP shows a result the same live and loaded ([#2988](https://github.com/0xPlaygrounds/rig/pull/2988)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a call is known by its reply and place; every stop goes through one owner ([#2985](https://github.com/0xPlaygrounds/rig/pull/2985)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] /model lists only models you can use now; a model that doesn't connect is refused and the current one stays ([#2983](https://github.com/0xPlaygrounds/rig/pull/2983)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] --tools limits only the run it is given to ([#2984](https://github.com/0xPlaygrounds/rig/pull/2984)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(driver)* [**breaking**] a streamed reply that sends only keep-alive bytes stalls ([#2982](https://github.com/0xPlaygrounds/rig/pull/2982)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(io)* [**breaking**] input and output are bounded; a stop is never blocked by output ([#2974](https://github.com/0xPlaygrounds/rig/pull/2974)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] only an explicit choice becomes the default; /model offers only models that answer ([#2973](https://github.com/0xPlaygrounds/rig/pull/2973)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(compaction)* [**breaking**] rewrite the summary, and never clear an output without notes or a same-bytes re-read ([#2975](https://github.com/0xPlaygrounds/rig/pull/2975)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one run contract for every front; unknown input is refused out loud ([#2972](https://github.com/0xPlaygrounds/rig/pull/2972)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] an @ reference attaches exactly the file that was picked ([#2962](https://github.com/0xPlaygrounds/rig/pull/2962)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(session)* [**breaking**] resume draws what live drew; an empty session is not kept ([#2964](https://github.com/0xPlaygrounds/rig/pull/2964)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one current agent per session; a fork says what it copies ([#2967](https://github.com/0xPlaygrounds/rig/pull/2967)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(http)* [**breaking**] an error keeps its retry class from where it was made ([#2969](https://github.com/0xPlaygrounds/rig/pull/2969)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] tool results show what they found ([#2965](https://github.com/0xPlaygrounds/rig/pull/2965)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a stopped turn keeps what streamed, and a stop is never an error ([#2966](https://github.com/0xPlaygrounds/rig/pull/2966)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tests)* test apps reach only scripted models; verify --affected builds only what it tests ([#2963](https://github.com/0xPlaygrounds/rig/pull/2963)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(acp)* rig acp, an ACP v1 front with a rig process per session, on rig-mcp-client ([#2948](https://github.com/0xPlaygrounds/rig/pull/2948)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mcp-client)* ship the MCP client as a rig plugin, off by default ([#2938](https://github.com/0xPlaygrounds/rig/pull/2938)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* add claude-cloak and harden Anthropic handling ([#2875](https://github.com/0xPlaygrounds/rig/pull/2875)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* [**breaking**] shell commands run to their end, and a remote command stops with its call ([#2874](https://github.com/0xPlaygrounds/rig/pull/2874)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* [**breaking**] AGENTS.md and skills as plugins, /login with API keys ([#2873](https://github.com/0xPlaygrounds/rig/pull/2873)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(agent)* [**breaking**] quitting mid-turn is a stop, and a restart carries nothing on by itself ([#2943](https://github.com/0xPlaygrounds/rig/pull/2943)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] headless runs hear SIGTERM in any setup, stop when nobody reads, and leave no tool processes behind ([#2942](https://github.com/0xPlaygrounds/rig/pull/2942)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* [**breaking**] a tool path starting with ~ is in the home directory ([#2940](https://github.com/0xPlaygrounds/rig/pull/2940)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* the system prompt lets the user's request set the answer ([#2939](https://github.com/0xPlaygrounds/rig/pull/2939)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* [**breaking**] stop forcing serde_json features ([#2934](https://github.com/0xPlaygrounds/rig/pull/2934)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* [**breaking**] fix picker layout, status text, edit diffs and paging ([#2917](https://github.com/0xPlaygrounds/rig/pull/2917)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] headless runs close what an exit cuts off, refuse unknown --tools, fail empty answers ([#2915](https://github.com/0xPlaygrounds/rig/pull/2915)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] Esc keeps the partial reply, /compact keeps history, accurate summaries, ctx ? after compaction ([#2918](https://github.com/0xPlaygrounds/rig/pull/2918)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] no self-extension prompt, a resume hint on exit, a session migration chain ([#2899](https://github.com/0xPlaygrounds/rig/pull/2899)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] a failed message goes back into the input ([#2895](https://github.com/0xPlaygrounds/rig/pull/2895)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* a 64k output budget, unknown-tool feedback, honest failed summaries, ordered headless JSON ([#2894](https://github.com/0xPlaygrounds/rig/pull/2894)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(catalog, compaction)* fit models' input limits with simpler overflow recovery ([#2880](https://github.com/0xPlaygrounds/rig/pull/2880)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(tui)* [**breaking**] resume and big streamed tool calls are fast: code is colored only on screen ([#2952](https://github.com/0xPlaygrounds/rig/pull/2952)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] extension points as ECS data and systems ([#2900](https://github.com/0xPlaygrounds/rig/pull/2900)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- provider knowledge out of generic code, typed lookups ([#2897](https://github.com/0xPlaygrounds/rig/pull/2897)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2897

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
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
