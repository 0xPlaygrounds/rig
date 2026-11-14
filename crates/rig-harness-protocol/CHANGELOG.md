# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-harness-protocol-v0.46.0...rig-harness-protocol-v0.47.0) - 2026-11-14

### Added

- *(rig-basics, rig-agent)* [login] app_name names the app on the ChatGPT sign-in pages; document live tool items after a sibling fails ([#3317](https://github.com/0xPlaygrounds/rig/pull/3317)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* warn once per start of a 0.46 home's plugins.toml and defaults.json ([#3239](https://github.com/0xPlaygrounds/rig/pull/3239)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* [**breaking**] rig update reinstalls rig and /update restarts the session on it ([#3224](https://github.com/0xPlaygrounds/rig/pull/3224)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(fold)* [**breaking**] rig-fold, an optional plugin that folds finished tool calls in the terminal view ([#3223](https://github.com/0xPlaygrounds/rig/pull/3223)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools,basics)* [**breaking**] remove the unused move tool and the checkpoint plugin ([#3048](https://github.com/0xPlaygrounds/rig/pull/3048)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness,cli,mcp-client)* [**breaking**] one settings file, plugins from the home list, full trust in projects, and the launcher in its own crate ([#3005](https://github.com/0xPlaygrounds/rig/pull/3005)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(usage,helpers,budget)* [**breaking**] one agent tree's spend is counted, and an opt-in budget holds ([#2994](https://github.com/0xPlaygrounds/rig/pull/2994)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-harness-protocol)* a settings edit is synced before it replaces the file ([#3278](https://github.com/0xPlaygrounds/rig/pull/3278)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* move a process-group list a reload cannot read aside instead of writing over it ([#3263](https://github.com/0xPlaygrounds/rig/pull/3263)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* [**breaking**] hold one build from describe to spawn and roll back per session ([#3266](https://github.com/0xPlaygrounds/rig/pull/3266)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* say the command line that runs a launcher word given after options ([#3259](https://github.com/0xPlaygrounds/rig/pull/3259)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* keep each launcher warning whole when it holds a line break ([#3258](https://github.com/0xPlaygrounds/rig/pull/3258)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* `rig plugin check --build` refuses settings a start would refuse ([#3253](https://github.com/0xPlaygrounds/rig/pull/3253)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* the version names its commit, the lock follows the checkout, and a source mismatch is refused ([#3221](https://github.com/0xPlaygrounds/rig/pull/3221)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* settings edits write through a dangling settings.toml symlink instead of replacing it ([#3225](https://github.com/0xPlaygrounds/rig/pull/3225)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* an existing private file's staged copy is never readable by others ([#3166](https://github.com/0xPlaygrounds/rig/pull/3166)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* keep prompt history and run logs private to the user ([#3164](https://github.com/0xPlaygrounds/rig/pull/3164)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* never discard a session whose agent logs hold a conversation ([#3143](https://github.com/0xPlaygrounds/rig/pull/3143)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* create session directories private to the user ([#3137](https://github.com/0xPlaygrounds/rig/pull/3137)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* rig -p refuses a plugin that does not compile, and plugin disable takes out an unknown plugin ([#3094](https://github.com/0xPlaygrounds/rig/pull/3094)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* an unknown plugin in settings refuses the start instead of running the previous build ([#3057](https://github.com/0xPlaygrounds/rig/pull/3057)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] every limit on asked work is a settings.toml key with a sensible default, and rig says what it changes ([#2995](https://github.com/0xPlaygrounds/rig/pull/2995)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2995
- *(io)* [**breaking**] input and output are bounded; a stop is never blocked by output ([#2974](https://github.com/0xPlaygrounds/rig/pull/2974)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(session)* [**breaking**] resume draws what live drew; an empty session is not kept ([#2964](https://github.com/0xPlaygrounds/rig/pull/2964)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a stopped turn keeps what streamed, and a stop is never an error ([#2966](https://github.com/0xPlaygrounds/rig/pull/2966)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-core, rig-basics)* the caller supplies the ChatGPT sign-in's browser opener ([#3294](https://github.com/0xPlaygrounds/rig/pull/3294)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] file errors name the path and step; only NotFound is absent ([#3275](https://github.com/0xPlaygrounds/rig/pull/3275)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] one AgentEnd record says how the agent ended ([#3274](https://github.com/0xPlaygrounds/rig/pull/3274)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] one Launch type builds every child the launcher starts ([#3271](https://github.com/0xPlaygrounds/rig/pull/3271)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- fix stale claims and commands in the library READMEs ([#3219](https://github.com/0xPlaygrounds/rig/pull/3219)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3219
- [**breaking**] remove duplicated tests, dead code and doc bloat across the workspace ([#3202](https://github.com/0xPlaygrounds/rig/pull/3202)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3202
- *(deps)* update rmcp, bevy, base64, signal-hook, testcontainers, surrealdb, similar, sha2 and syn ([#3089](https://github.com/0xPlaygrounds/rig/pull/3089)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

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
