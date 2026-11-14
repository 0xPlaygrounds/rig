# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-cli-v0.46.0...rig-cli-v0.47.0) - 2026-11-14

### Added

- *(rig-cli)* shorten the warning for a 0.46 home's plugins.toml and defaults.json ([#3245](https://github.com/0xPlaygrounds/rig/pull/3245)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* warn once per start of a 0.46 home's plugins.toml and defaults.json ([#3239](https://github.com/0xPlaygrounds/rig/pull/3239)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* [**breaking**] rig update reinstalls rig and /update restarts the session on it ([#3224](https://github.com/0xPlaygrounds/rig/pull/3224)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* rig plugin list groups rig's plugins and yours; each build writes a writing-rig-plugins skill ([#3218](https://github.com/0xPlaygrounds/rig/pull/3218)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness,cli,mcp-client)* [**breaking**] one settings file, plugins from the home list, full trust in projects, and the launcher in its own crate ([#3005](https://github.com/0xPlaygrounds/rig/pull/3005)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-cli)* a staged file that is not a program is passed over on macOS too ([#3307](https://github.com/0xPlaygrounds/rig/pull/3307)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* a staged build that cannot be started gives way to the previous build ([#3297](https://github.com/0xPlaygrounds/rig/pull/3297)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* a signal at a secret prompt puts the terminal back before the launcher ends ([#3281](https://github.com/0xPlaygrounds/rig/pull/3281)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* [**breaking**] hold one build from describe to spawn and roll back per session ([#3266](https://github.com/0xPlaygrounds/rig/pull/3266)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* install rig from crates.io with --locked ([#3267](https://github.com/0xPlaygrounds/rig/pull/3267)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* bound the describe wait including its stderr ([#3262](https://github.com/0xPlaygrounds/rig/pull/3262)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* keep exported secrets out of rig update's cargo install and rebuild ([#3265](https://github.com/0xPlaygrounds/rig/pull/3265)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* run rig update's cargo install away from the directory rig started in ([#3260](https://github.com/0xPlaygrounds/rig/pull/3260)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* say the command line that runs a launcher word given after options ([#3259](https://github.com/0xPlaygrounds/rig/pull/3259)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* write the writing-rig-plugins skill once, when rig sets up a new home ([#3247](https://github.com/0xPlaygrounds/rig/pull/3247)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* pass the home and RIG_SOURCE on absolute to processes started elsewhere ([#3254](https://github.com/0xPlaygrounds/rig/pull/3254)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* keep each launcher warning whole when it holds a line break ([#3258](https://github.com/0xPlaygrounds/rig/pull/3258)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* `rig plugin check --build` refuses settings a start would refuse ([#3253](https://github.com/0xPlaygrounds/rig/pull/3253)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* keep the session when the new launcher's agent refuses a /resume ([#3250](https://github.com/0xPlaygrounds/rig/pull/3250)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* the version names its commit, the lock follows the checkout, and a source mismatch is refused ([#3221](https://github.com/0xPlaygrounds/rig/pull/3221)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* plugin check says why a plugin's Cargo.toml can't be read ([#3214](https://github.com/0xPlaygrounds/rig/pull/3214)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* keep prompt history and run logs private to the user ([#3164](https://github.com/0xPlaygrounds/rig/pull/3164)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* never discard a session whose agent logs hold a conversation ([#3143](https://github.com/0xPlaygrounds/rig/pull/3143)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* a Ctrl-C'd trial build is not rejected, and running the previous build instead of a rejected one is said ([#3139](https://github.com/0xPlaygrounds/rig/pull/3139)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* the wheel scrolls the transcript and a drag copies the text under it ([#3104](https://github.com/0xPlaygrounds/rig/pull/3104)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* rig -p refuses a plugin that does not compile, and plugin disable takes out an unknown plugin ([#3094](https://github.com/0xPlaygrounds/rig/pull/3094)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* an unknown plugin in settings refuses the start instead of running the previous build ([#3057](https://github.com/0xPlaygrounds/rig/pull/3057)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-harness-protocol)* [**breaking**] file errors name the path and step; only NotFound is absent ([#3275](https://github.com/0xPlaygrounds/rig/pull/3275)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* launch decides each step from what the agent left; a guard restores the screen ([#3276](https://github.com/0xPlaygrounds/rig/pull/3276)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] one AgentEnd record says how the agent ended ([#3274](https://github.com/0xPlaygrounds/rig/pull/3274)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] one Launch type builds every child the launcher starts ([#3271](https://github.com/0xPlaygrounds/rig/pull/3271)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig)* an install line that works, a quick start, how to update, and what is per home ([#3217](https://github.com/0xPlaygrounds/rig/pull/3217)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] remove duplicated tests, dead code and doc bloat across the workspace ([#3202](https://github.com/0xPlaygrounds/rig/pull/3202)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3202
- *(rig-cli)* help-agents says a settings.toml the launcher can't read exits 1 ([#3134](https://github.com/0xPlaygrounds/rig/pull/3134)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
