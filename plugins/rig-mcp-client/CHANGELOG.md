# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.1](https://github.com/0xPlaygrounds/rig/compare/rig-mcp-client-v0.47.0...rig-mcp-client-v0.47.1) - 2026-11-25

### Other

- release v0.47.1 ([#3442](https://github.com/0xPlaygrounds/rig/pull/3442)) (by [github-actions[bot]](https://github.com/github-actions[bot])) - #3442

### Contributors

* [github-actions[bot]](https://github.com/github-actions[bot])
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-mcp-client-v0.46.0...rig-mcp-client-v0.47.0) - 2026-11-20

### Added

- *(rig-basics, rig-tui, rig-harness, rig-harness-protocol, rig-cli, rig-steel, rig-ecs)* [**breaking**] plugin tree, forks off by default, one first-model rule, and release fixes ([#3425](https://github.com/0xPlaygrounds/rig/pull/3425)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs)* [**breaking**] spawn_root takes the agent's bundle, so Add<Agent> observers see it ([#3399](https://github.com/0xPlaygrounds/rig/pull/3399)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] fs::replace creates a new file private; drop replace_private ([#3370](https://github.com/0xPlaygrounds/rig/pull/3370)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(acp)* load or resume of a session the client has open re-attaches to it ([#3228](https://github.com/0xPlaygrounds/rig/pull/3228)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness,cli,mcp-client)* [**breaking**] one settings file, plugins from the home list, full trust in projects, and the launcher in its own crate ([#3005](https://github.com/0xPlaygrounds/rig/pull/3005)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- launcher, tools, plugin and memory bugs from the idiom audit (F5) ([#3365](https://github.com/0xPlaygrounds/rig/pull/3365)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3365
- release blockers from the v0.46.0 upgrade test: /mcp approve recall, /fork context, Bevy pins, 0.46 home warnings, restore notice ([#3330](https://github.com/0xPlaygrounds/rig/pull/3330)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3330
- *(rig-rmcp)* a server that declares no tools is not asked for them ([#3296](https://github.com/0xPlaygrounds/rig/pull/3296)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-rmcp)* client_config offers the released MCP protocol; examples and docs use it ([#3289](https://github.com/0xPlaygrounds/rig/pull/3289)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-mcp-client)* attribute MCP tools to the plugin that added McpCorePlugin ([#3272](https://github.com/0xPlaygrounds/rig/pull/3272)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mcp-client)* a server that wants an OAuth sign-in says so in plain words ([#3229](https://github.com/0xPlaygrounds/rig/pull/3229)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mcp-client)* offer the released MCP protocol 2025-11-25, not rmcp's draft ([#3227](https://github.com/0xPlaygrounds/rig/pull/3227)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mcp-client)* refuse a secret variable in mcp.json ([#3133](https://github.com/0xPlaygrounds/rig/pull/3133)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] every limit on asked work is a settings.toml key with a sensible default, and rig says what it changes ([#2995](https://github.com/0xPlaygrounds/rig/pull/2995)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2995
- *(ecs)* [**breaking**] one spawn path for agents, with no depth limit, and requests owned by what made them ([#2991](https://github.com/0xPlaygrounds/rig/pull/2991)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(io)* [**breaking**] input and output are bounded; a stop is never blocked by output ([#2974](https://github.com/0xPlaygrounds/rig/pull/2974)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] only an explicit choice becomes the default; /model offers only models that answer ([#2973](https://github.com/0xPlaygrounds/rig/pull/2973)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one run contract for every front; unknown input is refused out loud ([#2972](https://github.com/0xPlaygrounds/rig/pull/2972)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-basics, rig-harness, rig-core)* [**breaking**] dedup the harness and rig-basics helpers ([#3347](https://github.com/0xPlaygrounds/rig/pull/3347)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] one copy of each duplicated rule in rig-tools, the launcher, MCP and the Claude plugin ([#3346](https://github.com/0xPlaygrounds/rig/pull/3346)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3346
- *(mcp-client)* Streamable HTTP loopback test and an MCP/ACP interop matrix ([#3222](https://github.com/0xPlaygrounds/rig/pull/3222)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] remove duplicated tests, dead code and doc bloat across the workspace ([#3202](https://github.com/0xPlaygrounds/rig/pull/3202)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3202
- drop build edges nothing uses ([#3144](https://github.com/0xPlaygrounds/rig/pull/3144)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3144
- *(harness)* [**breaking**] one project root for settings, MCP, AGENTS.md, skills and plugin data ([#3000](https://github.com/0xPlaygrounds/rig/pull/3000)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(acp)* rig acp, an ACP v1 front with a rig process per session, on rig-mcp-client ([#2948](https://github.com/0xPlaygrounds/rig/pull/2948)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mcp-client)* ship the MCP client as a rig plugin, off by default ([#2938](https://github.com/0xPlaygrounds/rig/pull/2938)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
