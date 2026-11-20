# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-claude-cloak-v0.46.0...rig-claude-cloak-v0.47.0) - 2026-11-20

### Added

- *(rig-memory, rig-basics, rig-ecs)* [**breaking**] compact the way pi does ([#3434](https://github.com/0xPlaygrounds/rig/pull/3434)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics, rig-tui, rig-harness, rig-harness-protocol, rig-cli, rig-steel, rig-ecs)* [**breaking**] plugin tree, forks off by default, one first-model rule, and release fixes ([#3425](https://github.com/0xPlaygrounds/rig/pull/3425)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs)* [**breaking**] spawn_root takes the agent's bundle, so Add<Agent> observers see it ([#3399](https://github.com/0xPlaygrounds/rig/pull/3399)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* a public loopback sign-in listener, used by the ChatGPT flow and the Claude plugin ([#3308](https://github.com/0xPlaygrounds/rig/pull/3308)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness,cli,mcp-client)* [**breaking**] one settings file, plugins from the home list, full trust in projects, and the launcher in its own crate ([#3005](https://github.com/0xPlaygrounds/rig/pull/3005)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-claude-cloak)* strict tool schemas are chosen by shape, not by another program's tool names ([#3293](https://github.com/0xPlaygrounds/rig/pull/3293)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-claude-cloak)* transport wrappers see the request the plugin sends, not the one it was handed ([#3280](https://github.com/0xPlaygrounds/rig/pull/3280)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(claude-cloak)* start the login browser without rig's secrets ([#3135](https://github.com/0xPlaygrounds/rig/pull/3135)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] one spawn path for agents, with no depth limit, and requests owned by what made them ([#2991](https://github.com/0xPlaygrounds/rig/pull/2991)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] /model lists only models you can use now; a model that doesn't connect is refused and the current one stays ([#2983](https://github.com/0xPlaygrounds/rig/pull/2983)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] only an explicit choice becomes the default; /model offers only models that answer ([#2973](https://github.com/0xPlaygrounds/rig/pull/2973)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(http)* [**breaking**] an error keeps its retry class from where it was made ([#2969](https://github.com/0xPlaygrounds/rig/pull/2969)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- [**breaking**] callers use the existing helpers (R8) ([#3362](https://github.com/0xPlaygrounds/rig/pull/3362)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3362
- *(rig-core)* [**breaking**] one public path for the shared sign-in types ([#3356](https://github.com/0xPlaygrounds/rig/pull/3356)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics, rig-harness, rig-core)* [**breaking**] dedup the harness and rig-basics helpers ([#3347](https://github.com/0xPlaygrounds/rig/pull/3347)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] one copy of each duplicated rule in rig-tools, the launcher, MCP and the Claude plugin ([#3346](https://github.com/0xPlaygrounds/rig/pull/3346)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3346
- *(rig-http)* [**breaking**] HttpClientExt takes and returns Bytes; dead rig-http and rig-core surface ([#3341](https://github.com/0xPlaygrounds/rig/pull/3341)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* content accessors, to_text, json_in_text and Aside::read replace about 45 copies ([#3335](https://github.com/0xPlaygrounds/rig/pull/3335)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-claude-cloak)* credentials are kept through rig-core's FileTokenStore ([#3305](https://github.com/0xPlaygrounds/rig/pull/3305)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] remove duplicated tests, dead code and doc bloat across the workspace ([#3202](https://github.com/0xPlaygrounds/rig/pull/3202)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3202
- *(deps)* update rmcp, bevy, base64, signal-hook, testcontainers, surrealdb, similar, sha2 and syn ([#3089](https://github.com/0xPlaygrounds/rig/pull/3089)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(rig-claude-cloak)* publish claude-cloak with every release ([#2891](https://github.com/0xPlaygrounds/rig/pull/2891)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* add claude-cloak and harden Anthropic handling ([#2875](https://github.com/0xPlaygrounds/rig/pull/2875)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(core)* [**breaking**] stop forcing serde_json features ([#2934](https://github.com/0xPlaygrounds/rig/pull/2934)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(agent)* [**breaking**] extension points as ECS data and systems ([#2900](https://github.com/0xPlaygrounds/rig/pull/2900)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- provider knowledge out of generic code, typed lookups ([#2897](https://github.com/0xPlaygrounds/rig/pull/2897)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2897
- *(core)* each wire owns where its reasoning blob lives ([#2896](https://github.com/0xPlaygrounds/rig/pull/2896)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
