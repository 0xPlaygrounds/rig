# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-acp-v0.46.0...rig-acp-v0.47.0) - 2026-11-14

### Added

- *(rig-ecs)* [**breaking**] the kernel owns the last save and the persistence barrier ([#3268](https://github.com/0xPlaygrounds/rig/pull/3268)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs)* [**breaking**] write a reply that did not stream as stream deltas ([#3264](https://github.com/0xPlaygrounds/rig/pull/3264)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(acp)* load or resume of a session the client has open re-attaches to it ([#3228](https://github.com/0xPlaygrounds/rig/pull/3228)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools,basics)* [**breaking**] remove the unused move tool and the checkpoint plugin ([#3048](https://github.com/0xPlaygrounds/rig/pull/3048)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness,cli,mcp-client)* [**breaking**] one settings file, plugins from the home list, full trust in projects, and the launcher in its own crate ([#3005](https://github.com/0xPlaygrounds/rig/pull/3005)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-core, rig-ecs, rig-basics)* [**breaking**] one reader and printer for a reasoning setting as text, so [model] effort = "high" works on budget-only models ([#3291](https://github.com/0xPlaygrounds/rig/pull/3291)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-acp)* re-attach with the same MCP servers in another order ([#3252](https://github.com/0xPlaygrounds/rig/pull/3252)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* the version names its commit, the lock follows the checkout, and a source mismatch is refused ([#3221](https://github.com/0xPlaygrounds/rig/pull/3221)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-acp)* show the editor a reply a plugin answers without streaming ([#3138](https://github.com/0xPlaygrounds/rig/pull/3138)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-acp)* rig acp refuses an option after acp instead of ignoring it ([#3142](https://github.com/0xPlaygrounds/rig/pull/3142)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness,rig-basics,rig-acp)* one answer rule for every front, and the JSON front refuses a lost answer ([#3140](https://github.com/0xPlaygrounds/rig/pull/3140)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] one spawn path for agents, with no depth limit, and requests owned by what made them ([#2991](https://github.com/0xPlaygrounds/rig/pull/2991)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(print,acp)* [**breaking**] --json names a call by its place (protocol v 2); ACP shows a result the same live and loaded ([#2988](https://github.com/0xPlaygrounds/rig/pull/2988)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a call is known by its reply and place; every stop goes through one owner ([#2985](https://github.com/0xPlaygrounds/rig/pull/2985)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] /model lists only models you can use now; a model that doesn't connect is refused and the current one stays ([#2983](https://github.com/0xPlaygrounds/rig/pull/2983)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(io)* [**breaking**] input and output are bounded; a stop is never blocked by output ([#2974](https://github.com/0xPlaygrounds/rig/pull/2974)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] only an explicit choice becomes the default; /model offers only models that answer ([#2973](https://github.com/0xPlaygrounds/rig/pull/2973)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one run contract for every front; unknown input is refused out loud ([#2972](https://github.com/0xPlaygrounds/rig/pull/2972)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one current agent per session; a fork says what it copies ([#2967](https://github.com/0xPlaygrounds/rig/pull/2967)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a stopped turn keeps what streamed, and a stop is never an error ([#2966](https://github.com/0xPlaygrounds/rig/pull/2966)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-harness-protocol)* [**breaking**] file errors name the path and step; only NotFound is absent ([#3275](https://github.com/0xPlaygrounds/rig/pull/3275)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] one AgentEnd record says how the agent ended ([#3274](https://github.com/0xPlaygrounds/rig/pull/3274)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] one Launch type builds every child the launcher starts ([#3271](https://github.com/0xPlaygrounds/rig/pull/3271)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mcp-client)* Streamable HTTP loopback test and an MCP/ACP interop matrix ([#3222](https://github.com/0xPlaygrounds/rig/pull/3222)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- drop build edges nothing uses ([#3144](https://github.com/0xPlaygrounds/rig/pull/3144)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3144
- *(redact)* [**breaking**] redaction is one optional plugin; secrets become placeholders a tool call can still use ([#2990](https://github.com/0xPlaygrounds/rig/pull/2990)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(acp)* rig acp, an ACP v1 front with a rig process per session, on rig-mcp-client ([#2948](https://github.com/0xPlaygrounds/rig/pull/2948)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
