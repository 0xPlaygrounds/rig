# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-harness-v0.46.0...rig-harness-v0.47.0) - 2026-11-20

### Added

- *(rig-basics, rig-memory, rig-ecs, rig-core)* [**breaking**] minimal compaction on the kernel's turns, queue and retries ([#3436](https://github.com/0xPlaygrounds/rig/pull/3436)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics, rig-tui, rig-harness, rig-harness-protocol, rig-cli, rig-steel, rig-ecs)* [**breaking**] plugin tree, forks off by default, one first-model rule, and release fixes ([#3425](https://github.com/0xPlaygrounds/rig/pull/3425)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol, rig-cli, rig-basics)* [**breaking**] plugins say what they add and what they work only with ([#3405](https://github.com/0xPlaygrounds/rig/pull/3405)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol, rig-cli, rig-basics)* [**breaking**] plugin tree areas and branch switches in [plugins] off ([#3404](https://github.com/0xPlaygrounds/rig/pull/3404)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs)* [**breaking**] spawn_root takes the agent's bundle, so Add<Agent> observers see it ([#3399](https://github.com/0xPlaygrounds/rig/pull/3399)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* rig plugin add and remove write [[plugins.extra]] ([#3398](https://github.com/0xPlaygrounds/rig/pull/3398)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* the rig mark pulses while the agent works ([#3379](https://github.com/0xPlaygrounds/rig/pull/3379)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics, rig-tui, rig-harness)* [**breaking**] /plugins ticks with Enter and asks to reload now or later on Esc ([#3382](https://github.com/0xPlaygrounds/rig/pull/3382)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* the status line shows how long the agent has been working ([#3377](https://github.com/0xPlaygrounds/rig/pull/3377)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs)* [**breaking**] the kernel owns the last save and the persistence barrier ([#3268](https://github.com/0xPlaygrounds/rig/pull/3268)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs)* [**breaking**] write a reply that did not stream as stream deltas ([#3264](https://github.com/0xPlaygrounds/rig/pull/3264)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* warn once per start of a 0.46 home's plugins.toml and defaults.json ([#3239](https://github.com/0xPlaygrounds/rig/pull/3239)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools,basics)* [**breaking**] remove the unused move tool and the checkpoint plugin ([#3048](https://github.com/0xPlaygrounds/rig/pull/3048)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness,cli,mcp-client)* [**breaking**] one settings file, plugins from the home list, full trust in projects, and the launcher in its own crate ([#3005](https://github.com/0xPlaygrounds/rig/pull/3005)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(usage,helpers,budget)* [**breaking**] one agent tree's spend is counted, and an opt-in budget holds ([#2994](https://github.com/0xPlaygrounds/rig/pull/2994)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-tui, rig-cli, rig-basics, rig-harness-protocol)* a pasted /plugins filter, the upgrade warning, the Bevy refusal and /compact's notice tell the truth ([#3438](https://github.com/0xPlaygrounds/rig/pull/3438)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs, rig-basics, rig-harness, rig-harness-protocol, rig-cli, rig-tui)* [**breaking**] right wording for missing keys, /plugins branches and plugin refusals ([#3432](https://github.com/0xPlaygrounds/rig/pull/3432)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* read the catalog and effect logs under serde_json/arbitrary_precision ([#3414](https://github.com/0xPlaygrounds/rig/pull/3414)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tui, rig-fold)* a refused key binding is shown once in the terminal view ([#3397](https://github.com/0xPlaygrounds/rig/pull/3397)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs, rig-harness)* failed restored turns keep finished results, a first model choice connects once, nested plugins own their work, deletes recover without RigHome ([#3393](https://github.com/0xPlaygrounds/rig/pull/3393)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli, rig)* [**breaking**] the rig package alone owns the `rig` binary, and `rig update` installs it ([#3389](https://github.com/0xPlaygrounds/rig/pull/3389)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs, rig-harness, rig-basics, rig-memory, rig-tui)* [**breaking**] twelve kernel, harness and plugin bugs ([#3361](https://github.com/0xPlaygrounds/rig/pull/3361)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* write no longer overwrites a file changed since the model read it ([#3331](https://github.com/0xPlaygrounds/rig/pull/3331)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- release blockers from the v0.46.0 upgrade test: /mcp approve recall, /fork context, Bevy pins, 0.46 home warnings, restore notice ([#3330](https://github.com/0xPlaygrounds/rig/pull/3330)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3330
- *(rig-harness)* move a process-group list a reload cannot read aside instead of writing over it ([#3263](https://github.com/0xPlaygrounds/rig/pull/3263)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] check a front-only plugin's settings while describing ([#3273](https://github.com/0xPlaygrounds/rig/pull/3273)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] a shell registered per call starts its commands as Processes say ([#3269](https://github.com/0xPlaygrounds/rig/pull/3269)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* keep each launcher warning whole when it holds a line break ([#3258](https://github.com/0xPlaygrounds/rig/pull/3258)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-cli)* `rig plugin check --build` refuses settings a start would refuse ([#3253](https://github.com/0xPlaygrounds/rig/pull/3253)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* keep prompt history and run logs private to the user ([#3164](https://github.com/0xPlaygrounds/rig/pull/3164)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* never discard a session whose agent logs hold a conversation ([#3143](https://github.com/0xPlaygrounds/rig/pull/3143)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness,rig-basics,rig-acp)* one answer rule for every front, and the JSON front refuses a lost answer ([#3140](https://github.com/0xPlaygrounds/rig/pull/3140)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* create session directories private to the user ([#3137](https://github.com/0xPlaygrounds/rig/pull/3137)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] every limit on asked work is a settings.toml key with a sensible default, and rig says what it changes ([#2995](https://github.com/0xPlaygrounds/rig/pull/2995)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2995
- *(redact)* a notice is redacted between the stages, before any front reads it ([#2993](https://github.com/0xPlaygrounds/rig/pull/2993)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] one spawn path for agents, with no depth limit, and requests owned by what made them ([#2991](https://github.com/0xPlaygrounds/rig/pull/2991)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(print,acp)* [**breaking**] --json names a call by its place (protocol v 3 by this release); ACP shows a result the same live and loaded ([#2988](https://github.com/0xPlaygrounds/rig/pull/2988)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a call is known by its reply and place; every stop goes through one owner ([#2985](https://github.com/0xPlaygrounds/rig/pull/2985)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] --tools limits only the run it is given to ([#2984](https://github.com/0xPlaygrounds/rig/pull/2984)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(io)* [**breaking**] input and output are bounded; a stop is never blocked by output ([#2974](https://github.com/0xPlaygrounds/rig/pull/2974)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(models)* [**breaking**] only an explicit choice becomes the default; /model offers only models that answer ([#2973](https://github.com/0xPlaygrounds/rig/pull/2973)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] every child process has one owner, enforced by a lint ([#2970](https://github.com/0xPlaygrounds/rig/pull/2970)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(compaction)* [**breaking**] rewrite the summary, and never clear an output without notes or a same-bytes re-read ([#2975](https://github.com/0xPlaygrounds/rig/pull/2975)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one run contract for every front; unknown input is refused out loud ([#2972](https://github.com/0xPlaygrounds/rig/pull/2972)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] an @ reference attaches exactly the file that was picked ([#2962](https://github.com/0xPlaygrounds/rig/pull/2962)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(session)* [**breaking**] resume draws what live drew; an empty session is not kept ([#2964](https://github.com/0xPlaygrounds/rig/pull/2964)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one current agent per session; a fork says what it copies ([#2967](https://github.com/0xPlaygrounds/rig/pull/2967)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] tool results show what they found ([#2965](https://github.com/0xPlaygrounds/rig/pull/2965)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a stopped turn keeps what streamed, and a stop is never an error ([#2966](https://github.com/0xPlaygrounds/rig/pull/2966)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-harness, rig-ecs, rig-tui)* a short, generic plugin guide that points at examples ([#3401](https://github.com/0xPlaygrounds/rig/pull/3401)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- minimal root README ([#3371](https://github.com/0xPlaygrounds/rig/pull/3371)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3371
- *(rig-ecs, rig-harness, rig-basics, rig-tools, rig-cli)* [**breaking**] one copy of each kernel and harness rule, and the shell's forwarding wrapper goes ([#3369](https://github.com/0xPlaygrounds/rig/pull/3369)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics, rig-harness, rig-core)* [**breaking**] dedup the harness and rig-basics helpers ([#3347](https://github.com/0xPlaygrounds/rig/pull/3347)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tui)* [**breaking**] one type per render input; TranscriptSlot loses its unused Ord ([#3345](https://github.com/0xPlaygrounds/rig/pull/3345)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs, rig-tui, rig-basics)* [**breaking**] NoticeLevel derives serde and Hash; kernel errors derive thiserror; DroppedOptions goes ([#3336](https://github.com/0xPlaygrounds/rig/pull/3336)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] remove harness, memory and cassette surface only tests used, and merge rig-memory's repeated rules ([#3337](https://github.com/0xPlaygrounds/rig/pull/3337)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3337
- *(rig-core, rig-harness)* one closest-name rule, public beside edit_distance ([#3311](https://github.com/0xPlaygrounds/rig/pull/3311)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] file errors name the path and step; only NotFound is absent ([#3275](https://github.com/0xPlaygrounds/rig/pull/3275)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness-protocol)* [**breaking**] one AgentEnd record says how the agent ended ([#3274](https://github.com/0xPlaygrounds/rig/pull/3274)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tools)* [**breaking**] one listing rule for both watch backends, and typed watch events ([#3270](https://github.com/0xPlaygrounds/rig/pull/3270)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig)* an install line that works, a quick start, how to update, and what is per home ([#3217](https://github.com/0xPlaygrounds/rig/pull/3217)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] remove duplicated tests, dead code and doc bloat across the workspace ([#3202](https://github.com/0xPlaygrounds/rig/pull/3202)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3202
- *(tools)* confinement on another machine compares path text and cannot see links there ([#3163](https://github.com/0xPlaygrounds/rig/pull/3163)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- drop build edges nothing uses ([#3144](https://github.com/0xPlaygrounds/rig/pull/3144)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #3144
- *(deps)* update rmcp, bevy, base64, signal-hook, testcontainers, surrealdb, similar, sha2 and syn ([#3089](https://github.com/0xPlaygrounds/rig/pull/3089)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one project root for settings, MCP, AGENTS.md, skills and plugin data ([#3000](https://github.com/0xPlaygrounds/rig/pull/3000)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(redact)* [**breaking**] redaction is one optional plugin; secrets become placeholders a tool call can still use ([#2990](https://github.com/0xPlaygrounds/rig/pull/2990)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Added

- *(acp)* rig acp, an ACP v1 front with a rig process per session, on rig-mcp-client ([#2948](https://github.com/0xPlaygrounds/rig/pull/2948)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(mcp-client)* ship the MCP client as a rig plugin, off by default ([#2938](https://github.com/0xPlaygrounds/rig/pull/2938)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-traffic)* record every provider HTTP exchange, on rig-cassette's engine ([#2893](https://github.com/0xPlaygrounds/rig/pull/2893)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-claude-cloak)* publish claude-cloak with every release ([#2891](https://github.com/0xPlaygrounds/rig/pull/2891)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* [**breaking**] AGENTS.md and skills as plugins, /login with API keys ([#2873](https://github.com/0xPlaygrounds/rig/pull/2873)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(tui)* [**breaking**] Up never loses what you typed, and a recalled command opens no list ([#2947](https://github.com/0xPlaygrounds/rig/pull/2947)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] quitting mid-turn is a stop, and a restart carries nothing on by itself ([#2943](https://github.com/0xPlaygrounds/rig/pull/2943)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] headless runs hear SIGTERM in any setup, stop when nobody reads, and leave no tool processes behind ([#2942](https://github.com/0xPlaygrounds/rig/pull/2942)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* the system prompt lets the user's request set the answer ([#2939](https://github.com/0xPlaygrounds/rig/pull/2939)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* [**breaking**] fix picker layout, status text, edit diffs and paging ([#2917](https://github.com/0xPlaygrounds/rig/pull/2917)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(corpus, tui)* [**breaking**] editor commands run through the shell, and their failures show ([#2914](https://github.com/0xPlaygrounds/rig/pull/2914)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] Esc keeps the partial reply, /compact keeps history, accurate summaries, ctx ? after compaction ([#2918](https://github.com/0xPlaygrounds/rig/pull/2918)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui)* [**breaking**] prompt history per project, the shown conversation's prompts first ([#2916](https://github.com/0xPlaygrounds/rig/pull/2916)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui, harness)* one Ctrl+D quits; a session with no messages isn't saved ([#2909](https://github.com/0xPlaygrounds/rig/pull/2909)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] no self-extension prompt, a resume hint on exit, a session migration chain ([#2899](https://github.com/0xPlaygrounds/rig/pull/2899)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] a failed message goes back into the input ([#2895](https://github.com/0xPlaygrounds/rig/pull/2895)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* a 64k output budget, unknown-tool feedback, honest failed summaries, ordered headless JSON ([#2894](https://github.com/0xPlaygrounds/rig/pull/2894)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(tui)* [**breaking**] resume and big streamed tool calls are fast: code is colored only on screen ([#2952](https://github.com/0xPlaygrounds/rig/pull/2952)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] extension points as ECS data and systems ([#2900](https://github.com/0xPlaygrounds/rig/pull/2900)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- provider knowledge out of generic code, typed lookups ([#2897](https://github.com/0xPlaygrounds/rig/pull/2897)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2897

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-harness-v0.44.0...rig-harness-v0.45.0) - 2026-10-19

### Added

- *(rig-basics)* the default plugins.toml no longer lists SessionPickerPlugin ([#2863](https://github.com/0xPlaygrounds/rig/pull/2863)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-tui)* [**breaking**] a flat look: no box drawing, call rows tinted by state, the status line in the input ([#2859](https://github.com/0xPlaygrounds/rig/pull/2859)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-basics)* [**breaking**] rig over its pipes: --json, -p as the same front, and the helper tool ([#2854](https://github.com/0xPlaygrounds/rig/pull/2854)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 4: fix what the benchmark found ([#2843](https://github.com/0xPlaygrounds/rig/pull/2843)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 2: deletions, forks are agents, one recovery path, Esc stops subagents ([#2841](https://github.com/0xPlaygrounds/rig/pull/2841)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 1: fail closed, one fixed system prompt, keys redacted, /plugins, kernel tests ([#2838](https://github.com/0xPlaygrounds/rig/pull/2838)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* smaller defaults, whole color-coded transcript, syntax highlighting ([#2832](https://github.com/0xPlaygrounds/rig/pull/2832)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] a thin coding agent over rig libraries, with rig-ecs and rig-tools ([#2771](https://github.com/0xPlaygrounds/rig/pull/2771)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-tools)* triobench round 6: shell output drops terminal escape codes, a cut keeps the first and last errors, a remote write cut off mid-way says whether it happened, an edit after a write is one call ([#2858](https://github.com/0xPlaygrounds/rig/pull/2858)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig)* [**breaking**] triobench round 1: subagent messages and reports, remote HOST:PATH paths, shell streams and cuts, search paths, OSC 133 turn marks ([#2849](https://github.com/0xPlaygrounds/rig/pull/2849)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] everything the pre-0.45 API review found ([#2848](https://github.com/0xPlaygrounds/rig/pull/2848)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2848

### Other

- a README that is a front door: the agent first, then the library; the agent's reference in docs/agent.md ([#2862](https://github.com/0xPlaygrounds/rig/pull/2862)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2862
- *(rig-basics)* [**breaking**] fold the headless runner into the print front ([#2856](https://github.com/0xPlaygrounds/rig/pull/2856)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 5: a smaller core, more plugins, fewer crates ([#2844](https://github.com/0xPlaygrounds/rig/pull/2844)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* publish rig-ecs, rig-harness, rig-tools and the plugin crates (onto main) ([#2836](https://github.com/0xPlaygrounds/rig/pull/2836)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- faster, simpler, smaller rig (one PR) ([#2835](https://github.com/0xPlaygrounds/rig/pull/2835)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2835
- fast, solid CI (one corpus build, hangs fail fast, heavy lanes in the queue) ([#2833](https://github.com/0xPlaygrounds/rig/pull/2833)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2833
- *(rig)* [**breaking**] measured performance pass on rig's libraries, with benchmarks ([#2796](https://github.com/0xPlaygrounds/rig/pull/2796)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
