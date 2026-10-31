# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.0](https://github.com/0xPlaygrounds/rig/compare/rig-ecs-v0.46.0...rig-ecs-v0.47.0) - 2026-10-31

### Fixed

- *(session)* [**breaking**] resume draws what live drew; an empty session is not kept ([#2964](https://github.com/0xPlaygrounds/rig/pull/2964)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] one current agent per session; a fork says what it copies ([#2967](https://github.com/0xPlaygrounds/rig/pull/2967)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tools)* [**breaking**] tool results show what they found ([#2965](https://github.com/0xPlaygrounds/rig/pull/2965)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] a stopped turn keeps what streamed, and a stop is never an error ([#2966](https://github.com/0xPlaygrounds/rig/pull/2966)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.46.0](https://github.com/0xPlaygrounds/rig/compare/v0.45.0...v0.46.0) - 2026-10-31

### Fixed

- *(agent)* [**breaking**] quitting mid-turn is a stop, and a restart carries nothing on by itself ([#2943](https://github.com/0xPlaygrounds/rig/pull/2943)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] Esc keeps the partial reply, /compact keeps history, accurate summaries, ctx ? after compaction ([#2918](https://github.com/0xPlaygrounds/rig/pull/2918)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tui, harness)* one Ctrl+D quits; a session with no messages isn't saved ([#2909](https://github.com/0xPlaygrounds/rig/pull/2909)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] no self-extension prompt, a resume hint on exit, a session migration chain ([#2899](https://github.com/0xPlaygrounds/rig/pull/2899)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* [**breaking**] a failed message goes back into the input ([#2895](https://github.com/0xPlaygrounds/rig/pull/2895)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(agent)* a 64k output budget, unknown-tool feedback, honest failed summaries, ordered headless JSON ([#2894](https://github.com/0xPlaygrounds/rig/pull/2894)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(catalog, compaction)* fit models' input limits with simpler overflow recovery ([#2880](https://github.com/0xPlaygrounds/rig/pull/2880)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(agent)* [**breaking**] extension points as ECS data and systems ([#2900](https://github.com/0xPlaygrounds/rig/pull/2900)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- provider knowledge out of generic code, typed lookups ([#2897](https://github.com/0xPlaygrounds/rig/pull/2897)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2897

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.45.0](https://github.com/0xPlaygrounds/rig/compare/rig-ecs-v0.44.0...rig-ecs-v0.45.0) - 2026-10-19

### Added

- *(harness)* [**breaking**] beat Pi, part 4: fix what the benchmark found ([#2843](https://github.com/0xPlaygrounds/rig/pull/2843)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 2: deletions, forks are agents, one recovery path, Esc stops subagents ([#2841](https://github.com/0xPlaygrounds/rig/pull/2841)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(harness)* [**breaking**] beat Pi, part 1: fail closed, one fixed system prompt, keys redacted, /plugins, kernel tests ([#2838](https://github.com/0xPlaygrounds/rig/pull/2838)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* [**breaking**] a thin coding agent over rig libraries, with rig-ecs and rig-tools ([#2771](https://github.com/0xPlaygrounds/rig/pull/2771)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- [**breaking**] everything the pre-0.45 API review found ([#2848](https://github.com/0xPlaygrounds/rig/pull/2848)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2848
- *(compaction)* a stop during a held summary cancels it ([#2846](https://github.com/0xPlaygrounds/rig/pull/2846)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] API review follow-up: defects, async storage, restored capabilities ([#2845](https://github.com/0xPlaygrounds/rig/pull/2845)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2845
- [**breaking**] one TurnPolicy per model turn owns its tool choice; Skip, hooks and the ECS checks read it ([#2757](https://github.com/0xPlaygrounds/rig/pull/2757)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2757
- *(rig-core)* [**breaking**] one meaning for a document's text on every wire (DocumentSourceKind::String becomes DocumentData::Text) ([#2754](https://github.com/0xPlaygrounds/rig/pull/2754)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* classify an in-band provider error envelope in one place ([#2758](https://github.com/0xPlaygrounds/rig/pull/2758)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-ecs)* read the save-failure notice one frame after the turn ends ([#2847](https://github.com/0xPlaygrounds/rig/pull/2847)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-harness)* publish rig-ecs, rig-harness, rig-tools and the plugin crates (onto main) ([#2836](https://github.com/0xPlaygrounds/rig/pull/2836)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- faster, simpler, smaller rig (one PR) ([#2835](https://github.com/0xPlaygrounds/rig/pull/2835)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2835
- fast, solid CI (one corpus build, hangs fail fast, heavy lanes in the queue) ([#2833](https://github.com/0xPlaygrounds/rig/pull/2833)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2833
- *(rig)* [**breaking**] measured performance pass on rig's libraries, with benchmarks ([#2796](https://github.com/0xPlaygrounds/rig/pull/2796)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] remove the rig-ecs crate and its tests, goldens and fixtures ([#2761](https://github.com/0xPlaygrounds/rig/pull/2761)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2761

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.44.0](https://github.com/0xPlaygrounds/rig/compare/rig-ecs-v0.43.0...rig-ecs-v0.44.0) - 2026-10-07

### Added

- typed generation options, model catalog, provider extensions and normalized replies ([#2750](https://github.com/0xPlaygrounds/rig/pull/2750)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2750
- [**breaking**] stream tool-call arguments as they arrive ([#2721](https://github.com/0xPlaygrounds/rig/pull/2721)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2721

### Fixed

- [**breaking**] unknown finish reasons are one outcome, a caller's choice, and never silent ([#2726](https://github.com/0xPlaygrounds/rig/pull/2726)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2726

### Other

- *(core)* [**breaking**] one name for the unknown-finish-reason setting ([#2734](https://github.com/0xPlaygrounds/rig/pull/2734)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- shape-matched cassettes, encoder snapshots, and a coverage gate ([#2720](https://github.com/0xPlaygrounds/rig/pull/2720)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2720
- *(message)* [**breaking**] item-shaped assistant history with provenance ([#2713](https://github.com/0xPlaygrounds/rig/pull/2713)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] type tool definition and tool choice names as ToolName ([#2689](https://github.com/0xPlaygrounds/rig/pull/2689)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2689
- *(ecs)* [**breaking**] typed checkpoint save/restore errors ([#2692](https://github.com/0xPlaygrounds/rig/pull/2692)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] stop flattening tool and MCP failures into strings ([#2687](https://github.com/0xPlaygrounds/rig/pull/2687)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2687
- *(tests)* share request fixtures and golden helpers outside provider tests ([#2677](https://github.com/0xPlaygrounds/rig/pull/2677)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* [**breaking**] one request-assembly pass and no hand-kept mirrors in rig-ecs ([#2672](https://github.com/0xPlaygrounds/rig/pull/2672)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] share structured-output policy between rig-agent and rig-ecs ([#2660](https://github.com/0xPlaygrounds/rig/pull/2660)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2660

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-ecs-v0.0.1...rig-ecs-v0.43.0) - 2026-09-30

### Added

- *(core)* [**breaking**] stateful handle round-trips and reasoning provenance ([#2578](https://github.com/0xPlaygrounds/rig/pull/2578)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- feat!(core, ecs): one provider vocabulary, lossless configuration ([#2548](https://github.com/0xPlaygrounds/rig/pull/2548)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2548
- feat!(core): Usage counters are Option<u64> — an absent counter is representable ([#2535](https://github.com/0xPlaygrounds/rig/pull/2535)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2535
- *(rig-ecs)* [**breaking**] the ECS contract under a long tool loop on five wires, with tool results as data, cached utterance views, bindings as data and runs as commands ([#2516](https://github.com/0xPlaygrounds/rig/pull/2516)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-ecs)* explicit stream delivery — StreamItemsDelivered, two consumers, five-wire matrix ([#2515](https://github.com/0xPlaygrounds/rig/pull/2515)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* add durable tool-turn checkpoint boundaries ([#2514](https://github.com/0xPlaygrounds/rig/pull/2514)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* expose typed content parts and shared binary assets ([#2513](https://github.com/0xPlaygrounds/rig/pull/2513)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443

### Fixed

- pre-0.43 release fixes (docs.rs, opt-in rig::cassette, rust-version, READMEs) ([#2638](https://github.com/0xPlaygrounds/rig/pull/2638)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2638
- the parity gaps the ECS matrices found, closed — Gemini's block is a refusal, a truncated reasoning-only turn commits nothing, images ride the regrouped stream ([#2509](https://github.com/0xPlaygrounds/rig/pull/2509)) ([#2510](https://github.com/0xPlaygrounds/rig/pull/2510)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2510
- *(rig-ecs)* a Retry written on an empty turn asks again instead of settling ([#2504](https://github.com/0xPlaygrounds/rig/pull/2504)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* restore the observe scrub helpers; a truncated stream is retryable ([#2502](https://github.com/0xPlaygrounds/rig/pull/2502)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(gemini, rig-ecs)* retry transient provider failures inside the run; block_reason=OTHER is not a refusal ([#2500](https://github.com/0xPlaygrounds/rig/pull/2500)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] the merge review's defects on main, each pinned by a matrix ([#2499](https://github.com/0xPlaygrounds/rig/pull/2499)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2499

### Other

- [**breaking**] replace NonEmpty<T> with Vec<T>, check empty turns at the request boundary ([#2640](https://github.com/0xPlaygrounds/rig/pull/2640)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2640
- *(rig-core)* [**breaking**] the driver alone writes provider, request id and raw ([#2626](https://github.com/0xPlaygrounds/rig/pull/2626)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] archimpro — one typed decoder and one fold per wire ([#2617](https://github.com/0xPlaygrounds/rig/pull/2617)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2617
- *(rig-core)* [**breaking**] a smaller model layer (Operation, Fold, Wire, Decoder, Transport) ([#2613](https://github.com/0xPlaygrounds/rig/pull/2613)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] provider clients own their transport; conversations use plain values ([#2611](https://github.com/0xPlaygrounds/rig/pull/2611)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2611
- *(rig-core)* [**breaking**] unify provider operation errors ([#2582](https://github.com/0xPlaygrounds/rig/pull/2582)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* pin every native cell to its own world golden ([#2572](https://github.com/0xPlaygrounds/rig/pull/2572)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tests)* stop comparing effect logs across runtimes in the ECS parity harness ([#2571](https://github.com/0xPlaygrounds/rig/pull/2571)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- simplify production comments and preserve contract rationale ([#2568](https://github.com/0xPlaygrounds/rig/pull/2568)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2568
- [**breaking**] establish host-owned assembly and execution lifetimes ([#2567](https://github.com/0xPlaygrounds/rig/pull/2567)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2567
- *(ecs)* [**breaking**] use one reflected content payload component ([#2559](https://github.com/0xPlaygrounds/rig/pull/2559)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] remove backwards-compatibility shims and rename aliases ([#2557](https://github.com/0xPlaygrounds/rig/pull/2557)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2557
- [**breaking**] consolidate record/replay in rig-cassette ([#2552](https://github.com/0xPlaygrounds/rig/pull/2552)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2552
- Unify every provider onto one wire model ([#2538](https://github.com/0xPlaygrounds/rig/pull/2538)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2538
- [**breaking**] retire five clippy heuristics and unbox everything they made us write ([#2536](https://github.com/0xPlaygrounds/rig/pull/2536)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2536
- refactor!(ecs, providers): delete the ECS message cache, the second task/handler storage, and test-only wire→core conversions ([#2534](https://github.com/0xPlaygrounds/rig/pull/2534)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2534
- a Bevy app — one pass per update, tasks and handlers as components, one RunPhase, split systems, reflected checkpoint ([#2529](https://github.com/0xPlaygrounds/rig/pull/2529)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2529
- delete the reflect and replay features and the source-shape guards ([#2523](https://github.com/0xPlaygrounds/rig/pull/2523)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2523
- *(rig-ecs)* delete duplicated scaffolding, lists and wrappers ([#2522](https://github.com/0xPlaygrounds/rig/pull/2522)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Reduce test-suite compilation and duplicated fixtures ([#2518](https://github.com/0xPlaygrounds/rig/pull/2518)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2518
- *(ecs)* images through record, history, tool loop, memory and scene resume on four wires ([#2512](https://github.com/0xPlaygrounds/rig/pull/2512)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* pin reasoning across six provider wires ([#2511](https://github.com/0xPlaygrounds/rig/pull/2511)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* the ECS contract under faults on six wires — setup, retryable statuses, truncated and error-bearing streams, refusals, failing tools, stops, scenes ([#2503](https://github.com/0xPlaygrounds/rig/pull/2503)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* the ECS contract on five more wires — OpenAI Chat, OpenAI Responses, Gemini, DeepSeek, Doubleword, Venice ([#2501](https://github.com/0xPlaygrounds/rig/pull/2501)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
