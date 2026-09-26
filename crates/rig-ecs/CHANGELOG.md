# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-ecs-v0.0.1...rig-ecs-v0.43.0) - 2026-09-26

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

- the parity gaps the ECS matrices found, closed — Gemini's block is a refusal, a truncated reasoning-only turn commits nothing, images ride the regrouped stream ([#2509](https://github.com/0xPlaygrounds/rig/pull/2509)) ([#2510](https://github.com/0xPlaygrounds/rig/pull/2510)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2510
- *(rig-ecs)* a Retry written on an empty turn asks again instead of settling ([#2504](https://github.com/0xPlaygrounds/rig/pull/2504)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* restore the observe scrub helpers; a truncated stream is retryable ([#2502](https://github.com/0xPlaygrounds/rig/pull/2502)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(gemini, rig-ecs)* retry transient provider failures inside the run; block_reason=OTHER is not a refusal ([#2500](https://github.com/0xPlaygrounds/rig/pull/2500)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] the merge review's defects on main, each pinned by a matrix ([#2499](https://github.com/0xPlaygrounds/rig/pull/2499)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2499

### Other

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
