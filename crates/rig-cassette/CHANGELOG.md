# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-cassette-v0.0.1...rig-cassette-v0.43.0) - 2026-09-26

### Added

- *(cassette)* [**breaking**] refuse account failures and stored state, keep failed attempts, and move recording tooling into xtask ([#2587](https://github.com/0xPlaygrounds/rig/pull/2587)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* [**breaking**] signature slots, upstream reasoning provenance and the legacy fixture corpus ([#2580](https://github.com/0xPlaygrounds/rig/pull/2580)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(core)* [**breaking**] stateful handle round-trips and reasoning provenance ([#2578](https://github.com/0xPlaygrounds/rig/pull/2578)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* [**breaking**] record provider-opaque fields verbatim ([#2577](https://github.com/0xPlaygrounds/rig/pull/2577)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* skip non-recordable tests during live re-recording ([#2556](https://github.com/0xPlaygrounds/rig/pull/2556)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* add durable tool-turn checkpoint boundaries ([#2514](https://github.com/0xPlaygrounds/rig/pull/2514)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443

### Fixed

- repair eleven downstream consumer conformance defects ([#2561](https://github.com/0xPlaygrounds/rig/pull/2561)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2561
- *(rig-cassette)* restore start_at and checkpoint_recording for consumers that stage candidates ([#2498](https://github.com/0xPlaygrounds/rig/pull/2498)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(rig-core)* [**breaking**] classify every request-building failure as a request error ([#2586](https://github.com/0xPlaygrounds/rig/pull/2586)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* [**breaking**] standardize modality request builders and unify GenAI spans ([#2583](https://github.com/0xPlaygrounds/rig/pull/2583)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* [**breaking**] unify provider operation errors ([#2582](https://github.com/0xPlaygrounds/rig/pull/2582)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* guard history round-trips and fix Groq and Ollama replay ([#2576](https://github.com/0xPlaygrounds/rig/pull/2576)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* cover long native ECS tasks and cache accounting ([#2573](https://github.com/0xPlaygrounds/rig/pull/2573)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(ecs)* pin every native cell to its own world golden ([#2572](https://github.com/0xPlaygrounds/rig/pull/2572)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(tests)* stop comparing effect logs across runtimes in the ECS parity harness ([#2571](https://github.com/0xPlaygrounds/rig/pull/2571)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- simplify production comments and preserve contract rationale ([#2568](https://github.com/0xPlaygrounds/rig/pull/2568)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2568
- [**breaking**] establish host-owned assembly and execution lifetimes ([#2567](https://github.com/0xPlaygrounds/rig/pull/2567)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2567
- [**breaking**] remove backwards-compatibility shims and rename aliases ([#2557](https://github.com/0xPlaygrounds/rig/pull/2557)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2557
- [**breaking**] consolidate record/replay in rig-cassette ([#2552](https://github.com/0xPlaygrounds/rig/pull/2552)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2552
- *(tests)* one cassette response-header reader, one raw-capture harness ([#2549](https://github.com/0xPlaygrounds/rig/pull/2549)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Reduce test-suite compilation and duplicated fixtures ([#2518](https://github.com/0xPlaygrounds/rig/pull/2518)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2518
- *(ecs)* the ECS contract under faults on six wires — setup, retryable statuses, truncated and error-bearing streams, refusals, failing tools, stops, scenes ([#2503](https://github.com/0xPlaygrounds/rig/pull/2503)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* make rig-cassette publishable ([#2497](https://github.com/0xPlaygrounds/rig/pull/2497)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
