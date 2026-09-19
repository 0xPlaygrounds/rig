# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.43.0](https://github.com/0xPlaygrounds/rig/compare/rig-cassette-v0.0.1...rig-cassette-v0.43.0) - 2026-09-19

### Added

- *(ecs)* add durable tool-turn checkpoint boundaries ([#2514](https://github.com/0xPlaygrounds/rig/pull/2514)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- [**breaking**] the effect-bus critical path — one protocol, one channel, typed views ([#2443](https://github.com/0xPlaygrounds/rig/pull/2443)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2443

### Fixed

- *(rig-cassette)* restore start_at and checkpoint_recording for consumers that stage candidates ([#2498](https://github.com/0xPlaygrounds/rig/pull/2498)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Other

- *(tests)* one cassette response-header reader, one raw-capture harness ([#2549](https://github.com/0xPlaygrounds/rig/pull/2549)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- Reduce test-suite compilation and duplicated fixtures ([#2518](https://github.com/0xPlaygrounds/rig/pull/2518)) (by [gold-silver-copper](https://github.com/gold-silver-copper)) - #2518
- *(ecs)* the ECS contract under faults on six wires — setup, retryable statuses, truncated and error-bearing streams, refusals, failing tools, stops, scenes ([#2503](https://github.com/0xPlaygrounds/rig/pull/2503)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(cassette)* make rig-cassette publishable ([#2497](https://github.com/0xPlaygrounds/rig/pull/2497)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
