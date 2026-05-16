# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.37.1](https://github.com/0xPlaygrounds/rig/compare/rig-v0.37.0...rig-v0.37.1) - 2026-05-16

### Other

- *(gemini)* add streaming metadata cassettes ([#1777](https://github.com/0xPlaygrounds/rig/pull/1777)) (by @gold-silver-copper)
- Add replayable provider cassette tests ([#1769](https://github.com/0xPlaygrounds/rig/pull/1769)) (by @gold-silver-copper) - #1769

### Contributors

* @gold-silver-copper
## [0.37.0](https://github.com/0xPlaygrounds/rig/compare/rig-v0.36.0...rig-v0.37.0) - 2026-05-13

### Added

- *(openrouter)* add transcription (STT) and audio generation (TTS) support ([#1757](https://github.com/0xPlaygrounds/rig/pull/1757)) (by @fversaci)
- *(rig-bedrock)* add structured output support via Converse API ([#1667](https://github.com/0xPlaygrounds/rig/pull/1667)) (by @jdwil)
- *(memory)* Rig-managed conversation memory + rig-memory companion crate ([#1702](https://github.com/0xPlaygrounds/rig/pull/1702)) (by @ForeverAngry)
- add copilot model listing ([#1700](https://github.com/0xPlaygrounds/rig/pull/1700)) (by @BigtoC) - #1700

### Fixed

- *(gemini)* Token usage correctness for posthog llm analytics ([#1761](https://github.com/0xPlaygrounds/rig/pull/1761)) (by @mateobelanger)
- *(core)* [**breaking**] make Chat append messages to caller history ([#1733](https://github.com/0xPlaygrounds/rig/pull/1733)) (by @gold-silver-copper)

### Other

- Clean up root facade features and integration docs ([#1764](https://github.com/0xPlaygrounds/rig/pull/1764)) (by @gold-silver-copper) - #1764
- fix "a ancient" grammar in glarb-glarb sample text ([#1755](https://github.com/0xPlaygrounds/rig/pull/1755)) (by @abhicris) - #1755
- *(deps)* bump lopdf from 0.36.0 to 0.40.0 ([#1754](https://github.com/0xPlaygrounds/rig/pull/1754)) (by @dependabot[bot])
- *(deps)* bump quick-xml from 0.39.2 to 0.39.4 ([#1752](https://github.com/0xPlaygrounds/rig/pull/1752)) (by @dependabot[bot])
- *(deps)* bump tonic-build from 0.14.5 to 0.14.6 ([#1751](https://github.com/0xPlaygrounds/rig/pull/1751)) (by @dependabot[bot])
- Move reusable test doubles into rig_core::test_utils ([#1745](https://github.com/0xPlaygrounds/rig/pull/1745)) (by @gold-silver-copper) - #1745
- workspace and docs cleanup ([#1742](https://github.com/0xPlaygrounds/rig/pull/1742)) (by @gold-silver-copper) - #1742
- openrouter vars ([#1741](https://github.com/0xPlaygrounds/rig/pull/1741)) (by @gold-silver-copper) - #1741
- Add provider file ID support for document inputs ([#1740](https://github.com/0xPlaygrounds/rig/pull/1740)) (by @gold-silver-copper) - #1740
- add smoke test for completion across all Copilot models ([#1730](https://github.com/0xPlaygrounds/rig/pull/1730)) (by @BigtoC) - #1730
- bump dependencies ([#1728](https://github.com/0xPlaygrounds/rig/pull/1728)) (by @gold-silver-copper) - #1728
- remove needless files ([#1715](https://github.com/0xPlaygrounds/rig/pull/1715)) (by @gold-silver-copper) - #1715
- AGENTS.MD, CONTRIBUTING.MD, and docs ([#1714](https://github.com/0xPlaygrounds/rig/pull/1714)) (by @gold-silver-copper) - #1714
- Add Bedrock integration tests ([#1707](https://github.com/0xPlaygrounds/rig/pull/1707)) (by @gold-silver-copper) - #1707

### Contributors

* @gold-silver-copper
* @fversaci
* @mateobelanger
* @abhicris
* @jdwil
* @dependabot[bot]
* @ForeverAngry
* @BigtoC
