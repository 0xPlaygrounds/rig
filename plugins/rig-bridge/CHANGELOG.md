# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.47.1](https://github.com/0xPlaygrounds/rig/compare/rig-bridge-v0.47.0...rig-bridge-v0.47.1) - 2026-11-24

### Added

- *(rig-bridge)* Esc stops the program's turn, and a new process goes on from the session the program saved ([#3463](https://github.com/0xPlaygrounds/rig/pull/3463)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-watch)* rig-watch, an optional plugin that wakes the agent when a file it waits on changes ([#3462](https://github.com/0xPlaygrounds/rig/pull/3462)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-core)* refresh model catalog (Gemini 4 Neon, Copilot MAI-Code 2, Bedrock/Azure/Copilot GPT-6.2) ([#3458](https://github.com/0xPlaygrounds/rig/pull/3458)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-bridge)* populate reasoning effort controls and reuse warm sessions across effort changes ([#3455](https://github.com/0xPlaygrounds/rig/pull/3455)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-bridge)* add live quota probes for Grok, Antigravity, and Copilot in /bridge usage ([#3454](https://github.com/0xPlaygrounds/rig/pull/3454)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-bridge)* other coding agents' CLIs as model providers ([#3445](https://github.com/0xPlaygrounds/rig/pull/3445)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Fixed

- *(rig-bridge)* rename Copilot CLI bridge vendor to copilot-cli to avoid built-in copilot catalog collision ([#3459](https://github.com/0xPlaygrounds/rig/pull/3459)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-bridge)* emit per-turn Copilot usage deltas, tighten scratch-dir note, and share nested TMPDIR guard ([#3457](https://github.com/0xPlaygrounds/rig/pull/3457)) (by [gold-silver-copper](https://github.com/gold-silver-copper))
- *(rig-bridge)* harden handshakes, probe isolation, Codex 0.171 flags, and warm recovery ([#3453](https://github.com/0xPlaygrounds/rig/pull/3453)) (by [gold-silver-copper](https://github.com/gold-silver-copper))

### Contributors

* [gold-silver-copper](https://github.com/gold-silver-copper)
