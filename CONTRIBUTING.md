# Contributing to Rig

Thank you for considering contributing to Rig! Here are some guidelines to help you get started.

General guidelines and requested contributions can be found in the [How to Contribute](https://docs.rig.rs/docs/how_to_contribute) section of the documentation. Repository layout, architecture, and engineering rules are in `AGENTS.md`; verification and publication workflow in [DEVELOPING.md](DEVELOPING.md); test commands in `tests/README.md`.

## Issues
Before reporting an issue, please check existing or similar issues that are currently tracked.

Additionally, please ensure that if you are submitting a bug ticket (ie, something doesn't work) that the bug is reproducible. If we cannot reproduce the bug, your ticket is likely to be marked either `wontfix` or closed (although it's likely we'll take note of it in case there's a secondary occurrence).

## Pull Requests

Contributions are always encouraged and welcome. Before creating a pull request, create a new issue that tracks that pull request describing the problem in more detail. Pull request descriptions should include information about its implementation, especially if it makes changes to existing abstractions.

PRs should be small and focused and should avoid interacting with multiple facets of the library. This may result in a larger PR being split into two or more smaller PRs. Commit messages should follow the [Conventional Commit](https://conventionalcommits.org/en/v1.0.0) format (prefixing with `feat`, `fix`, etc.) as this integrates into our auto-releases via a [release-plz](https://github.com/MarcoIeni/release-plz) Github action.

Do not edit `CHANGELOG.md`, `crates/*/CHANGELOG.md` or `MIGRATING.md` in a pull request; CI fails the PR if you do. Put changelog bullets and migration notes in the PR description under `## Changelog` and `## Migration` (the PR template has both); the release PR regenerates both files from them.

Unless the PR is for something minor (ie a typo), please ensure that an issue has been opened for the feature or work you would like to contribute beforehand. By opening an issue, a discussion can be held beforehand on scoping the work effectively and ensuring that the work is in line with the vision for Rig. Without any linked issues, your PR may be liable to be closed if we (the maintainers) do not feel that your PR is within scope for the library.

It is also highly suggested to comment on issues you're interested in working on. By doing so, it allows others to see that something is being worked on and therefore avoids frustrating situations, such as multiple contributors opening a PR for the same issue. In such a case, any duplicate PRs will be closed unless it is clear that the original contributor is unable to continue the work.

You can link your PR back to a given issue by writing `Fixes #999` in your PR message; this auto-links the issue and closes it once the PR has been merged.

**Working on your first Pull Request?** You can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://kcd.im/pull-request)

### Code Contribution Guidelines

We will not strictly enforce guidelines because we want to make it as easy as possible to contribute to Rig, but we advise contributors to stick to three policies:
- Use docstrings on any new public items (structs, enums, methods whether free-standing or associated).
- Use full syntax for trait bounds where possible. This makes the code much easier to read.
- If your PR adds functionality, it must include relevant tests that pass. Provider behavior changes should usually include cassette-backed regression tests; user-facing changes should also update examples or documentation as appropriate.

The workspace enforces strict clippy lints, including forbidding `unwrap`, `expect`, `todo`, and `unimplemented`. Prefer explicit error types, `?`, and complete edge-case handling. Provider and vector-store implementation requirements are in `AGENTS.md`.

Each PR will be taken on a case-by-case basis.

### PRs that will be rejected
Not every contribution is within the scope of the repo. Out of scope includes but is not limited to:
- Changes that would force model provider integrations to diverge from the original API (eg adding a field to the OpenAI API that does not exist there for the sake of another model provider)
- Lazy workarounds: `String` error types, scattered `.unwrap()` calls, stubbed error handling, incomplete edge-case handling
- TODO comments, placeholder implementations, or `unimplemented!()`
- Raw `Send`/`Sync` where `WasmCompatSend`/`WasmCompatSync` should be used
- Unclear code that needs comments to explain what it is doing instead of being refactored for readability
- Major architectural changes, new abstractions, or public API reshaping without prior discussion
- Arbitrary markdown files. The only markdown files we allow are ones that are already traditional convention (DEVELOPING.md, CONTRIBUTING.md, ARCHITECTURE.md, ... etc)
- Duplicates of other PRs

This will be reviewed on a case by case basis, but generally unjustifiable breaking changes are much more likely to be rejected.

## Project Structure

Rig is a monorepo: the root `rig` facade, `crates/rig-core`, and companion `crates/rig-*` crates (see `AGENTS.md` for the full layout). `rig-core` avoids adding many dependencies and only contains simple provider integrations on top of the base abstractions; side crates add first-party behavior with heavier dependencies (for example, `rig-mongodb` depends on `mongodb`).

If you are unsure whether a side-crate should live in the main repo, spin up a personal repo containing your crate and create an issue making the case for integrating and maintaining it here.

## Developing

Setup is similar to most Rust projects:

```bash
git clone https://github.com/0xplaygrounds/rig
cd rig
cargo test
```

CI enforces both `clippy` and `fmt`. These broad commands are available for deliberate local debugging, not mandatory prepublication checks:

```bash
cargo clippy --all-features --all-targets
cargo fmt -- --check
```

Run the smallest useful local check for changed behavior and leave comprehensive testing to CI; see [DEVELOPING.md](DEVELOPING.md). Core, cassette, live, and integration test commands are in `tests/README.md`. Mention in your PR whether cassettes were replayed, recorded, or not applicable.
