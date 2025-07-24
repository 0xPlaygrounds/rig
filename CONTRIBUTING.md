# Contributing to Rig

Thank you for considering contributing to Rig! Here are some guidelines to help you get started.

General guidelines and requested contributions can be found in the [How to Contribute](https://docs.rig.rs/docs/how_to_contribute) section of the documentation.

## Issues
Before reporting an issue, please check existing or similar issues that are currently tracked.

Additionally, please ensure that if you are submitting a bug ticket (ie, something doesn't work) that the bug is reproducible. If we cannot reproduce the bug, your ticket is likely to be marked either `wontfix` or closed (although it's likely we'll take note of it in case there's a secondary occurrence).

## Pull Requests

Contributions are always encouraged and welcome. Before creating a pull request, create a new issue that tracks that pull request describing the problem in more detail. Pull request descriptions should include information about it's implementation, especially if it makes changes to existing abstractions.

PRs should be small and focused and should avoid interacting with multiple facets of the library. This may result in a larger PR being split into two or more smaller PRs. Commit messages should follow the [Conventional Commit](https://conventionalcommits.org/en/v1.0.0) format (prefixing with `feat`, `fix`, etc.) as this integrates into our auto-releases via a [release-plz](https://github.com/MarcoIeni/release-plz) Github action.

Unless the PR is for something minor (ie a typo), please ensure that an issue has been opened for the feature or work you would like to contribute beforehand. By opening an issue, a discussion can be held beforehand on scoping the work effectively and ensuring that the work is in line with the vision for Rig. Without any linked issues, your PR may be liable to be closed if we (the maintainers) do not feel that your PR is within scope for the library.

It is also highly suggested to comment on issues you're interested in working on. By doing so, it allows others to see that something is being worked on and therefore avoids frustrating situations, such as multiple contributors opening a PR for the same issue. In such a case, any duplicate PRs will be closed unless it is clear that the original contributor is unable to continue the work.

You can link your PR back to a given issue by writing the following in your PR message:
```md
Fixes #999
```

This will then auto-link issue 999 (for example) and will automatically close the issue once the PR has been merged.

**Working on your first Pull Request?** You can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://kcd.im/pull-request)

## Project Structure

Rig is split up into multiple crates in a monorepo structure. The main crate `rig-core` contains all of the foundational abstractions for building with LLMs. This crate avoids adding many new dependencies to keep to lean and only really contains simple provider integrations on top of the base layer of abstractions. Side crates are leveraged to help add important first-party behavior without over burdening the main library with dependencies. For example, `rig-mongodb` contains extra dependencies to be able to interact with `mongodb` as a vector store.

If you are unsure whether a side-crate should live in the main repo, you can spin up a personal repo containing your crate and create an issue in our repo making the case on whether this side-crate should be integrated in the main repo and maintained by the Rig team.


## Developing

### Setup

This should be similar to most Rust projects.

```bash
git clone https://github.com/0xplaygrounds/rig
cd rig
cargo test
```

### Clippy and Fmt

We enforce both `clippy` and `fmt` for all pull requests.

```bash
cargo clippy --all-features --all-targets
cargo fmt -- --check
```

If you have the `just` task runner installed, you can also run `just` (or `just ci`).

### Tests

Make sure to test against the test suite before making a pull request.

```bash
cargo test
```
