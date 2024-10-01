# Contributing to Rig

Thank you for considering contributing to Rig! Here are some guidelines to help you get started.

## Issues

Before reporting an issue, please check existing or similar issues that are currently tracked.

## Pull Requests

Contributions are always encouraged and welcome. Before creating a pull request, create a new issue that tracks that pull request describing the problem in more detail. Pull request descriptions should include information about it's implementation, especially if it makes changes to existing abstractions.

PRs should be small and focused and should avoid interacting with multiple facets of the library. This may result in a larger PR being split into two or more smaller PRs. Commit messages should follow the [Conventional Commit](conventionalcommits.org/en/v1.0.0) format (prefixing with `feat`, `fix`, etc.) as this integrates into our auto-releases via a [release-plz](https://github.com/MarcoIeni/release-plz) Github action.

## Project Structure

Rig is split up into multiple crates in a monorepo structure. The main crate `rig-core` contains all of the foundational abstractions for building with LLMs. This crate avoids adding many new dependencies to keep to lean and only really contains simple provider integrations on top of the base layer of abstractions. Side crates are leveraged to help add important first-party behavior without over burdening the main library with dependencies. For example, `rig-mongodb` contains extra dependencies to be able to interact with `mongodb` as a vector store.

If you are unsure whether a side-crate should live in the main repo, you can spin up a personal repo containing your crate and create an issue in our repo making the case on whether this side-crate should be integrated in the main repo and maintained by the Rig team.


## Developing

### Setup

This should be similar to most rust projects.

```bash
git clone https://github.com/0xplaygrounds/rig
cd rig
cargo test
```

### Clippy and Fmt

We enforce both `clippy` and `fmt` for all pull requests.

```bash
cargo clippy -- -D warnings
```

```bash
cargo fmt
```


### Tests

Make sure to test against the test suite before making a pull request.

```bash
cargo test
```
