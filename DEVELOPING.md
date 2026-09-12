# Development verification

One list of checks, `xtask/src/verify/checks.rs`, is the whole verification
policy. CI runs each check in its own job with `cargo xtask verify --check
<id>`; locally the planner selects from the same list by what changed.

```sh
cargo xtask verify --changed                       # the edit/check loop
cargo xtask verify --changed --dry-run             # show the selection only
cargo xtask verify --pr --base origin/feat/effect-bus   # before publishing a PR
cargo xtask verify --full                          # everything, including the slow lanes
cargo xtask verify --check core-all                # one check by id
```

Every selected check executes, every time; nothing is reused or recorded
between runs. A failed step stops the run and prints which checks did not run.
Verification always replays cassettes (`RIG_PROVIDER_TEST_MODE=replay`), never
records, and never regenerates goldens.

## What each mode selects

`--changed` compares the working tree (staged, unstaged and untracked files)
to `HEAD`, or to the merge base of `--base REF`. Provider source or cassette
edits run that provider's complete test target with all facade features.
Package edits run the package's tests under all its features and compile its
reverse dependencies, including examples. Documentation edits run docs and
doctests. Shared inputs (manifests, the lockfile, the toolchain, Cargo and
nextest configuration, CI, xtask, scripts, test support, the facade source)
and unknown files select the full plan rather than guessing. An edit only to
this file selects nothing.

`--pr` requires the intended base and selects every check the PR gate runs,
plus two slow lanes when the diff touches their inputs:

- `full-tests`, the all-features workspace run with the Docker-backed storage
  suites and the facade feature-forwarding guard, for the storage crates,
  `tests/integrations/**`, `test-support/**`, `.config/**`, the root manifest
  and lockfile, xtask, the shared setup action and `slow.yaml`
  (`checks::full_lane`);
- `dependency-floors`, the workspace built at its declared minimum versions,
  for Cargo's resolver inputs: any manifest, the lockfile, the toolchain,
  `.cargo/**`, the floor checker, xtask, the setup action and `slow.yaml`
  (`checks::floor_lane`).

CI's `slow.yaml` asks the same planner (`cargo xtask verify --lanes --base
<base>`) on every PR, so local and hosted selection cannot disagree. A
source-only change that starts using an API newer than a declared floor is
caught on the merge queue, the nightly schedule or the release gate rather
than on its own PR; that tradeoff is deliberate.

`--full` runs every check, including both slow lanes.

## Prerequisites

The repository toolchain (`rust-toolchain.toml`), nextest (0.9.91 or newer
honours the test priorities in `.config/nextest.toml`; older versions warn
and run every test in default order), Clippy, rustfmt, protoc, Docker for the
storage suites, Node, wasm-bindgen-test-runner at the lock file's
`wasm-bindgen` version, and Python 3.12+ for the floor checker and its
isolation tests. The planner
probes for what the selected checks need before compiling anything; a missing
tool is a failed run.

## Hosted CI

- `ci.yaml`, the PR gate: guards (fmt, source guards, layout, scenario
  catalog, xtask's tests), release-document freeze, the default-feature type
  check, the default/bedrock sweep with the scenario registration check, the
  cross-crate guards (core-all, bus verification, macro hygiene, out-of-facade
  conformance), rig-derive, loom, doctests, docs, clippy, the wasm checks with
  the two native-only diagnostics, and the wasm test suites. Apart from the
  release-document freeze (a shell script) every job is checkout,
  `.github/actions/rust-setup`, a job-level rust-cache where a warm entry
  exists, and one or more `cargo xtask verify --check` steps.
- `slow.yaml`: a plan job runs `verify --lanes` on PRs and gates the
  all-features run and the dependency floors on its answer; the merge queue,
  the 06:00 UTC schedule, manual dispatch and the release gate run both.
- `cd.yaml`: both gates on every push to `main`, then release-plz.
- `cache-warm.yaml`: on trusted pushes, compiles the default/bedrock test
  graph (`default-test-build`, development base only) and the all-features
  test graph (`full-test-build`, shared key `all-features`, both branches)
  with nextest `--no-run`, executing nothing. Only pushes save caches; PRs
  restore their base branch's entries. rust-cache runs at job level
  everywhere so a failed or cancelled job saves nothing, and warms are never
  cancelled mid-build.

Tokens are read-only except release-plz. No job receives provider secrets:
cassettes replay with a dummy key and live tests are `#[ignore]`.

## Adding a check

Add it to `checks::all()` with a stable id, decide which selection rule owns
it in `selection.rs` (or leave it in the fast set every `--pr` runs), add a
`cargo xtask verify --check <id>` step to the matching ci.yaml job, and update
the tests in `xtask/src/verify/tests.rs` that pin ids and lanes. Checks that
shell out to a nested Cargo build (the facade guard, macro hygiene) get a
nextest priority so they start first.

## PR completion

Finish the implementation and review before the expensive final plan:
inspect the complete diff against the actual merge base, run formatting and
targeted checks, obtain an independent full-diff review and fix confirmed
findings, then run `cargo xtask verify --pr --base <intended-base>` on the
frozen tree. After a later fix, rerun the affected checks and the review.
Publish only after that; then watch committed-head CI and review threads.
Keep progress notes outside the repository. Generated release documents stay
untouched (see AGENTS.md).
