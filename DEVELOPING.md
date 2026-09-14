# Development verification

`xtask/src/verify/checks.rs` defines the available checks shared by CI and
the optional local planner. The default workflow is minimal relevant local
checks, prompt authorized publication, and comprehensive GitHub CI.

Before publication, inspect the intended diff against its actual merge base,
preserve unrelated changes, check formatting/whitespace as applicable, and run
the smallest useful check for the changed behavior. A focused regression test
or narrow compile check can suffice for code. For documentation or instructions,
review the diff and links/consistency; do not run Rust compilation, full docs,
or workspace tests merely because instructions changed.

Select local checks by relevance and cost. Workspace-wide tests, all-features
builds, every provider/example, full docs/doctests, WASM matrices, Docker suites,
and dependency-floor checks belong in CI by default. Explain the concrete need
before an expensive local check; do not invent hypothetical risks to justify
a full local gate. Broad planner commands remain available for explicit requests
or deliberate debugging, not routine publication prerequisites:

```sh
cargo xtask verify --changed --dry-run             # optional selection preview
cargo xtask verify --changed                       # optional; only if selection is suitably small
cargo xtask verify --pr --base origin/feat/effect-bus   # optional broad plan; use actual base
cargo xtask verify --full                          # optional exhaustive verification
cargo xtask verify --check core-all                # explicit check by id; assess its cost first
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

`--changed` is optional, not a promise of cheap verification. Inspect its dry-run
selection before using it when it might expand broadly. If shared inputs or
other changes select the full plan, choose explicit small checks or leave
comprehensive execution to CI rather than running the selected plan locally.

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
Install only prerequisites needed for the small local checks you choose.
Expensive CI-only prerequisites are not required to publish a PR.

## Hosted CI

- `ci.yaml`, the PR gate: guards (fmt, source guards, layout, xtask's tests), release-document freeze, the default-feature type
  check, the default/bedrock sweep, the
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

The default sweep excludes root ECS/corpus parity tests and rig-verify.
`ecs-parity` owns those root tests, the golden pairing guard, and rig-verify's
separate `world_replay` target. `bus-verification` owns the remaining rig-verify
tests and compiles the explicitly invoked `run_assembly_cost` benchmark.
Default-member and standalone package graphs remain separate executions:
JSON ordering/float parsing and allocator features differ between them. Both
root parity configurations belong to `ecs-parity`, including the extracted ECS
helper regressions in `rig-test-support`; both rig-verify configurations
are split between that lane (`world_replay`) and `bus-verification` (other tests).
Default-member executions and standalone root parity retain two retries;
standalone rig-verify retains zero. The all-feature, wasm, and loom checks also
retain their distinct configuration coverage even when test names overlap.
The standalone default-feature type check selects both the facade and
`rig-test-support`, so extracted helper regression bodies still compile there.

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

Finish implementation, examples, tests, migration notes, and scope review.
Inspect the complete intended diff against its actual merge base, including
staged, unstaged, and relevant untracked files. Complete minimal relevant local
checks and independent full-diff review before authorized publication.
Validate findings; fix confirmed P0/P1 and in-scope lower-severity issues, or
document why lower-severity findings should remain. After fixes, run useful
targeted local checks and the final independent review; repeat the review/fix
loop if new confirmed P0/P1 issues appear. Do not restart the full local suite
after each fix or rerun unchanged checks without a concrete reason.

Commit/push/open a non-draft PR promptly when the task authorizes it. Do not
delay publication to duplicate CI locally. This policy does not independently
authorize commits, pushes, PR creation, merging, or comments on PRs/issues.
Do not comment on PRs/issues unless asked.

After publication, inspect required checks on the current committed head and
actionable review feedback. CI is the comprehensive verification gate before
claiming fully verified or ready to merge. Confirm selected jobs cover the task's
comprehensive acceptance criteria; report and address missing coverage within
scope. Absent or skipped coverage is not successful verification. Preserve CI
coverage, required checks, feature-isolation matrices, regression tests, and
assertions; never disable jobs or skip failing checks to speed publication.
Fix in-scope failures and feedback, run only useful targeted local checks, and
push when authorized for CI verification. Report unrelated failures or missing
prerequisites without broadening scope.

Report publication and verification separately: “PR opened; CI pending” is a
valid publication handoff. An ordinary request to open a PR does not require
waiting for all CI; continue monitoring when the task asks for completion through
CI. Do not claim fully verified or ready to merge with required checks pending
or failing, confirmed P0/P1 findings, or unresolved required work. Keep the
handoff concise: PR link, review findings, local checks actually run, CI status,
and concrete blockers.

Keep progress notes outside the repository. Generated release documents stay
untouched (see AGENTS.md).
