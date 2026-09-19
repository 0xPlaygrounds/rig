# Development verification

`xtask/src/verify/checks.rs` defines the available checks shared by CI and
the optional local planner. The default workflow is minimal relevant local
checks, independent review, prompt authorized publication, and comprehensive
GitHub CI. Broad local verification is not a prerequisite for creating or
updating a PR.

Before publication, inspect the intended diff against its actual merge base,
preserve unrelated changes, check formatting/whitespace as applicable, and run
the smallest useful check for the changed behavior. A focused regression test
or narrow compile check can suffice for code. For documentation or instructions,
review the diff and links/consistency; do not run Rust compilation, full docs,
or workspace tests merely because instructions changed.

Workspace-wide tests, all-features builds, every provider/example, full
docs/doctests, WASM matrices, Docker suites, and dependency-floor checks belong
in CI by default. Explain the concrete need before an expensive local check;
do not invent hypothetical risks to justify a full local gate. Install only the
prerequisites the small local checks you choose need. The planner commands
remain available for explicit requests or deliberate debugging:

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

## Choosing a focused check

| Change | Useful local evidence |
| --- | --- |
| Documentation or agent instructions only | Full diff, relative links, and consistency with the referenced code/configuration; no Rust build |
| Test implementation | Run the affected tests; use package-scoped Clippy for new lint-sensitive patterns |
| Public API or feature boundary | Focused behavior test and compilation of known consumers in the affected configuration |
| Target-specific implementation | Inspect both branches; check the affected target when available, and distinguish source review from executed checks |
| CI failure | Read the failed job's diagnostics, fix the cause, then run the narrow owning check |

These are selection guidelines, not a checklist to run in full. Command recipes
and test-specific pitfalls live in [tests/README.md](tests/README.md#core-tests).
Serialize Cargo commands sharing a target directory unless they can make real
progress independently; concurrent commands often just wait for the build lock.

## Reading CI failures

Start with check names and status on the current PR head, then inspect only
failed-job diagnostics and enough surrounding context to understand the failure.
Retain the fetched log outside the repository; search it instead of repeatedly
fetching or printing the entire workflow log.

If `gh run view --log-failed` cannot read a completed job while its workflow is
still running, fetch that job directly:

```sh
gh api repos/0xPlaygrounds/rig/actions/jobs/JOB_ID/logs
```

Capture the response before extracting diagnostics; replace `JOB_ID` with the
failed job's ID. A truncated log is not evidence that later errors are absent.

## What each mode selects

`--changed` compares the working tree (staged, unstaged and untracked files)
to `HEAD`, or to the merge base of `--base REF`. Provider source or cassette
edits run that provider's complete test target with all facade features.
Package edits run the package's tests under all its features and compile its
reverse dependencies, including examples. Documentation edits run docs and
doctests. Shared inputs (manifests, the lockfile, the toolchain, Cargo and
nextest configuration, CI, xtask, scripts, test support, the facade source)
and unknown files select the full plan rather than guessing. An edit only to
this file selects nothing. Because of that, `--changed` is not a promise of
cheap verification: inspect its dry-run selection first, and if it selects the
full plan, choose explicit small checks or leave execution to CI.

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
isolation tests. The planner probes for what the selected checks need before
compiling anything; a missing tool is a failed run.

## Hosted CI

- `ci.yaml`, the PR gate: guards (fmt, source guards, layout, xtask's tests),
  release-document freeze, the default-feature type check, the default/bedrock
  sweep, the cross-crate guards (core-all, bus verification, macro hygiene,
  out-of-facade conformance), rig-derive, loom, doctests, docs, clippy, the
  wasm checks with the two native-only diagnostics, and the wasm test suites.
  Apart from the release-document freeze (a shell script) every job is
  checkout, `.github/actions/rust-setup`, a job-level rust-cache where a warm
  entry exists, and one or more `cargo xtask verify --check` steps.
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

Every cassette-backed test source lives in `crates/rig-cassette`: the provider
targets, the ECS/corpus parity cells inside them, the cache-prefix guard, and
the verification targets `verify` and `world_replay`, and the effect-log and
runtime-adapter library regressions. The unpublished `rig-cassette-minimal`
runner in `tests/minimal/Cargo.toml` shares those verification entrypoints and
the effect-log/classic-replay regression sources. It enables only cassette's
`agent,ecs` features, without native HTTP, the facade or provider helpers.
The facade keeps live-only provider suites, `core`, feature guards and
integrations; it re-exports cassette's logs and enables its classic adapter
with the facade's `agent` feature. Both runtimes remain free of normal
dependencies on the concrete cassette/log implementation.

The default sweep excludes the parity cells and the two verification targets;
the cassette library's own tests and every provider target stay in it.
`ecs-parity` owns the parity cells (both configurations, including the
extracted ECS helper regressions in `rig-test-support`), the golden pairing
guard that stayed in the facade's `core` target, and the separate
`world_replay` target; `bus-verification` owns `verify` and the minimal
`effect_log` regression target. `core-all` retains the migrated classic replay
unit tests under its all-features graph. Provider parity excludes the
two verification binaries from the `corpus_`/`ecs_` pattern, whose module names
would otherwise match. The standalone parity filter also explicitly includes
the extracted `rig-test-support` regressions. Minimal verification executions
select `package(rig-cassette-minimal)`; their `serde_json` graph has neither
`preserve_order` nor `float_roundtrip`, unlike the unified executions through
`rig-cassette`. The dependency-graph guard checks the resolved features of the
packages selected by the CI commands, not just their names. Default-member
executions and standalone parity retain two retries; minimal verification
retains zero. The minimal runner is outside default-members and excluded from
`full-tests`, where its shared sources already execute through `rig-cassette`.
Shared sources under cassette's `tests/`, `src/effect_log/` and
`src/agent/replay*` also select the minimal runner in changed-file plans.
`wasm-rig-cassette` builds the minimal library and the independently enabled
`agent`, `ecs` and combined integrations; the HTTP engine is native-only.
The wasm and loom checks retain their distinct configuration coverage even
when test names overlap. The standalone default-feature type check selects both the
facade and `rig-test-support`, so extracted helper regression bodies still
compile there. A `provider-<name>` check is planned against whichever package
declares that target.

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
document why lower-severity findings should remain. After fixes, rerun only
useful targeted checks and the independent review; repeat if new confirmed
P0/P1 issues appear. Do not restart the full local suite after each fix.

Commit/push/open a non-draft PR promptly when the task authorizes it; do not
delay publication to duplicate CI locally. This policy does not independently
authorize commits, pushes, PR creation, merging, or comments on PRs/issues.
Do not open draft PRs or comment on PRs/issues unless asked.

After publication, inspect required checks on the current committed head and
actionable review feedback. CI is the comprehensive verification gate. Confirm
selected jobs cover the task's acceptance criteria; report and address missing
coverage within scope — absent or skipped coverage is not successful
verification. Preserve CI coverage, required checks, feature-isolation
matrices, regression tests, and assertions; never disable jobs or skip failing
checks to speed publication. Fix in-scope failures with targeted checks and
push when authorized; report unrelated failures or missing prerequisites
without broadening scope.

Report publication and verification separately: "PR opened; CI pending" is a
valid publication handoff, and an ordinary request to open a PR does not require
waiting for CI; continue monitoring when the task asks for completion through
CI. Never claim fully verified or ready to merge with required checks pending
or failing, confirmed P0/P1 findings, or unresolved required work. Keep the
handoff concise: PR link, review findings, local checks actually run, CI status,
and concrete blockers. Keep one verification ledger outside the repository
when needed; reuse its commands/results in the PR and handoff rather than
maintaining duplicate progress reports. Generated release documents stay
untouched (see `AGENTS.md`).
