# Development verification

Use the Rust planner in `xtask` for the ordinary edit/check loop:

```sh
cargo xtask verify --changed
cargo xtask verify --changed --dry-run
cargo xtask verify --pr --base origin/feat/effect-bus
cargo xtask verify --full
```

`--changed` selects checks for staged, unstaged and untracked working changes.
With `--base REF`, it additionally includes the complete diff from that ref's
merge base. Without it, the comparison is explicitly to `HEAD`, not to `main`.
Provider edits and cassette-only edits run the complete provider target,
with all facade features, including audio/image/websocket tests and safety
assertions. This uses the existing all-feature configuration rather than
maintaining a second list of provider feature gates. Package edits run package tests and compile
Cargo's reverse dependencies, including examples; dependent tests and the
complete feature/platform matrix belong to PR/full validation. Shared support,
manifests, dependencies, verification tooling, and unknown inputs broaden the
plan conservatively. Read the printed selection and skip reasons.

`--pr` requires an explicit intended base and includes every existing fast CI
check. Changes affecting the full lane also select full verification. It is the
local execution part of preparing a PR, not the independent review or remote CI
gate. `--full` executes supported workspace tests, all example/consumer targets,
feature combinations, documentation, dependency-floor checks, concurrency
models, and the native/WASM matrix. Ignored tests, including live-provider,
model, and service tests, remain opt-in; neither mode records cassettes or
regenerates goldens.

Full validation needs the repository toolchain, nextest, Clippy, rustfmt,
protoc, Docker for service integrations, Node, and wasm-bindgen-test-runner
matching the lockfile. The dependency-floor script and its isolation tests also need Python 3.11+.
CI installs its required tools through `.github/actions/rust-setup`. A missing
prerequisite is a failed/incomplete check, never successful verification.

## Narrow manual checks

```sh
cargo nextest run --locked -p rig --features bedrock --test anthropic --retries 0
cargo test --locked -p rig-ecs --all-features
cargo test --locked -p rig-service-tests --features sqlite --test integrations
cargo xtask verify --check core-all
```

The eight vector-store suites still live under `tests/integrations/`, but their
unpublished runner is `rig-service-tests`. This removes their database/container
dev-dependencies from `-p rig` provider builds. Their identities change from
`rig::integrations::<service>::<test>` to
`rig-service-tests::integrations::<service>::<test>`; test functions and assertions
are retained. Bedrock integration tests remain in the root `rig` package.
The two LanceDB tests use separate temporary stores, cleaned up automatically,
so running the new package does not create database files in the repository.
The current agent/ECS scenario catalog contains no service-integration mappings.
The service runner is a default workspace member and full CI enables all its
features. `cargo test -p rig --all-features` alone no longer runs these services.

## Build and result reuse

Use a stable target directory and a few configurations: native default/bedrock,
native all-features, and browser WASM. Local debug/incremental settings remain
unchanged. The planner starts Cargo children serially and uses an OS lock to
avoid two planners competing in the same target directory. It never stops an
unrelated Cargo process; Cargo may still wait for another process's build lock.
Do not run independent verification planners against the same build artifacts.

On macOS, a heavily accumulated `target/debug/deps` directory can also slow
test execution: system proxy discovery calls CoreFoundation bundle discovery,
which enumerates the executable directory. A fresh
or smaller target can help in that situation; use one deliberately for the
verification session, rather than a new directory for every check. This is an
observed platform/cache limitation, not a change to Rig's proxy support.

Changed mode can reuse successful local checks; `--no-reuse` forces execution.
PR mode executes by default (`--reuse` is an explicit local opt-in); full mode
and CI always execute. `full-tests` and `dependency-floors` always execute even
with `--reuse`, because external services and dependency resolution can change.
`doctests` and `package-rig-fastembed` also always execute and write no success
receipt: their model-loading examples depend on mutable external model files.
These commands explicitly set `HF_HOME` and `FASTEMBED_CACHE_DIR` to
`<target>/verify/fastembed-cache`. A pre-existing ignored default model cache is
moved there before fingerprints, preserving downloads and internal relative
links. Conflicts, tracked caches, and unsafe links fail before checks. Repository
source inputs remain checked before and after these fresh executions.
Result files under `target/verify/` are disposable and
contain the latest success fingerprint, per-input hashes, configuration hash, and
measured elapsed time for each check. Per-check `.log` files contain subprocess
output and phase boundaries; internal source guards print diagnostics on the
console. The active subprocess emits a heartbeat every 15 seconds, without an
ETA. Dry runs explain selection, fresh-execution policy, and receipt mismatches.
Cheap prerequisite probes precede execution. Selected nested compile fixtures
resolve their ignored Cargo lockfiles before any check fingerprints are taken;
these lockfiles remain verification inputs. This avoids invalidating earlier
checks when a nested Cargo invocation first creates or updates its lockfile.
The six known ignored fixture locks select their owning tests in changed mode;
unknown generated inputs still select conservative full coverage.
Failure or catchable interruption
prints remaining work and one continuation command; the planner never loops
automatically. Uncatchable termination (such as SIGKILL or power loss) cannot
print a summary, but the active check has no success receipt.
There is no evidence archive, download step, historical review log, or parity
verdict. Deleting the files simply causes fresh verification.

Fingerprints include commands, repository source/fixture/lock/configuration
bytes, test selection, environment, tool versions, ancestor/user Cargo
configuration, and user nextest configuration. A changed input invalidates reuse; a failed rerun removes prior
success. Inputs changing during a check prevent recording a reusable result.
Unsupported inputs such as symlinks execute fresh without a reusable receipt.
This guide is reporting-only and is not a compilation or test input; a change
only to `DEVELOPING.md` selects no checks in changed mode. Other Markdown,
including tracked progress files, remains conservatively covered. PR/full
requirements are unchanged by this reporting-only exception.

Dependency floors resolve freshly in a temporary sibling copy of the current
working files, including uncommitted and ignored generated inputs. Only Git
metadata and known build directories are omitted; symlinks and external path
dependencies fail explicitly. The caller's lockfile is never written or restored.
Floor builds use the stable `<target>/dependency-floors` directory, so lowered
dependency artifacts do not displace the normal locked build configuration.
`python3 scripts/check-dependency-floors.py --keep` retains the temporary source
and lowered lockfile for inspection, without modifying the caller. Resolution
and compilation have separate measured durations. Unreachable floors retain
explicit lower-bound caveats. A hard kill can leave a temporary source copy;
its printed path identifies it, and it is never reused as a successful check.

CI uses the same check definitions in `xtask/src/verify/checks.rs`, while retaining
separate jobs for parallelism and platform setup. The guards nextest profile
runs telemetry and span-safety tests without retries in the existing all-feature
run. Other tests retain their prior retry policy. Command-line retry settings
would override these per-test rules, so that run deliberately does not pass
`--retries`. See [nextest retry precedence](https://nexte.st/docs/features/retries/).

Only trusted upstream pushes may write shared caches. One warming job targets
`feat/effect-bus` and the default/bedrock test configuration; fork PRs and PR merge
refs remain read-only. PRs can restore their base branch's caches under
[GitHub's cache visibility rules](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching).
Other matrix configurations are not multiplied into development-base caches.
The warmer's explicit `default-test-build` check compiles the identical
default/bedrock artifacts with nextest `--no-run`; it makes no test-execution
claim and is never substituted for `default-tests` in verification. Conformance
passes explicit Cargo test-target selectors matching its existing nextest
filter, preserving the four executed binaries and their package/feature graph
while avoiding compilation of unused harnesses.
Cache warming still consumes runner time and quota; its benefit can only be
measured after the workflow lands and trusted pushes populate that base cache.

## PR completion

Finish implementation and review before starting expensive final verification:

1. Inspect the complete intended diff against the actual merge base, including
   staged, unstaged, deleted, renamed, and relevant untracked files.
2. Complete implementation, examples, tests, migration notes, and scope review.
   Generated release documents remain subject to the repository's release policy.
3. Run formatting, targeted tests, and cheap checks; resolve findings.
4. Obtain a fresh independent full-diff review and fix confirmed findings before
   launching expensive final checks. Validate findings against current code;
   fix P0/P1 issues and address or justify in-scope lower-severity findings.
5. Freeze verification inputs and run the required final plan. Keep progress
   logs and reports outside the repository (for example in a sibling reviews
   directory). Freezing inputs does not require committing unfinished work.
6. If a later fix is needed, rerun affected checks and the final review. Use
   `cargo xtask verify --pr --base <intended-base-ref> --reuse` to continue with
   matching successes under the reuse rules above. Broad input changes may
   invalidate every result; `full-tests` and `dependency-floors` must still run
   fresh. Report every mandatory fresh check still outstanding.

This reduces avoidable restarts; defects can still require another run. Never
count a successful command on old inputs as verification of the current tree.
Publish a normal PR only after initial verification and review. Inspect
committed-head required CI and unresolved actionable review threads afterward.
Pending/failed checks and unresolved required work mean the PR is incomplete.
Report unrelated blockers without broadening the change to fix them.
