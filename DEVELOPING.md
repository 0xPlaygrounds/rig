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
matching the lockfile. The existing dependency-floor script also needs Python.
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
which enumerates the executable directory. Profiling the old local cache found
this path dominating cassette startup with 384,122 directory entries. A fresh
or smaller target can help in that situation; use one deliberately for the
verification session, rather than a new directory for every check. This is an
observed platform/cache limitation, not a change to Rig's proxy support.

Changed mode can reuse successful local checks; `--no-reuse` forces execution.
PR mode executes by default (`--reuse` is an explicit local opt-in); full mode
and CI always execute. `full-tests` and `dependency-floors` always execute even
with `--reuse`, because external services and dependency resolution can change.
Result files under `target/verify/` are disposable and
contain only the latest success fingerprint and elapsed time for each check.
There is no evidence archive, download step, historical review log, or parity
verdict. Deleting the files simply causes fresh verification.

Fingerprints include commands, repository source/fixture/lock/configuration
bytes, test selection, environment, tool versions, ancestor/user Cargo
configuration, and user nextest configuration. A changed input invalidates reuse; a failed rerun removes prior
success. Inputs changing during a check prevent recording a reusable result.
Unsupported inputs such as symlinks execute fresh without a reusable receipt.
This guide is reporting-only and is not a compilation or test input.

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
Cache warming still consumes runner time and quota; its benefit can only be
measured after the workflow lands and trusted pushes populate that base cache.

## PR completion

Inspect the complete intended diff against its merge base, including staged,
unstaged and relevant untracked files. Run targeted checks first, then the full
required PR plan. Obtain a fresh independent review of the complete diff;
validate findings, fix confirmed P0/P1 issues and in-scope lower-severity issues,
and rerun affected checks and review after fixes. Publish a normal PR only after
initial verification and review. Inspect committed-head required CI and
unresolved actionable review threads afterward. Pending/failed checks and
unresolved required work mean the PR is incomplete. Report unrelated blockers
without broadening the change to fix them.

## Measurements

Initial comparable cold check, Rust 1.95.0, aarch64-apple-darwin, native dev
profile with normal local debuginfo/incremental settings and default features:

| Workflow | Before | After | Configuration |
| --- | ---: | ---: | --- |
| Cold narrow provider type-check | 44.13s | 27.05s | `cargo check --locked -p rig --test anthropic --timings` |
| Warm no-op check | 0.435s | 0.355s | `cargo check --locked -p rig --test anthropic` |
| Warm ECS implementation edit | 0.933s | 0.950s | `cargo check --locked -p rig-ecs --all-features --lib` |
| Warm single cassette replay | 1.012s | 0.922s | Command below |
| Warm fixture-only edit | 0.984s | 0.900s | Same replay command |
| Warm example implementation edit | 0.382s | 0.381s | `cargo check --locked -p agent_no_tokio` |
| Warm Rust documentation edit | 2.279s | 2.217s | `cargo doc --locked -p rig-ecs --all-features --no-deps` |

The baseline is `de065e9be`, the same source tree as `45216c03e`. Body-edit
and planner measurements use candidate `6aafd32ef`. The cold and other warm
measurements used the preceding working candidate with the same narrow-provider
dependency split, before temporary-store and Git-fixture cleanup. Both cold
measurements used initially empty dedicated target directories, serially, on
the same machine with the registry already available. This measures compilation,
not registry downloads, provider latency, or CI cache restoration. It is one
sample (about 39% lower wall time), not a universal speedup guarantee.

Warm values are medians of three runs on an Apple M2 Max with 64 GiB RAM,
nextest 0.9.67. Each checkout started with an empty target directory and ran
the same configuration warmups before measurement. No measured run reported
a Cargo lock wait. The ECS implementation edit changed `WorldOutcome::order`
to a saturating increment; the example changed its frame duration from 16 to
17 milliseconds. These were compile-only edits, restored byte-for-byte after
each trial. The documentation edit added a crate-doc line; the fixture edit
added a harmless response header to the consumed cassette. The ECS result is
a small regression in this sample; the example/docs differences do not establish
a meaningful speedup. These direct Cargo measurements isolate
the dependency boundary and are not timings of the complete planner checks.

The actual planner, using the existing local validation target at `6aafd32ef`,
took a median **0.274s** for a clean `cargo xtask verify --changed` (three runs).
The example implementation edit selected formatting, the example test harness,
and consumer compilation: its first execution took **32.301s**, then three
identical invocations reused all three checks in a median **1.843s**. The first
execution included compilation with the existing cache; it is not a comparable
cold-build sample or evidence that a changed input may reuse stale success.
The tests separately exercise fixture/configuration/command invalidation.

Local validation passed formatting, source guards, 38 tooling tests and strict
Clippy, 7,941 default tests (203 skipped), 8,306 all-feature workspace tests
(213 skipped), core/bus/conformance/derive/macro-hygiene checks, documentation,
loom, all-target workspace compilation, eight WASM compilation configurations,
all three executed WASM suites, and both expected native-only diagnostics.
The final full suite ran in 653.588s with no failures or retries. Dependency-floor
validation passed with 52 downgrades, 24 already at floor, and 44 floors constrained
by transitives and checked at the lowest admitted versions. Node was v24.10.0;
the executed WASM suites used an isolated wasm-bindgen runner 0.2.118 matching
the lockfile. Required committed-head CI remains a separate PR gate.

The single-test replay command (native default features plus bedrock) was:

```sh
cargo nextest run --locked -p rig --features bedrock --test anthropic --retries 0 \
  -E 'test(ecs_endings::tool_call_delta_stop_effect_log_is_the_golden_fixture)'
```

The baseline committed-head GitHub runs were
[Lint & Test](https://github.com/0xPlaygrounds/rig/actions/runs/34055590165)
and [Nightly Full Test](https://github.com/0xPlaygrounds/rig/actions/runs/34055590122),
on Ubuntu x64 with Rust 1.95.0, incremental compilation disabled and
`line-tables-only` dev/test debuginfo. Neither test job restored a Rust cache;
lookup took about 0.17s in the default job and 0.05s in the all-feature job.
The default job reported zero sccache hits. Default test compilation took
7m00s and execution 289.043s (7,942 passed); all-feature compilation took
10m58s and execution 575.039s (8,025 passed). The complete all-feature job
took 21m24s, including setup and post steps. These are CI measurements,
not comparable to local Mac compilation or claims about the unpopulated
development-base cache.
