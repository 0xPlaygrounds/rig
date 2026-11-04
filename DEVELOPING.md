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
prerequisites the small local checks you choose need. Iterate on one package
with the `local` nextest profile ([Local builds](#local-builds)).

Before pushing, run what CI's plan for the change selects (`--affected`): the
tests of every package that can observe the change, each with only the
features that observe it, and every other check whose build compiles a changed
package. The Docker suites and the macOS job are left to CI. A change to a
shared input (the workspace manifest, the lockfile, CI, xtask, nextest or Cargo
configuration, test support, scripts) selects every check; `--dry-run` shows the
selection first.

```sh
cargo xtask verify --affected --base origin/main   # before every push
```

Other modes, for explicit requests or deliberate debugging:

```sh
cargo xtask verify --plan --base origin/main       # print the plan CI would run
cargo xtask verify --full                          # every check; assess its cost first
cargo xtask verify --check core-all                # one check by id, in full
```

Every selected check executes, every time; nothing is reused or recorded
between runs. Locally a failed step stops the run and prints which checks did
not run; a CI job runs each of its selected checks and fails naming every
failed one.
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

## What the plan selects

`xtask/src/verify/plan.rs` reads the diff from the merge base of `--base`
(staged, unstaged and untracked files) and the workspace's `cargo metadata`:

- Each changed path belongs to the package whose directory holds it; the
  facade owns only `src/`, `tests/`, `benches/` and `README.md` at the root.
  A path in no package, or a shared input, runs every check in full. Markdown
  files at the root (but README.md), `docs/` and `prompts/` run only the tree
  guards.
- The tests (`tests`), doctests, docs and Clippy run over every package that
  can observe a changed one: a changed package with all its features; any
  other package with its default features plus each feature that brings in a
  package depending on a changed one, or adds features to a changed package.
  So a vector-store change runs the store, the facade with only that store's
  feature, and `rig-service-tests` with only that store's Docker suite.
  Dev-dependencies count for a package's own tests only, never transitively.
- A package whose sources name a path outside it (`"../rig-cassette/fixtures"`
  from its manifest directory, or an `include_str!`) reads it: a change there
  runs that package's own tests with all features.
- Every other check names what it builds in `plan::builds` and runs when one
  of those packages compiles a changed one. A member manifest also selects the
  dependency floors and the compile guards.

The merge queue, the nightly schedule, manual dispatch and every push to
`main` run the full plan. A source-only change that starts using an API newer
than a declared floor is caught there rather than on its own PR; that tradeoff
is deliberate.

## Local builds

- `cargo nextest run --locked --profile local -p <crate> [filter]` is the inner
  loop. The `local` profile has no retries, stops at the first failure, and
  skips the service suites and the trybuild cases (`--ignore-default-filter`
  runs them). Tests that launch a nested Cargo build are `#[ignore]`d; their
  `guard-*` checks run them with `--run-ignored only`.
- The dev profile keeps line tables for workspace code and no debuginfo for
  dependencies. For a debugger, add `--config 'profile.dev.debug=true' --config
  'profile.dev.package."*".debug=true'`; `CARGO_PROFILE_DEV_DEBUG` alone does
  not reach dependencies.
- Keep incremental compilation on; CI sets `CARGO_INCREMENTAL=0` because its
  runners start fresh.
- Worktrees of this repository can share one target directory: set
  `build.target-dir` in a `.cargo/config.toml` above them. Dependencies then
  build once instead of once per worktree; builds that run at the same time
  wait on the directory's lock.
- Give rust-analyzer its own target directory
  (`rust-analyzer.cargo.targetDir: true`) so it does not hold that lock while
  you build.
- On macOS, add your terminal to System Settings > Privacy & Security >
  Developer Tools; otherwise each new test binary is scanned on its first run.

## Prerequisites

The repository toolchain (`rust-toolchain.toml`), nextest (0.9.130 or newer,
for the profile inheritance in `.config/nextest.toml` and the thread count it
gives tests), Clippy, rustfmt, protoc, Docker for the
storage suites, Node, wasm-bindgen-test-runner at the lock file's
`wasm-bindgen` version, and Python 3.12+ for the floor checker and its
isolation tests. The planner probes for what the selected checks need before
compiling anything; a missing tool is a failed run.

## Hosted CI

- `ci.yaml`, the gate: a plan job (`verify --plan`) and one job per group of
  checks (`checks::job`): guards, the release-document freeze, the runtime
  suite (`stable / test`), the Docker suites (`stable / services`), the
  cross-crate tripwires (`stable / test guards`), the effect corpus, the
  compile guards, doctests, docs, Clippy, the default-feature type check, the
  wasm checks, kernel-only rig-ecs, the dependency floors and macOS. Each job
  runs `cargo xtask verify --job <id>`, and is skipped when the plan selects
  none of its checks. `stable / required` fails unless every planned job
  succeeded and every other job was skipped: it is the one context branch
  protection and the merge queue require.
- `cd.yaml`: the full gate on every push to `main`, then release-plz. Only
  these pushes save caches; PRs restore their base branch's entries.
  rust-cache runs at job level everywhere so a failed or cancelled job saves
  nothing.

Each package links its integration tests into one binary, `integration`, with
one module per suite (see [tests/README.md](tests/README.md)). Every
cassette-backed test source lives in `crates/rig-cassette`: the provider
suites, the agent corpus cells inside them, the cache-prefix guard, the
verification suite `verify`, and the effect-log and runtime-adapter library
regressions. The unpublished `rig-cassette-minimal` runner in
`tests/minimal/Cargo.toml` shares the verification suite and the
effect-log/classic-replay regression sources. It enables only cassette's
`agent` feature, without native HTTP, the facade or provider helpers.
The facade keeps live-only provider suites, `core`, feature guards and
integrations; it re-exports cassette's logs and enables its classic adapter
with the facade's `agent` feature. The agent runtime remains free of normal
dependencies on the concrete cassette/log implementation.

`effect-corpus` owns the corpus cells (both configurations, including the
extracted `corpus_` regressions in `rig-test-support`) and the golden pairing
guard that stayed in the facade's `core` suite; `bus-verification` owns
`verify` and the minimal `effect_log` regression suite. `core-all` retains
the migrated classic replay unit tests under its all-features graph. The
corpus filter excludes the `verify` suite from the `corpus_` pattern, whose
module names would otherwise match. Minimal verification executions select
`package(rig-cassette-minimal)`, whose graph has no native HTTP engine, facade
or provider helpers. The dependency-graph guard checks the resolved features
of the packages selected by the CI commands, not just their names.
Default-member executions and the standalone corpus run retain two retries;
minimal verification retains zero. The minimal runner is outside
default-members and excluded from `tests`, where its shared sources
already execute through `rig-cassette`. `wasm-rig-cassette` builds the minimal library
and the `agent` integration; the HTTP engine is native-only. The wasm checks
retain their distinct configuration coverage even when test names overlap. The standalone default-feature type check selects both the facade and
`rig-test-support`, so extracted helper regression bodies still compile
there.

Tokens are read-only except release-plz. No job receives provider secrets:
cassettes replay with a dummy key and live tests are `#[ignore]`.

## Adding a check

Add it to `checks::all()` with a stable id, give it a job in `checks::job`
(a new job also goes in ci.yaml and in `stable / required`'s `needs`), and
say what it builds in `plan::builds` (or make it package-scoped in
`plan::SCOPED`, or always-on in `plan::always`). The tests in
`xtask/src/verify/plan/tests.rs` check that every check's job runs it. A test
that shells out to a nested Cargo build is `#[ignore]`d with a `guard-*`
check of its own (`checks::guard`).

## Refreshing the model catalog

`cargo xtask catalog sync` regenerates `crates/rig-core/src/catalog/models.json`
from models.dev and `xtask/src/catalog/review.json` (module docs:
`xtask/src/catalog.rs`; the schema: `crates/rig-core/TYPED_OPTIONS.md`
section 8). Save models.dev once, so every run reads the same data:

```sh
curl -sSfL https://models.dev/api.json -o ../api.json   # curl.exe in PowerShell
cargo xtask catalog sync --from ../api.json
```

- **A lost capability stops the sync**, which then writes nothing and names
  each row and the key that answers it. Answer it in review.json on that row:
  set the key to the old value if models.dev is wrong, or to the new one if it
  is right, with a `source`. A value a join copies (Bedrock and Vertex from
  Anthropic, OpenRouter's reasoning translation, `gemini-grpc` from Gemini)
  is no answer, so each of those rows needs its own. Keeping the old value
  makes the loss disappear. For a row models.dev lists, a kept value that
  differs from models.dev's shows under "Reviewed values models.dev
  disagrees with"; a mirror row's or a `rig` fact's does not.
- **A `source`** says what proved the fact: a doc URL, a cassette, a live
  check. Or it starts with `unverified:` and says where the fact came from
  (`unverified: copied from openai/gpt-6.1-sol`). Never claim a check that was
  not made. Every review row a change adds or edits needs its own when it
  sets a capability, `limit` or a `rig` fact; one that only copies a row with
  `from` may take the copied row's. The old rows `unsourced.txt` lists are
  held to it once they change.
- **Check each list under "Reasoning controls now listed"**: rig sent any
  reasoning control to those models before and now refuses what the list
  lacks, so a wrong list there is a new refusal.
- **Paste the report** sync prints into the PR under "Catalog changes". It
  goes to stdout and refusals to stderr, so
  `cargo xtask catalog sync --from ../api.json > ../catalog-changes.md`
  captures the report alone. A later sync reports only what changed since
  the last one; to see the whole refresh again, restore main's catalog
  first: `git checkout origin/main -- crates/rig-core/src/catalog/models.json`.
- **Decide every removed row in the PR**: retired (say how you know), or kept
  with a review row that carries its facts and a `name`. A public model
  constant naming it is deleted or kept to match;
  `tests/core/catalog_constants.rs` enforces it.
- **A reviewed fact that changes what rig sends or refuses** gets a focused
  behaviour test.
- **Review the request-body golden against the report** before regenerating
  it (`tests/core/request_bodies.rs`): every change there should trace to a
  line of the report.

## PR completion

Finish implementation, examples, tests, migration notes, and scope review.
Inspect the complete intended diff against its actual merge base, including
staged, unstaged, and relevant untracked files. Complete minimal relevant local
checks and independent full-diff review before authorized publication.
Validate findings; fix confirmed P0/P1 and in-scope lower-severity issues, or
document why lower-severity findings should remain. After fixes, rerun only
useful targeted checks and the independent review; repeat if new confirmed
P0/P1 issues appear. Do not restart the full local suite after each fix.

Commit/push/open a PR promptly when the task authorizes it; do not delay
publication to duplicate CI locally. This policy does not independently
authorize commits, pushes, PR creation, merging, or comments on PRs/issues.
Do not comment on PRs/issues unless asked.

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
