# ECS consumer recording and replay

The headless maintenance consumer inspects a disposable `greeting.txt`, proposes
an edit, applies the host's approval decision, writes the file and validates
it. The executable and `tests/ecs_consumer.rs` instantiate the same ECS systems.
Provider cases use real completion adapters; synthetic cases use an explicitly
scripted completion handler. Both use real workspace tools.

The `repository-repair` matrix gives the same consumer a dependency-free Rust
project with a failing pagination test. The genuine provider reads the contract,
diagnoses the failure, writes a desired-behavior regression, observes it fail,
chooses a production repair, requests the host approval decision and validates
its result. The host does not supply replacement contents to the provider.
Synthetic cases supply known stimuli for refusal, timeout and replay assertions.

Run commands from the repository root:

```sh
cargo run -p rig --example ecs-consumer -- plan
cargo run -p rig --example ecs-consumer -- verify
cargo run -p rig --example ecs-consumer -- verify --matrix provider
cargo run -p rig --example ecs-consumer -- verify --case stream-single-events
cargo run -p rig --example ecs-consumer -- replay --case anthropic-stream
cargo run -p rig --example ecs-consumer -- resume --case anthropic-stream --cut after-write
cargo test -p rig --test ecs_consumer
```

`registry.rs` defines stable case IDs, provider, tool concurrency, intake bound,
handler serialization, inspection arrival order, stream group size, archetype
perturbation and host approval input. `--case` accepts comma-separated IDs;
unknown or duplicate IDs fail instead of silently reducing the matrix. `--matrix`
selects the registry's matrix name. `plan` and `list` never call a provider.
The plan lists required fixture paths and applicable producer paths. Executed
reports include each selected case's status and coverage grouped by every
registry axis; planning alone never counts as a pass.

The original corpus has 32 cases: six genuine provider lanes (unary and stream
for Anthropic, OpenAI and Gemini) and 26 synthetic policy/fault cases. The six
original captures used 32 live requests in total. Reuse those cassettes offline for
subsequent golden derivation; synthetic variants incur no provider calls.
The repair extension adds ten cases, for 42 total (35 synthetic, seven genuine
provider lanes) and 217 fixture files. Its accepted Opus 5 capture used nine live
requests. Earlier repair attempts remain unaccepted or historical local candidates;
they are not counted as additional accepted provider lanes.

| Matrix | Assertions and reproduction selection |
| --- | --- |
| Provider | `--matrix provider`: matched tool-call requests, actual edit and validation, unary/stream conventions for all three providers |
| Concurrency | `--matrix concurrency`: tool limit 1/2, same-key serial/concurrent custom work, reversed/coincident arrivals, intake bound and spawn history |
| Stream | `--matrix stream`: grouped/interleaved delivery, partial cancellation, error before/after Final, final-only replay and policy metadata refusals |
| Approval | `--matrix approval`: denial, cancellation, failed write; approved writes and published receipts are checked in successful cases |
| Identity | `--matrix identity`: model/capability/policy mismatch, multiple scopes, uncalled grant, missing/conflicting metadata and unexpected dispatch |
| Persistence | `--matrix persistence`: zero-progress unary/stream restart, unknown extensions and unsafe prefixes; successful cases also compare after-write resume |
| Lifecycle | `--matrix lifecycle`: removed model, cancellation before serve/background loser, and lost write outcome requiring reconciliation |
| Repository repair | `--matrix repository-repair`: model-directed unary capture plus nine synthetic cases for approval, denial/cancellation, insufficient repair, stale approval, timeout, streaming groups and lost edit outcome |
| Custom answers | `--matrix custom_answers`: seven JSON value shapes through typed answers, replay and scene load |

Every `verify` case runs its owning producer, policy replay and each available
resume cut. A case that ends before a completed approved write reports that
resume is inapplicable. These ECS policy observations are not declared as
supported by the legacy corpus interpreters.

## Recording and deliberately accepting artifacts

Verification, derivation, effect replay and resume force offline replay even
when `RIG_PROVIDER_TEST_MODE=record` is present. They require no provider keys.
The transport permits only the selected local cassette server; redirects and
automatic retries are disabled. The existing cassette engine matches outgoing
requests and checks that recorded interactions were consumed.

Only `record` makes live calls, through that engine's recorder. Set the relevant
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY` in the environment and
select a small case explicitly:

```sh
cargo run -p rig --example ecs-consumer -- record --case anthropic-unary
```

The command prints its candidate path. Use that path in the following steps:

```sh
cargo run -p rig --example ecs-consumer -- derive --case anthropic-unary --candidate .ecs-consumer/candidates/REPLACE_WITH_RECORDED_CASE
cargo run -p rig --example ecs-consumer -- replay --case anthropic-unary --candidate .ecs-consumer/candidates/REPLACE_WITH_DERIVED_CASE
cargo run -p rig --example ecs-consumer -- promote --case anthropic-unary --candidate .ecs-consumer/candidates/REPLACE_WITH_DERIVED_CASE
```

`derive` produces a **new** candidate after running the scrubbed cassette,
checking independent application assertions, replaying the resulting log in a
fresh world, and comparing supported resume paths. Inspect its JSON artifacts
and provider YAML before promotion. `promote` reruns the producer/replay checks,
compares the candidate to that fresh result, validates its hashes and secret
checks, prints semantic JSON differences, then installs the files. A failed
derivation bundle is not an accepted baseline. These local fixture operations
do not commit or publish changes.

Each live recording invocation shares a limit of 32 transport requests,
300 seconds and zero retries. Genuine repair requests explicitly allow 8,192
output tokens, including provider reasoning tokens; historical greeting and
synthetic requests keep their 512-token limits. The plan reports each case’s
cap and the invocation’s maximum before sending traffic. Offline commands give
each selected case its limits independently, and the report totals all local
requests. This lets a growing replay matrix run without increasing the live
allowance. Recording reserves the last ten seconds for finalization; every
repair subprocess also receives the remaining absolute execution deadline.
Completed traffic is periodically scrubbed into `provider.partial.yaml` in the
unaccepted candidate. Empty snapshots are skipped. An execution/finalization
failure retains available traffic and `capture-failure.json`; it cannot create
successful capture provenance. In-flight requests or unfinished stream responses
are not guaranteed to be retained. The consumer also has a bounded turn
count (8 greeting / 24 repair) and an execution guard (120 seconds for greeting,
300 seconds for repair, further capped by the recording execution deadline). The plan names the provider and model
before any request. Capture metadata retains request use, limits, model and
revision where available; a missing historical capture manifest remains absent.
Generation manifests retain schema version, source, revision, dirty-worktree
status and artifact/cassette SHA-256 hashes. Revision metadata is provenance,
not a semantic golden field that changes on every ordinary verification.

## What the artifacts prove

There are four distinct kinds of evidence:

1. Provider YAML contains scrubbed HTTP requests/responses from genuine live
   traffic. Synthetic policy cases never substitute for this evidence.
2. `effects.json` is the existing `EffectLog`, produced by the ECS consumer over
   scrubbed cassette replay. It retains requests, descriptors, scopes, causal
   IDs, stream events/errors, delivery partitions and published tool data.
3. `observations.json` records what the consumer saw after the entire Collect
   set: partial streams, outcomes, the first visible inspection, approval input
   and published write receipts. `application.json` records actual file state,
   write count and the run result. Independent assertions require the approved
   target contents, validation and exactly one write for the greeting task, or
   unchanged contents and zero writes when its approval is denied or cancelled.
   Repair cases instead check the documented behavior, immutable files, exact
   approvals and two edits (regression then production). Production denial or
   cancellation retains only the approved regression edit.
4. `checkpoints.json` stores a supported library scene plus explicitly declared
   application resources: file image, mutation ledger, approval/proposal state,
   observations and stable logical IDs. Host input is in the case configuration;
   observed decisions are never replay instructions.

Canonicalization sorts handler descriptors by key and renumbers **observed**
delivery batches consecutively. It preserves each batch's complete contents,
partition and ordering. Empty elapsed passes are not inputs to this consumer's
policy. Nothing strips requests, capabilities, scopes, errors or causal IDs.

The schedule wrapper buffers adapter output independently of HTTP chunking.
Explicit gates release a complete configured stream group before permitting
Collect, then acknowledge that collection before releasing the next group.
Single-event and multi-event cases therefore exercise different policy-visible
boundaries. Inspection tools can arrive read-first, search-first or together;
the first visible answer becomes the provisional view. This is controlled
consumer policy replay, not equality with arbitrary timing of a live capture.
Live runs have separate scrubbed evidence and application invariants.

Effect replay rebuilds handlers from JSON metadata, constructs the same program
in a fresh world, calls `check_replayable` and uses `Replay::policy_visible`.
The program creates effects and IDs itself. Recorded mutations are projected
only into a newly owned disposable workspace, with a duplicate-write guard.
There is no fallback to a live provider or mutation handler.

The `after-write` cut retains completed streams and published receipts. Resume
JSON-round-trips the scene and host state, rebinds the log tail, restores the
registered run extension and creates new effects with later IDs. It compares
the continuation and complete application result against uninterrupted
execution. This is not an arbitrary mid-flight fork. An external operation
whose outcome was never persisted still requires idempotency or reconciliation;
cancellation cannot roll back an external write.

`zero-progress-unary` and `zero-progress-stream` also save a
`before-first-delivery` cut after dispatch and before world serving/Collect.
Resume verifies completed state and the next-ID counter immediately after load,
then compares the continued recording. Unknown extensions and unfinished
delivered stream prefixes must fail before spawning any saved entities:

```sh
cargo run -p rig --example ecs-consumer -- resume --case zero-progress-stream --cut before-first-delivery
cargo run -p rig --example ecs-consumer -- verify --case replay-guarantees,external-write-outcome-lost
```

The lost-write fault removes a persisted answer after a real disposable-file
write, while retaining the operation ledger. Reissuing that operation is refused
as a duplicate and requires explicit reconciliation. This demonstrates the
external-effect hazard; it does not provide generic exactly-once persistence.
`replay-guarantees` separately checks a folded log's final application result
and refuses using that log for policy-visible replay.

## Agent repair workflow

1. Run `plan`, select the relevant matrix and verify it offline. Reports and
   failure bundles are under ignored `.ecs-consumer/`; preserve the failing
   candidate, case configuration and command before changing code.
   `failure.json` supplies structured reproduction arguments and fixture hashes;
   runtime failures also preserve the observed trace, effect log and unfinished
   logical effect IDs. Artifact validation/redaction applies to these reports.
2. Reproduce the smallest selected case. A JSON difference names its pointer
   and artifact; inspect the surrounding effect, observations and checkpoint.
   Distinguish request mismatch, producer error, application invariant failure,
   replay divergence and an unavailable fixture.
3. Add a regression asserting the intended contract. Use independent release
   gates for scheduling failures, not sleeps, retries or a weaker overlap check.
4. Fix the confirmed production or harness defect. Rerun the frozen regression,
   its matrix and affected repository checks. Never regenerate expectations just
   because the previous expectation failed.
5. For an intentional semantic change, derive a separate candidate, inspect the
   difference, then promote it. Preserve the old failure evidence. Obtain an
   independent review before publishing a PR, with the repository's full PR gate.

The consumer exposed a premature missing-effect replay refusal after restoring
a completed tool batch, and successful unpaced collection after a replay
refusal. `bus_delivery.rs` now includes regressions for a multi-stage continuation
after Collect and effects created after refusal. The runner also rejects a
handler panic even when the handler emitted a successful terminal answer.

Add scenarios through the executable registry and shared consumer systems.
Document whether each scenario requires a genuine cassette, a synthetic fault,
or only an existing runtime contract test. New required cases need real
assertions and reproduction evidence; a name or a passing plan is not coverage.

The legacy cross-runtime corpus remains separate equivalence evidence. This
consumer does not establish arbitrary full-world determinism, automatically
save application resources, or justify extracting another runtime crate.

## Model-directed Rust repair

The genuine `repair-anthropic-unary` lane selects Claude Opus 5
(`claude-opus-5`) with its default adaptive thinking enabled. New live repair
work must use a current capable model such as Opus or GPT Sol; do not downgrade
to older or mini models to save tokens. The configured OpenAI repair alternative
is `gpt-5.6-sol`. Verify available API IDs before adding a new capture lane.
Historical greeting cassettes retain their recorded model identities.
Capture/provenance manifests and the plan report the selected model. Use stable case IDs from
`plan --matrix repository-repair`. Only the explicit
`record` command uses a provider key:

```sh
cargo run -p rig --example ecs-consumer -- record --case repair-anthropic-unary
cargo run -p rig --example ecs-consumer -- derive --case repair-anthropic-unary --candidate .ecs-consumer/candidates/REPLACE_WITH_CAPTURE
cargo run -p rig --example ecs-consumer -- verify --case repair-anthropic-unary --candidate .ecs-consumer/candidates/REPLACE_WITH_DERIVATION
cargo run -p rig --example ecs-consumer -- resume --case repair-anthropic-unary --candidate .ecs-consumer/candidates/REPLACE_WITH_DERIVATION --cut after-write
```

Review the actual diagnosis, both patch diffs, compiler/test evidence, host
approval observations and candidate JSON differences before using `promote`.
A failed or partial genuine repair capture is evidence for debugging; it cannot
be promoted as a successful live capture. Do not repeat paid calls until a
specific failure has been diagnosed and any confirmed harness defect has a
failing regression and a verified fix.

The project copied from `tests/fixtures/repository_repair` deliberately stays
broken in the repository. Only `tests/regression.rs` and `src/lib.rs` are writable
in each disposable project; existing tests, manifest, lockfile and README are
immutable. Tool schemas expose those paths and named validation phases. A
proposal first runs bounded rustfmt on scratch text, then returns the formatted
exact diff and operation identity. The original request, formatter argv/status/output
and whether formatting changed bytes are recorded. Formatting never changes the
project or runs after approval; Collect checks this evidence against the issued
proposal and returned bytes. Calling `repo_apply_patch`
asks the separate host policy for its declared decision before applying it.
Changing contents or the source digest invalidates the approval. The bounded
workflow accepts one regression edit and one production edit per run.

Initial and regression validation require executed behavioral failures against
the unchanged production image. Compile errors, ignored/empty/passing tests,
timeouts and truncated output cannot establish the regression. Final validation
requires all original and added tests, formatting, and a host-owned contract
oracle linked to the same compiled library. The oracle checks 245 boundary rows
and borrowing against independent expectations. It is finite behavior evidence;
source review must still reject hardcoded answers or harness-dependent code.
The final model message cannot override the tool results.

Repair validation uses macOS `sandbox-exec` or Linux `bwrap`, cleared environment,
read-only project/artifacts and separate writable scratch. It runs Cargo offline
and locked with immutable dependency-free metadata and no build scripts. Linux
requires working unprivileged user namespaces and bubblewrap; namespace failure
is an error. Commands have bounded wall time, CPU, output, descriptors and
individual file size. Linux also applies a virtual memory bound; no hard macOS
memory or aggregate disk quota is claimed. This boundary supports the small
fixture, not arbitrary hostile repositories.

Validation results become policy state at Collect. A model that batches an edit
and its validation gets a recoverable instruction to validate on the following
turn, after observing the edit receipt. Replay validates the issued operation
and typed publication before projecting a write into a new owned workspace;
it restores recorded process results without running compilers or tests.
The `after-write` repair cut is after the completed production edit and before
final validation. It carries the full project image, exact approvals, reports
and mutation ledger. Fresh-world continuation preserves that completed edit;
a missing saved outcome requires reconciliation, never a duplicate write.
