# rig-cassette

Rig's record/replay home: effect logs and runtime replay adapters, the native
HTTP cassette engine, the committed fixture corpora, and their verification suites.

| part | path | published |
|---|---|---|
| effect logs and checkpoints | `src/effect_log/` | yes, always |
| classic-agent replay adapter | `src/agent/` | yes, `agent` feature |
| native HTTP engine | `src/http/` | yes, `http` feature |
| provider cassettes | `fixtures/cassettes/<provider>/...yaml` | no (`exclude`) |
| agent effect-log goldens | `fixtures/effects/<name>.effects.json` | no (`exclude`) |
| cassette provider suites | `tests/<provider>.rs`, `tests/providers/`, `tests/common/` | no (`exclude`) |
| effect-bus verification | `tests/verify/` | no (`exclude`) |
| minimal verification runner | `tests/minimal/Cargo.toml` (shared test sources) | no (`publish = false`, `exclude`) |

## Features and dependency direction

Defaults are empty. An effect-log-only consumer uses:

```toml
rig-cassette = { version = "0.42.0", default-features = false }
```

| features | public modules | additional normal dependencies |
|---|---|---|
| none | `effect_log` | core contracts, futures and serialization only |
| `jsonl` | `effect_log`, `effect_log::jsonl` (JSON-lines files, native) | none |
| `agent` | `effect_log`, `agent` | `rig-agent`, without its default features |
| `http` | `effect_log`, `http` | the native HTTP server/client engine, Tokio, ordered/round-trip JSON |
| `bedrock` | `effect_log`, `http` | `http` plus Smithy event-stream decoding |

Effect logs alone acquire neither the agent runtime, HTTP clients/servers,
Tokio nor AWS dependencies. The agent integration is selectable on its own and
does not enable HTTP. These guarantees concern the selected normal dependency
graph; Cargo can still unify features requested by other dependencies in the
same build.

The dependency direction is cassette → runtime → core. The runtime does not
depend on cassette, including through optional features. The `rig` facade re-exports
this crate as `rig::cassette` behind its opt-in `cassette` feature; with the
facade's `agent` feature it also enables the cassette agent adapter. Direct
minimal consumers should depend on `rig-cassette`, rather than the facade's
default transport configuration.

The facade and provider helpers needed by this package's tests remain
version-less path dev-dependencies, omitted from its published manifest.
The facade dev-dependency explicitly enables its capability features so provider
targets retain the same test inventory without relying on workspace defaults.
Independent downstream graph guards live in `src/http/paths.rs` and
`tests/core/dependency_graph.rs`; they distinguish native-engine isolation from
the minimal library and the agent adapter.

## Effect logs and runtime integration

`rig_cassette::effect_log` owns `EffectLog`, `EffectLogRecorder`,
`EffectLogReplayer`, `Checkpoint`, `RequestCheck`, header validation and stable
hashing. The log/checkpoint wire formats and fingerprint inputs are unchanged.
The recorder implements `rig_core::serve::Recorder`; the replayer implements
the ordinary core handler interface.

Effect records require `tool_output`: `null` means no publication and an object
means published values, including an empty map. Omitting it cannot establish
whether values were lost, so decoding rejects omission. Replayers publish these
values before resolving outcomes, including errors. Handlers must publish before
resolving and must exclude inbound context, secrets, and live capabilities.
Custom outcomes keep their JSON value in an explicit `payload` field. Nested tool
identities retain origin tags; bare-string identities are rejected. Logs validate
their data rather than a global format number; unknown header fields, including
`format`, are rejected. The checkpoint envelope has its own version and does not convert
nested payloads.

Replay preserves recorded semantic families, model identities, and capabilities.
Required keys include all scoped program rows; conflicting declarations are
rejected. Callers reapply executable middleware. Inferred descriptors for logs
without declarations cannot establish verified program compatibility.

Delivery batches are optional metadata a runtime may record about its
consumer-visible schedule. The shared replayer supplies exchanges and event
sequences; the classic bus records no delivery batches. Without delivery
metadata and retained stream items, replay cannot prove exact partial state or
first-visible answer policies. Batches are not clocks or external-side-effect
guarantees. Stream error positions preserve ordering even around a final event;
a folded outcome cannot reconstruct that order. Empty metadata is omitted, and
logs without error positions cannot prove the original error-item sequence.

For a classic agent, enable `agent`, retain an `EffectLogRecorder`, and pass a
clone to `AgentBuilder::record_to`. Use `keeping_stream_events()` instead of
`new()` when original stream item boundaries are required. After driving the
agent, import `rig_cassette::agent::AgentReplayExt` and call
`agent.stamp(recorder.take())` (or stamp `recorder.log()` for a snapshot).
The extension trait also supplies `run_spec_hash` and `check_replayable`.
For a host-owned bus, attach the recorder with `BusDriver::record_to`; an agent
over that bus cannot install its own recorder. Replay registration is
`rig_cassette::agent::replay::register_all` (or `register_all_checking`).

Migration: replace the removed `rig-effect-log` dependency with `rig-cassette`
and its `effect_log` module. Replace implicit agent recording/log getters with
an explicit recorder and the agent extension trait. Move runtime replay imports
to the cassette adapters; native cassette imports now start with
`rig_cassette::http` and require `http` (or `bedrock`). No old package or API
aliases remain.

## The engine
Enable `http` and import engine APIs from `rig_cassette::http`. This feature is
native-only and intentionally opts into both JSON features listed above.


Pass a fixture root containing provider directories to `ProviderCassette::start`,
`start_via(RecordVia::Direct, ..)`, `cassette_path` and every `recorded_*` reader:

```text
<fixture-root>/anthropic/completion.yaml
<fixture-root>/openai/nested/scenario.yaml
```

`CassetteSpec::new` preserves strict interaction order; `.unordered()` permits
matching any unused interaction. `finish` checks complete consumption and shuts
down the replay server. A replay session dropped without `finish` panics when
it still holds an unplayed interaction or a refused request, so a test that
returns early cannot pass on a recording it never played to the end. The guard
stays silent while the thread is already panicking, for a fully played session,
and after `finish_after_test_result` returns a test's own error. Invalid
fixtures and failed assertions panic, preserving the original test-support
behavior. Applications that record or replay outside a test use
`ProviderCassette::try_start_at` and `try_finish`, which return a
`CassetteError` for each of those failures instead. `try_finish` shuts the
session down whatever its outcome, so dropping it afterwards never panics.

`RIG_PROVIDER_TEST_MODE` defaults to `replay`. `record` contacts the configured
upstream and overwrites the selected fixture after scrubbing; `start` and
`start_via` read the mode from the environment, and a recording reaches the
provider through the proxy (`start`) or directly (`start_via(RecordVia::Direct, ..)`).
Consumers that stage candidates use `ProviderCassette::start_at` with an explicit
mode and exact path: a live capture records into a candidate path, and a
verification pass replays with `CassetteMode::Replay` even when the environment
asks for recording, so a candidate is never implicitly promoted to a fixture.
While a live run is in progress, `checkpoint_recording` writes the completed,
scrubbed exchanges to a partial path without finalizing the recording.

`RIG_CASSETTE_SNAPSHOTS` is off unless set. With `check`, a replay session
also compares each request body with the fixture's request snapshot,
`<fixture>.requests.json` (`request_snapshot`), and `finish` returns
`CassetteError::SnapshotMismatch` naming each difference by JSON pointer.
With `write`, the session rewrites the snapshot from the requests it received.
A snapshot holds only how a sent body differs from its recording, compared
after scrubbing as canonical JSON or as multipart parts, so a fixture whose
requests equal their recordings has none. This repository sets `check` for
its own cassette tests.

Replay compares request bodies exactly unless `RIG_CASSETTE_MATCHING=shape`
is set or the spec says `.shape_matched()`; `.exact_matched()` keeps exact
matching whatever the variable says. Shape matching still compares the
method, path, query and recorded headers exactly, but compares bodies by a
coarse key: the field structure, the order of arrays of objects, and the
values of `model`, `role`, `name` and `stream`, with every other value, the
types of content, tool schemas and tool arguments erased. A change to how a
request is written then replays without a new recording, and with snapshots
on in `check` mode the session fails until the snapshot holds the change.
Unordered shape matching serves, among the interactions with the request's
key, the one whose body is closest to the request's. This
repository sets `shape` for its own cassette tests.

A recording only becomes the fixture when the test passed and the recording is
clean. A failed test's exchanges, and a recording `finish` refuses, go under
`attempt_root()` (`RIG_CASSETTE_ATTEMPT_DIR`, default `cassette-attempts` in the
target directory). `finish` refuses two kinds of recording:

- one holding an account failure that the cell did not declare: a refused
  credential, a spent quota, a rate limit or an empty balance, as classified
  by `reply_account_failure`, including a failure delivered inside a 2xx
  stream. Cells declare one with `CassetteSpec::expects_account_failure`,
  `ProviderCassette::expect_account_failure` or `bogus_api_key`.
- one in which an OpenAI or xAI Responses request stored a response that the
  session never deleted (`cassette_stored_state`).

A relay in front of the proxy, and the direct recorder for unary and
multipart requests, append every created stored response, file, cache,
interaction, conversation and vector store to the `ledger` module's
`ledger.jsonl` before the reply reaches the test. `ledger::clean_up` deletes
what the ledger still holds. The `cassette_tool` example exposes both checks
and the cleanup pass.

`DirectRecorder`, its request/response types and `DirectRecordingHttpClient`
preserve binary bodies that a text proxy cannot record. When the proxy omits a
non-UTF-8 multipart request body, replay accepts multipart input without comparing
its missing bytes; provider unit tests must cover the multipart shape. Other
requests with an absent recorded body must be empty. SSE and ordinary binary
responses require only `http`. Enable `bedrock` for Smithy event-stream decoding
and scrubbing; its Smithy dependencies are absent otherwise.

The engine retains credential scrubbing, strict request matching, and safety
validation. Provider-issued values (ids, signatures, encrypted reasoning,
cache keys, cursors, generated images) are recorded verbatim, and request
matching compares recorded bytes as they are. Repository-specific source
scans and fixture censuses remain with each caller. Rig's adapter in
`test-support/rig-test-support/src/cassettes.rs` is a thin path binding that
supplies `crates/rig-cassette/fixtures/cassettes`; a downstream can supply
`fixtures/cassettes` of its own instead. `Retry-After` response headers retain
canonical seconds or HTTP dates for replay diagnostics; malformed values are
discarded instead of persisting arbitrary server text. Home-directory paths are
scrubbed separately because credential scans cannot identify operator names or
cache layouts. Only paths at the beginning of a string value are eligible,
preserving embedded public URLs and the model basename. Spaces remain part of
the path to avoid leaking account names.

## The corpora

`fixtures/cassettes/<provider>/` holds the recorded HTTP interactions the
provider suites in this package replay. `fixtures/effects/*.effects.json` holds
agent-produced effect logs. All fixtures are data: they are
regenerated by their producer, never edited by hand, and never regenerated to
make a check pass. `.gitattributes` exempts the cassettes from the
blank-at-eof whitespace check because SSE bodies legitimately end in a blank
line.

No corpus is embedded with `include_*!`; the binaries read them at runtime
from `CARGO_MANIFEST_DIR`, which is why `exclude` can keep them out of the
published tarball (the crates.io 10 MiB upload limit is enforced server-side
and never surfaces in `cargo publish --dry-run`).

## The recording loop

Background, not an instruction to record: recording contacts a real provider
and is a deliberate, separately authorized act.

A golden is produced from the recorded HTTP in three steps:

1. **Record the HTTP.** The producer test runs against the real provider under
   `RIG_PROVIDER_TEST_MODE=record`, on its own exact test filter, and writes a
   cassette under `fixtures/cassettes/`. The golden call is a no-op in that
   mode.
2. **Produce the agent golden.** The same producer runs again in replay mode under
   `RIG_REGENERATE_GOLDEN=1`, so the golden is generated from the *replayed*
   cassette and holds exactly the bytes replay serves.
3. **Replay the goldens.** The suite here replays them with no provider behind any
   key. A change in what the program asks (a kind), what it was answered (an
   outcome) or how a stream was delivered (its events) fails the replay naming
   the record and the JSON pointer of the difference. Fix forward, and
   re-record live when the change is intended, never by hand-editing a golden.

Hooks are program (the header names them; a different stack is refused before
the first dispatch); tools are record (a replayer answers them); nothing the
engine mints is random, so the same program produces the same log twice.

Agent producers live in this package (`tests/providers/*/cassette/corpus_*.rs`)
and the root package (`tests/core/golden_*.rs`); the shared matrix driver is
`tests/common/corpus_matrix.rs` with its `corpus_matrix/` modules.
`tests/core/golden_pairing.rs` enforces one producer per golden.

## The verification suite

`tests/verify/main.rs` is the single entrypoint: every matrix below is a module
of that target, not a target of its own, so the shared corpus implementation
(`tests/corpus/mod.rs`, which holds the program table and the dimension table of
an effect trace as a whole) is compiled once.

The unpublished `rig-cassette-minimal` package in `tests/minimal/Cargo.toml`
points at the same `verify` entrypoint and reuses the effect-log and classic
replay unit-test sources through its `effect_log` target. It depends on
`rig-cassette` with only `agent`, without the native HTTP engine, facade or
provider helpers. Testing the main cassette package does not prove this
isolation: its dev-dependencies enable the native engine.

CI runs the minimal targets separately with zero retries, and the shared
verification target through the default-member graph with two. The all-features
workspace run excludes the nested package to avoid repeating the same sources
with unified features. `core-all` separately retains the migrated classic replay
unit tests; the default/all-features cassette library executes all migrated
library regressions. The planner tracks both the shared test entrypoints and
the shared library regression sources.

| matrix | module | subject |
|---|---|---|
| — | `golden_replay.rs` | the original corpus: ten goldens, both agent interpreters |
| A | `corpus_retrieval.rs` | retrieval effects |
| B | `corpus_hooks.rs` | the hook surface |
| C | `corpus_serving.rs` | serving policy, routing and bus ownership |
| D | `corpus_outcome.rs` | continuation, cancellation and failure outcomes |
| E | `corpus_request_shape.rs` | request-shape axes that change the spec hash |
| F | `corpus_endings.rs` | hook-ended runs |
| G | `corpus_invalid.rs` | invalid tool calls, streamed and ignored |
| H | `corpus_output.rs` | output modes |
| I | `corpus_host.rs` | a host's own families over the host's bus |
| J | `corpus_memory.rs` | memory operations |
| K | `corpus_delta.rs` | the delta wire |
| L | `corpus_resume.rs` | resumption under everything |
| M | `corpus_shaping.rs` | per-turn shaping |
| N | `corpus_breadth.rs` | provider breadth for the pass-2 shapes |
| O | `corpus_oracle.rs` | the oracle and the header |
| P | `corpus_layers.rs` | layers |
| Q | `corpus_causal.rs` | causal dispatch |
| R | `corpus_checkpoint.rs` | checkpoints and hash-checked replay |
| S | `corpus_header.rs` | the header's new types |
| T | `corpus_leftovers.rs` | the leftovers of the #2443 review, and `Denied` |

Each module carries its own dimension table, its cells and what it found; that
header is the census, not this list. Beside the matrices, `golden_refusal.rs`
proves a stale golden refuses rather than passing on a different trace,
`record_replay.rs` and `durable_execution.rs` pin recording and interrupted
runs, `interpreters_agree.rs` states interpreter agreement as a proptest
property, and `log_header.rs` pins the header.

The two agent interpreters (the classic runner and the direct `AgentRun`
driver, both in `tests/corpus/mod.rs`) replay the goldens their matrices
enumerate, including cancelled streams: the replayer answers the record as the
cancel it was, after the events it kept. A producer whose golden the cassette
prune replaced (`coverage/pruned.tsv`) replays its log record by record through
its own replayers by effect id instead of comparing it with a committed file.

## What lives where

- here: effect logs, concrete replay adapters, the cassette engine,
  `fixtures/cassettes/`, agent goldens in `fixtures/effects/`, every
  cassette-backed provider suite with its producers, and the behaviour of the
  bus and the agent over it (record and replay, durable execution, the two
  interpreters agreeing);
- the root package's `tests/core`: guards that scan the source tree and the
  fixture runners (they need the repository root), plus the one-producer-per-
  golden pairing guard;
- the root package's `tests/providers`: the live-only provider suites of
  providers with no recorded corpus;
- `rig-core`/`rig-agent` unit tests: anything that needs crate-private types
  (the loom models among them).

## Running

```sh
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette-minimal --test effect_log --retries 0
RIG_PROVIDER_TEST_MODE=replay cargo test --locked -p rig-cassette --lib --all-features
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette --all-features -E 'binary(verify)'
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette-minimal --all-features --retries 0
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette --all-features --test <provider>
```

`RIG_REGENERATE_GOLDEN` must be unset for all of them. The lane owners of these
executions, including their retry policies and the default-member graph twins,
are in `xtask/src/verify/checks.rs` (`default-tests`, `effect-corpus`,
`bus-verification`).
