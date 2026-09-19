# rig-cassette

Rig's record/replay home: effect logs and runtime replay adapters, the native
HTTP cassette engine, both committed fixture corpora, and their verification suites.

| part | path | published |
|---|---|---|
| effect logs and checkpoints | `src/effect_log/` | yes, always |
| classic-agent replay adapter | `src/agent/` | yes, `agent` feature |
| ECS replay adapter | `src/ecs/` | yes, `ecs` feature |
| native HTTP engine | `src/http/` | yes, `http` feature |
| provider cassettes | `fixtures/cassettes/<provider>/...yaml` | no (`exclude`) |
| effect-log goldens | `fixtures/effects/<name>.effects.json` | no (`exclude`) |
| cassette provider suites | `tests/<provider>.rs`, `tests/providers/`, `tests/common/` | no (`exclude`) |
| effect-bus verification | `tests/verify/`, `tests/world_replay.rs` | no (`exclude`) |
| minimal verification runner | `tests/minimal/Cargo.toml` (shared test sources) | no (`publish = false`, `exclude`) |

## Features and dependency direction

Defaults are empty. An effect-log-only consumer uses:

```toml
rig-cassette = { version = "0.42.0", default-features = false }
```

| features | public modules | additional normal dependencies |
|---|---|---|
| none | `effect_log` | core contracts, futures and serialization only |
| `agent` | `effect_log`, `agent` | `rig-agent`, without its default features |
| `ecs` | `effect_log`, `ecs` | `rig-ecs` and Bevy |
| `http` | `effect_log`, `http` | the native HTTP server/client engine, Tokio, ordered/round-trip JSON |
| `bedrock` | `effect_log`, `http` | `http` plus Smithy event-stream decoding |

Effect logs alone acquire neither runtime, Bevy, HTTP clients/servers, Tokio
nor AWS dependencies. Agent and ECS integration are independently selectable:
neither enables HTTP or `serde_json/preserve_order` / `float_roundtrip`, and
neither acquires the other runtime. These guarantees concern the selected
normal dependency graph; Cargo can still unify features requested by other
dependencies in the same build.

The dependency direction is cassette → runtime → core. Neither runtime depends
on cassette, including through optional features. The `rig` facade re-exports
this crate as `rig::cassette`; its `agent` feature enables the cassette agent
adapter. Direct minimal consumers should depend on `rig-cassette`, rather than
the facade's default transport configuration.

The facade and provider helpers needed by this package's tests remain
version-less path dev-dependencies, omitted from its published manifest.
The facade dev-dependency explicitly enables its capability features so provider
targets retain the same test inventory without relying on workspace defaults.
Independent downstream graph guards live in `src/http/paths.rs` and
`tests/core/dependency_graph.rs`; they distinguish native-engine isolation from
the minimal and independently enabled runtime adapters.

## Effect logs and runtime integration

`rig_cassette::effect_log` owns `EffectLog`, `EffectLogRecorder`,
`EffectLogReplayer`, `Checkpoint`, `RequestCheck`, header validation and stable
hashing. The log/checkpoint wire formats and fingerprint inputs are unchanged.
The recorder implements `rig_core::serve::Recorder`; the replayer implements
the ordinary core handler interface.

For a classic agent, enable `agent`, retain an `EffectLogRecorder`, and pass a
clone to `AgentBuilder::record_to`. Use `keeping_stream_events()` instead of
`new()` when original stream item boundaries are required. After driving the
agent, import `rig_cassette::agent::AgentReplayExt` and call
`agent.stamp(recorder.take())` (or stamp `recorder.log()` for a snapshot).
The extension trait also supplies `run_spec_hash` and `check_replayable`.
For a host-owned bus, attach the recorder with `BusDriver::record_to`; an agent
over that bus cannot install its own recorder. Replay registration is
`rig_cassette::agent::replay::register_all` (or `register_all_checking`).

For ECS, enable `ecs`. Install any recorder through
`rig_ecs::bus::Recording::install`, or use
`rig_cassette::ecs::EffectLogResource::install` to retain the concrete handle as
a resource too. Add `rig_cassette::ecs::ReplayPlugin` after `RigPlugin` or
`BusPlugin`, then call `Replay::register(&mut World, &EffectLog)` before issuing
replayed effects. `Replay::load` restores recorded effect ids;
`Replay::policy_visible()` additionally requires the recorded delivery contract.
`ReplayDelivery`, `ReplayFailure` and program identity helpers live in
`rig_cassette::ecs` / `rig_cassette::ecs::identity`. World checkpoint state and
generic execution/observation mechanisms remain in `rig-ecs`.

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
`start_via(Transport::Direct, ..)`, `cassette_path` and every `recorded_*` reader:

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
behavior.

`RIG_PROVIDER_TEST_MODE` defaults to `replay`. `record` contacts the configured
upstream and overwrites the selected fixture after scrubbing; `start` and
`start_via` read the mode from the environment, and a recording reaches the
provider through the proxy (`start`) or directly (`start_via(Transport::Direct, ..)`).
Consumers that stage candidates use `ProviderCassette::start_at` with an explicit
mode and exact path: a live capture records into a candidate path, and a
verification pass replays with `CassetteMode::Replay` even when the environment
asks for recording, so a candidate is never implicitly promoted to a fixture.
While a live run is in progress, `checkpoint_recording` writes the completed,
scrubbed exchanges to a partial path without finalizing the recording.

`DirectRecorder`, its request/response types and `DirectRecordingHttpClient`
preserve binary bodies that a text proxy cannot record. SSE and ordinary binary
responses require only `http`. Enable `bedrock` for Smithy event-stream decoding
and scrubbing; its Smithy dependencies are absent otherwise.

The engine retains secret and generated-identifier scrubbing, strict request
matching, and safety validation. Repository-specific source scans and fixture
censuses remain with each caller. Rig's adapter in
`test-support/rig-test-support/src/cassettes.rs` is a thin path binding that
supplies `crates/rig-cassette/fixtures/cassettes`; a downstream can supply
`fixtures/cassettes` of its own instead. `Retry-After` response headers retain
canonical seconds or HTTP dates for replay diagnostics; malformed values are
discarded instead of persisting arbitrary server text. Generated request IDs
remain placeholdered.

## The corpora

`fixtures/cassettes/<provider>/` holds the recorded HTTP interactions the
provider suites in this package replay. `fixtures/effects/` holds the
effect-log goldens the verification suite replays. Both are data: they are
re-recorded by their producer, never edited by hand, and never regenerated to
make a check pass. `.gitattributes` exempts the cassettes from the
blank-at-eof whitespace check because SSE bodies legitimately end in a blank
line.

Neither corpus is embedded with `include_*!`; the binaries read them at runtime
from `CARGO_MANIFEST_DIR`, which is why `exclude` can keep them out of the
published tarball (the crates.io 10 MiB upload limit is enforced server-side
and never surfaces in `cargo publish --dry-run`).

## The recording loop

Background, not an instruction to record: recording contacts a real provider
and is a deliberate, separately authorized act.

A golden effect log is produced in two stages, and replayed in a third:

1. **Record the HTTP.** The producer test runs against the real provider under
   `RIG_PROVIDER_TEST_MODE=record`, on its own exact test filter, and writes a
   cassette under `fixtures/cassettes/`. The golden call is a no-op in that
   mode.
2. **Produce the golden.** The same producer runs again in replay mode under
   `RIG_REGENERATE_GOLDEN=1`, so the golden is generated from the *replayed*
   cassette and holds the cassette's placeholders rather than live ids.
3. **Replay the golden.** The suite here replays it with no provider behind any
   key. A change in what the program asks (a kind), what it was answered (an
   outcome) or how a stream was delivered (its events) fails the replay naming
   the record and the JSON pointer of the difference. Fix forward, and
   re-record live when the change is intended — never by hand-editing a golden.

Hooks are program (the header names them; a different stack is refused before
the first dispatch); tools are record (a replayer answers them); nothing the
engine mints is random, so the same program produces the same log twice.

The producers live in this package (`tests/providers/*/cassette/corpus_*.rs`)
and the root package (`tests/core/golden_*.rs`), and are paired one-to-one with
the goldens by `tests/core/golden_pairing.rs`.

## The verification suite

`tests/verify/main.rs` is the single entrypoint: every matrix below is a module
of that target, not a target of its own, so the shared corpus implementation
(`tests/corpus/mod.rs`, which holds the program table and the dimension table of
an effect trace as a whole) is compiled once. `tests/world_replay.rs` is the one
separate target, so a lane can select or exclude the world interpreter without
touching the rest.

The unpublished `rig-cassette-minimal` package in `tests/minimal/Cargo.toml`
points at these same two entrypoints and reuses the effect-log and classic
replay unit-test sources through its `effect_log` target. It depends on
`rig-cassette` with only `agent,ecs`, without the native HTTP engine, facade or
provider helpers. Its standalone graph lacks `serde_json/preserve_order` and
`serde_json/float_roundtrip`. Testing the main cassette package does not prove
this isolation: its dev-dependencies enable the native engine.

CI runs the minimal targets separately with zero retries, and the shared
verification targets through the default-member graph with two. The all-features
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

Every golden is replayed by the ECS world interpreter in `world_replay.rs`, one
row per golden, counted against the corpus; the two agent interpreters (the
classic runner and the direct `AgentRun` driver, both in `tests/corpus/mod.rs`)
replay the goldens their matrices enumerate, including cancelled streams — the
replayer answers the record as the cancel it was, after the events it kept.

## What lives where

- here: effect logs, concrete replay adapters, the cassette engine, both
  corpora, every cassette-backed provider suite with its golden producers,
  and the behaviour of the bus and the agent
  over it (record and replay, durable execution, the three interpreters
  agreeing);
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
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette --all-features -E 'binary(world_replay)'
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette-minimal --all-features --retries 0
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette --all-features --test <provider>
```

`RIG_REGENERATE_GOLDEN` must be unset for all of them. The lane owners of these
executions, including their retry policies and the default-member graph twins,
are in `xtask/src/verify/checks.rs` (`default-tests`, `ecs-parity`,
`bus-verification`).
