# rig-cassette

Rig's record/replay home: the cassette engine, both committed fixture corpora,
and the effect bus's behavioural verification suite.

| part | path | published |
|---|---|---|
| engine | `src/` | yes |
| provider cassettes | `fixtures/cassettes/<provider>/...yaml` | no (`exclude`) |
| effect-log goldens | `fixtures/effects/<name>.effects.json` | no (`exclude`) |
| cassette provider suites | `tests/<provider>.rs`, `tests/providers/`, `tests/common/` | no (`exclude`) |
| effect-bus verification | `tests/verify/`, `tests/world_replay.rs` | no (`exclude`) |
| minimal verification runner | `tests/minimal/Cargo.toml` (shared test sources) | no (`publish = false`, `exclude`) |

Everything here shares a subject — a recording and the program that replays it
— not a dependency graph. The engine's normal dependencies are `rig-core` and
`rig-reqwest`: no agent runtime, no facade, no consumer registry, no fixture
inventory. Everything the suites need on top of that (the `rig` facade,
`rig-test-support`, `rig-agent`, `rig-ecs`, `rig-effect-log`, the Bevy crates,
`proptest`) is a version-less path **dev-dependency**, so Cargo omits it from
the published manifest and it never reaches a downstream's normal graph;
`src/paths.rs` pins that with a `cargo tree -e normal` probe over an
independent downstream package, and the repository's
`tests/core/dependency_graph.rs` pins it from the other side. Dev-dependencies
do not qualify the engine's runtime independence.

The edge runs one way at the package level: the facade no longer depends on
this crate at all. The dev-dependency on the facade enables its capability
features (`audio`, `image`, `derive`, `websocket`, `bedrock`, …)
unconditionally, so a provider target enumerates the same tests in every lane
instead of shrinking silently when a lane omits `--all-features`.

## The engine

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
responses work with no default features. Enable `bedrock` for Smithy event-stream
decoding and scrubbing; its Smithy dependencies are absent otherwise.

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
provider suites in the root package replay. `fixtures/effects/` holds the
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
points at these same two entrypoints. It deliberately has no dependency on the
cassette engine, facade or provider helpers, preserving replay without
`serde_json/preserve_order` and `serde_json/float_roundtrip`. Selecting
`rig-cassette` alone cannot provide that configuration: its engine enables both
features unconditionally. CI runs the nested package separately with zero
retries, and the shared targets through the default-member graph with two.
The all-features workspace run excludes the nested package to avoid repeating
the same sources with unified features.

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

- here: the cassette engine, both corpora, every cassette-backed provider
  suite with its golden producers, and the behaviour of the bus and the agent
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
RIG_PROVIDER_TEST_MODE=replay cargo test --locked -p rig-cassette --lib --no-default-features
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
