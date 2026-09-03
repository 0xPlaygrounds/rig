//! The effect bus's behavioural verification suite.
//!
//! Empty on purpose: every test is an integration test under `tests/`,
//! written against the public API of `rig_core` and `rig_agent` only. This
//! crate exists so those suites have a home that builds in seconds (the root
//! `rig` package pulls every provider and the cassette machinery), their own
//! dev-dependencies, and one place a reviewer looks for "what does the bus
//! promise, and which test proves it".
//!
//! The effect corpus (`fixtures/*.effects.json`, `tests/golden_replay.rs`
//! and the `tests/corpus_*.rs` matrices) is the loop this crate exists
//! for. Write or edit a program; run it once against the cassette
//! transport with `record_effects()` (the root suite's producer, under
//! `RIG_REGENERATE_GOLDEN=1`); commit the golden; replay it here with no
//! provider behind any key. A new scenario is recorded on the producer's
//! exact test filter with `RIG_PROVIDER_TEST_MODE=record` (the golden call
//! is a no-op in that mode, so the cassette is written), replayed once on
//! the same filter, and its golden is then generated in replay mode
//! (`RIG_REGENERATE_GOLDEN=1`), so it holds the cassette's placeholders,
//! not live ids. A change in what the program asks (a kind), what it was
//! answered (an outcome) or how a stream was delivered (its events) fails
//! the replay naming the record and the JSON pointer of the difference —
//! fix forward, and re-record live when the change is intended, never by
//! hand-editing a golden. Hooks are program (the header names them; a
//! different stack is refused before the first dispatch); tools are record
//! (a replayer answers them); nothing the engine mints is random, so the
//! same program produces the same log twice.
//!
//! The corpus is a set of matrices, each a module under `tests/` with its
//! dimension table, its cells and what it found, over the two interpreters
//! and the program table in `tests/corpus/mod.rs` (which also holds the
//! dimension table of an effect trace as a whole):
//!
//! | matrix | module | cells |
//! |---|---|---|
//! | the original corpus | `golden_replay.rs` | 10 goldens, three providers and a mock |
//! | A, retrieval effects | `corpus_retrieval.rs` | 12 goldens, gemini and openai |
//! | B, the hook surface | `corpus_hooks.rs` | 12 goldens, one per hook decision |
//! | C, serving policy, routing and bus ownership | `corpus_serving.rs` | 12 goldens |
//! | D, continuation, cancellation and failure outcomes | `corpus_outcome.rs` | 8 goldens and 4 resume rows |
//! | E, request-shape axes | `corpus_request_shape.rs` | 13 goldens, one axis each |
//!
//! Every golden is replayed by both interpreters (a cancelled stream by
//! both too: the replayer answers the record as the cancel it was, after
//! the events it kept). The
//! producers are the `golden_fixture` tests of the root suite
//! (`tests/providers/*/cassette/corpus_*.rs`, `tests/core/golden_*.rs`),
//! paired one-to-one with the goldens by `tests/core/golden_pairing.rs`.
//!
//! What lives where:
//! - here: behaviour of the bus and the agent over it (record and replay,
//!   durable execution, the two interpreters agreeing);
//! - the root package's `tests/core`: guards that scan the source tree and
//!   the fixture runners (they need the repository root);
//! - the root package's `tests/providers`: cassette-backed provider suites;
//! - `rig-core`/`rig-agent` unit tests: anything that needs crate-private
//!   types (the loom models among them).
