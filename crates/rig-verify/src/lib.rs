//! The effect bus's behavioural verification suite.
//!
//! Empty on purpose: every test is an integration test under `tests/`,
//! written against the public API of `rig_core` and `rig_agent` only. This
//! crate exists so those suites have a home that builds in seconds (the root
//! `rig` package pulls every provider and the cassette machinery), their own
//! dev-dependencies, and one place a reviewer looks for "what does the bus
//! promise, and which test proves it".
//!
//! The effect corpus (`fixtures/*.effects.json`, `tests/golden_replay.rs`)
//! is the loop this crate exists for. Write or edit a program; run it once
//! against the cassette transport with `record_effects()` (the root suite's
//! producer, under `RIG_REGENERATE_GOLDEN=1`); commit the golden; replay it
//! here with no provider behind any key. A change in what the program asks
//! (a kind), what it was answered (an outcome) or how a stream was delivered
//! (its events) fails the replay naming the record and the JSON pointer of
//! the difference — fix forward, and re-record live when the change is
//! intended, never by hand-editing a golden. Hooks are program (the header
//! names them; a different stack is refused before the first dispatch);
//! tools are record (a replayer answers them); nothing the engine mints is
//! random, so the same program produces the same log twice.
//!
//! What lives where:
//! - here: behaviour of the bus and the agent over it (record and replay,
//!   durable execution, the two interpreters agreeing);
//! - the root package's `tests/core`: guards that scan the source tree and
//!   the fixture runners (they need the repository root);
//! - the root package's `tests/providers`: cassette-backed provider suites;
//! - `rig-core`/`rig-agent` unit tests: anything that needs crate-private
//!   types (the loom models among them).
