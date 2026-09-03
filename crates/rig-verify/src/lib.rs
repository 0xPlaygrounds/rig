//! The effect bus's behavioural verification suite.
//!
//! Empty on purpose: every test is an integration test under `tests/`,
//! written against the public API of `rig_core` and `rig_agent` only. This
//! crate exists so those suites have a home that builds in seconds (the root
//! `rig` package pulls every provider and the cassette machinery), their own
//! dev-dependencies, and one place a reviewer looks for "what does the bus
//! promise, and which test proves it".
//!
//! What lives where:
//! - here: behaviour of the bus and the agent over it (record and replay,
//!   durable execution, the two interpreters agreeing);
//! - the root package's `tests/core`: guards that scan the source tree and
//!   the fixture runners (they need the repository root);
//! - the root package's `tests/providers`: cassette-backed provider suites;
//! - `rig-core`/`rig-agent` unit tests: anything that needs crate-private
//!   types (the loom models among them).
