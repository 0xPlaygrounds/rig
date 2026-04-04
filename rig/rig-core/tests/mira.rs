//! Mira integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test mira`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test mira mira::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "mira/mod.rs"]
mod mira;
