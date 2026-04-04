//! Galadriel integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test galadriel`
//!
//! Run the ignored smoke test with:
//! `cargo test -p rig-core --test galadriel galadriel::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "galadriel/mod.rs"]
mod galadriel;
