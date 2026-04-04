//! Moonshot integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test moonshot`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test moonshot -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test moonshot moonshot::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "moonshot/mod.rs"]
mod moonshot;
