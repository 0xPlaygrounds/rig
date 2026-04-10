//! Moonshot integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test moonshot`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test moonshot -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test moonshot moonshot::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "moonshot/mod.rs"]
mod moonshot;
