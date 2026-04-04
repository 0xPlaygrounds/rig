//! Together integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test together`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test together -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test together together::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "together/mod.rs"]
mod together;
