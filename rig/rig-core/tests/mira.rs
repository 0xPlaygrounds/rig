//! Mira integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test mira`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test mira -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test mira mira::models::list_models_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "mira/mod.rs"]
mod mira;
