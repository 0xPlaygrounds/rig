//! Copilot integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test copilot`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test copilot -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test copilot copilot::routing::codex_models_route_through_responses -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "copilot/mod.rs"]
mod copilot;
