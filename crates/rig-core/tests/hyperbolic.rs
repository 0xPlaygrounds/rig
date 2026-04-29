#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

//! Hyperbolic integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test hyperbolic`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test hyperbolic -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test hyperbolic hyperbolic::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "hyperbolic/mod.rs"]
mod hyperbolic;
