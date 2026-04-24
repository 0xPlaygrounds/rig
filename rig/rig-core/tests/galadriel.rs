#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

//! Galadriel integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test galadriel`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test galadriel -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run the ignored smoke test with:
//! `cargo test -p rig-core --test galadriel galadriel::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "galadriel/mod.rs"]
mod galadriel;
