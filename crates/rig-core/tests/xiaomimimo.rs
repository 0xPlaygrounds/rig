#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

//! Xiaomi MiMo integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test xiaomimimo`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test xiaomimimo -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Use XIAOMI_MIMO_API_KEY to set the api key
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test xiaomimimo xiaomimimo::anthropic::anthropic_compatible_completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "xiaomimimo/mod.rs"]
mod xiaomimimo;
