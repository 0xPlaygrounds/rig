#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

//! Mistral integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test mistral`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test mistral -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test mistral mistral::transcription::transcription_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "mistral/mod.rs"]
mod mistral;
