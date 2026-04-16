//! VoyageAI integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test voyageai`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test voyageai -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run the ignored smoke test with:
//! `cargo test -p rig-core --test voyageai voyageai::embeddings::embeddings_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "voyageai/mod.rs"]
mod voyageai;
