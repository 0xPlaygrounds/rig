//! Cohere integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test cohere`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test cohere cohere::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "cohere/mod.rs"]
mod cohere;
