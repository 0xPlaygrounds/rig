//! Groq integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test groq`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test groq -- --ignored`
//!
//! Run the ignored smoke test with:
//! `cargo test -p rig-core --test groq groq::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "groq/mod.rs"]
mod groq;
