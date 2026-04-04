//! Groq integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test groq`
//!
//! Run the ignored smoke test with:
//! `cargo test -p rig-core --test groq groq::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "groq/mod.rs"]
mod groq;
