//! Ollama integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test ollama`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test ollama -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test ollama ollama::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "ollama/mod.rs"]
mod ollama;
