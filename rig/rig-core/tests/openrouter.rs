//! OpenRouter integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test openrouter`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test openrouter -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test openrouter openrouter::agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "openrouter/mod.rs"]
mod openrouter;
