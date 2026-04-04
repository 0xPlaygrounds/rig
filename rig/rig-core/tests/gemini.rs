//! Gemini integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test gemini`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test gemini -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test gemini gemini::agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "gemini/mod.rs"]
mod gemini;
