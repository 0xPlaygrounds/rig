//! xAI integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test xai`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test xai xai::agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "xai/mod.rs"]
mod xai;
