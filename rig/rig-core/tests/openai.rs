//! OpenAI integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test openai`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test openai openai::agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "openai/mod.rs"]
mod openai;
