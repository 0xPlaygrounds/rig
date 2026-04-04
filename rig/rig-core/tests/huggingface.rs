//! Hugging Face integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test huggingface`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test huggingface -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test huggingface huggingface::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "huggingface/mod.rs"]
mod huggingface;
