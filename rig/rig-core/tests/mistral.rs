//! Mistral integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test mistral`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test mistral -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test mistral mistral::transcription::transcription_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "mistral/mod.rs"]
mod mistral;
