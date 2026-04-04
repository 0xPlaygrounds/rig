//! Azure integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test azure`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test azure -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test azure azure::transcription::transcription_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "azure/mod.rs"]
mod azure;
