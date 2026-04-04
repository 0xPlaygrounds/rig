//! VoyageAI integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test voyageai`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test voyageai -- --ignored`
//!
//! Run the ignored smoke test with:
//! `cargo test -p rig-core --test voyageai voyageai::embeddings::embeddings_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "voyageai/mod.rs"]
mod voyageai;
