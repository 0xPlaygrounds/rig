//! VoyageAI integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test voyageai`
//!
//! Run the ignored smoke test with:
//! `cargo test -p rig-core --test voyageai embeddings::embeddings_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "voyageai/embeddings.rs"]
mod embeddings;
