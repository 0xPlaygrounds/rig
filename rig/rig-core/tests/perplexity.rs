//! Perplexity integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test perplexity`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test perplexity -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test perplexity perplexity::perplexity_agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "perplexity/mod.rs"]
mod perplexity;
