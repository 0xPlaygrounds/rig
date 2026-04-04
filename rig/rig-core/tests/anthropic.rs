//! Anthropic integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test anthropic`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test anthropic -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test anthropic anthropic::agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "anthropic/mod.rs"]
mod anthropic;
