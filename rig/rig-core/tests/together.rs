//! Together integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test together`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test together -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test together together::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "together/mod.rs"]
mod together;
