//! Hyperbolic integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test hyperbolic`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test hyperbolic -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test hyperbolic hyperbolic::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "hyperbolic/mod.rs"]
mod hyperbolic;
