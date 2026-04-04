//! Moonshot integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test moonshot`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test moonshot agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "moonshot/agent.rs"]
mod agent;
#[path = "moonshot/context.rs"]
mod context;
