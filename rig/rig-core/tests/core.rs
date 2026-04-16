//! Core integration tests that are not provider-specific.
//!
//! Run the target with:
//! `cargo test -p rig-core --test core`
//!
//! Run a single test with:
//! `cargo test -p rig-core --test core core::prompt_response_messages::standard_prompt_returns_string`

#[path = "core/mod.rs"]
mod core;
#[path = "common/reasoning.rs"]
mod reasoning;
