#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

//! Core integration tests that are not provider-specific.
//!
//! Run the target with:
//! `cargo test -p rig --test core`
//!
//! Run a single test with:
//! `cargo test -p rig --test core core::prompt_response_messages::standard_prompt_returns_string`

#[path = "core/mod.rs"]
mod core;
use rig_test_support::goldens;
use rig_test_support::reasoning;
