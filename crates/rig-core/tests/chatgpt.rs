#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

//! ChatGPT integration tests.
//!
//! Run the provider target with:
//! `cargo test -p rig-core --test chatgpt`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test chatgpt -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test chatgpt chatgpt::completion::system_messages_are_lifted_into_instructions -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "chatgpt/mod.rs"]
mod chatgpt;
