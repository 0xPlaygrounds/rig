//! OpenAI integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test openai`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test openai -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable rate-limit,
//! quota, and load-related flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test openai openai::agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "openai/mod.rs"]
mod openai;
