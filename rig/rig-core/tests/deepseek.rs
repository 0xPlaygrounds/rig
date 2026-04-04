//! DeepSeek integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test deepseek`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test deepseek deepseek::agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "deepseek/mod.rs"]
mod deepseek;
