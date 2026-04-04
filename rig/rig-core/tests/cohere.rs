//! Cohere integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test cohere`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test cohere agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "cohere/agent.rs"]
mod agent;
#[path = "cohere/streaming.rs"]
mod streaming;
#[path = "cohere/streaming_tools.rs"]
mod streaming_tools;
#[path = "cohere/tools.rs"]
mod tools;
