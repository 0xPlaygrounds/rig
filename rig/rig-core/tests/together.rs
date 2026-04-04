//! Together integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test together`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test together agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "together/agent.rs"]
mod agent;
#[path = "together/context.rs"]
mod context;
#[path = "together/embeddings.rs"]
mod embeddings;
#[path = "together/streaming.rs"]
mod streaming;
#[path = "together/streaming_tools.rs"]
mod streaming_tools;
#[path = "together/tools.rs"]
mod tools;
