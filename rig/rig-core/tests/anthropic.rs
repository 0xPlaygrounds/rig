//! Anthropic integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test anthropic`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test anthropic agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "anthropic/agent.rs"]
mod agent;
#[path = "anthropic/reasoning_roundtrip.rs"]
mod reasoning_roundtrip;
#[path = "anthropic/reasoning_tool_roundtrip.rs"]
mod reasoning_tool_roundtrip;
#[path = "anthropic/streaming.rs"]
mod streaming;
#[path = "anthropic/streaming_tools.rs"]
mod streaming_tools;
#[path = "anthropic/structured_output.rs"]
mod structured_output;
