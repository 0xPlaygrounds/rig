//! OpenRouter integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test openrouter`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test openrouter agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "openrouter/agent.rs"]
mod agent;
#[path = "openrouter/reasoning_roundtrip.rs"]
mod reasoning_roundtrip;
#[path = "openrouter/reasoning_tool_roundtrip.rs"]
mod reasoning_tool_roundtrip;
#[path = "openrouter/streaming_tools.rs"]
mod streaming_tools;
