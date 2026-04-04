//! Gemini integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test gemini`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test gemini agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "gemini/agent.rs"]
mod agent;
#[path = "gemini/embeddings.rs"]
mod embeddings;
#[path = "gemini/extractor.rs"]
mod extractor;
#[path = "gemini/reasoning_roundtrip.rs"]
mod reasoning_roundtrip;
#[path = "gemini/reasoning_tool_roundtrip.rs"]
mod reasoning_tool_roundtrip;
#[path = "gemini/streaming.rs"]
mod streaming;
#[path = "gemini/streaming_tools.rs"]
mod streaming_tools;
#[path = "gemini/structured_output.rs"]
mod structured_output;
