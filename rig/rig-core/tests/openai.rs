//! OpenAI integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test openai`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test openai agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "openai/agent.rs"]
mod agent;
#[cfg(feature = "audio")]
#[path = "openai/audio_generation.rs"]
mod audio_generation;
#[path = "openai/extractor.rs"]
mod extractor;
#[path = "openai/extractor_usage.rs"]
mod extractor_usage;
#[cfg(feature = "image")]
#[path = "openai/image_generation.rs"]
mod image_generation;
#[path = "openai/permission_control.rs"]
mod permission_control;
#[path = "openai/reasoning_roundtrip.rs"]
mod reasoning_roundtrip;
#[path = "openai/reasoning_tool_roundtrip.rs"]
mod reasoning_tool_roundtrip;
#[path = "openai/response_schema.rs"]
mod response_schema;
#[path = "openai/responses_input_item.rs"]
mod responses_input_item;
#[path = "openai/streaming.rs"]
mod streaming;
#[path = "openai/streaming_tools.rs"]
mod streaming_tools;
#[path = "openai/structured_output.rs"]
mod structured_output;
