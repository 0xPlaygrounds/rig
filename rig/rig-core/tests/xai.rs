//! xAI integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test xai`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test xai agent::completion_smoke -- --ignored`

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "xai/agent.rs"]
mod agent;
#[cfg(feature = "audio")]
#[path = "xai/audio_generation.rs"]
mod audio_generation;
#[path = "xai/context.rs"]
mod context;
#[cfg(feature = "image")]
#[path = "xai/image_generation.rs"]
mod image_generation;
#[path = "xai/loaders.rs"]
mod loaders;
#[path = "xai/reasoning_roundtrip.rs"]
mod reasoning_roundtrip;
#[path = "xai/reasoning_tool_roundtrip.rs"]
mod reasoning_tool_roundtrip;
#[path = "xai/streaming.rs"]
mod streaming;
#[path = "xai/tools.rs"]
mod tools;
