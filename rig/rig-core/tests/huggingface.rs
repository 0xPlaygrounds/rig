//! Hugging Face integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test huggingface`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test huggingface agent::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "huggingface/agent.rs"]
mod agent;
#[path = "huggingface/context.rs"]
mod context;
#[cfg(feature = "image")]
#[path = "huggingface/image_generation.rs"]
mod image_generation;
#[path = "huggingface/loaders.rs"]
mod loaders;
#[path = "huggingface/streaming.rs"]
mod streaming;
#[path = "huggingface/tools.rs"]
mod tools;
