//! Portable streaming values re-exported for the classic agent runtime.
//!
//! The streaming prompting *traits* (`StreamingPrompt`/`StreamingChat`) are
//! gone: use [`Agent::stream_prompt`](crate::Agent::stream_prompt) and
//! [`Agent::stream_chat`](crate::Agent::stream_chat), which return the
//! [`StreamingPromptRequest`](crate::agent::StreamingPromptRequest) driver.

pub use rig_core::streaming::*;
