//! Integration tests: ignored provider-backed smoke tests for prompt context.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "context/moonshot.rs"]
mod moonshot;

#[path = "context/together.rs"]
mod together;

#[path = "context/xai.rs"]
mod xai;

#[path = "context/huggingface.rs"]
mod huggingface;
