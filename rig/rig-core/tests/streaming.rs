//! Integration tests: ignored provider-backed smoke tests for streaming prompt responses.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "streaming/openai.rs"]
mod openai;

#[path = "streaming/anthropic.rs"]
mod anthropic;

#[path = "streaming/cohere.rs"]
mod cohere;

#[path = "streaming/together.rs"]
mod together;

#[path = "streaming/xai.rs"]
mod xai;

#[path = "streaming/gemini.rs"]
mod gemini;

#[path = "streaming/huggingface.rs"]
mod huggingface;
