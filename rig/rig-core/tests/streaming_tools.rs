//! Integration tests: ignored provider-backed smoke tests for streaming with tools.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "streaming_tools/openai.rs"]
mod openai;

#[path = "streaming_tools/anthropic.rs"]
mod anthropic;

#[path = "streaming_tools/cohere.rs"]
mod cohere;

#[path = "streaming_tools/together.rs"]
mod together;

#[path = "streaming_tools/openrouter.rs"]
mod openrouter;

#[path = "streaming_tools/gemini.rs"]
mod gemini;
