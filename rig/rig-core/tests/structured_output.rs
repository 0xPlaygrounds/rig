//! Integration tests: ignored provider-backed smoke tests for structured output.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "structured_output/openai.rs"]
mod openai;

#[path = "structured_output/anthropic.rs"]
mod anthropic;

#[path = "structured_output/gemini.rs"]
mod gemini;
