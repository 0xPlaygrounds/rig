//! Integration tests: ignored provider-backed smoke tests for extractors.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "extractor_live/openai.rs"]
mod openai;

#[path = "extractor_live/deepseek.rs"]
mod deepseek;

#[path = "extractor_live/gemini.rs"]
mod gemini;
