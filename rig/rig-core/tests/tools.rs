//! Integration tests: ignored provider-backed smoke tests for agent tools.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "tools/cohere.rs"]
mod cohere;

#[path = "tools/deepseek.rs"]
mod deepseek;

#[path = "tools/together.rs"]
mod together;

#[path = "tools/xai.rs"]
mod xai;

#[path = "tools/huggingface.rs"]
mod huggingface;

#[path = "tools/mira.rs"]
mod mira;
