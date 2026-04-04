//! Integration tests: ignored provider-backed smoke tests for loader-backed context.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "loaders/xai.rs"]
mod xai;

#[path = "loaders/huggingface.rs"]
mod huggingface;
