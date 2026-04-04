#![cfg(feature = "image")]
//! Integration tests: ignored provider-backed smoke tests for image generation.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "image_generation/openai.rs"]
mod openai;

#[path = "image_generation/hyperbolic.rs"]
mod hyperbolic;

#[path = "image_generation/huggingface.rs"]
mod huggingface;

#[path = "image_generation/xai.rs"]
mod xai;
