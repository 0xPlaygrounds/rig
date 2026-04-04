#![cfg(feature = "audio")]
//! Integration tests: ignored provider-backed smoke tests for audio generation.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "audio_generation/openai.rs"]
mod openai;

#[path = "audio_generation/hyperbolic.rs"]
mod hyperbolic;

#[path = "audio_generation/xai.rs"]
mod xai;
