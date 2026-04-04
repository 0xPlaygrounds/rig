//! Integration tests: ignored provider-backed smoke tests for bare embeddings.
//!
//! These tests make live provider API calls and expect the relevant API key to
//! already be exported in the shell environment.

#[path = "common/support.rs"]
mod support;

#[path = "embeddings/gemini.rs"]
mod gemini;

#[path = "embeddings/together.rs"]
mod together;

#[path = "embeddings/voyageai.rs"]
mod voyageai;
