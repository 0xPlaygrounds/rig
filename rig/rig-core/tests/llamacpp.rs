//! llama.cpp OpenAI-compatible integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test llamacpp`
//!
//! Run all ignored provider-backed tests serially with:
//! `cargo test -p rig-core --test llamacpp -- --ignored --test-threads=1`
//!
//! Use `--test-threads=1` because these ignored tests talk to real model
//! backends, and running them concurrently creates avoidable load-related
//! flakiness.
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test llamacpp llamacpp::agent::completion_smoke -- --ignored`
//!
//! Optional environment variables:
//! - `LLAMACPP_API_BASE_URL` (default: `http://localhost:8080/v1`)
//! - `LLAMACPP_API_KEY` (default: `none`)
//! - `LLAMACPP_MODEL` (default: `model`)

#[path = "common/support.rs"]
mod support;

#[path = "llamacpp/mod.rs"]
mod llamacpp;
