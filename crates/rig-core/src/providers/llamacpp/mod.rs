//! llama.cpp model identifiers and typed response timings.
//!
//! [`crate::providers::openai::wire::LLAMACPP`] defaults to
//! `http://localhost:8080/v1`. `LLAMACPP_API_BASE_URL` overrides the endpoint;
//! `LLAMACPP_API_KEY` supplies optional bearer authentication for secured servers.
//!
//! ```no_run
//! use rig_core::providers::{llamacpp, openai::wire::{LLAMACPP, OpenAI}};
//! let wire = OpenAI::from_env_with(&LLAMACPP)?.chat(llamacpp::LLAMA_CPP);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod completion;

pub use completion::{CompletionResponse, LLAMA_CPP, Timings};
