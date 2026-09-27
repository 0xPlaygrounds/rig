//! llama.cpp model identifiers and typed response timings.
//!
//! [`from_env`] and [`new`] build a client on the
//! [`LLAMACPP`](crate::providers::openai::wire::LLAMACPP) dialect, which defaults to
//! `http://localhost:8080/v1`. `LLAMACPP_API_BASE_URL` overrides the endpoint;
//! `LLAMACPP_API_KEY` supplies optional bearer authentication for secured servers.
//!
//! ```no_run
//! use rig_core::providers::llamacpp;
//! let model = llamacpp::from_env()?.chat(llamacpp::LLAMA_CPP);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod completion;

pub use completion::{CompletionResponse, LLAMA_CPP, Timings};

crate::client::macros::openai_vendor!(crate::providers::openai::wire::LLAMACPP, "llama.cpp");
