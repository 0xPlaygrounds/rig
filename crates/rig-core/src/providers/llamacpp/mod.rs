//! llama.cpp (`llama-server`) API client and Rig integration.
//!
//! [llama.cpp](https://github.com/ggml-org/llama.cpp) ships `llama-server`, an
//! OpenAI-compatible HTTP server that serves a GGUF model from local hardware.
//! Started with no arguments beyond a model it listens on
//! `http://localhost:8080` and exposes `/v1/chat/completions`, `/v1/embeddings`,
//! `/v1/models`, `/v1/rerank` and more.
//!
//! # This module replaces `providers::llamafile`
//!
//! Rig used to reach the same server through a provider named after Mozilla's
//! [llamafile](https://github.com/Mozilla-Ocho/llamafile) distribution. A
//! `.llamafile` bundles the *same* llama.cpp server into a single executable
//! and serves the *same* OpenAI-compatible API, so one provider covers both:
//! point [`Client`] at a running `.llamafile` and everything works exactly as
//! it does against `llama-server`. The rename is to the name people search for.
//! `llamafile::Client::from_url(url)` becomes `llamacpp::Client::from_url(url)`
//! and nothing else changes; see `MIGRATING.md`.
//!
//! # Base URL
//!
//! [`Client::from_url`] and the builder's `base_url` take the server root
//! (`http://localhost:8080`); the `/v1` prefix is this provider's business, not
//! the caller's. Passing a URL that already ends in `/v1` is also accepted and
//! does not double up — see [`LlamacppExt::build_uri`].
//!
//! # Authentication
//!
//! A local `llama-server` needs no credential, and the default client sends no
//! `Authorization` header at all. A server started with `--api-key <key>`
//! rejects everything else with 401, so pass the key and the client sends
//! `Authorization: Bearer <key>`:
//!
//! ```no_run
//! # use rig_core::providers::llamacpp;
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // No credential — the common local case.
//! let local = llamacpp::Client::from_url("http://localhost:8080")?;
//!
//! // `llama-server --api-key hunter2`
//! let secured = llamacpp::Client::builder()
//!     .api_key("hunter2")
//!     .base_url("http://localhost:8080")
//!     .build()?;
//! # let _ = (local, secured);
//! # Ok(())
//! # }
//! ```
//!
//! # Capabilities
//!
//! `completion`, `embeddings`, `model_listing` and `rerank` are implemented.
//! `transcription`, `image_generation` and `audio_generation` are
//! [`Nothing`](crate::client::Nothing); see [`client`] for the reason attached
//! to each.
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     client::CompletionClient,
//!     completion::CompletionModel,
//!     providers::llamacpp,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new llama.cpp client (defaults to http://localhost:8080)
//! let client = llamacpp::Client::from_url("http://localhost:8080")?;
//!
//! // Send a completion request with a preamble.
//! let model = client.completion_model(llamacpp::LLAMA_CPP);
//! let request = model
//!     .completion_request("Hello!")
//!     .preamble("You are a helpful assistant.".to_string())
//!     .build();
//! let response = model.completion(request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod rerank;

pub use client::{Client, ClientBuilder, LlamacppApiKey, LlamacppBuilder, LlamacppExt};
pub use completion::{CompletionModel, LLAMA_CPP};
pub use embedding::EmbeddingModel;
pub use rerank::RerankModel;
