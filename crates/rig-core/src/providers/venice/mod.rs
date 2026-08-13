//! Venice AI API client and Rig integration.
//!
//! [Venice](https://docs.venice.ai/overview/about-venice) is a privacy-focused
//! inference provider whose chat-completions endpoint is a drop-in replacement
//! for OpenAI's. This integration covers the capabilities Rig has traits for:
//!
//! - [`CompletionModel`] — chat completions, streaming, tools, vision, and
//!   structured output, plus Venice's own [`VeniceParameters`] request block
//!   (web search, thinking control, characters);
//! - [`EmbeddingModel`] — `POST /embeddings`;
//! - [`VeniceModelLister`] — `GET /models`;
//! - [`ImageGenerationModel`] — Venice's native `POST /image/generate`
//!   (feature `image`);
//! - [`AudioGenerationModel`] — `POST /audio/speech` (feature `audio`);
//! - [`TranscriptionModel`] — `POST /audio/transcriptions`.
//!
//! Venice's video, image-editing, music, web-augmentation (`/augment/*`),
//! crypto-RPC, character, API-key and billing endpoints have no corresponding
//! Rig trait and are deliberately not wrapped here.
//!
//! Set `VENICE_API_KEY` (and optionally `VENICE_BASE_URL`) to use
//! [`ProviderClient::from_env`](crate::client::ProviderClient::from_env).
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::CompletionModel,
//!     providers::venice,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = venice::Client::from_env()?;
//! let model = client.completion_model(venice::QWEN3_5_9B);
//! let request = model.completion_request("What is Rig?").build();
//! let response = model.completion(request).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! # Venice-specific request parameters
//! ```no_run
//! use rig_core::client::{CompletionClient, ProviderClient};
//! use rig_core::completion::CompletionModel;
//! use rig_core::providers::venice;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = venice::Client::from_env()?;
//! let model = client.completion_model(venice::QWEN3_5_9B);
//! let request = model
//!     .completion_request("What shipped in Rust this month?")
//!     .additional_params(
//!         venice::VeniceParameters::new()
//!             .enable_web_search(venice::WebSearchMode::Auto)
//!             .enable_web_citations(true)
//!             .into_additional_params(),
//!     )
//!     .build();
//!
//! // `raw_completion` keeps Venice's own blocks, including the citations
//! // web search returns.
//! let response = model.raw_completion(request).await?;
//! for citation in response.web_search_citations() {
//!     println!("{} — {}", citation.title, citation.url);
//! }
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod client;
pub mod completion;
pub mod embedding;
#[cfg(feature = "image")]
pub mod image_generation;
pub mod transcription;

#[cfg(feature = "audio")]
pub use audio_generation::*;
pub use client::{Client, ClientBuilder, VENICE_API_BASE_URL, VeniceExt, VeniceModelLister};
pub use completion::*;
pub use embedding::*;
#[cfg(feature = "image")]
pub use image_generation::*;
pub use transcription::*;
