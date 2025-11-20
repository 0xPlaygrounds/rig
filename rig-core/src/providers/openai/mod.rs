//! OpenAI API client and Rig integration
//!
//! This module provides support for both OpenAI APIs:
//! - **Responses API** (default): The newer API supporting reasoning, structured outputs, and advanced features
//! - **Completions API**: The traditional Chat Completions API
//!
//! # Responses API (Default)
//! ```
//! use rig::providers::openai;
//!
//! let client = openai::Client::new("YOUR_API_KEY");
//! let gpt4o = client.completion_model(openai::GPT_4O);
//! ```
//!
//! # Completions API
//! ```
//! use rig::providers::openai;
//!
//! let client = openai::CompletionsClient::new("YOUR_API_KEY");
//! let gpt4o = client.completion_model(openai::GPT_4O);
//! ```
//!
//! # Switching Between APIs
//! ```
//! use rig::providers::openai;
//!
//! let responses_client = openai::Client::new("YOUR_API_KEY");
//! let completions_client = responses_client.completions_api();
//! let back_to_responses = completions_client.responses_api();
//! ```
pub mod client;
pub mod completion;
pub mod embedding;
pub mod responses_api;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;

pub mod transcription;

pub use client::{
    Client, ClientBuilder, CompletionsClient, CompletionsClientBuilder, OpenAICompletionsExt,
    OpenAIResponsesExt,
};
pub use completion::*;
pub use embedding::*;

#[cfg(feature = "audio")]
pub use audio_generation::{AudioGenerationModel, TTS1, TTS1hd};

#[cfg(feature = "image")]
pub use image_generation::*;
pub use streaming::*;
pub use transcription::*;
