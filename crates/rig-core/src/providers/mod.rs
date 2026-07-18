//! Provider integrations included in `rig-core`.
//!
//! - Anthropic
//! - Azure OpenAI
//! - ChatGPT and GitHub Copilot auth-backed clients
//! - Cohere
//! - DeepSeek
//! - Gemini
//! - Groq
//! - Hugging Face
//! - Hyperbolic
//! - Llamafile
//! - MiniMax
//! - Mira
//! - Mistral
//! - Moonshot
//! - Ollama
//! - OpenAI
//! - OpenRouter
//! - Perplexity
//! - Together
//! - Voyage AI
//! - xAI
//! - Xiaomi MiMo
//! - Z.ai
//!
//! Each provider module defines a `Client` type and model types for the
//! capabilities it supports. Capability traits such as
//! [`CompletionClient`](crate::client::CompletionClient) and
//! [`EmbeddingsClient`](crate::client::EmbeddingsClient) are implemented only
//! when the provider declares that capability.
//!
//! # Provider implementation checklist
//!
//! When adding or changing a provider, verify that the integration includes:
//!
//! - for OpenAI-chat-compatible APIs: completions driven by
//!   [`GenericCompletionModel`](crate::providers::openai::completion::GenericCompletionModel)
//!   via an
//!   [`OpenAICompatibleProvider`](crate::providers::openai::completion::OpenAICompatibleProvider)
//!   impl on the provider extension (never a hand-rolled completion model,
//!   request struct, or message conversion — dialect differences go in the
//!   trait's hooks);
//! - public `Client` and `ClientBuilder` aliases with the correct generics,
//!   including a `ClientBuilder` API-key generic matching `ProviderBuilder::ApiKey`;
//! - the `Provider`, `ProviderBuilder`, `Capabilities`, and `ProviderClient`
//!   implementations;
//! - explicit API-key marker/auth types with redacted debug behavior for
//!   credential-bearing values;
//! - model constants where they are useful and current;
//! - request conversion from Rig request types, such as
//!   [`CompletionRequest`](crate::completion::CompletionRequest), without
//!   inventing unsupported provider API fields;
//! - response conversion into Rig response types, including usage and tool or
//!   multimodal content where applicable;
//! - streaming support when the provider supports streaming;
//! - provider-response error preservation plus `ProviderResponseExt` and
//!   telemetry fields consistent with nearby providers where applicable;
//! - unit, cassette, or live-test coverage appropriate to the changed behavior;
//! - root facade feature/docs updates for companion provider crates; and
//! - examples and documentation that match the actual API, feature flags, and
//!   credential requirements.
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::CompletionModel,
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::from_env()?;
//! let model = client.completion_model(openai::GPT_5_2);
//! let response = model
//!     .completion(model.completion_request("Hello").build())
//!     .await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```
pub mod anthropic;
pub mod azure;
pub mod chatgpt;
pub mod cohere;
pub mod copilot;
pub mod deepseek;
pub mod doubleword;
pub mod gemini;
pub mod groq;
pub mod huggingface;
pub mod hyperbolic;
pub(crate) mod internal;
pub mod llamafile;
pub mod minimax;
pub mod mira;
pub mod mistral;
pub mod moonshot;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod voyageai;
pub mod xai;
pub mod xiaomimimo;
pub mod zai;
