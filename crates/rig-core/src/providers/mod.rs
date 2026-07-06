//! Provider integrations included in `rig-core`.
//!
//! - Anthropic
//! - Azure OpenAI
//! - ChatGPT and GitHub Copilot auth-backed clients
//! - Cohere
//! - DeepSeek
//! - Galadriel
//! - Gemini
//! - Groq
//! - Hugging Face
//! - Hyperbolic
//! - LiteLLM
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
//!     agent::AgentBuilder,
//!     client::{CompletionClient, ProviderClient},
//!     providers::openai,
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the OpenAI client
//! let openai = openai::Client::from_env()?;
//!
//! // Create a model and initialize an agent
//! let model = openai.completion_model(openai::GPT_5_2);
//!
//! let agent = AgentBuilder::new(model)
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ")
//!     .build();
//!
//! // Alternatively, you can initialize an agent directly
//! let agent = openai.agent(openai::GPT_5_2)
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ")
//!     .build();
//! # Ok(())
//! # }
//! ```
pub mod anthropic;
pub mod azure;
pub mod chatgpt;
pub mod cohere;
pub mod copilot;
pub mod deepseek;
pub mod galadriel;
pub mod gemini;
pub mod groq;
pub mod huggingface;
pub mod hyperbolic;
pub(crate) mod internal;
pub mod litellm;
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
