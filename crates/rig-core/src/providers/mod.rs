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
