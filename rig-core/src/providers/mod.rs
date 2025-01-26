//! This module contains clients for the different LLM providers that Rig supports.
//!
//! Currently, the following providers are supported:
//! - Cohere
//! - OpenAI
//! - Perplexity
//! - Anthropic
//! - Google Gemini
//! - xAI
//! - EternalAI
//! - DeepSeek
//!
//! Each provider has its own module, which contains a `Client` implementation that can
//! be used to initialize completion and embedding models and execute requests to those models.
//!
//! The clients also contain methods to easily create higher level AI constructs such as
//! agents and RAG systems, reducing the need for boilerplate.
//!
//! # Example
//! ```
//! use rig::{providers::openai, agent::AgentBuilder};
//!
//! // Initialize the OpenAI client
//! let openai = openai::Client::new("your-openai-api-key");
//!
//! // Create a model and initialize an agent
//! let gpt_4o = openai.completion_model("gpt-4o");
//!
//! let agent = AgentBuilder::new(gpt_4o)
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ")
//!     .build();
//!
//! // Alternatively, you can initialize an agent directly
//! let agent = openai.agent("gpt-4o")
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ")
//!     .build();
//! ```
//! Note: The example above uses the OpenAI provider client, but the same pattern can
//! be used with the Cohere provider client.
pub mod anthropic;
pub mod cohere;
pub mod deepseek;
pub mod gemini;
pub mod hyperbolic;
pub mod openai;
pub mod perplexity;
pub mod xai;
