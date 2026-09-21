//! Provider configurations, endpoint wires, and model identifiers.
//!
//! Bind a configured wire to a transport to execute requests. For serialized
//! provider selection, use [`registry::ProviderRef`] instead of a concrete type.
//!
//! ```no_run
//! use rig_core::providers::openai::{self, wire::OpenAI};
//! let wire = OpenAI::from_env()?.chat(openai::GPT_5_2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
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
pub mod internal;
pub mod llamacpp;
pub mod minimax;
pub mod mira;
pub mod mistral;
pub mod moonshot;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod perplexity;
pub mod registry;
pub mod together;
pub mod venice;
pub mod voyageai;
pub mod xai;
pub mod xiaomimimo;
pub mod zai;
