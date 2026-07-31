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
//! Each provider module exposes a `functions` submodule: the provider as plain
//! data plus free functions. A serde `Config` (base URL, credential
//! [location](descriptor::ApiKeyLocation), model, extra headers), a
//! [`ProviderDescriptor`] capability sheet, pure `build_request`/`parse_response`
//! functions, and async `complete`/`open_stream` (plus `embed`, `transcribe`,
//! `generate_image`, `generate_audio`, `rerank`, `list_models` where the
//! provider supports them) over the shared
//! [`HttpRuntime`](crate::http_runtime::HttpRuntime).
//!
//! There is no client type and no capability trait: a provider supports a
//! capability exactly when its `functions` module exposes the corresponding
//! function, and the descriptor records the wire-level knobs
//! (`supports_tools`, `max_embedding_documents`, …) that request building
//! consults.
//!
//! Every `Config` has `new(model)` for explicit construction and
//! `from_env(model)` for the provider's conventional environment variables.
//!
//! # Provider implementation checklist
//!
//! When adding or changing a provider, verify that the integration includes:
//!
//! - a `functions` module with a serde `Config` (`new` + `from_env`), a
//!   `DESCRIPTOR`, pure `build_request`/`parse_response`, and async wrappers
//!   over [`HttpRuntime`](crate::http_runtime::HttpRuntime);
//! - for OpenAI-chat-compatible APIs: a `build_body` composed from
//!   `openai::functions::compatible_typed_request` and
//!   `compatible_body_value`, with the provider's own
//!   dialect steps in between (never a hand-rolled request struct or message
//!   conversion), and a `STREAM_DIALECT` built from the descriptor;
//! - credentials expressed as [`ApiKeyLocation`]
//!   so they are resolved at request time and redacted in `Debug`;
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
//!     completion::AssistantContent,
//!     http_runtime::HttpRuntime,
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // The provider is plain data; the runtime owns the HTTP transport.
//! let cfg = openai::functions::Config::from_env(openai::GPT_5_2)?;
//! let rt = HttpRuntime::new();
//!
//! let request =
//!     rig_core::completion::CompletionRequest::builder("Discuss the fate of Middle Earth.")
//!         .preamble(
//!             "You are Gandalf the white and you will be conversing with other \
//!             powerful beings to discuss the fate of Middle Earth.",
//!         )
//!         .build();
//! let response = openai::functions::complete(&cfg, &rt, request).await?;
//! for item in response.choice {
//!     if let AssistantContent::Text(text) = item {
//!         println!("{}", text.text);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
pub mod descriptor;

pub use descriptor::{
    ApiKeyError, ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var,
    required_env_var,
};

pub mod verify;

pub use verify::VerifyError;

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
