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
//! - llama.cpp (`llama-server`, and llamafile)
//! - MiniMax
//! - Mira
//! - Mistral
//! - Moonshot
//! - Ollama
//! - OpenAI
//! - OpenRouter
//! - Perplexity
//! - Together
//! - Venice
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
//! Every concrete provider is behind its same-named Cargo feature. The default
//! build exposes only provider-neutral contracts and shared infrastructure;
//! `providers-all` is the explicit aggregate for the full module tree.
//!
//! # Provider implementation checklist
//!
//! When adding or changing a provider, verify that the integration includes:
//!
//! - for OpenAI-chat-compatible APIs: completions driven by
//!   `openai_compatible::completion::GenericCompletionModel`
//!   via an `openai_compatible::completion::OpenAICompatibleProvider`
//!   impl on the provider extension (never a hand-rolled completion model,
//!   request struct, or message conversion — dialect differences go in the
//!   trait's hooks);
//! - a same-named feature in `rig-core`, `rig-reqwest`, and the root `rig`
//!   facade, inclusion in each `providers-all` aggregate, and corresponding
//!   module/re-export/test gates;
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
//!   multimodal content where applicable, built through the
//!   [`CompletionResponse`](crate::completion::CompletionResponse) `new`/`with_*`
//!   builders rather than a struct literal — the `with_*_finish_reason` setters
//!   are what apply
//!   [`FinishReason::reconcile_with_output`](crate::completion::FinishReason::reconcile_with_output);
//! - a finish-reason mapping covering every value the provider can report,
//!   with anything unrecognized preserved verbatim in
//!   [`FinishReason::Other`](crate::completion::FinishReason::Other) rather
//!   than guessed at;
//! - a shared conversion (one used by several OpenAI-compatible providers)
//!   that takes the provider descriptor name as an input instead of hardcoding
//!   one, so a reused wire type cannot mislabel its provider;
//! - `raw_completion` and `raw_stream` inherent methods returning the
//!   provider's own wire types, with the normalized
//!   [`CompletionModel`](crate::completion::CompletionModel) methods delegating
//!   to them so there is exactly one request path either way;
//! - streaming support when the provider supports streaming;
//! - provider-response error preservation plus `ProviderResponseExt` and
//!   telemetry fields consistent with nearby providers where applicable;
//! - unit, cassette, or live-test coverage appropriate to the changed behavior;
//! - root facade feature/docs updates for companion provider crates; and
//! - examples and documentation that match the actual API, feature flags, and
//!   credential requirements.
//!
//! # Example
//! ```ignore
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::{AssistantContent, CompletionModel},
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the OpenAI client
//! let openai = openai::Client::from_env()?;
//!
//! // Create a model and send a low-level completion request.
//! let model = openai.completion_model(openai::GPT_5_2);
//! let request = model
//!     .completion_request("Discuss the fate of Middle Earth.")
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ".to_string())
//!     .build();
//! let response = model.completion(request).await?;
//! for item in response.choice {
//!     if let AssistantContent::Text(text) = item {
//!         println!("{}", text.text);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
#[cfg(any(
    feature = "anthropic",
    feature = "minimax",
    feature = "moonshot",
    feature = "xiaomimimo",
    feature = "zai"
))]
#[doc(hidden)]
#[path = "anthropic/mod.rs"]
pub mod anthropic_compatible;
#[cfg(feature = "anthropic")]
#[cfg_attr(docsrs, doc(cfg(feature = "anthropic")))]
pub use anthropic_compatible as anthropic;
#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub mod azure;
#[cfg(feature = "chatgpt")]
#[cfg_attr(docsrs, doc(cfg(feature = "chatgpt")))]
pub mod chatgpt;
#[cfg(feature = "cohere")]
#[cfg_attr(docsrs, doc(cfg(feature = "cohere")))]
pub mod cohere;
#[cfg(feature = "copilot")]
#[cfg_attr(docsrs, doc(cfg(feature = "copilot")))]
pub mod copilot;
#[cfg(feature = "deepseek")]
#[cfg_attr(docsrs, doc(cfg(feature = "deepseek")))]
pub mod deepseek;
#[cfg(feature = "doubleword")]
#[cfg_attr(docsrs, doc(cfg(feature = "doubleword")))]
pub mod doubleword;
#[cfg(feature = "gemini")]
#[cfg_attr(docsrs, doc(cfg(feature = "gemini")))]
pub mod gemini;
#[cfg(feature = "groq")]
#[cfg_attr(docsrs, doc(cfg(feature = "groq")))]
pub mod groq;
#[cfg(feature = "huggingface")]
#[cfg_attr(docsrs, doc(cfg(feature = "huggingface")))]
pub mod huggingface;
#[cfg(feature = "hyperbolic")]
#[cfg_attr(docsrs, doc(cfg(feature = "hyperbolic")))]
pub mod hyperbolic;
pub mod internal;
#[cfg(feature = "llamacpp")]
#[cfg_attr(docsrs, doc(cfg(feature = "llamacpp")))]
pub mod llamacpp;
#[cfg(feature = "minimax")]
#[cfg_attr(docsrs, doc(cfg(feature = "minimax")))]
pub mod minimax;
#[cfg(feature = "mira")]
#[cfg_attr(docsrs, doc(cfg(feature = "mira")))]
pub mod mira;
#[cfg(feature = "mistral")]
#[cfg_attr(docsrs, doc(cfg(feature = "mistral")))]
pub mod mistral;
#[cfg(feature = "moonshot")]
#[cfg_attr(docsrs, doc(cfg(feature = "moonshot")))]
pub mod moonshot;
#[cfg(feature = "ollama")]
#[cfg_attr(docsrs, doc(cfg(feature = "ollama")))]
pub mod ollama;
// OpenAI-compatible providers share this protocol implementation, while the
// concrete OpenAI namespace is only exported when `openai` is enabled.
#[cfg(any(
    feature = "azure",
    feature = "chatgpt",
    feature = "copilot",
    feature = "deepseek",
    feature = "doubleword",
    feature = "groq",
    feature = "huggingface",
    feature = "hyperbolic",
    feature = "llamacpp",
    feature = "minimax",
    feature = "mira",
    feature = "mistral",
    feature = "moonshot",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "together",
    feature = "venice",
    feature = "xai",
    feature = "xiaomimimo",
    feature = "zai",
))]
#[doc(hidden)]
#[path = "openai/mod.rs"]
pub mod openai_compatible;
#[cfg(feature = "openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai")))]
pub use openai_compatible as openai;
#[cfg(feature = "openrouter")]
#[cfg_attr(docsrs, doc(cfg(feature = "openrouter")))]
pub mod openrouter;
#[cfg(feature = "perplexity")]
#[cfg_attr(docsrs, doc(cfg(feature = "perplexity")))]
pub mod perplexity;
#[cfg(feature = "together")]
#[cfg_attr(docsrs, doc(cfg(feature = "together")))]
pub mod together;
#[cfg(feature = "venice")]
#[cfg_attr(docsrs, doc(cfg(feature = "venice")))]
pub mod venice;
#[cfg(feature = "voyageai")]
#[cfg_attr(docsrs, doc(cfg(feature = "voyageai")))]
pub mod voyageai;
#[cfg(feature = "xai")]
#[cfg_attr(docsrs, doc(cfg(feature = "xai")))]
pub mod xai;
#[cfg(feature = "xiaomimimo")]
#[cfg_attr(docsrs, doc(cfg(feature = "xiaomimimo")))]
pub mod xiaomimimo;
#[cfg(feature = "zai")]
#[cfg_attr(docsrs, doc(cfg(feature = "zai")))]
pub mod zai;
