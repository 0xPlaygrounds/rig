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
//! Each provider module defines a configuration type plus one
//! [`Wire`](crate::wire::Wire) per endpoint it speaks. Binding that
//! configuration to a transport yields a [`Bound`](crate::driver::Bound), and
//! a capability is a method on it — `completion(model)`, `embedding(model,
//! ndims)`, `models()` — present exactly when the provider declares the
//! matching wire through [`HasCompletion`](crate::wire::HasCompletion),
//! [`HasEmbedding`](crate::driver::HasEmbedding) and their siblings.
//!
//! A caller that names a provider in *code* reaches for the module directly.
//! A caller that names one in *data* — a config file, a database row, a
//! saved world — reaches for [`registry`]: a validated
//! [`ProviderId`](registry::ProviderId) selects a vendor and a protocol
//! family and yields that provider's preset configuration, and a
//! [`ProviderConfig`](registry::ProviderConfig) carries an explicit one.
//! Both travel as a [`ProviderRef`](registry::ProviderRef).
//!
//! # Provider implementation checklist
//!
//! When adding or changing a provider, verify that the integration includes:
//!
//! - for OpenAI-chat-compatible APIs: completions driven by the shared
//!   [`Chat`](crate::providers::openai::wire::Chat) wire and its
//!   [`ChatDecoder`](crate::providers::openai::wire::ChatDecoder), with the
//!   provider contributing one
//!   [`Dialect`](crate::providers::openai::wire::Dialect) **const** beside
//!   [`OPENAI`](crate::providers::openai::wire::OPENAI) (never a hand-rolled
//!   wire, request struct, or message conversion — dialect differences are
//!   fields on that const, and a quirk no field can express is one arm of its
//!   [`BodyRewrite`](crate::providers::openai::wire::BodyRewrite));
//! - a configuration type built by `new(key)` / `from_env()?` and joined to a
//!   transport by [`Bind::bind`](crate::driver::Bind::bind) or `bound()?`,
//!   plus root re-exports of the configuration type and every wire type;
//! - one [`Wire`](crate::wire::Wire) impl per endpoint and one `Has*` impl per
//!   supported capability, so a capability the provider does not have is a
//!   method that does not exist rather than one that fails at runtime;
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
//! - the provider's own reply document carried verbatim on
//!   [`CompletionResponse::raw`](crate::completion::CompletionResponse::raw),
//!   so a caller who wants it typed deserializes the provider's own reply type
//!   out of it — one request path, and no second normalization;
//! - streaming support when the provider supports streaming;
//! - the provider's error body preserved verbatim, through
//!   [`WireError::http_response`](crate::wire::WireError::http_response) for a
//!   non-success reply and
//!   [`WireError::provider_body`](crate::wire::WireError::provider_body) for
//!   an error envelope on a 200, plus telemetry fields consistent with nearby
//!   providers;
//! - unit, cassette, or live-test coverage appropriate to the changed behavior;
//! - root facade feature/docs updates for companion provider crates; and
//! - examples and documentation that match the actual API, feature flags, and
//!   credential requirements.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model that sends a request.
//! ```no_run
//! use rig_core::{
//!     completion::{AssistantContent, CompletionRequestBuilder, CompletionResponse},
//!     providers::openai::{self, wire::OpenAI},
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Read `OPENAI_API_KEY` into the configuration and pick a model.
//! let model = OpenAI::from_env()?.chat(openai::GPT_5_2);
//!
//! // A low-level completion request, ready for `model.completion(request)`.
//! let request = CompletionRequestBuilder::unbound("Discuss the fate of Middle Earth.")
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ".to_string())
//!     .build();
//! # let _ = (model, request);
//! # Ok(())
//! # }
//!
//! fn print_text(response: CompletionResponse) {
//!     for item in response.choice {
//!         if let AssistantContent::Text(text) = item {
//!             println!("{}", text.text);
//!         }
//!     }
//! }
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
