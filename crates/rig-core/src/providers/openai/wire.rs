//! The OpenAI wires: one chat-completions wire, N dialects.
//!
//! Every OpenAI-shaped provider in this crate speaks the same three
//! endpoints with the same bytes; what differs is a base URL, an env var, a
//! path, a handful of flags, and — for a few of them — one rewrite of the
//! serialized body. So there is one [`Chat`] wire and one [`ChatDecoder`],
//! and a provider is a [`Dialect`] **const value** ([`OPENAI`], [`GROQ`],
//! [`DEEPSEEK`], …) rather than a type implementing a trait.
//!
//! ```
//! use rig_core::providers::openai;
//!
//! // Official OpenAI.
//! let openai = openai::wire::OpenAI::new("sk-…");
//! // The same wire, pointed at Groq.
//! let groq = openai::wire::OpenAI::new("gsk_…").with_dialect(&openai::wire::GROQ);
//! assert_eq!(groq.base_url, "https://api.groq.com/openai/v1");
//! ```

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::driver::{HasEmbedding, HasModelListing, HasTranscription, HasVerify};
use crate::wire::{HasCompletion, Secret};

mod chat;
mod dialects;
/// The wire's reply shapes. `pub(crate)` rather than private because the
/// chat-completions model this wire replaces still names the same DTOs, and
/// there must be exactly one definition of each while both exist.
pub(crate) mod dto;
mod modality;
mod observation;

pub use chat::{Chat, ChatDecoder, ChatEvent};
pub use dialects::*;
pub use dto::{ChatChoice, ChatFrame, ChatUsage, FinishReason, StreamingCompletionResponse};
pub use modality::{
    EmbeddingDatum, Embeddings, EmbeddingsDecoder, EmbeddingsReply, ModelEntry, Models,
    ModelsDecoder, ModelsReply, Transcriptions, TranscriptionsDecoder, Verify, VerifyDecoder,
};

#[cfg(feature = "image")]
pub use modality::{Images, ImagesDecoder};
#[cfg(feature = "audio")]
pub use modality::{Speech, SpeechDecoder};

/// How a dialect authenticates.
///
/// Three shapes, all observed: the bearer token everyone but Azure uses,
/// Azure's own `api-key` header, and llama.cpp's optional key — a local
/// server started without `--api-key` rejects a request that carries an
/// `Authorization` header it was not configured for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Auth {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `Authorization: Bearer <key>`, omitted entirely when the key is empty.
    OptionalBearer,
    /// Azure's `api-key: <key>`.
    ApiKeyHeader,
}

/// How a dialect addresses a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    /// The endpoint is a path under the base URL and the model rides in the
    /// request body — every dialect but Azure.
    Path,
    /// Azure: the model is a *deployment* in the URL
    /// (`{base}/openai/deployments/{model}/chat/completions?api-version=…`)
    /// and the body carries no `model` field.
    AzureDeployment,
}

/// How a dialect spells the output-token cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputCap {
    /// `max_tokens`, for every endpoint not observed to reject it.
    Legacy,
    /// `max_completion_tokens` for OpenAI's reasoning families, which answer
    /// a `max_tokens` request with `Unsupported parameter`. Scoped to the
    /// model, because this same wire reaches compatible servers that know
    /// only the legacy field.
    OpenAiReasoningFamilies,
}

/// Which field a dialect takes an embedding width in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimensionsField {
    /// The OpenAI-compatible `dimensions` field.
    Dimensions,
    /// Mistral's `output_dimension`.
    OutputDimension,
    /// The server ignores any width field, so none is sent (`llama-server`
    /// reads no such field and would answer 200 with the native width).
    Ignored,
}

/// The rewrite a dialect applies to the serialized chat body.
///
/// This is the escape hatch the contract allows: a quirk no flag can express
/// is one arm of a `match dialect.quirks.rewrite` inside [`Chat::encode`],
/// in one file, instead of a `finalize_request_body` override per provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyRewrite {
    /// Send the OpenAI-compatible body unchanged.
    None,
    /// Groq: fold `additional_params.tools` (its compound-system native
    /// tools) into `compound_custom.enabled_tools` so they do not clobber
    /// the function-tool array on serialization.
    GroqCompoundTools,
    /// Hugging Face's router: qualify the model identifier for sub-providers
    /// that demand one (Fireworks).
    HuggingFaceRouter,
    /// DeepSeek: string-flattened content, `content: ""` on tool-call-only
    /// assistant turns, `index` on echoed tool calls, and forced tool
    /// choices suppressed unless thinking is explicitly disabled.
    DeepSeek,
    /// Mira's gateway: plain `{role, content}` history, names stripped,
    /// content-part arrays flattened.
    Mira,
    /// Perplexity: plain text history with strict user/assistant
    /// alternation, text-only arrays flattened.
    Perplexity,
    /// Hyperbolic: tool-exchange remnants stripped, content-part arrays kept
    /// (its vision models need them).
    Hyperbolic,
    /// Mistral: `any` for a forced tool choice, the choice relaxed to `auto`
    /// beside a structured response format, `prefix` on assistant turns and
    /// `reasoning_content` removed.
    Mistral,
    /// llama.cpp: refuse a specific-function tool choice, which
    /// `llama-server` silently treats as `auto`.
    LlamaCpp,
    /// Moonshot: refuse a specific-function tool choice and coerce
    /// `required` to `auto` with a steering message.
    Moonshot,
    /// OpenRouter: ephemeral `cache_control` on the system prompt when
    /// prompt caching is on, and `reasoning_content` respelled `reasoning`.
    OpenRouter,
}

/// What a dialect's embeddings endpoint accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EmbeddingQuirks {
    /// Most inputs the provider embeds in one request.
    pub max_documents: usize,
    /// Whether a successful reply must carry usage.
    pub requires_usage: bool,
    /// Whether the provider accepts `encoding_format`.
    pub supports_encoding_format: bool,
    /// Whether the provider accepts `user`.
    pub supports_user: bool,
    /// Whether the model is a body field (false for Azure, which addresses a
    /// deployment through the URL).
    pub sends_model_field: bool,
    /// Which field a requested width goes in.
    pub dimensions: DimensionsField,
}

impl EmbeddingQuirks {
    /// OpenAI's own embeddings contract, which most dialects inherit.
    pub const fn openai() -> Self {
        Self {
            max_documents: 1024,
            requires_usage: true,
            supports_encoding_format: true,
            supports_user: true,
            sends_model_field: true,
            dimensions: DimensionsField::Dimensions,
        }
    }
}

/// Everything about a dialect that is not its identity: paths, capability
/// flags, and the one body rewrite it needs.
///
/// `#[non_exhaustive]` because the constants live in this crate and a new
/// quirk must not be a breaking change for a host that stored a wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Quirks {
    /// How the dialect authenticates.
    pub auth: Auth,
    /// How the dialect addresses a model.
    pub routing: Routing,
    /// The chat-completions path, relative to the base URL.
    pub completion_path: &'static str,
    /// The embeddings path.
    pub embeddings_path: &'static str,
    /// The model-listing path.
    pub models_path: &'static str,
    /// The path a credential check hits. Empty means the dialect offers no
    /// check that does not consume tokens (Azure, Perplexity, Copilot).
    pub verify_path: &'static str,
    /// The transcription path.
    pub transcription_path: &'static str,
    /// The image-generation path.
    pub image_generation_path: &'static str,
    /// The speech path.
    pub audio_generation_path: &'static str,
    /// Whether `tools`/`tool_choice` reach the provider at all.
    pub supports_tools: bool,
    /// Whether `output_schema` maps to `response_format`.
    pub supports_response_format: bool,
    /// Whether this server honours an image inside a `role:"tool"` message.
    pub supports_image_tool_results: bool,
    /// Whether a streaming request asks for the usage chunk through
    /// `stream_options`.
    pub stream_include_usage: bool,
    /// Whether the backend can emit a whole tool call in one chunk.
    pub emits_complete_single_chunk_tool_calls: bool,
    /// How the dialect spells the output-token cap.
    pub output_cap: OutputCap,
    /// Whether an upstream-native `finish_reason` is consulted when the
    /// normalized one is absent — a gateway property (OpenRouter).
    pub native_finish_reason: bool,
    /// Whether the dialect emits `reasoning_details` entries (OpenRouter's
    /// encrypted reasoning blobs and replay signatures).
    pub reasoning_details: bool,
    /// The rewrite this dialect applies to the serialized chat body.
    pub rewrite: BodyRewrite,
    /// What the embeddings endpoint accepts.
    pub embedding: EmbeddingQuirks,
}

impl Quirks {
    /// OpenAI's own contract, which every dialect starts from and overrides
    /// only where it was measured to differ.
    pub const fn openai() -> Self {
        Self {
            auth: Auth::Bearer,
            routing: Routing::Path,
            completion_path: "/chat/completions",
            embeddings_path: "/embeddings",
            models_path: "/models",
            verify_path: "/models",
            transcription_path: "/audio/transcriptions",
            image_generation_path: "/images/generations",
            audio_generation_path: "/audio/speech",
            supports_tools: true,
            supports_response_format: true,
            supports_image_tool_results: false,
            stream_include_usage: true,
            emits_complete_single_chunk_tool_calls: false,
            output_cap: OutputCap::OpenAiReasoningFamilies,
            native_finish_reason: false,
            reasoning_details: false,
            rewrite: BodyRewrite::None,
            embedding: EmbeddingQuirks::openai(),
        }
    }
}

/// One OpenAI-shaped provider, as a `const` value.
///
/// A dialect is an *identity* plus its [`Quirks`]. There is one constant per
/// provider in this crate ([`OPENAI`], [`AZURE`], … — see the
/// [`dialects`](self) list), and adding a provider that speaks this wire is
/// adding a constant, not a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dialect {
    /// The provider descriptor name, as records and telemetry name it.
    pub name: &'static str,
    /// The default base URL.
    pub base_url: &'static str,
    /// The environment variable holding the credential.
    pub api_key_env: &'static str,
    /// The environment variable overriding the base URL, when the provider
    /// has one.
    pub base_url_env: Option<&'static str>,
    /// The reply header carrying the provider's transport request id.
    pub request_id_header: Option<&'static str>,
    /// Everything that is not identity.
    pub quirks: Quirks,
}

/// A dialect serializes as its [`name`](Dialect::name), and deserializes by
/// looking that name up among this module's constants.
///
/// A wire is plain data a host may store in a scene, a component or a config
/// file, so it must be serializable — but a dialect is an *identity*, not a
/// payload: its fields are `&'static str`, which cannot be deserialized at
/// all, and two dialects that agreed on every field but one would still be
/// two different providers. So the name is the whole wire format, and a name
/// this build does not know is an error rather than a silently
/// half-constructed provider.
impl Serialize for Dialect {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name)
    }
}

impl<'de> Deserialize<'de> for Dialect {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        dialects::by_name(&name).copied().ok_or_else(|| {
            serde::de::Error::custom(format!("unknown OpenAI-compatible dialect `{name}`"))
        })
    }
}

/// An OpenAI-shaped provider's configuration: plain data, key redacted.
///
/// Holds no transport and no type parameter, so a host can store one. Pair
/// it with a socket through [`Bound`](crate::driver::Bound) to get a model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpenAI {
    /// The credential. Never serialized (see [`Secret`]).
    pub api_key: Secret,
    /// The base URL every path resolves against.
    pub base_url: String,
    /// Which OpenAI-shaped provider this is.
    pub dialect: Dialect,
    /// Azure's `api-version` query parameter, which every Azure route
    /// requires. `None` for every other dialect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

impl OpenAI {
    /// Official OpenAI, with `api_key`.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::with_key(&OPENAI, api_key)
    }

    /// `dialect` with `api_key`, at the dialect's default base URL.
    pub fn with_key(dialect: &Dialect, api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: dialect.base_url.to_owned(),
            dialect: *dialect,
            api_version: None,
        }
    }

    /// Official OpenAI from `OPENAI_API_KEY`, with `OPENAI_BASE_URL`
    /// overriding the base URL — the variables the client read.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_with(&OPENAI)
    }

    /// `dialect` from its own `api_key_env` and `base_url_env`.
    ///
    /// Azure additionally reads `AZURE_API_VERSION`, because every Azure
    /// route carries it and there is no default that would not silently
    /// address the wrong API.
    pub fn from_env_with(dialect: &Dialect) -> Result<Self, EnvError> {
        let api_key = env::required(dialect.api_key_env)?;
        let base_url = match dialect.base_url_env {
            Some(name) => env::optional(name)?,
            None => None,
        };
        let api_version = match dialect.quirks.routing {
            Routing::AzureDeployment => {
                Some(env::required(dialects::AZURE_API_VERSION_ENV)?)
            }
            Routing::Path => None,
        };
        Ok(Self {
            api_key: api_key.into(),
            base_url: base_url.unwrap_or_else(|| dialect.base_url.to_owned()),
            dialect: *dialect,
            api_version,
        })
    }

    /// Point this configuration at another dialect, taking that dialect's
    /// default base URL.
    pub fn with_dialect(mut self, dialect: &Dialect) -> Self {
        self.base_url = dialect.base_url.to_owned();
        self.dialect = *dialect;
        self
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set Azure's `api-version`.
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    /// The chat-completions wire for `model`.
    pub fn chat(&self, model: impl Into<String>) -> Chat {
        Chat::new(self.clone(), model)
    }

    /// The embeddings wire for `model`.
    pub fn embeddings(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        Embeddings::new(self.clone(), model, ndims)
    }

    /// The transcription wire for `model`.
    pub fn transcriptions(&self, model: impl Into<String>) -> Transcriptions {
        Transcriptions::new(self.clone(), model)
    }

    /// The model-listing wire.
    pub fn models(&self) -> Models {
        Models::new(self.clone())
    }

    /// The credential-check wire.
    pub fn verify_wire(&self) -> Verify {
        Verify::new(self.clone())
    }

    /// The image-generation wire for `model`.
    #[cfg(feature = "image")]
    pub fn images(&self, model: impl Into<String>) -> Images {
        Images::new(self.clone(), model)
    }

    /// The speech wire for `model`.
    #[cfg(feature = "audio")]
    pub fn speech(&self, model: impl Into<String>) -> Speech {
        Speech::new(self.clone(), model)
    }

    /// Resolve `path` against the base URL, applying Azure's
    /// deployment-in-URL routing when the dialect uses it.
    pub(crate) fn uri(&self, path: &str, model: Option<&str>) -> String {
        match (self.dialect.quirks.routing, model) {
            (Routing::AzureDeployment, Some(model)) => format!(
                "{}/openai/deployments/{}{}?api-version={}",
                self.base_url.trim_end_matches('/'),
                model.trim_start_matches('/'),
                path,
                self.api_version.as_deref().unwrap_or_default(),
            ),
            _ => format!("{}{}", self.base_url.trim_end_matches('/'), path),
        }
    }

    /// Apply the dialect's authentication to a request builder.
    pub(crate) fn authenticate(&self, builder: http::request::Builder) -> http::request::Builder {
        match self.dialect.quirks.auth {
            Auth::Bearer => {
                builder.header("Authorization", format!("Bearer {}", self.api_key.expose()))
            }
            Auth::OptionalBearer if self.api_key.is_empty() => builder,
            Auth::OptionalBearer => {
                builder.header("Authorization", format!("Bearer {}", self.api_key.expose()))
            }
            Auth::ApiKeyHeader => builder.header("api-key", self.api_key.expose()),
        }
    }
}

impl HasCompletion for OpenAI {
    type Wire = Chat;

    fn completion(&self, model: impl Into<String>) -> Chat {
        self.chat(model)
    }
}

impl HasEmbedding for OpenAI {
    type Wire = Embeddings;

    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        self.embeddings(model, ndims)
    }
}

impl HasTranscription for OpenAI {
    type Wire = Transcriptions;

    fn transcription(&self, model: impl Into<String>) -> Transcriptions {
        self.transcriptions(model)
    }
}

impl HasModelListing for OpenAI {
    type Wire = Models;

    fn model_listing(&self) -> Models {
        self.models()
    }
}

impl HasVerify for OpenAI {
    type Wire = Verify;

    fn verify(&self) -> Verify {
        self.verify_wire()
    }
}

#[cfg(feature = "image")]
impl crate::driver::HasImageGeneration for OpenAI {
    type Wire = Images;

    fn image_generation(&self, model: impl Into<String>) -> Images {
        self.images(model)
    }
}

#[cfg(feature = "audio")]
impl crate::driver::HasAudioGeneration for OpenAI {
    type Wire = Speech;

    fn audio_generation(&self, model: impl Into<String>) -> Speech {
        self.speech(model)
    }
}

#[cfg(test)]
mod tests;
