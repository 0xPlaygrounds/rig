//! OpenAI-compatible configurations, dialect policies, and endpoint wires.
//! A [`Dialect`](crate::providers::openai::wire::Dialect) selects request and response policies; [`OpenAI`] holds
//! credentials and overrides. Bind an endpoint wire to a transport to execute it.
//!
//! ```
//! use rig_core::providers::openai::{OpenAI, Route, wire::OpenAiWire};
//! let provider = OpenAI::new("key").with_route(Route::Chat);
//! assert!(matches!(provider.completion("gpt-5.2"), OpenAiWire::Chat(_)));
//! ```

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::driver::{Bound, HasEmbedding, HasModelListing, HasRerank, HasTranscription, HasVerify};
use crate::wire::{HasCompletion, Secret};

use super::responses_api::SystemInstructionsPlacement;
use super::responses_api::wire::Responses;

mod chat;
mod dialects;
/// Shared Chat Completions response shapes.
pub(crate) mod dto;
mod modality;
mod route;

pub use chat::{Chat, ChatDecoder, ChatEvent};
pub use dialects::*;
pub use dto::{ChatChoice, ChatFrame, ChatUsage, FinishReason, StreamingCompletionResponse};
pub use modality::{
    Embeddings, EmbeddingsDecoder, ModelEntry, Models, ModelsDecoder, ModelsReply, Rerank,
    RerankDecoder, RerankReply, RerankResultEntry, RerankUsage, Transcriptions,
    TranscriptionsDecoder, Verify, VerifyDecoder,
};
pub use route::{OpenAiDecoder, OpenAiEvent, OpenAiWire, Route};

#[cfg(feature = "image")]
pub use modality::{ImageDatum, Images, ImagesDecoder, ImagesEvent, ImagesReply};
#[cfg(feature = "audio")]
pub use modality::{Speech, SpeechDecoder};

/// Credential header policy, including omission of empty optional tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Auth {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `Authorization: Bearer <key>`, omitted entirely when the key is empty.
    OptionalBearer,
    /// Azure's `api-key: <key>`.
    ApiKeyHeader,
}

/// Alternative credential environment variable and its authentication policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthAlternative {
    /// The variable holding this credential.
    pub api_key_env: &'static str,
    /// How it is sent.
    pub auth: Auth,
}

/// Backend selected through the Hugging Face router.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubRoute {
    /// Hugging Face's own inference backend: the only one that serves
    /// transcription and image generation.
    #[default]
    HFInference,
    /// Together AI, through the router.
    Together,
    /// SambaNova, through the router.
    SambaNova,
    /// Fireworks AI, which addresses models by a qualified id.
    Fireworks,
    /// Hyperbolic, through the router.
    Hyperbolic,
    /// Nebius, through the router.
    Nebius,
    /// Novita, through the router.
    Novita,
    /// A route this build does not name.
    Custom(String),
}

impl SubRoute {
    /// The router's slug for this sub-provider.
    pub fn slug(&self) -> &str {
        match self {
            Self::HFInference => "hf-inference/models",
            Self::Together => "together",
            Self::SambaNova => "sambanova",
            Self::Fireworks => "fireworks-ai",
            Self::Hyperbolic => "hyperbolic",
            Self::Nebius => "nebius",
            Self::Novita => "novita",
            Self::Custom(route) => route,
        }
    }

    /// Qualify Fireworks model identifiers unless already prefixed.
    /// Return other sub-routes' identifiers unchanged.
    pub fn model_identifier(&self, model: &str) -> String {
        const FIREWORKS_PREFIX: &str = "accounts/fireworks/models/";
        match self {
            Self::Fireworks if !model.starts_with(FIREWORKS_PREFIX) => {
                format!("{FIREWORKS_PREFIX}{model}")
            }
            _ => model.to_owned(),
        }
    }

    /// Whether this sub-provider serves the endpoints that address the model
    /// through the URL (transcription, image generation).
    pub fn serves_model_routed_endpoints(&self) -> bool {
        matches!(self, Self::HFInference)
    }
}

impl std::fmt::Display for SubRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.slug())
    }
}

impl From<&str> for SubRoute {
    fn from(route: &str) -> Self {
        Self::Custom(route.to_owned())
    }
}

impl From<String> for SubRoute {
    fn from(route: String) -> Self {
        Self::Custom(route)
    }
}

/// Paired request and response formats for image generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImageBody {
    /// `{model, prompt, size}`, answered with `data[].b64_json`.
    #[default]
    OpenAi,
    /// xAI: `{model, prompt, response_format, aspect_ratio}` and no `size`,
    /// answered with `data[].b64_json` and no `created`.
    Xai,
    /// `{model_name, prompt, height, width}`, answered with `images[].image`.
    Hyperbolic,
    /// `{model, prompt, width, height}`, answered with base64 strings in `images`.
    Venice,
    /// `{inputs, parameters: {width, height}}`, answered with raw image bytes.
    /// The model is addressed through the URL path.
    HuggingFace,
}

/// Which body a speech endpoint takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpeechBody {
    /// OpenAI: `{model, input, voice, speed}`.
    #[default]
    OpenAi,
    /// xAI: `{text, voice_id, language}`, with `eve` as the default voice.
    Xai,
    /// Hyperbolic: `{language, speaker, text, speed}`, answered with
    /// `{"audio": "<base64>"}` rather than the audio bytes themselves.
    ///
    /// It addresses this endpoint by *language*, so the identifier a caller
    /// passes as the model is the language tag (`"EN"`).
    Hyperbolic,
}

/// Which body a transcription endpoint takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptionBody {
    /// OpenAI: a `multipart/form-data` upload with the audio as a file part
    /// beside `model`, `language`, `prompt` and `temperature`.
    Multipart,
    /// OpenRouter: a JSON body whose audio rides base64-encoded under
    /// `input_audio`, with its container format beside it
    /// (`{"input_audio": {"data": "…", "format": "mp3"}, "model": …}`).
    /// The gateway's speech-to-text route serves only this shape, and has no
    /// top-level `prompt` field at all.
    InputAudioJson,
}

/// How a dialect addresses a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    /// Resolve the endpoint under the base URL and send the model in the body.
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

impl DimensionsField {
    /// The body field a requested width goes in, or `None` when the dialect
    /// reads no width field at all.
    ///
    /// The encoder puts a width in this field and a refusal names it, so
    /// both spell it from here rather than from two matching literals.
    pub const fn name(self) -> Option<&'static str> {
        match self {
            Self::Dimensions => Some("dimensions"),
            Self::OutputDimension => Some("output_dimension"),
            Self::Ignored => None,
        }
    }
}

/// Which widths a request may name for one embedding model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcceptedWidths {
    /// The model emits one width and reads no width field, so any value but
    /// its own native width is a request for a parameter the provider does
    /// not accept there, and is refused as a request error.
    Fixed,
    /// The model truncates to any width in `min..=max`, and anything else is
    /// refused with [`requirement`](Self::Range::requirement).
    Range {
        /// Narrowest width the provider honours.
        min: usize,
        /// Widest width the provider honours.
        max: usize,
        /// Static error clause describing the accepted bounds.
        /// Must agree with `min` and `max`.
        requirement: &'static str,
    },
}

/// One embedding model's width contract: the width it returns unasked, and
/// the widths it will honour when asked.
///
/// Stated per model rather than per dialect because a dialect serves models
/// of different widths, and a model's default is not always its maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelWidth {
    /// The model identifier, as the `model` field spells it.
    pub model: &'static str,
    /// Default width reported when no width is requested, or `None` if unknown.
    /// Unknown widths report zero through [`crate::embeddings::EmbeddingModel::ndims`].
    pub default: Option<usize>,
    /// The widths a request may name.
    pub accepted: AcceptedWidths,
}

/// Dialect-specific transformation of the serialized chat request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyRewrite {
    /// Send the OpenAI-compatible body unchanged.
    None,
    /// Groq: fold `additional_params.tools` (its compound-system native
    /// tools) into `compound_custom.enabled_tools` so they do not clobber
    /// the function-tool array on serialization, and replay assistant turns
    /// without `reasoning_content`, which Groq rejects.
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

/// Reranking endpoint policy. An empty [`Self::path`] disables reranking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RerankQuirks {
    /// The rerank path, or empty when the dialect offers none.
    pub path: &'static str,
    /// Most documents the provider accepts in one request.
    pub max_documents: usize,
    /// Whether the model is a body field.
    pub sends_model_field: bool,
}

impl RerankQuirks {
    /// The signal for a dialect with no reranking endpoint.
    pub const fn unsupported() -> Self {
        Self {
            path: "",
            max_documents: 0,
            sends_model_field: true,
        }
    }
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
    /// Model width contracts used for capability reporting and request validation.
    /// Consulted before the shared OpenAI model-width table.
    pub widths: &'static [ModelWidth],
    /// The `requirement` clause refusing a declared width of zero, or
    /// `None` for a dialect that lets zero through as rig's own "unknown"
    /// sentinel rather than a claim.
    pub refuse_zero_width: Option<&'static str>,
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
            // Shared OpenAI model widths are resolved separately.
            widths: &[],
            // Zero represents an unknown width, not a requested dimension.
            refuse_zero_width: None,
        }
    }
}

/// The caller identity a gateway requires on every request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// The `originator` header's default value.
    pub originator: &'static str,
    /// The environment variable overriding `originator`.
    pub originator_env: &'static str,
    /// The environment variable overriding `user-agent`.
    pub user_agent_env: &'static str,
    /// Whether every request carries a fresh `session_id` header.
    pub session_ids: bool,
}

/// The identity a gateway requires on every request, resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallerIdentity {
    /// The `originator` header.
    pub originator: String,
    /// The `user-agent` header.
    pub user_agent: String,
}

/// The user agent a gateway that asks for one is told: the crate, the host,
/// and who is calling.
fn default_user_agent(originator: &str) -> String {
    format!(
        "rig/{} ({} {}; {originator})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

/// Request restrictions and response handling for a Responses dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesContract {
    /// Standard Responses behavior with independently configured instruction placement.
    OpenAi,
    /// xAI's `/v1/responses`: it answers a success with its error envelope
    /// as the whole body and publishes a finished tool call at
    /// `output_item.done`, and its native structured output does not
    /// compose with tool calls. The stream's own `error` event is not this:
    /// that is protocol on every dialect and the decoder always reads it.
    Xai,
    /// Always-streamed Codex responses with optional content-type and envelope fields.
    /// Requests omit sampling controls, storage, metadata, and structured output.
    Codex,
}

/// What a dialect's Responses endpoint is: where it lives, where the system
/// preamble goes, and which contract it speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResponsesQuirks {
    /// The endpoint path, appended to the base URL.
    pub path: &'static str,
    /// Where Rig's system instructions go in the request. Not part of
    /// [`Self::contract`]: OpenAI's contract is served with all three
    /// placements (OpenAI's own `instructions`, Copilot's and OpenRouter's
    /// `system` items in `input`).
    pub system_instructions: SystemInstructionsPlacement,
    /// Which contract this dialect's endpoint speaks.
    pub contract: ResponsesContract,
    /// Whether newly constructed Responses wires normalize tools for strict validation.
    pub strict_tools_by_default: bool,
}

impl ResponsesQuirks {
    /// OpenAI's own Responses contract.
    pub const fn openai() -> Self {
        Self {
            path: "/responses",
            system_instructions: SystemInstructionsPlacement::Instructions,
            contract: ResponsesContract::OpenAi,
            strict_tools_by_default: false,
        }
    }
}

/// Optional executable extensions to the shared OpenAI dialect.
///
/// Store this value in a `static`: equality means the same extension definition,
/// not equality of function addresses (which code generation can merge or duplicate).
/// This lets named dialect persistence reject replaced hooks without interpreting
/// their behavior or pretending arbitrary callbacks can be serialized.
#[derive(Debug)]
pub struct DialectHooks {
    /// Derive a default endpoint from a credential. `None` uses the dialect's
    /// static URL. Called only at construction, never on credential replacement.
    pub default_endpoint: Option<fn(&str) -> Option<String>>,
    /// Select the default route for a model, unless configuration chose a route.
    pub model_route: Option<fn(&str) -> Route>,
    /// Apply the completion envelope after shared authentication and identity.
    /// Called once by either completion encoder; builder errors remain attached
    /// and are returned when the encoder finishes the request.
    pub completion_envelope: Option<CompletionEnvelope>,
    /// Stamp the envelope every modality request (embeddings, listing,
    /// verification, transcription, images, speech) carries, on the finished
    /// request. Runs after shared authentication, so a hook may replace the
    /// credential header rather than add a second one.
    pub modality_envelope: Option<ModalityEnvelope>,
}

/// A dialect's modality-request headers, applied to the built request.
pub type ModalityEnvelope =
    fn(&OpenAI, &mut http::Request<crate::wire::Body>) -> Result<(), http::Error>;

/// A dialect's completion headers, applied to the authenticated request builder.
pub type CompletionEnvelope = fn(
    &OpenAI,
    &crate::completion::CompletionRequest,
    http::request::Builder,
) -> http::request::Builder;

impl PartialEq for DialectHooks {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for DialectHooks {}

/// Everything about a dialect that is not its identity: paths, capability
/// flags, and the one body rewrite it needs.
///
/// `#[non_exhaustive]` because the constants live in this crate and a new
/// quirk must not be a breaking change for a host that stored a wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Quirks {
    /// Provider-owned extensions for defaults and completion headers.
    pub hooks: Option<&'static DialectHooks>,
    /// How the dialect authenticates.
    pub auth: Auth,
    /// How the dialect addresses a model.
    pub routing: Routing,
    /// Which completion endpoint [`OpenAI::completion`] builds: the
    /// dialect's flagship. Chat Completions is the one endpoint every
    /// dialect serves, so it is the baseline; OpenAI itself, xAI and ChatGPT
    /// serve `/responses` as their primary API and say so.
    pub completion_route: Route,
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
    /// Whether to send `response_format` with tools before any tool result.
    /// When false, defer the format until a tool result exists to avoid suppressing calls.
    pub response_format_with_tools: bool,
    /// Whether this server honours an image inside a `role:"tool"` message.
    pub supports_image_tool_results: bool,
    /// Whether a streaming request asks for the usage chunk through
    /// `stream_options`.
    pub stream_include_usage: bool,
    /// Whether the backend can emit a whole tool call in one chunk.
    pub emits_complete_single_chunk_tool_calls: bool,
    /// How the dialect spells the output-token cap.
    pub output_cap: OutputCap,
    /// Whether to consult upstream-native finish reasons when normalized ones are absent.
    pub native_finish_reason: bool,
    /// Whether the dialect emits `reasoning_details` entries (OpenRouter's
    /// encrypted reasoning blobs and replay signatures).
    pub reasoning_details: bool,
    /// Whether reasoning belongs to the upstream model's family rather than
    /// to this dialect: a gateway relays each upstream's own reasoning state,
    /// valid only for that family ([`upstream_reasoning_issuer`]).
    pub upstream_reasoning_issuer: bool,
    /// Whether a bare JSON string is accepted as a text-only completion reply.
    pub accepts_bare_string_reply: bool,
    /// Whether document and file inputs may use provider file IDs.
    pub accepts_file_ids: bool,
    /// The rewrite this dialect applies to the serialized chat body.
    pub rewrite: BodyRewrite,
    /// Paths that strip a trailing `/v1` from the configured base URL.
    pub root_relative_routes: &'static [&'static str],
    /// Whether the model is the modality endpoint's *path* rather than a
    /// body field. Hugging Face's router addresses transcription and image
    /// generation as `/{model}`; everyone else uses a fixed path.
    pub model_is_modality_path: bool,
    /// Which body the image endpoint takes.
    pub image_body: ImageBody,
    /// Which body the transcription endpoint takes.
    pub transcription_body: TranscriptionBody,
    /// Which body the speech endpoint takes.
    pub speech_body: SpeechBody,
    /// What the embeddings endpoint accepts.
    pub embedding: EmbeddingQuirks,
    /// What the rerank endpoint accepts.
    pub rerank: RerankQuirks,
    /// A second environment variable naming the base URL, kept because the
    /// provider documents both spellings.
    pub base_url_env_alias: Option<&'static str>,
    /// The environment variable naming the account a credential belongs to,
    /// sent as `ChatGPT-Account-Id`.
    pub account_id_env: Option<&'static str>,
    /// Instructions this gateway expects every turn to carry, merged ahead
    /// of the caller's preamble.
    pub default_instructions: Option<&'static str>,
    /// The environment variable overriding [`Self::default_instructions`].
    pub instructions_env: Option<&'static str>,
    /// The caller identity this gateway requires on every request.
    pub identity: Option<Identity>,
    /// What the Responses endpoint accepts.
    pub responses: ResponsesQuirks,
}

impl Quirks {
    /// Baseline compatible endpoint policies with Chat Completions routing and
    /// the legacy `max_tokens` cap. Dialects override supported differences.
    pub const fn openai() -> Self {
        Self {
            hooks: None,
            auth: Auth::Bearer,
            routing: Routing::Path,
            completion_route: Route::Chat,
            completion_path: "/chat/completions",
            embeddings_path: "/embeddings",
            models_path: "/models",
            verify_path: "/models",
            transcription_path: "/audio/transcriptions",
            image_generation_path: "/images/generations",
            audio_generation_path: "/audio/speech",
            supports_tools: true,
            supports_response_format: true,
            response_format_with_tools: false,
            supports_image_tool_results: false,
            stream_include_usage: true,
            emits_complete_single_chunk_tool_calls: false,
            output_cap: OutputCap::Legacy,
            native_finish_reason: false,
            reasoning_details: false,
            upstream_reasoning_issuer: false,
            accepts_bare_string_reply: false,
            accepts_file_ids: true,
            rewrite: BodyRewrite::None,
            embedding: EmbeddingQuirks::openai(),
            // OpenAI has no reranking endpoint, and neither does any dialect
            // on this wire but llama.cpp.
            rerank: RerankQuirks::unsupported(),
            root_relative_routes: &[],
            model_is_modality_path: false,
            image_body: ImageBody::OpenAi,
            speech_body: SpeechBody::OpenAi,
            transcription_body: TranscriptionBody::Multipart,
            base_url_env_alias: None,
            account_id_env: None,
            default_instructions: None,
            instructions_env: None,
            identity: None,
            responses: ResponsesQuirks::openai(),
        }
    }
}

/// Provider identity, endpoint defaults, and shared-wire policies.
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
    /// A second credential this dialect accepts, with its own variable and
    /// header. `None` for every dialect but Azure.
    pub alternate_auth: Option<AuthAlternative>,
    /// Everything that is not identity.
    pub quirks: Quirks,
}

impl Dialect {
    /// Create a dialect with [`Quirks::openai`] and the supplied identity.
    /// URL overrides, request-ID headers, and alternative credentials are unset.
    pub const fn gateway(
        name: &'static str,
        base_url: &'static str,
        api_key_env: &'static str,
    ) -> Self {
        Self {
            name,
            base_url,
            api_key_env,
            base_url_env: None,
            request_id_header: None,
            alternate_auth: None,
            quirks: Quirks::openai(),
        }
    }
}

/// Serialize the registered dialect name, rejecting unregistered or modified definitions.
/// Deserialization resolves that name from this build's registry.
impl Serialize for Dialect {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let registered = dialects::by_name(self.name) == Some(self);
        crate::providers::internal::named_dialect::serialize(
            serializer, "OpenAI", self.name, registered,
        )
    }
}

impl<'de> Deserialize<'de> for Dialect {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        crate::providers::internal::named_dialect::deserialize(deserializer, "OpenAI", |name| {
            dialects::by_name(name).copied()
        })
    }
}

/// Serializable provider configuration without a transport.
/// Credentials are redacted and omitted from serialization. Construct endpoint
/// wires and bind them through [`Bound`] to execute requests.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAI {
    /// The credential. Never serialized (see [`Secret`]).
    pub api_key: Secret,
    /// The base URL every path resolves against.
    pub base_url: String,
    /// Which OpenAI-shaped provider this is.
    pub dialect: Dialect,
    /// The completion endpoint this configuration uses when asked for "a
    /// completion", when it differs from the dialect's flagship
    /// ([`Quirks::completion_route`]). Set by [`with_route`](Self::with_route).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<Route>,
    /// Azure's `api-version` query parameter, which every Azure route
    /// requires. `None` for every other dialect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    /// Azure versions its speech endpoint separately from the rest, so a
    /// speech request carries this `api-version` instead of
    /// [`Self::api_version`]. `None` falls back to `api_version`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_api_version: Option<String>,
    /// How this configuration's credential is sent. Taken from the dialect,
    /// except when the credential came from the dialect's
    /// [`alternate_auth`](Dialect::alternate_auth) variable, which has its
    /// own header.
    pub auth: Auth,
    /// Which sub-provider the Hugging Face router forwards to. `None` behaves
    /// as [`SubRoute::HFInference`], the router's own default. `None` for
    /// every other dialect, which routes nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_route: Option<SubRoute>,
    /// The account the credential belongs to, when the gateway asks which
    /// (`ChatGPT-Account-Id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    /// Instructions merged ahead of every Responses turn's preamble, when
    /// the gateway expects some.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The caller identity, when the gateway requires one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<CallerIdentity>,
    /// Responses instruction placement override. `None` uses the dialect default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_instructions: Option<SystemInstructionsPlacement>,
}

impl OpenAI {
    /// Official OpenAI, with `api_key`.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::with_key(&OPENAI, api_key)
    }

    /// `dialect` with `api_key`, at the dialect's default base URL and with
    /// the instructions and caller identity its gateway expects, if any.
    pub fn with_key(dialect: &Dialect, api_key: impl Into<Secret>) -> Self {
        let quirks = &dialect.quirks;
        let api_key = api_key.into();
        let base_url = quirks
            .hooks
            .and_then(|hooks| hooks.default_endpoint)
            .and_then(|endpoint| endpoint(api_key.expose()))
            .unwrap_or_else(|| dialect.base_url.to_owned());
        Self {
            api_key,
            base_url,
            dialect: *dialect,
            route: None,
            // Azure deployment URLs require an explicit API version.
            api_version: match quirks.routing {
                Routing::AzureDeployment => Some(dialects::AZURE_DEFAULT_API_VERSION.to_owned()),
                Routing::Path => None,
            },
            audio_api_version: None,
            auth: quirks.auth,
            sub_route: None,
            account_id: None,
            instructions: quirks.default_instructions.map(str::to_owned),
            identity: quirks.identity.map(|identity| CallerIdentity {
                originator: identity.originator.to_owned(),
                user_agent: default_user_agent(identity.originator),
            }),
            system_instructions: None,
        }
    }

    /// `dialect` with the credential it accepts through its
    /// [`alternate_auth`](Dialect::alternate_auth) variable, sent with that
    /// alternative's header.
    ///
    /// Azure's account key and its Entra bearer token are both credentials
    /// for the same account but go out under different headers, so which one
    /// is held has to be recorded rather than guessed from the value.
    pub fn with_alternate_key(dialect: &Dialect, api_key: impl Into<Secret>) -> Self {
        let auth = dialect
            .alternate_auth
            .map_or(dialect.quirks.auth, |alternative| alternative.auth);
        Self {
            auth,
            ..Self::with_key(dialect, api_key)
        }
    }

    /// Read `OPENAI_API_KEY` and the optional `OPENAI_BASE_URL` override.
    /// Return an environment error for missing credentials or invalid values.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_with(&OPENAI)
    }

    /// `dialect` from its own `api_key_env` and `base_url_env` (or the
    /// alias its quirks name), plus whatever else its gateway reads: the
    /// account id, the default instructions and the caller identity.
    ///
    /// Azure additionally reads `AZURE_API_VERSION`, because every Azure
    /// route carries it and there is no default that would not silently
    /// address the wrong API.
    pub fn from_env_with(dialect: &Dialect) -> Result<Self, EnvError> {
        let quirks = &dialect.quirks;
        // Alternative credentials require their own header policy; prefer the primary.
        let (api_key, auth) = match dialect.alternate_auth {
            Some(alternative) => match env::optional(dialect.api_key_env)? {
                Some(api_key) => (api_key, quirks.auth),
                None => match env::optional(alternative.api_key_env)? {
                    Some(api_key) => (api_key, alternative.auth),
                    None => {
                        return Err(EnvError::Invalid {
                            name: dialect.api_key_env,
                            detail: format!(
                                "either `{}` or `{}` must be set",
                                dialect.api_key_env, alternative.api_key_env
                            ),
                        });
                    }
                },
            },
            None => (env::required(dialect.api_key_env)?, quirks.auth),
        };
        let mut provider = Self::with_key(dialect, api_key);
        provider.auth = auth;
        for name in [dialect.base_url_env, quirks.base_url_env_alias]
            .into_iter()
            .flatten()
        {
            if let Some(base_url) = env::optional(name)? {
                provider.base_url = base_url;
                break;
            }
        }
        // Azure speech uses an independently versioned endpoint.
        if let Routing::AzureDeployment = quirks.routing {
            provider.api_version = Some(env::required(dialects::AZURE_API_VERSION_ENV)?);
            provider.audio_api_version = env::optional(dialects::AZURE_AUDIO_API_VERSION_ENV)?
                .or_else(|| Some(dialects::AZURE_DEFAULT_AUDIO_API_VERSION.to_owned()));
        }
        if let Some(name) = quirks.account_id_env {
            provider.account_id = env::optional(name)?;
        }
        if let Some(name) = quirks.instructions_env
            && let Some(instructions) = env::optional(name)?
            && !instructions.trim().is_empty()
        {
            provider.instructions = Some(instructions);
        }
        if let (Some(identity), Some(resolved)) = (quirks.identity, provider.identity.as_mut()) {
            if let Some(originator) =
                env::optional(identity.originator_env)?.filter(|value| !value.is_empty())
            {
                resolved.originator = originator;
                resolved.user_agent = default_user_agent(&resolved.originator);
            }
            if let Some(user_agent) =
                env::optional(identity.user_agent_env)?.filter(|value| !value.is_empty())
            {
                resolved.user_agent = user_agent;
            }
        }
        Ok(provider)
    }

    /// Point this configuration at another dialect: the same credential,
    /// with everything else at that dialect's defaults.
    pub fn with_dialect(self, dialect: &Dialect) -> Self {
        Self::with_key(dialect, self.api_key)
    }

    /// Route through a Hugging Face sub-provider.
    pub fn with_sub_route(mut self, sub_route: SubRoute) -> Self {
        self.sub_route = Some(sub_route);
        self
    }

    /// Send the credential with `auth`'s header.
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
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

    /// Set the `api-version` Azure's speech endpoint is versioned by.
    pub fn with_audio_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.audio_api_version = Some(api_version.into());
        self
    }

    /// Name the account the credential belongs to (`ChatGPT-Account-Id`).
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Merge these instructions ahead of every Responses turn's preamble.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Put Rig's system instructions somewhere other than the dialect's
    /// default placement, for every Responses wire this configuration
    /// builds.
    pub fn with_system_instructions_placement(
        mut self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        self.system_instructions = Some(placement);
        self
    }

    /// Send Rig's system instructions as `system` messages in `input`, for a
    /// backend that rejects or ignores top-level `instructions`.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }

    /// Where a Responses wire built from this configuration puts Rig's
    /// system instructions: the dialect's placement unless
    /// [`with_system_instructions_placement`](Self::with_system_instructions_placement)
    /// chose another.
    pub fn system_instructions_placement(&self) -> SystemInstructionsPlacement {
        self.system_instructions
            .unwrap_or(self.dialect.quirks.responses.system_instructions)
    }

    /// Override dialect and model-specific routing for [`Self::completion`].
    pub fn with_route(mut self, route: Route) -> Self {
        self.route = Some(route);
        self
    }

    /// The configured route or dialect's static default. A model-route hook
    /// may refine the default when a completion wire is constructed.
    pub fn completion_route(&self) -> Route {
        self.route.unwrap_or(self.dialect.quirks.completion_route)
    }

    /// The completion wire for `model` on this configuration's
    /// [`completion_route`](Self::completion_route): Responses for OpenAI,
    /// xAI and ChatGPT, model-dependent routing when a dialect supplies it, and
    /// Chat Completions for other compatible gateways, unless
    /// [`with_route`](Self::with_route) chose the other one.
    pub fn completion(&self, model: impl Into<String>) -> OpenAiWire {
        OpenAiWire::new(self.clone(), model)
    }

    /// The Responses wire for `model`: `POST /responses`.
    pub fn responses(&self, model: impl Into<String>) -> Responses {
        Responses::new(self.clone(), model)
    }

    /// The chat-completions wire for `model`.
    pub fn chat(&self, model: impl Into<String>) -> Chat {
        Chat::new(self.clone(), model)
    }

    /// The embeddings wire for `model`.
    pub fn embeddings(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        Embeddings::new(self.clone(), model, ndims)
    }

    /// The rerank wire for `model`.
    pub fn reranker(&self, model: impl Into<String>) -> Rerank {
        Rerank::new(self.clone(), model)
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

    pub(crate) fn completion_headers(
        &self,
        request: &crate::completion::CompletionRequest,
        builder: http::request::Builder,
    ) -> http::request::Builder {
        let builder = self.headers(builder);
        match self
            .dialect
            .quirks
            .hooks
            .and_then(|hooks| hooks.completion_envelope)
        {
            Some(envelope) => envelope(self, request, builder),
            None => builder,
        }
    }

    /// Resolve `path` against the base URL, applying Azure's
    /// deployment-in-URL routing when the dialect uses it.
    pub(crate) fn uri(&self, path: &str, model: Option<&str>) -> String {
        self.uri_versioned(path, model, self.api_version.as_deref())
    }

    /// [`Self::uri`] with an explicit `api-version`, for the one endpoint
    /// Azure versions separately (speech).
    pub(crate) fn uri_versioned(
        &self,
        path: &str,
        model: Option<&str>,
        api_version: Option<&str>,
    ) -> String {
        match (self.dialect.quirks.routing, model) {
            (Routing::AzureDeployment, Some(model)) => format!(
                "{}/openai/deployments/{}{}?api-version={}",
                self.base_url.trim_end_matches('/'),
                model.trim_start_matches('/'),
                path,
                api_version.unwrap_or_default(),
            ),
            _ => format!("{}{}", self.base(path), path),
        }
    }

    /// The base URL `path` resolves against: the configured one, with the
    /// version segment dropped for a route the dialect serves at the root.
    fn base(&self, path: &str) -> &str {
        let base = self.base_url.trim_end_matches('/');
        if self.dialect.quirks.root_relative_routes.contains(&path) {
            return base.strip_suffix("/v1").unwrap_or(base);
        }
        base
    }

    /// The `api-version` a speech request carries.
    #[cfg(feature = "audio")]
    pub(crate) fn speech_api_version(&self) -> Option<&str> {
        self.audio_api_version
            .as_deref()
            .or(self.api_version.as_deref())
    }

    /// The sub-provider the Hugging Face router forwards to. `None` on the
    /// configuration means the router's own default.
    pub(crate) fn route(&self) -> std::borrow::Cow<'_, SubRoute> {
        match &self.sub_route {
            Some(route) => std::borrow::Cow::Borrowed(route),
            None => std::borrow::Cow::Owned(SubRoute::default()),
        }
    }

    /// Return `model` for Azure deployment routing, otherwise `None`.
    pub(crate) fn deployment<'a>(&self, model: &'a str) -> Option<&'a str> {
        match self.dialect.quirks.routing {
            Routing::AzureDeployment => Some(model),
            Routing::Path => None,
        }
    }

    /// Resolve a fixed or model-addressed modality URL.
    /// Return an error if the selected sub-route does not serve model-routed endpoints.
    pub(crate) fn modality_uri(
        &self,
        endpoint: &str,
        fixed: &'static str,
        model: &str,
    ) -> Result<String, String> {
        if !self.dialect.quirks.model_is_modality_path {
            return Ok(self.uri(fixed, self.deployment(model)));
        }
        let route = self.route();
        if !route.serves_model_routed_endpoints() {
            return Err(format!(
                "{endpoint} endpoint is not supported yet for {route}"
            ));
        }
        Ok(format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            model.trim_start_matches('/')
        ))
    }

    /// Apply the dialect's authentication to a request builder.
    pub(crate) fn authenticate(&self, builder: http::request::Builder) -> http::request::Builder {
        match self.auth {
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

    /// Apply authentication, configured identity, account, and per-request session headers.
    pub(crate) fn headers(&self, builder: http::request::Builder) -> http::request::Builder {
        let mut builder = self.authenticate(builder);
        if let Some(identity) = &self.identity {
            builder = builder
                .header("originator", &identity.originator)
                .header(http::header::USER_AGENT, &identity.user_agent);
        }
        if self
            .dialect
            .quirks
            .identity
            .is_some_and(|identity| identity.session_ids)
        {
            // Session identity must be fresh for each request.
            builder = builder.header("session_id", crate::providers::chatgpt::session_id());
        }
        if let Some(account_id) = &self.account_id {
            builder = builder.header("ChatGPT-Account-Id", account_id);
        }
        builder
    }
}

/// Build completion wires using configured, model-specific, or dialect routing.
impl HasCompletion for OpenAI {
    type Wire = OpenAiWire;

    fn completion(&self, model: impl Into<String>) -> OpenAiWire {
        self.completion(model)
    }
}

/// Explicit endpoint constructors that retain the bound transport.
impl<H: Clone> Bound<OpenAI, H> {
    /// The chat-completions wire for `model`, on this socket.
    pub fn chat(&self, model: impl Into<String>) -> Bound<Chat, H> {
        Bound::new(self.wire.chat(model), self.http.clone())
    }

    /// The Responses wire for `model`, on this socket.
    pub fn responses(&self, model: impl Into<String>) -> Bound<Responses, H> {
        Bound::new(self.wire.responses(model), self.http.clone())
    }
}

impl HasEmbedding for OpenAI {
    type Wire = Embeddings;

    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        self.embeddings(model, ndims)
    }
}

impl HasRerank for OpenAI {
    type Wire = Rerank;

    fn rerank(&self, model: impl Into<String>) -> Rerank {
        self.reranker(model)
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

/// The issuer of reasoning a gateway relays from `model` (`vendor/name`):
/// `anthropic` for Claude, whose thinking signatures Anthropic documents as
/// valid across its platforms and which verified as valid between OpenRouter
/// and the Claude API in both directions; `<gateway>/<vendor>` for every
/// other family, whose signatures and ciphertext are bound to the gateway's
/// upstream account.
pub fn upstream_reasoning_issuer(gateway: &str, model: &str) -> String {
    let vendor = model_vendor(model);
    if vendor == "anthropic" {
        vendor.to_owned()
    } else {
        format!("{gateway}/{vendor}")
    }
}

fn model_vendor(model: &str) -> &str {
    let vendor = model.split_once('/').map_or(model, |(vendor, _)| vendor);
    vendor.trim_start_matches('~')
}

/// The reasoning issuers a request to `model` over `dialect` replays.
///
/// A dialect without upstream issuers replays its own reasoning. A gateway
/// replays the requested model's family, plus reasoning stamped with the
/// gateway alone, which predates upstream issuers. A model under the
/// gateway's own vendor (`openrouter/auto`) or a preset (`@preset/name`)
/// names no family, so the turn may be served by any of them: it replays
/// every family the gateway relayed and Claude reasoning from any surface,
/// rather than dropping reasoning a Claude tool loop must send back. An
/// unrecognised vendor is its own family.
pub(crate) fn replay_issuers(dialect: &Dialect, model: &str) -> Vec<String> {
    let gateway = dialect.name;
    if !dialect.quirks.upstream_reasoning_issuer {
        return vec![gateway.to_owned()];
    }
    if model.starts_with('@') || model_vendor(model) == gateway {
        return vec![
            "anthropic".to_owned(),
            format!("{gateway}/"),
            gateway.to_owned(),
        ];
    }
    vec![
        upstream_reasoning_issuer(gateway, model),
        gateway.to_owned(),
    ]
}

#[cfg(test)]
mod tests;
