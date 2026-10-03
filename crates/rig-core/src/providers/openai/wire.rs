//! OpenAI-compatible configurations, dialect policies, and endpoint wires.
//! A [`Dialect`](crate::providers::openai::wire::Dialect) selects request
//! and response policies; [`OpenAIConfig`](crate::providers::openai::OpenAIConfig)
//! holds credentials and overrides. An
//! [`OpenAI`](crate::providers::openai::OpenAI) client puts
//! the configuration on a transport and builds each endpoint's model.
//!
//! ```
//! use rig_core::providers::openai::{OpenAIConfig, Route, wire::OpenAiWire};
//!
//! let openai = OpenAIConfig::new("key").with_route(Route::Chat).client();
//! assert!(matches!(openai.completion("gpt-5.2").wire, OpenAiWire::Chat(_)));
//! ```

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::wire::Secret;

use super::responses_api::SystemInstructionsPlacement;
use super::responses_api::wire::Responses;

mod auth;
mod chat;
mod dialects;
/// The merge that assembles a streamed provider object from its fragments.
pub(crate) mod dto;
mod modality;
mod route;

use auth::default_user_agent;
pub use auth::{Auth, AuthAlternative, CallerIdentity, Identity};
pub use chat::Chat;
pub use dialects::*;
pub use modality::{
    AcceptedWidths, DimensionsField, EmbeddingQuirks, Embeddings, EmbeddingsDecoder, ImageBody,
    ModelEntry, ModelWidth, Models, ModelsDecoder, ModelsReply, Rerank, RerankDecoder,
    RerankQuirks, RerankReply, RerankResultEntry, RerankUsage, SpeechBody, TranscriptionBody,
    Transcriptions, TranscriptionsDecoder, Verify, VerifyDecoder,
};
pub use route::{OpenAiDecoder, OpenAiEvent, OpenAiWire, Route};

#[cfg(feature = "image")]
pub use modality::{ImageDatum, Images, ImagesDecoder, ImagesEvent, ImagesReply};
#[cfg(feature = "audio")]
pub use modality::{Speech, SpeechDecoder};

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

/// Dialect-specific transformation of the serialized chat request.
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
    /// assistant turns, and forced tool choices suppressed unless thinking
    /// is explicitly disabled. Its reasoning field is
    /// [`Quirks::reasoning_field`].
    DeepSeek,
    /// Mira's gateway: content-part arrays flattened to strings.
    Mira,
    /// Perplexity: text-only arrays flattened.
    Perplexity,
    /// Mistral: `any` for a forced tool choice, the choice relaxed to `auto`
    /// beside a structured response format, its own content chunks, and
    /// `content` on every assistant turn.
    Mistral,
    /// llama.cpp: refuse a specific-function tool choice, which
    /// `llama-server` silently treats as `auto`.
    LlamaCpp,
    /// Moonshot: refuse a specific-function tool choice and coerce
    /// `required` to `auto` with a steering message.
    Moonshot,
    /// OpenRouter: ephemeral `cache_control` on the system prompt when
    /// prompt caching is on.
    OpenRouter,
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
    fn(&OpenAIConfig, &mut http::Request<crate::wire::Body>) -> Result<(), http::Error>;

/// A dialect's completion headers, applied to the authenticated request builder.
pub type CompletionEnvelope = fn(
    &OpenAIConfig,
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
    /// Which completion endpoint
    /// [`OpenAI::completion`](crate::providers::openai::OpenAI::completion) builds: the
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
    /// Where the server takes `system` messages that come after the
    /// conversation begins.
    pub later_system: crate::completion::LaterSystem,
    /// Whether a streaming request asks for the usage chunk through
    /// `stream_options`.
    pub stream_include_usage: bool,
    /// How the dialect spells the output-token cap.
    pub output_cap: OutputCap,
    /// Whether to consult upstream-native finish reasons when normalized ones are absent.
    pub native_finish_reason: bool,
    /// The finish reasons the dialect documents beyond the ones every Chat
    /// dialect shares (`stop`, `length`, `tool_calls`, `content_filter` and
    /// their compatible spellings). Any other reason fails the turn.
    pub finishes: &'static [(&'static str, crate::completion::FinishReason)],
    /// Whether every reply states why it stopped (pi's
    /// `supportsFinishReason`). A dialect whose server never sends a reason
    /// says `false`, and its replies then end as a stop.
    pub states_finish_reason: bool,
    /// The field rebuilt reasoning goes under, which every assistant message
    /// then carries, empty when the turn has none (pi's
    /// `requiresReasoningContentOnAssistantMessages`). `None` sends
    /// reasoning only under the field it arrived in.
    pub reasoning_field: Option<&'static str>,
    /// Whether `completion_tokens_details.reasoning_tokens` can be trusted as
    /// a part of `completion_tokens`, as OpenAI documents it. A dialect whose
    /// replies report more reasoning than completion leaves the count
    /// unreported, so [`Usage`](crate::completion::Usage) never reports more
    /// reasoning than output.
    pub reliable_reasoning_count: bool,
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
            later_system: crate::completion::LaterSystem::InPlace,
            stream_include_usage: true,
            output_cap: OutputCap::Legacy,
            native_finish_reason: false,
            finishes: &[],
            states_finish_reason: true,
            reasoning_field: None,
            reliable_reasoning_count: true,
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

    /// These quirks without the streamed usage chunk: a streaming request
    /// sends no `stream_options`, so streamed usage reports `None`.
    pub const fn without_stream_usage(mut self) -> Self {
        self.stream_include_usage = false;
        self
    }

    /// These quirks without structured output: `output_schema` does not map
    /// to `response_format`.
    pub const fn without_response_format(mut self) -> Self {
        self.supports_response_format = false;
        self
    }
}

impl Default for Quirks {
    /// [`Quirks::openai`].
    fn default() -> Self {
        Self::openai()
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

    /// This dialect with `quirks`.
    pub const fn with_quirks(mut self, quirks: Quirks) -> Self {
        self.quirks = quirks;
        self
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

/// The settings of an OpenAI-shaped provider: serializable, and the
/// credential is never serialized. [`connect`](Self::connect) puts it on a
/// transport as an [`OpenAI`](super::OpenAI) client.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAIConfig {
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

impl OpenAIConfig {
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
        let (api_key, auth) = Self::credential_from_env(dialect)?;
        Self::from_env_with_credential(dialect, api_key, auth)
    }

    /// [`Self::from_env_with`] with the credential already read.
    pub(crate) fn from_env_with_credential(
        dialect: &Dialect,
        api_key: String,
        auth: Auth,
    ) -> Result<Self, EnvError> {
        let quirks = &dialect.quirks;
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

    /// Override dialect and model-specific routing for the client's
    /// [`completion`](crate::providers::openai::OpenAI::completion).
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
    pub(crate) fn completion(&self, model: impl Into<String>) -> OpenAiWire {
        OpenAiWire::new(self.clone(), model)
    }

    /// The Responses wire for `model`: `POST /responses`.
    pub(crate) fn responses(&self, model: impl Into<String>) -> Responses {
        Responses::new(self.clone(), model)
    }

    /// The chat-completions wire for `model`, whatever the dialect's
    /// [`completion_route`](Quirks::completion_route).
    pub fn chat(&self, model: impl Into<String>) -> Chat {
        Chat::new(self.clone(), model)
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
}

#[cfg(test)]
mod tests;
