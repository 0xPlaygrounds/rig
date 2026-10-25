//! Configuration and wires for Messages-format completion, model listing,
//! and credential verification. [`Dialect`] describes endpoint defaults and capabilities.
//!
//! ```no_run
//! use rig_core::providers::anthropic::{Anthropic, completion::CLAUDE_SONNET_4_6};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Anthropic::from_env()?.completion(CLAUDE_SONNET_4_6);
//! # Ok(())
//! # }
//! ```

use crate::catalog::ModelFacts;
use crate::client::env::{self, EnvError};
use crate::completion::options::FinalBody;
use crate::completion::{CompletionRequest, ProviderCapabilities};
use crate::error::EncodeError;
use crate::operation::Completion;
use crate::providers::internal::named_dialect;
use crate::wire::{Body, Capabilities, Descriptor, Encoded, Framing, Mode, Secret, Wire};
use serde::{Deserialize, Serialize};

use super::completion::{
    CacheTtl, default_max_tokens_for_model, document_source, image_source,
    rejects_forced_tool_choice,
};
use super::streaming::MessagesDecoder;

/// Endpoint defaults and capabilities for a Messages-format provider.
/// Serializes by registered name. Serialization rejects modified or unregistered
/// dialects; deserialization rejects unknown names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dialect {
    /// The provider descriptor name, as records and telemetry spell it.
    pub name: &'static str,
    /// The default base URL.
    pub base_url: &'static str,
    /// The environment variable carrying the API key.
    pub api_key_env: &'static str,
    /// The environment variable overriding the base URL, when the provider
    /// documents one.
    pub base_url_env: Option<&'static str>,
    /// The reply header carrying the provider's transport request id.
    pub request_id_header: Option<&'static str>,
    /// Everything that is not identity.
    pub quirks: Quirks,
}

/// Token defaults, schema capabilities and request envelope for a
/// Messages-format dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Quirks {
    /// How `max_tokens` is defaulted when the caller sets none. Anthropic
    /// requires the field, so a wire that cannot supply it fails the request
    /// rather than sending one the API rejects.
    pub max_tokens: MaxTokens,
    /// Whether the provider implements Anthropic's constrained tool schemas.
    /// A gateway that does not leaves Rig-generated tools unchanged.
    pub strict_tool_schemas: bool,
    /// Whether the provider takes thinking back without a signature, as
    /// Kimi does. Elsewhere unsigned thinking
    /// replays as text.
    pub unsigned_thinking: bool,
    /// How a request with tools asks the provider to stream tool input.
    pub tool_input_streaming: ToolInputStreaming,
    /// How the credential is sent.
    pub auth: Auth,
    /// The header every completion request carries a session id in: the
    /// request's cache key when it has one, else a fresh id.
    pub session_header: Option<&'static str>,
    /// Whether `output_schema` maps to `output_config.format`. A dialect
    /// that does not read it gets no schema, with a warning. An agent run
    /// with tools then enforces the output through its output tool; one
    /// without tools gets unconstrained output.
    pub supports_output_format: bool,
}

/// How a Messages-format provider takes its credential.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Auth {
    /// The `x-api-key` header, always sent.
    #[default]
    ApiKey,
    /// `Authorization: Bearer`, sent only when there is a credential, for a
    /// local server that needs none and a hosted one that refuses
    /// `x-api-key`.
    OptionalBearer,
}

/// How a request with tools asks a Messages-format provider to stream each
/// call's input as the model writes it. Without it, Anthropic holds a long
/// argument value back and sends it in one burst. Unary requests ask too, so
/// a tool definition, which a prompt cache keys on, is the same whether a
/// turn streams.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolInputStreaming {
    /// `eager_input_streaming: true` on each Rig tool definition.
    #[default]
    Eager,
    /// The `fine-grained-tool-streaming-2025-05-14` beta flag, for a
    /// provider that rejects the per-tool field.
    BetaHeader,
    /// Neither: the provider streams tool input as it chooses.
    Off,
}

impl Quirks {
    /// Anthropic's own contract.
    pub const fn anthropic() -> Self {
        Self {
            max_tokens: MaxTokens::ByModel,
            strict_tool_schemas: true,
            unsigned_thinking: false,
            tool_input_streaming: ToolInputStreaming::Eager,
            auth: Auth::ApiKey,
            session_header: None,
            supports_output_format: true,
        }
    }

    /// Default to 4096 output tokens without constrained tool schemas, with
    /// eager tool input, which a Messages-format provider takes unless it is
    /// known to reject it.
    pub const fn gateway() -> Self {
        Self {
            max_tokens: MaxTokens::Fixed(4096),
            strict_tool_schemas: false,
            unsigned_thinking: false,
            tool_input_streaming: ToolInputStreaming::Eager,
            auth: Auth::ApiKey,
            session_header: None,
            supports_output_format: true,
        }
    }
}

/// Where a dialect's `max_tokens` default comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxTokens {
    /// Anthropic's published per-model output limits.
    ByModel,
    /// One ceiling for every model, which is all a gateway documents.
    Fixed(u64),
}

/// The API name the Messages wire reports for every dialect.
pub(crate) const MESSAGES_API: &str = "anthropic.messages";

/// Anthropic itself.
pub const ANTHROPIC: Dialect = Dialect {
    name: super::PROVIDER_NAME,
    base_url: "https://api.anthropic.com",
    api_key_env: "ANTHROPIC_API_KEY",
    base_url_env: Some("ANTHROPIC_BASE_URL"),
    request_id_header: Some("request-id"),
    quirks: Quirks::anthropic(),
};

/// Registered Messages-format dialects in declaration order.
const ALL: &[&Dialect] = &[
    &ANTHROPIC,
    &ZAI,
    &MINIMAX,
    &MOONSHOT,
    &XIAOMIMIMO,
    &OLLAMA,
    &OPENCODE,
    &OPENCODE_GO,
];

/// Every Messages-format dialect this build knows, in declaration order.
pub fn all() -> impl Iterator<Item = &'static Dialect> {
    ALL.iter().copied()
}

impl Dialect {
    /// The dialect this crate ships under `name`.
    pub fn by_name(name: &str) -> Option<Self> {
        all().copied().find(|dialect| dialect.name == name)
    }
}

impl Serialize for Dialect {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let registered = Self::by_name(self.name).as_ref() == Some(self);
        named_dialect::serialize(serializer, "Anthropic", self.name, registered)
    }
}

impl<'de> Deserialize<'de> for Dialect {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        named_dialect::deserialize(deserializer, "Anthropic", Self::by_name)
    }
}

/// Build a Messages-format dialect with `request-id` response headers and
/// [`Quirks::gateway`] defaults. Only registered dialects support serialization.
pub const fn compatible(
    name: &'static str,
    base_url: &'static str,
    api_key_env: &'static str,
    base_url_env: Option<&'static str>,
) -> Dialect {
    Dialect {
        name,
        base_url,
        api_key_env,
        base_url_env,
        request_id_header: Some("request-id"),
        quirks: Quirks::gateway(),
    }
}

impl Dialect {
    /// The `max_tokens` this dialect defaults `model` to, by the output
    /// limit `facts` give it where the dialect defaults by model.
    pub fn default_max_tokens(&self, facts: &ModelFacts, model: &str) -> Option<u64> {
        match self.quirks.max_tokens {
            MaxTokens::ByModel => default_max_tokens_for_model(facts, model),
            MaxTokens::Fixed(tokens) => Some(tokens),
        }
    }
}

/// Z.AI's Anthropic-format endpoint.
pub const ZAI: Dialect = compatible(
    crate::providers::zai::PROVIDER_NAME,
    "https://api.z.ai/api/anthropic",
    "ZAI_API_KEY",
    Some("ZAI_ANTHROPIC_API_BASE"),
);

/// MiniMax's Anthropic-format endpoint.
pub const MINIMAX: Dialect = compatible(
    crate::providers::minimax::PROVIDER_NAME,
    "https://api.minimax.io/anthropic",
    "MINIMAX_API_KEY",
    Some("MINIMAX_ANTHROPIC_API_BASE"),
);

/// Moonshot's Anthropic-format endpoint, which sends and takes back
/// thinking without a signature.
pub const MOONSHOT: Dialect = Dialect {
    quirks: Quirks {
        unsigned_thinking: true,
        ..Quirks::gateway()
    },
    ..compatible(
        crate::providers::moonshot::PROVIDER_NAME,
        "https://api.moonshot.ai/anthropic",
        "MOONSHOT_API_KEY",
        Some("MOONSHOT_ANTHROPIC_API_BASE"),
    )
};

/// Xiaomi MiMo's Anthropic-format endpoint.
pub const XIAOMIMIMO: Dialect = compatible(
    crate::providers::xiaomimimo::PROVIDER_NAME,
    "https://api.xiaomimimo.com/anthropic",
    "XIAOMI_MIMO_API_KEY",
    Some("XIAOMI_MIMO_ANTHROPIC_API_BASE"),
);

/// An Ollama daemon's Messages-format endpoint. A local daemon takes no
/// credential; Ollama's cloud takes its key only as a bearer token. Its
/// thinking carries no signature, and it constrains no output to a schema.
/// It lists no models in Anthropic's shape: list them through the
/// [`Ollama`](crate::providers::ollama::Ollama) client.
pub const OLLAMA: Dialect = Dialect {
    quirks: Quirks {
        auth: Auth::OptionalBearer,
        unsigned_thinking: true,
        // Its `output_config` reads only `effort`.
        supports_output_format: false,
        ..Quirks::gateway()
    },
    ..compatible(
        crate::providers::ollama::PROVIDER_NAME,
        "http://localhost:11434",
        "OLLAMA_API_KEY",
        Some("OLLAMA_API_BASE_URL"),
    )
};

/// OpenCode Zen's Messages-format endpoint, for the models its catalog
/// serves there.
pub const OPENCODE: Dialect = compatible(
    crate::providers::opencode::ZEN,
    "https://opencode.ai/inference/anthropic",
    "OPENCODE_API_KEY",
    Some("OPENCODE_ANTHROPIC_BASE_URL"),
);

/// OpenCode Go's Messages-format endpoint, which refuses a request without
/// an `x-opencode-session` header.
pub const OPENCODE_GO: Dialect = Dialect {
    quirks: Quirks {
        session_header: Some(crate::providers::opencode::SESSION_HEADER),
        ..Quirks::gateway()
    },
    ..compatible(
        crate::providers::opencode::GO,
        "https://opencode.ai/inference/go/anthropic",
        "OPENCODE_API_KEY",
        Some("OPENCODE_GO_ANTHROPIC_BASE_URL"),
    )
};

/// The settings of a Messages-format provider: serializable, and the
/// credential is never serialized. [`connect`](Self::connect) puts it on a
/// transport as an [`Anthropic`](super::Anthropic) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicConfig {
    /// The API key, sent as the dialect's [`Auth`] says.
    pub api_key: Secret,
    /// The API root, without a trailing `/v1`.
    pub base_url: String,
    /// The `anthropic-version` header.
    pub version: String,
    /// The `anthropic-beta` flags, joined with commas when non-empty.
    pub betas: Vec<String>,
    /// Which Messages-format provider this is.
    pub dialect: Dialect,
    /// Where a request names its model and API version.
    #[serde(default, skip_serializing_if = "Addressing::is_body")]
    pub addressing: Addressing,
}

/// Where a Messages request names its model and API version.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Addressing {
    /// `model` in the body and `version` as the `anthropic-version` header,
    /// as Anthropic's own API takes them.
    #[default]
    Body,
    /// No `model` in the body, because the URL names it, and `version` as
    /// the body's `anthropic_version`, as cloud platforms take them. Such a
    /// platform builds its URL with [`Messages::encode_with`].
    Url,
}

impl Addressing {
    /// Whether this is Anthropic's own [`Addressing::Body`].
    fn is_body(&self) -> bool {
        *self == Self::Body
    }
}

impl AnthropicConfig {
    /// Anthropic itself, with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::with_key(&ANTHROPIC, api_key)
    }

    /// `dialect` with `api_key`, at the dialect's default base URL and
    /// with default settings. The base URL is normalized as
    /// [`with_base_url`](Self::with_base_url) does.
    pub fn with_key(dialect: &Dialect, api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: normalize_base_url(dialect.base_url),
            version: super::completion::ANTHROPIC_VERSION_LATEST.to_owned(),
            betas: Vec::new(),
            dialect: *dialect,
            addressing: Addressing::Body,
        }
    }

    /// Anthropic from `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_with(&ANTHROPIC)
    }

    /// A Messages-format provider from the variables its dialect names. A
    /// dialect whose credential is optional ([`Auth::OptionalBearer`]) takes
    /// an unset key as none.
    pub fn from_env_with(dialect: &Dialect) -> Result<Self, EnvError> {
        let api_key = match dialect.quirks.auth {
            Auth::ApiKey => env::required(dialect.api_key_env)?,
            Auth::OptionalBearer => env::optional(dialect.api_key_env)?.unwrap_or_default(),
        };
        Self::from_env_with_credential(dialect, api_key)
    }

    /// `dialect` with `api_key` and the base URL its environment variable
    /// names, if set.
    pub(crate) fn from_env_with_credential(
        dialect: &Dialect,
        api_key: impl Into<Secret>,
    ) -> Result<Self, EnvError> {
        let mut provider = Self::with_key(dialect, api_key);
        if let Some(name) = dialect.base_url_env
            && let Some(base_url) = env::optional(name)?
        {
            provider.base_url = normalize_base_url(&base_url);
        }
        Ok(provider)
    }

    /// Pin the `anthropic-version` header.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Request an `anthropic-beta` flag.
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    /// Point the wire at another API root.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = normalize_base_url(base_url.as_ref());
        self
    }

    /// The Messages wire for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Messages {
        let model = model.into();
        Messages {
            default_max_tokens: self
                .dialect
                .default_max_tokens(ModelFacts::builtin(), &model),
            provider: self.clone(),
            model,
            prompt_caching: false,
            static_prefix_cache_ttl: None,
            strict_tools: false,
            tool_input_streaming: self.dialect.quirks.tool_input_streaming,
            facts: ModelFacts::default(),
        }
    }

    /// The request headers every Messages-format endpoint takes.
    pub(super) fn headers(&self, builder: http::request::Builder) -> http::request::Builder {
        self.headers_with(builder, &self.betas)
    }

    /// [`Self::headers`], with `betas` as the `anthropic-beta` flags.
    fn headers_with(
        &self,
        builder: http::request::Builder,
        betas: &[String],
    ) -> http::request::Builder {
        let builder = match self.dialect.quirks.auth {
            Auth::ApiKey => builder.header("x-api-key", self.api_key.expose()),
            Auth::OptionalBearer if self.api_key.is_empty() => builder,
            Auth::OptionalBearer => builder.header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.expose()),
            ),
        };
        let builder = builder.header("anthropic-version", &self.version);
        if betas.is_empty() {
            builder
        } else {
            builder.header("anthropic-beta", betas.join(","))
        }
    }

    /// The configured `anthropic-beta` flags, then each of `extra` not
    /// among them.
    fn betas_with(&self, extra: &[&str]) -> Vec<String> {
        let mut betas = self.betas.clone();
        for extra in extra {
            if !betas.iter().any(|beta| beta == extra) {
                betas.push((*extra).to_owned());
            }
        }
        betas
    }
}

/// Trim the suffixes a user may paste from the API docs, so a base URL that
/// already names the endpoint does not produce `/v1/messages/v1/messages`.
pub fn normalize_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    for suffix in ["/v1/messages", "/messages", "/v1"] {
        if let Some(stripped) = trimmed.strip_suffix(suffix) {
            return stripped.to_owned();
        }
    }
    trimmed.to_owned()
}

/// The Messages wire: `POST /v1/messages`, SSE when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Messages {
    /// The provider this wire speaks to.
    pub provider: AnthropicConfig,
    /// The model to address.
    pub model: String,
    /// What `max_tokens` defaults to when the caller sets none. Anthropic
    /// requires the field; `None` means a request without one is rejected
    /// here rather than by the API.
    pub default_max_tokens: Option<u64>,
    /// Manual prompt caching: `cache_control` breakpoints on the system
    /// prompt, the last tool definition, and the last content block of the
    /// last message. The request's
    /// [`cache`](crate::completion::GenerationOptions::cache) sets the
    /// top-level marker and its TTL.
    pub prompt_caching: bool,
    /// TTL for the static prefix (tools + system), independent of the
    /// conversation tail. `None` inherits the top-level TTL.
    pub static_prefix_cache_ttl: Option<CacheTtl>,
    /// Whether Rig-generated tools request the provider's strict validation.
    pub strict_tools: bool,
    /// How a request with tools asks for tool input as it is written; the
    /// dialect's by default.
    #[serde(default)]
    pub tool_input_streaming: ToolInputStreaming,
    /// The model facts the encoder reads and replies are priced by.
    #[serde(skip)]
    pub facts: ModelFacts,
}

impl Messages {
    /// The same wire, encoding with `facts` and pricing its replies by
    /// them. The `max_tokens` a request without one defaults to becomes the
    /// dialect's default for the model under `facts`; set another with
    /// [`Self::with_default_max_tokens`] after this.
    pub fn with_facts(mut self, facts: ModelFacts) -> Self {
        self.default_max_tokens = self
            .provider
            .dialect
            .default_max_tokens(&facts, &self.model);
        self.facts = facts;
        self
    }

    /// Set the `max_tokens` a request without one defaults to.
    pub fn with_default_max_tokens(mut self, tokens: u64) -> Self {
        self.default_max_tokens = Some(tokens);
        self
    }

    /// Enable cache breakpoints on the system prompt, final tool, and final message block.
    /// With a request [`cache`](crate::completion::GenerationOptions::cache), the
    /// provider owns the moving message breakpoint and the markers take its TTL.
    /// Existing tool markers are preserved and count toward the four-breakpoint budget.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Cache the static prefix (tool definitions and system prompt) at its
    /// own TTL, independent of the conversation tail's.
    ///
    /// Mark the final tool definition and system prompt at `ttl` without
    /// changing the conversation tail's TTL.
    ///
    /// ```no_run
    /// use rig_core::providers::anthropic::completion::{CLAUDE_SONNET_4_6, CacheTtl};
    /// use rig_core::providers::anthropic::Anthropic;
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut messages = Anthropic::from_env()?.completion(CLAUDE_SONNET_4_6);
    /// messages.wire = messages.wire.with_static_prefix_cache_ttl(CacheTtl::OneHour);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// One-hour markers must precede five-minute markers. A five-minute
    /// prefix under a request with
    /// [`CacheRetention::Long`](crate::completion::CacheRetention::Long) fails
    /// during encoding.
    /// Each marker must meet the model's minimum cacheable prompt length.
    pub fn with_static_prefix_cache_ttl(mut self, ttl: CacheTtl) -> Self {
        self.static_prefix_cache_ttl = Some(ttl);
        self
    }

    /// Request Anthropic's strict tool use for every Rig-generated tool.
    ///
    /// Anthropic constrains tool inputs to the supported JSON Schema subset
    /// when `strict: true` is present on a tool definition. Rig sanitizes
    /// each generated schema for that subset and leaves provider-specific
    /// tools supplied through `additional_params` unchanged. Unsupported
    /// validation keywords are retained only as model guidance in schema
    /// descriptions; neither Anthropic nor Rig enforces them, so validate
    /// tool inputs before execution when those constraints matter.
    ///
    /// Anthropic caches compiled schemas for up to 24 hours: do not include
    /// PHI in schema property names, enum or const values, or regex
    /// patterns. A dialect that does not implement constrained tool schemas
    /// ignores this.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Choose how a request with tools asks for tool input as the model
    /// writes it, for a gateway whose support differs from its dialect's:
    /// [`ToolInputStreaming::BetaHeader`] for one that rejects the per-tool
    /// field, [`ToolInputStreaming::Off`] for one that rejects both.
    ///
    /// ```no_run
    /// use rig_core::providers::anthropic::{Anthropic, completion::CLAUDE_SONNET_4_6};
    /// use rig_core::providers::anthropic::ToolInputStreaming;
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut messages = Anthropic::from_env()?.completion(CLAUDE_SONNET_4_6);
    /// messages.wire = messages.wire.with_tool_input_streaming(ToolInputStreaming::BetaHeader);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tool_input_streaming(mut self, streaming: ToolInputStreaming) -> Self {
        self.tool_input_streaming = streaming;
        self
    }
}

/// A Messages request encoded but not yet addressed or authenticated: what
/// a platform serving the Messages API under its own envelope sends.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Prepared {
    /// The model the request addresses.
    pub model: String,
    /// The request body.
    pub body: FinalBody,
    /// The `anthropic-beta` flags: the configured ones, then the ones the
    /// request needs.
    pub betas: Vec<String>,
}

impl Messages {
    /// Encode `request` as [`Wire::encode`] does, with `envelope` building
    /// the HTTP request in place of `POST /v1/messages` with an API key.
    /// The framing, request-id header and projection stay this wire's, so
    /// a platform that serves the Messages API at another address, with
    /// other credentials or another body envelope, reads replies as this
    /// wire does.
    ///
    /// # Errors
    ///
    /// When the request does not encode, or `envelope` fails.
    pub fn encode_with(
        &self,
        request: CompletionRequest,
        mode: Mode,
        envelope: impl FnOnce(Prepared) -> Result<http::Request<Body>, EncodeError>,
    ) -> Result<Encoded, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let fine_grained = !request.tools.is_empty()
            && self.tool_input_streaming == ToolInputStreaming::BetaHeader;
        let body = super::completion::body(self, &request, mode)?;
        let betas: Vec<&str> = [
            body.get("thinking")
                .and_then(|thinking| thinking.get("block_binding"))
                .is_some_and(serde_json::Value::is_object)
                .then_some(super::completion::THINKING_BINDING_BETA),
            fine_grained.then_some(super::completion::FINE_GRAINED_TOOL_STREAMING_BETA),
        ]
        .into_iter()
        .flatten()
        .collect();
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Anthropic completion request",
            &body,
        );
        let request = envelope(Prepared {
            model,
            body,
            betas: self.provider.betas_with(&betas),
        })?;
        Ok(Encoded::new(
            request,
            match mode {
                Mode::Unary => Framing::Whole,
                Mode::Streaming => Framing::Sse,
            },
        )
        .with_request_id_header(self.provider.dialect.request_id_header)
        .with_projection(MessagesDecoder::project))
    }
}

impl Wire for Messages {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Event = super::streaming::MessagesEvent;
    type Decoder<'id> = MessagesDecoder;
    type Reassembler = super::streaming::document::Message;

    /// Constrained output decoding does not suppress strict tool calls. A
    /// dialect that reads no output format leaves the output to the agent's
    /// tool.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
            .model(self.model.as_str())
            .facts(&self.facts)
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default()
                    .with_native_output_tool_composition(
                        self.provider.dialect.quirks.supports_output_format,
                    )
                    .with_forced_tool_choice_rejected(rejects_forced_tool_choice(
                        &self.facts,
                        &self.model,
                    )),
            ))
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let session = self.provider.dialect.quirks.session_header.map(|name| {
            let id = crate::providers::internal::request_session_id(&request);
            (name, id)
        });
        self.encode_with(request, mode, |prepared| {
            let mut builder = self
                .provider
                .headers_with(
                    http::Request::post(format!("{}/v1/messages", self.provider.base_url)),
                    &prepared.betas,
                )
                .header(http::header::CONTENT_TYPE, "application/json");
            if let Some((name, id)) = session {
                builder = builder.header(name, id);
            }
            Ok(builder.body(prepared.body.into_body())?)
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        MessagesDecoder::new(self.provider.dialect.quirks.unsigned_thinking)
    }
}

impl crate::completion::ReplayTarget for Messages {
    /// Section 6.1 of the typed-options design, by dialect and model class.
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        super::options::map_options(self, request, fields)
    }

    fn facts(&self) -> Option<&ModelFacts> {
        Some(&self.facts)
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static(MESSAGES_API)
    }

    fn provider(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Messages takes images in user turns and tool results, never in
    /// assistant turns, on models that read images: every Claude model, and
    /// on another dialect each model its catalog entry lists as reading
    /// them, or for a model the catalog does not list each dialect's
    /// documented vision naming.
    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        let rule: Option<fn(&str) -> bool> = match self.provider.dialect.name {
            name if name == ANTHROPIC.name => None,
            name if name == ZAI.name => Some(crate::providers::zai::reads_images),
            name if name == MOONSHOT.name => Some(crate::providers::moonshot::reads_images),
            name if name == MINIMAX.name => Some(crate::providers::minimax::reads_images),
            name if name == XIAOMIMIMO.name => Some(crate::providers::xiaomimimo::reads_images),
            _ => Some(|_| true),
        };
        let images = rule.is_none_or(|rule| {
            self.facts
                .reads_images_or(self.provider.dialect.name, model, rule)
        });
        crate::completion::Accepts {
            user_images: images,
            assistant_images: false,
            tool_result_images: images,
            tools: true,
            // Thinking replays only with its signature.
            foreign_reasoning: false,
        }
    }

    /// The encoder carries images by typed base64, URL or file id in user
    /// turns and tool results, and documents by file id, PDF data or URL,
    /// or the text they hold. Assistant images, audio and video it never
    /// carries.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        use crate::completion::{Media, Place};
        match media {
            Media::Image(image, Place::User | Place::ToolResult) => image_source(image).is_some(),
            Media::Document(document) => document_source(document).is_some(),
            Media::Image(_, Place::Assistant) | Media::Audio(_) | Media::Video(_) => false,
        }
    }

    /// Anthropic takes call ids of `[a-zA-Z0-9_-]`, at most 64 long.
    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _source: Option<&crate::message::Origin>,
    ) -> String {
        crate::providers::internal::wire_ids::legal_call_id(id, 64)
    }

    /// An edited call keeps the `caller` that ties it to the code
    /// execution that made it.
    fn identity(&self, item: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        item.get("caller")
            .filter(|_| item.get("type").and_then(serde_json::Value::as_str) == Some("tool_use"))
            .map(|caller| serde_json::Map::from_iter([("caller".to_owned(), caller.clone())]))
            .unwrap_or_default()
    }

    /// A model that takes no system message inside `messages` gets every
    /// one folded into the leading system prompt.
    fn later_system(&self, model: &str) -> crate::completion::LaterSystem {
        if super::completion::takes_mid_conversation_system(&self.facts, model) {
            crate::completion::LaterSystem::InPlace
        } else {
            crate::completion::LaterSystem::Leading
        }
    }

    /// A container item is request state, never content, and unsigned
    /// thinking that is redacted or blank has nothing to send.
    /// Claude models whose thinking binds to the tools and system prompt it
    /// was made under.
    fn binds_context(&self, model: &str) -> bool {
        self.provider.dialect.name == ANTHROPIC.name
            && super::completion::binds_context(&self.facts, model)
    }

    /// A request in adaptive thinking asks Anthropic to drop a block bound
    /// to another context (`drop_block`), so its turns replay verbatim.
    /// The thinking read is what the body sends: the mapped `reasoning`,
    /// with `additional_params` over it.
    fn drops_unbound_items(&self, request: &CompletionRequest) -> bool {
        let model = request.model.as_deref().unwrap_or(&self.model);
        let thinking = crate::completion::options::param(self, request, "thinking");
        super::completion::drops_unbound_thinking(self, model, thinking.as_ref())
    }

    /// What the encoder sends for the block, so the two never disagree.
    fn sends_alone(&self, block: &crate::message::AssistantContent) -> bool {
        let ids = crate::providers::internal::wire_ids::WireIds::default();
        super::completion::assistant_part(block, self, &ids).is_some()
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    fn blob_slots(&self) -> &'static [&'static str] {
        &["/signature", "/data"]
    }

    /// A server tool's use (`server_tool_use`, `mcp_tool_use`) and the
    /// `*_tool_result` that answers it by `tool_use_id`.
    fn hosted_pair(
        &self,
        item: &serde_json::Value,
    ) -> Option<(crate::completion::Pairing, String)> {
        use crate::completion::Pairing;
        let kind = item.get("type")?.as_str()?;
        let (side, key) = if kind.ends_with("_tool_use") {
            (Pairing::Use, "id")
        } else if kind.ends_with("_tool_result") {
            (Pairing::Result, "tool_use_id")
        } else {
            return None;
        };
        Some((side, item.get(key)?.as_str()?.to_owned()))
    }
}

#[cfg(test)]
mod tests;
