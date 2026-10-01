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

use crate::client::env::{self, EnvError};
use crate::completion::{CompletionRequest, ProviderCapabilities};
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::message::Issuer;
use crate::model::{ModelInfo, ModelList};
pub use crate::operation::VerifyDecoder;
use crate::operation::{Completion, ModelListing, ModelPage, Verify as VerifyOp};
use crate::providers::internal::named_dialect;
use crate::wire::Flow;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Framing, Mode, Out, Secret, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::completion::{
    AnthropicCompletionRequest, AnthropicRequestParams, CacheTtl, ToolDefinition,
    binds_thinking_blocks, default_max_tokens_for_model, rejects_forced_tool_choice,
    sanitize_strict_tool_schema,
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

/// Token defaults and schema capabilities for a Messages-format dialect.
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
    /// Whether the provider takes `thinking.block_binding`. When it does not,
    /// [`ThinkingPrefixMismatch`] has no effect and no binding beta is added.
    pub thinking_block_binding: bool,
}

impl Quirks {
    /// Anthropic's own contract.
    pub const fn anthropic() -> Self {
        Self {
            max_tokens: MaxTokens::ByModel,
            strict_tool_schemas: true,
            thinking_block_binding: true,
        }
    }

    /// Default to 4096 output tokens without constrained tool schemas or
    /// thinking-block binding controls.
    pub const fn gateway() -> Self {
        Self {
            max_tokens: MaxTokens::Fixed(4096),
            strict_tool_schemas: false,
            thinking_block_binding: false,
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

/// Anthropic itself.
pub const ANTHROPIC: Dialect = Dialect {
    name: "anthropic",
    base_url: "https://api.anthropic.com",
    api_key_env: "ANTHROPIC_API_KEY",
    base_url_env: Some("ANTHROPIC_BASE_URL"),
    request_id_header: Some("request-id"),
    quirks: Quirks::anthropic(),
};

/// Registered Messages-format dialects in declaration order.
const ALL: &[&Dialect] = &[&ANTHROPIC, &ZAI, &MINIMAX, &MOONSHOT, &XIAOMIMIMO];

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

/// The strict-tool transform Anthropic's constrained decoding needs.
pub(crate) fn strict_tool_transform(tool: &mut ToolDefinition) {
    sanitize_strict_tool_schema(&mut tool.input_schema);
    tool.strict = true;
}

impl Dialect {
    /// The `max_tokens` this dialect defaults `model` to.
    pub fn default_max_tokens(&self, model: &str) -> Option<u64> {
        match self.quirks.max_tokens {
            MaxTokens::ByModel => default_max_tokens_for_model(model),
            MaxTokens::Fixed(tokens) => Some(tokens),
        }
    }
}

/// Z.AI's Anthropic-format endpoint.
pub const ZAI: Dialect = compatible(
    "zai",
    "https://api.z.ai/api/anthropic",
    "ZAI_API_KEY",
    Some("ZAI_ANTHROPIC_API_BASE"),
);

/// MiniMax's Anthropic-format endpoint.
pub const MINIMAX: Dialect = compatible(
    "minimax",
    "https://api.minimax.io/anthropic",
    "MINIMAX_API_KEY",
    Some("MINIMAX_ANTHROPIC_API_BASE"),
);

/// Moonshot's Anthropic-format endpoint.
pub const MOONSHOT: Dialect = compatible(
    "moonshot",
    "https://api.moonshot.ai/anthropic",
    "MOONSHOT_API_KEY",
    Some("MOONSHOT_ANTHROPIC_API_BASE"),
);

/// Xiaomi MiMo's Anthropic-format endpoint.
pub const XIAOMIMIMO: Dialect = compatible(
    "xiaomimimo",
    "https://api.xiaomimimo.com/anthropic/v1",
    "XIAOMI_MIMO_API_KEY",
    Some("XIAOMI_MIMO_ANTHROPIC_API_BASE"),
);

/// The settings of a Messages-format provider: serializable, and the
/// credential is never serialized. [`connect`](Self::connect) puts it on a
/// transport as an [`Anthropic`](super::Anthropic) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicConfig {
    /// The API key, sent as `x-api-key`.
    pub api_key: Secret,
    /// The API root, without a trailing `/v1`.
    pub base_url: String,
    /// The `anthropic-version` header.
    pub version: String,
    /// The `anthropic-beta` flags, joined with commas when non-empty.
    pub betas: Vec<String>,
    /// Which Messages-format provider this is.
    pub dialect: Dialect,
    /// What the API does with a replayed thinking block whose conversation
    /// prefix changed.
    #[serde(default, skip_serializing_if = "ThinkingPrefixMismatch::is_default")]
    pub thinking_prefix_mismatch: ThinkingPrefixMismatch,
}

/// The `anthropic-beta` flag that `thinking.block_binding` requires.
pub const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";

/// What the API does with a replayed thinking block whose conversation prefix
/// changed since the block was produced, for example because the tool list
/// changed between turns.
///
/// It applies only to models that bind thinking blocks to their conversation
/// (Claude Opus 5.5, Fable 5.1 and Fable 5) and only to requests that replay
/// a `thinking` or `redacted_thinking` block. Other requests are unchanged. A
/// `thinking.block_binding` set through `additional_params` takes precedence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingPrefixMismatch {
    /// Send `thinking.block_binding.prefix_mismatch_behavior: "drop_block"`
    /// with the [`THINKING_BINDING_BETA`] flag, so the API drops the stale
    /// block and answers the request.
    #[default]
    DropBlock,
    /// Send no `block_binding`, so the API keeps its default and rejects the
    /// request with a 400.
    Reject,
}

impl ThinkingPrefixMismatch {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl AnthropicConfig {
    /// Anthropic itself, with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::with_key(&ANTHROPIC, api_key)
    }

    /// `dialect` with `api_key`, at the dialect's default base URL and
    /// with default settings.
    pub fn with_key(dialect: &Dialect, api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: dialect.base_url.to_owned(),
            version: super::completion::ANTHROPIC_VERSION_LATEST.to_owned(),
            betas: Vec::new(),
            dialect: *dialect,
            thinking_prefix_mismatch: ThinkingPrefixMismatch::default(),
        }
    }

    /// Anthropic from `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_with(&ANTHROPIC)
    }

    /// A Messages-format provider from the variables its dialect names.
    pub fn from_env_with(dialect: &Dialect) -> Result<Self, EnvError> {
        let mut provider = Self::with_key(dialect, env::required(dialect.api_key_env)?);
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

    /// Choose what the API does with a replayed thinking block whose
    /// conversation prefix changed. See [`ThinkingPrefixMismatch`].
    pub fn with_thinking_prefix_mismatch(mut self, behavior: ThinkingPrefixMismatch) -> Self {
        self.thinking_prefix_mismatch = behavior;
        self
    }

    /// Point the wire at another API root.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = normalize_base_url(base_url.as_ref());
        self
    }

    /// The Messages wire for `model`.
    pub(crate) fn completion(&self, model: impl Into<String>) -> Messages {
        let model = model.into();
        Messages {
            default_max_tokens: self.dialect.default_max_tokens(&model),
            provider: self.clone(),
            model,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
            strict_tools: false,
        }
    }

    /// The model-listing wire.
    pub(crate) fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }

    /// The credential-check wire.
    pub(crate) fn verify(&self) -> Verify {
        Verify {
            provider: self.clone(),
        }
    }

    /// The request headers every Messages-format endpoint takes.
    fn headers(&self, builder: http::request::Builder) -> http::request::Builder {
        self.headers_with_beta(builder, None)
    }

    /// [`Self::headers`], adding `beta` unless the configured flags name it.
    fn headers_with_beta(
        &self,
        builder: http::request::Builder,
        beta: Option<&str>,
    ) -> http::request::Builder {
        let builder = builder
            .header("x-api-key", self.api_key.expose())
            .header("anthropic-version", &self.version);
        let beta = beta.filter(|beta| {
            !self
                .betas
                .iter()
                .flat_map(|flags| flags.split(','))
                .any(|flag| flag.trim() == *beta)
        });
        let betas: Vec<&str> = self.betas.iter().map(String::as_str).chain(beta).collect();
        if betas.is_empty() {
            builder
        } else {
            builder.header("anthropic-beta", betas.join(","))
        }
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
    /// last message.
    pub prompt_caching: bool,
    /// Anthropic's automatic prompt caching: one top-level `cache_control`
    /// the API advances as the conversation grows.
    pub automatic_caching: bool,
    /// TTL for the automatic breakpoint. `None` takes the API default.
    pub automatic_caching_ttl: Option<CacheTtl>,
    /// TTL for the static prefix (tools + system), independent of the
    /// conversation tail. `None` inherits the top-level TTL.
    pub static_prefix_cache_ttl: Option<CacheTtl>,
    /// Whether Rig-generated tools request the provider's strict validation.
    pub strict_tools: bool,
}

impl Messages {
    /// Set the `max_tokens` a request without one defaults to.
    pub fn with_default_max_tokens(mut self, tokens: u64) -> Self {
        self.default_max_tokens = Some(tokens);
        self
    }

    /// Enable cache breakpoints on the system prompt, final tool, and final message block.
    /// With automatic caching, the provider owns the moving message breakpoint.
    /// Existing tool markers are preserved and count toward the four-breakpoint budget.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Enable top-level `cache_control`, advancing the last cacheable block
    /// as the conversation grows. No beta header is required.
    /// Caching is skipped below the model-specific minimum prompt length.
    ///
    /// ```no_run
    /// use rig_core::providers::anthropic::completion::CLAUDE_SONNET_4_6;
    /// use rig_core::providers::anthropic::Anthropic;
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut messages = Anthropic::from_env()?.completion(CLAUDE_SONNET_4_6);
    /// messages.wire = messages.wire.with_automatic_caching();
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_automatic_caching(mut self) -> Self {
        self.automatic_caching = true;
        self
    }

    /// Automatic caching with the one-hour TTL rather than the default five
    /// minutes. Identical to [`Self::with_automatic_caching`] but sets
    /// `ttl: "1h"` on the top-level `cache_control` field.
    ///
    /// ```no_run
    /// use rig_core::providers::anthropic::completion::CLAUDE_SONNET_4_6;
    /// use rig_core::providers::anthropic::Anthropic;
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut messages = Anthropic::from_env()?.completion(CLAUDE_SONNET_4_6);
    /// messages.wire = messages.wire.with_automatic_caching_1h();
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_automatic_caching_1h(mut self) -> Self {
        self.automatic_caching = true;
        self.automatic_caching_ttl = Some(CacheTtl::OneHour);
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
    /// messages.wire = messages.wire.with_automatic_caching().with_static_prefix_cache_ttl(CacheTtl::OneHour);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// One-hour markers must precede five-minute markers. A five-minute
    /// prefix with [`Self::with_automatic_caching_1h`] fails during encoding.
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

    /// The typed request body, shared by both modes.
    fn body(
        &self,
        mut request: CompletionRequest,
        mode: Mode,
    ) -> Result<serde_json::Value, EncodeError> {
        if request.max_tokens.is_none() {
            let Some(tokens) = self.default_max_tokens else {
                return Err(EncodeError::request(
                    "`max_tokens` must be set for Anthropic",
                ));
            };
            request.max_tokens = Some(tokens);
        }
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        // Only reasoning this dialect issued is replayed here.
        let issuers = [Issuer::from_static(self.provider.dialect.name)];
        let request = request.replayable_to(&issuers)?;
        let typed = AnthropicCompletionRequest::try_from_params(
            AnthropicRequestParams {
                model: &model,
                issuers: &issuers,
                request,
                prompt_caching: self.prompt_caching,
                automatic_caching: self.automatic_caching,
                automatic_caching_ttl: self.automatic_caching_ttl.clone(),
                static_prefix_cache_ttl: self.static_prefix_cache_ttl.clone(),
            },
            (self.strict_tools && self.provider.dialect.quirks.strict_tool_schemas)
                .then_some(strict_tool_transform as fn(&mut ToolDefinition)),
        )?;
        let mut body = serde_json::to_value(&typed)?;
        if self.provider.dialect.quirks.thinking_block_binding
            && self.provider.thinking_prefix_mismatch == ThinkingPrefixMismatch::DropBlock
            && binds_thinking_blocks(&model)
            && typed.replays_thinking()
        {
            drop_stale_thinking_blocks(&mut body);
        }
        if mode == Mode::Unary {
            return Ok(body);
        }
        // Anthropic rejects tool_choice without tools.
        if let Some(map) = body.as_object_mut() {
            map.insert("stream".to_owned(), serde_json::Value::Bool(true));
            if map.contains_key("tools") {
                map.entry("tool_choice")
                    .or_insert_with(|| serde_json::json!({ "type": "auto" }));
            } else {
                map.remove("tool_choice");
            }
        }
        Ok(body)
    }
}

/// Merge `block_binding: {prefix_mismatch_behavior: "drop_block"}` into the
/// body's `thinking`, keeping a caller's own `block_binding`. Disabled thinking
/// rejects the field, so it is left alone.
fn drop_stale_thinking_blocks(body: &mut serde_json::Value) {
    let Some(body) = body.as_object_mut() else {
        return;
    };
    let thinking = body
        .entry("thinking")
        .or_insert_with(|| serde_json::json!({}));
    let Some(thinking) = thinking.as_object_mut() else {
        return;
    };
    if thinking.get("type").and_then(serde_json::Value::as_str) == Some("disabled") {
        return;
    }
    thinking
        .entry("block_binding")
        .or_insert_with(|| serde_json::json!({ "prefix_mismatch_behavior": "drop_block" }));
}

impl Wire for Messages {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = MessagesDecoder<'id>;

    /// Constrained output decoding does not suppress strict tool calls.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default()
                    .with_native_output_tool_composition(true)
                    .with_forced_tool_choice_rejected(rejects_forced_tool_choice(&self.model)),
            ))
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let body = self.body(request, mode)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Anthropic completion request",
            &body,
        );
        // The field needs its beta flag, whoever set it.
        let binding_beta = (self.provider.dialect.quirks.thinking_block_binding
            && body.pointer("/thinking/block_binding").is_some())
        .then_some(THINKING_BINDING_BETA);
        let request = self
            .provider
            .headers_with_beta(
                http::Request::post(format!("{}/v1/messages", self.provider.base_url)),
                binding_beta,
            )
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
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

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        MessagesDecoder::new()
    }
}

/// The model-listing wire: `GET /v1/models`, cursor-paged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The provider this wire speaks to.
    pub provider: AnthropicConfig,
}

impl Wire for Models {
    type Op = ModelListing;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ModelsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
    }

    fn encode(&self, cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        Ok(Encoded::new(
            self.models_request(cursor.as_deref())?,
            Framing::Whole,
        ))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ModelsDecoder
    }
}

impl Models {
    /// One page's request, after `cursor` when the previous page named one.
    fn models_request(&self, cursor: Option<&str>) -> Result<http::Request<Body>, EncodeError> {
        let uri = match cursor {
            Some(cursor) => format!(
                "{}{}",
                self.provider.base_url,
                crate::providers::internal::with_query_pairs("/v1/models", &[("after_id", cursor)],)
            ),
            None => format!("{}/v1/models", self.provider.base_url),
        };
        self.provider
            .headers(http::Request::get(uri))
            .body(Body::empty())
            .map_err(EncodeError::from)
    }
}

/// One page of `GET /v1/models`.
#[derive(Debug, Deserialize)]
#[doc(hidden)]
pub struct ModelsPage {
    data: Vec<ModelEntry>,
    #[serde(default)]
    has_more: bool,
    #[serde(default)]
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    display_name: String,
}

impl From<ModelEntry> for ModelInfo {
    fn from(entry: ModelEntry) -> Self {
        ModelInfo::new(entry.id, entry.display_name)
    }
}

/// Decodes `GET /v1/models` and the cursor Anthropic names.
pub struct ModelsDecoder;

impl<'id> Decoder<'id, ModelListing> for ModelsDecoder {
    type Event = ModelsPage;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["data"])
    }

    fn decode(
        &mut self,
        page: Self::Event,
        out: Out<'id, ModelListing>,
    ) -> Result<Flow, ProviderError> {
        // Missing or empty cursors would repeatedly fetch page one, even with has_more.
        let next = page
            .last_id
            .filter(|cursor| page.has_more && !cursor.is_empty());
        Ok(out.end(ModelPage {
            models: ModelList::new(page.data.into_iter().map(ModelInfo::from).collect()),
            next,
        }))
    }
}

/// The credential-check wire: `GET /v1/models`, status only, decoded by
/// the shared [`VerifyDecoder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verify {
    /// The provider this wire speaks to.
    pub provider: AnthropicConfig,
}

impl Wire for Verify {
    type Op = VerifyOp;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = VerifyDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = self
            .provider
            .headers(http::Request::get(format!(
                "{}/v1/models",
                self.provider.base_url
            )))
            .body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        VerifyDecoder
    }
}

#[cfg(test)]
mod tests;
