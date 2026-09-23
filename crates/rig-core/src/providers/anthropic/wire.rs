//! Configuration and wires for Messages-format completion, model listing,
//! and credential verification. [`Dialect`] describes endpoint defaults and capabilities.
//!
//! ```no_run
//! use rig_core::providers::anthropic::{Anthropic, completion::CLAUDE_SONNET_4_6};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Anthropic::from_env()?.messages(CLAUDE_SONNET_4_6);
//! # Ok(())
//! # }
//! ```

use crate::client::env::{self, EnvError};
use crate::completion::{CompletionRequest, ProviderCapabilities};
use crate::error::EncodeError;
use crate::model::{Model, ModelList};
pub use crate::operation::VerifyDecoder;
use crate::operation::{Completion, ModelListing, Verify as VerifyOp};
use crate::providers::internal::named_dialect;
use crate::wire::{
    Body, Decoder, Encoded, Framing, HasCompletion, Mode, Output, Secret, Sink, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::completion::{
    AnthropicCompletionRequest, AnthropicRequestParams, CacheTtl, ToolDefinition,
    default_max_tokens_for_model, sanitize_strict_tool_schema,
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
}

impl Quirks {
    /// Anthropic's own contract.
    pub const fn anthropic() -> Self {
        Self {
            max_tokens: MaxTokens::ByModel,
            strict_tool_schemas: true,
        }
    }

    /// Default to 4096 output tokens without constrained tool schemas.
    pub const fn gateway() -> Self {
        Self {
            max_tokens: MaxTokens::Fixed(4096),
            strict_tool_schemas: false,
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

/// The shared configuration of an Anthropic-format provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Anthropic {
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
}

impl Anthropic {
    /// Anthropic itself, with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::with_dialect(api_key, &ANTHROPIC)
    }

    /// A Messages-format provider with default settings.
    pub fn with_dialect(api_key: impl Into<Secret>, dialect: &Dialect) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: dialect.base_url.to_owned(),
            version: super::completion::ANTHROPIC_VERSION_LATEST.to_owned(),
            betas: Vec::new(),
            dialect: *dialect,
        }
    }

    /// Anthropic from `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_with(&ANTHROPIC)
    }

    /// A Messages-format provider from the variables its dialect names.
    pub fn from_env_with(dialect: &Dialect) -> Result<Self, EnvError> {
        let mut provider = Self::with_dialect(env::required(dialect.api_key_env)?, dialect);
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
    pub fn messages(&self, model: impl Into<String>) -> Messages {
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
    pub fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }

    /// The credential-check wire.
    pub fn verify(&self) -> Verify {
        Verify {
            provider: self.clone(),
        }
    }

    /// The request headers every Messages-format endpoint takes.
    fn headers(&self, builder: http::request::Builder) -> http::request::Builder {
        let builder = builder
            .header("x-api-key", self.api_key.expose())
            .header("anthropic-version", &self.version);
        if self.betas.is_empty() {
            builder
        } else {
            builder.header("anthropic-beta", self.betas.join(","))
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
    pub provider: Anthropic,
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
    /// use rig_core::providers::anthropic::wire::Anthropic;
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let messages = Anthropic::from_env()?
    ///     .messages(CLAUDE_SONNET_4_6)
    ///     .with_automatic_caching();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// On a [`Bound`](crate::driver::Bound) the same option is forwarded
    /// with `bound.map_wire(|wire| wire.with_automatic_caching())`.
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
    /// use rig_core::providers::anthropic::wire::Anthropic;
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let messages = Anthropic::from_env()?
    ///     .messages(CLAUDE_SONNET_4_6)
    ///     .with_automatic_caching_1h();
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
    /// use rig_core::providers::anthropic::wire::Anthropic;
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let messages = Anthropic::from_env()?
    ///     .messages(CLAUDE_SONNET_4_6)
    ///     .with_automatic_caching()
    ///     .with_static_prefix_cache_ttl(CacheTtl::OneHour);
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
        let typed = AnthropicCompletionRequest::try_from_params(
            AnthropicRequestParams {
                model: &model,
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

impl Wire for Messages {
    type Op = Completion;
    type Decoder = MessagesDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let body = self.body(request, mode)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Anthropic completion request",
            &body,
        );
        let request = self
            .provider
            .headers(http::Request::post(format!(
                "{}/v1/messages",
                self.provider.base_url
            )))
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(
            request,
            match mode {
                Mode::Unary => Framing::Whole,
                Mode::Streaming => Framing::Sse,
            },
        )
        .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        MessagesDecoder::new(self.provider.dialect.name)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Constrained output decoding does not suppress strict tool calls.
        ProviderCapabilities::default().with_native_output_tool_composition(true)
    }
}

/// The model-listing wire: `GET /v1/models`, cursor-paged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The provider this wire speaks to.
    pub provider: Anthropic,
}

impl Wire for Models {
    type Op = ModelListing;
    type Decoder = ModelsDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        Ok(Encoded::new(self.models_request(None)?, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        ModelsDecoder {
            provider: self.provider.clone(),
            next: None,
        }
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

impl From<ModelEntry> for Model {
    fn from(entry: ModelEntry) -> Self {
        Model::new(entry.id, entry.display_name)
    }
}

/// Decodes `GET /v1/models`, following the cursor Anthropic names.
pub struct ModelsDecoder {
    provider: Anthropic,
    /// The cursor the last page named, when it claimed more pages.
    next: Option<String>,
}

impl Decoder<ModelListing> for ModelsDecoder {
    type Event = ModelsPage;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["data"])
    }

    fn interpret(&mut self, page: Self::Event, out: &mut Output<ModelListing>) {
        // Missing or empty cursors would repeatedly fetch page one, even with has_more.
        self.next = page
            .last_id
            .filter(|cursor| page.has_more && !cursor.is_empty());
        out.push(Ok(ModelList::new(
            page.data.into_iter().map(Model::from).collect(),
        )));
    }

    fn continuation(&self) -> Option<http::Request<Body>> {
        let cursor = self.next.as_deref()?;
        Models {
            provider: self.provider.clone(),
        }
        .models_request(Some(cursor))
        .ok()
    }
}

/// The credential-check wire: `GET /v1/models`, status only, decoded by
/// the shared [`VerifyDecoder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verify {
    /// The provider this wire speaks to.
    pub provider: Anthropic,
}

impl Wire for Verify {
    type Op = VerifyOp;
    type Decoder = VerifyDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
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

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        VerifyDecoder
    }
}

impl HasCompletion for Anthropic {
    type Wire = Messages;

    fn completion(&self, model: impl Into<String>) -> Messages {
        self.messages(model)
    }
}

impl crate::driver::HasModelListing for Anthropic {
    type Wire = Models;

    fn model_listing(&self) -> Models {
        self.models()
    }
}

impl crate::driver::HasVerify for Anthropic {
    type Wire = Verify;

    fn verify(&self) -> Verify {
        Anthropic::verify(self)
    }
}

#[cfg(test)]
mod tests;
