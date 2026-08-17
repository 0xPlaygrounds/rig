//! Completion request, response, and provider trait definitions.
//!
//! Provider integrations implement [`CompletionModel`] and translate
//! [`CompletionRequest`] into their native HTTP request format.
//!
//! # Low-level request example
//!
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::{AssistantContent, CompletionModel},
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::from_env()?;
//! let model = client.completion_model(openai::GPT_5_2);
//!
//! let request = model
//!     .completion_request("Who are you?")
//!     .preamble("You are a concise assistant.".to_string())
//!     .temperature(0.5)
//!     .build();
//!
//! let response = model.completion(request).await?;
//! for item in response.choice {
//!     if let AssistantContent::Text(text) = item {
//!         println!("{}", text.text);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use super::message::{AssistantContent, DocumentMediaType};
use crate::http_client;
use crate::message::ToolChoice;
use crate::provider_response;
use crate::streaming::StreamingCompletionResponse;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{
    json_utils,
    message::{Message, UserContent},
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Add, AddAssign};
use thiserror::Error;

// Errors
/// Errors returned by completion models.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
/// These recover the provider's raw HTTP status and response body so you can
/// branch on a provider error code or surface a precise diagnostic. The same
/// helpers are available on `EmbeddingError`, `ImageGenerationError`,
/// `AudioGenerationError`, `TranscriptionError`, and `RerankError`.
///
/// ```
/// use rig_core::completion::CompletionError;
///
/// /// Log the provider's raw error response when a completion fails.
/// fn report(error: &CompletionError) {
///     if let Some(status) = error.provider_response_status() {
///         // Note: this can be a 2xx status for providers that return an error
///         // envelope alongside a success status — the error itself means failure.
///         eprintln!("provider returned HTTP {status}");
///     }
///     match error.provider_response_json() {
///         Ok(Some(json)) => eprintln!("provider error payload: {json}"),
///         Ok(None) => eprintln!("no provider response body (e.g. a transport error)"),
///         Err(_) => eprintln!(
///             "provider response body was not valid JSON: {:?}",
///             error.provider_response_body(),
///         ),
///     }
/// }
/// ```
#[derive(Debug, Error)]
pub enum CompletionError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Url error (e.g.: invalid URL)
    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[cfg(not(target_family = "wasm"))]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + 'static>),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the completion model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),

    /// Raw error response preserved from the completion model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(CompletionError);

impl CompletionError {
    /// Maps an SSE transport error into a completion error without flattening HTTP failures.
    ///
    /// Non-success HTTP responses remain [`CompletionError::HttpError`] so provider response
    /// helpers can read status and body. Other transport failures keep the existing
    /// [`CompletionError::ProviderError`] display string behavior.
    pub(crate) fn from_stream_transport(error: http_client::Error) -> Self {
        if error.non_success_status().is_some() {
            Self::HttpError(error)
        } else {
            Self::ProviderError(error.to_string())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Document {
    /// Stable document identifier included in the serialized context block.
    pub id: String,
    /// Text content passed to the model as retrieval or static context.
    pub text: String,
    /// Additional string metadata rendered before the document text.
    #[serde(flatten)]
    pub additional_props: HashMap<String, String>,
}

impl std::fmt::Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            concat!("<file id: {}>\n", "{}\n", "</file>\n"),
            self.id,
            if self.additional_props.is_empty() {
                self.text.clone()
            } else {
                let mut sorted_props = self.additional_props.iter().collect::<Vec<_>>();
                sorted_props.sort_by(|a, b| a.0.cmp(b.0));
                let metadata = sorted_props
                    .iter()
                    .map(|(k, v)| format!("{k}: {v:?}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("<metadata {} />\n{}", metadata, self.text)
            }
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool name exposed to the model. It must match the registered tool name.
    pub name: String,
    /// Human-readable description sent to the model.
    pub description: String,
    /// JSON Schema describing tool arguments.
    pub parameters: serde_json::Value,
}

/// Provider-native tool definition.
///
/// Stored under `additional_params.tools` and forwarded by providers that support
/// provider-managed tools.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProviderToolDefinition {
    /// Tool type/kind name as expected by the target provider (for example `web_search`).
    #[serde(rename = "type")]
    pub kind: String,
    /// Additional provider-specific configuration for this hosted tool.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub config: serde_json::Map<String, serde_json::Value>,
}

impl ProviderToolDefinition {
    /// Creates a provider-hosted tool definition by type.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            config: serde_json::Map::new(),
        }
    }

    /// Adds a provider-specific configuration key/value.
    pub fn with_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
}

/// Why the model stopped generating, normalized across providers.
///
/// Providers report this under different names and vocabularies
/// (`finish_reason`, `stop_reason`, `stopReason`, …). Each provider's response
/// conversion maps its wire value onto these variants and preserves anything
/// unmapped verbatim in [`FinishReason::Other`], so a provider adding a new
/// terminal reason never silently reads as a natural stop. Closes #2090/#1886.
///
/// Provider *failure* statuses that arrive with parseable output (a Gemini
/// Interactions `failed`/`cancelled` interaction, a Cohere `ERROR`) follow one
/// policy: the response converts normally and the status is preserved verbatim
/// as [`FinishReason::Other`], leaving the caller to decide whether a
/// failure-flagged-but-parseable turn is usable. Statuses that arrive with no
/// usable output surface as errors instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of the response.
    Stop,
    /// The response hit the output-token limit.
    Length,
    /// The model stopped to call one or more tools.
    ToolCalls,
    /// The provider filtered the content.
    ContentFilter,
    /// A provider-specific reason outside the normalized vocabulary, carried
    /// verbatim in the provider's own wire spelling.
    Other(String),
}

impl FinishReason {
    /// Reconcile a provider's reported reason with what the turn actually
    /// produced.
    ///
    /// Several providers report a plain `stop` on a turn that carried tool
    /// calls (OpenAI-compatible gateways are the usual offenders). A caller
    /// branching on [`FinishReason::ToolCalls`] to decide whether to run tools
    /// would then miss the call entirely, so a natural stop is upgraded
    /// whenever the turn emitted at least one tool call.
    ///
    /// Only [`FinishReason::Stop`] is upgraded: `Length`, `ContentFilter`, and
    /// `Other` describe terminations that remain true regardless of the
    /// content, and overriding them would lose information.
    ///
    /// This is the single place the upgrade happens. Construct normalized
    /// responses through [`CompletionResponse::with_finish_reason`] or
    /// [`CompletionResponse::with_optional_finish_reason`] (and, for streams,
    /// [`crate::streaming::normalize_stream`]) so it is always applied.
    pub fn reconcile_with_output(self, has_tool_call: bool) -> Self {
        if has_tool_call && matches!(self, Self::Stop) {
            Self::ToolCalls
        } else {
            self
        }
    }

    /// Whether the provider cut the turn short instead of letting the model
    /// finish.
    ///
    /// A turn that ended this way can legitimately carry *no content at all* —
    /// an output-token cap consumed entirely by hidden reasoning tokens, or a
    /// filter that removed everything the model produced — and the reason is
    /// then the only diagnostic the caller has. Normalization keeps such an
    /// empty turn rather than rejecting it as a malformed response, so a
    /// caller can tell "you hit the cap" from "the provider misbehaved".
    ///
    /// [`Stop`](Self::Stop) and [`ToolCalls`](Self::ToolCalls) describe turns
    /// that ran to completion, so an empty one really is a provider defect;
    /// [`Other`](Self::Other) is unclassified and gets the strict treatment —
    /// it carries a provider's own wire spelling with no normalized meaning.
    ///
    /// This is also the set rig-agent has a remedy for when a turn arrives
    /// without an answer ("raise `max_tokens`" / "the provider filtered the
    /// response"), and that is the same question: the reasons a provider may
    /// hand back an answerless turn are the reasons there is something useful
    /// to say about it. Both sides read this predicate so they cannot drift.
    pub fn truncated_output(&self) -> bool {
        matches!(self, Self::Length | Self::ContentFilter)
    }
}

/// General completion response struct: the completion choice plus normalized
/// response metadata. The completion choice contains one or more assistant
/// content items.
///
/// This type is concrete — it carries no provider-typed payload. Callers who
/// hold a concrete model and need a provider's own wire response *typed* call
/// that model's inherent `raw_completion` method, which performs the same
/// request and returns the provider's native type. Callers who do not hold the
/// concrete model — an agent erases it at construction — read the same value,
/// serialized, from [`CompletionResponse::raw`], which every provider seam
/// populates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "CompletionResponseRepr")]
pub struct CompletionResponse {
    /// The completion choice (represented by one or more assistant message content)
    /// returned by the completion model provider
    pub choice: Vec<AssistantContent>,
    /// Tokens used during prompting and responding
    pub usage: Usage,
    /// The identifier the provider assigned to the *assistant message* itself,
    /// when it issued one — an OpenAI Responses output-message `msg_` ID or an
    /// Anthropic `msg_` ID. Only IDs the provider would recognize on a replayed
    /// assistant message belong here; identifiers that name the whole response
    /// (an OpenAI chat `chatcmpl-` ID, a Gemini `responseId`) go in
    /// [`CompletionResponse::response_id`] instead.
    ///
    /// The Responses API path uses it to pair reasoning input items with their
    /// output items across turns, and it is what agent history promotes into
    /// [`Message::Assistant`]'s `id`.
    #[serde(default)]
    pub message_id: Option<String>,
    /// The identifier the provider assigned to the response as a whole, when it
    /// reported one — an OpenAI chat `chatcmpl-` ID, a Gemini `responseId`, a
    /// Cohere generation ID. Response-scoped: useful for logging, telemetry
    /// (`gen_ai.response.id`), and support requests, but never replayed to a
    /// provider as a message ID.
    #[serde(default)]
    pub response_id: Option<String>,
    /// The provider's transport-level request identifier, taken from the HTTP
    /// response headers (Anthropic `request-id`, OpenAI/xAI `x-request-id`) or
    /// the provider SDK's response metadata (Bedrock) — the id provider
    /// support asks for when investigating a request. Never the body's
    /// `message.id`/response id; those are [`Self::message_id`] and
    /// [`Self::response_id`]. `None` means the provider did not report one —
    /// that is a documented outcome (e.g. Gemini sends no id header), never an
    /// error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Why the model stopped generating, when the provider reported it.
    ///
    /// Private so that every write flows through
    /// [`CompletionResponse::with_finish_reason`] /
    /// [`CompletionResponse::with_optional_finish_reason`], which apply
    /// [`FinishReason::reconcile_with_output`] — a direct assignment would
    /// silently skip the tool-call upgrade. Read via
    /// [`CompletionResponse::finish_reason`].
    #[serde(default)]
    finish_reason: Option<FinishReason>,
    /// Stable descriptor name of the provider that produced this response, for
    /// example `"openai"`. Always populated, including for responses derived
    /// from a stream that ended before its terminal record.
    pub provider: String,
    /// Provider-reported model identifier for the response.
    ///
    /// This is the model named by the wire response, not the model that was
    /// requested; it is `None` when the provider reports no identifier.
    #[serde(default)]
    pub model: Option<String>,
    /// The provider's own response for this call: the value the model's
    /// inherent `raw_completion` would have returned, serialized. It is the
    /// response as rig's wire type parsed it — fields that type does not model
    /// are not here. Every provider seam populates it, unconditionally — the
    /// same parity the pre-normalization `raw_response: T` had.
    ///
    /// An escape hatch for provider-specific data rig does not normalize — it
    /// never replaces a normalized field, and every normalized field means the
    /// same thing whatever this holds. `Value::Null` means the value was built
    /// without a provider behind it — [`CompletionResponse::new`] without
    /// `with_raw` (test doubles, hand-built responses), or a response
    /// persisted before the field existed — never that the provider sent
    /// nothing: no provider seam produces `Null`.
    ///
    /// Typed access is recoverable: provider raw types are `Deserialize`, so
    /// `provider::CompletionResponse::deserialize(&raw)` returns the
    /// provider's own type, and [`NormalizeCompletionResponse`] converts
    /// forward.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

/// Response identity metadata for one completed model call: which provider
/// objects this exact attempt produced. The three axes stay distinct —
/// message-scoped, response-scoped, and transport — and every field is `None`
/// when the provider did not report it: a documented outcome, never an error.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseIdentity {
    /// Provider-assigned *assistant message* ID (e.g. an Anthropic or OpenAI
    /// Responses `msg_…`) — an ID the provider would recognize on a replayed
    /// assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Provider-assigned *response-scoped* ID (e.g. an OpenAI `chatcmpl-` or
    /// `resp_…` ID) — names the whole response, never replayed as a message
    /// ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The provider's *transport* request id (HTTP response header such as
    /// Anthropic `request-id`, or provider SDK response metadata) — the id
    /// provider support asks for. Never the body's message/response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

impl CompletionResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(choice: Vec<AssistantContent>, usage: Usage, provider: impl Into<String>) -> Self {
        Self {
            choice,
            usage,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            finish_reason: None,
            provider: provider.into(),
            model: None,
            raw: serde_json::Value::Null,
        }
    }

    /// Why the model stopped generating, when the provider reported it.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason.clone()
    }

    /// This response's identity metadata as one [`ResponseIdentity`] carrier.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: self.message_id.clone(),
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }

    /// Attach the normalized finish reason, reconciled against the choice via
    /// [`FinishReason::reconcile_with_output`].
    pub fn with_finish_reason(self, finish_reason: FinishReason) -> Self {
        self.with_optional_finish_reason(Some(finish_reason))
    }

    /// Attach the normalized finish reason when the provider reported one.
    ///
    /// This is the `Option` form of [`CompletionResponse::with_finish_reason`]
    /// and applies the same reconciliation. Provider conversions that hold an
    /// `Option<FinishReason>` use this rather than assigning the field, so the
    /// tool-call upgrade is never skipped.
    pub fn with_optional_finish_reason(mut self, finish_reason: Option<FinishReason>) -> Self {
        let has_tool_call = self
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)));
        self.finish_reason =
            finish_reason.map(|reason| reason.reconcile_with_output(has_tool_call));
        self
    }
}

crate::provider_response::response_metadata_setters!(CompletionResponse);

/// Wire-shape mirror of [`CompletionResponse`], used only for deserialization.
///
/// Serde must never construct an invariant-bearing value structurally: a plain
/// derive would let `"finish_reason":"stop"` skip
/// [`FinishReason::reconcile_with_output`] and `"message_id":""` skip the
/// empty-string filtering. This mirror deserializes the exact wire shape and
/// [`From`] funnels it through [`CompletionResponse::new`] and the `with_*`
/// setters, so every deserialized value satisfies the same invariants as a
/// constructed one. Serialization stays derived on [`CompletionResponse`]
/// itself, so the wire format is unchanged.
#[derive(Deserialize)]
struct CompletionResponseRepr {
    choice: Vec<AssistantContent>,
    usage: Usage,
    #[serde(default)]
    message_id: Option<String>,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    provider_request_id: Option<String>,
    #[serde(default)]
    finish_reason: Option<FinishReason>,
    provider: String,
    #[serde(default)]
    model: Option<String>,
    // `default` because persisted responses predate the field; a missing key
    // loads as `Null`, which is exactly what "no provider response behind this
    // value" means.
    #[serde(default)]
    raw: serde_json::Value,
}

impl From<CompletionResponseRepr> for CompletionResponse {
    fn from(repr: CompletionResponseRepr) -> Self {
        let CompletionResponseRepr {
            choice,
            usage,
            message_id,
            response_id,
            provider_request_id,
            finish_reason,
            provider,
            model,
            raw,
        } = repr;
        Self::new(choice, usage, provider)
            .with_optional_message_id(message_id)
            .with_optional_response_id(response_id)
            .with_optional_provider_request_id(provider_request_id)
            .with_optional_finish_reason(finish_reason)
            .with_optional_model(model)
            .with_raw(raw)
    }
}

/// Convert a provider's own completion payload into the normalized
/// [`CompletionResponse`].
///
/// The provider descriptor name is an *input* rather than something the
/// conversion knows, because several providers share one wire shape — the
/// OpenAI chat-completions payload is used by more than a dozen of them. A
/// conversion that hardcoded a name would mislabel every provider but one, and
/// a placeholder overwritten by the caller would be correct only by convention.
///
/// This is a trait rather than `TryFrom<(&str, T)>` for a concrete reason:
/// a tuple is not a local type, so `impl TryFrom<(&str, TheirResponse)> for
/// CompletionResponse` is rejected by the orphan rule in any crate other than
/// `rig-core`. Implementing this trait on a provider's own response type is
/// allowed anywhere, which keeps provider extensions implementable outside this
/// crate.
pub trait NormalizeCompletionResponse {
    /// Normalize this payload, attributing it to `provider`.
    fn normalize(self, provider: &str) -> Result<CompletionResponse, CompletionError>;
}

/// Struct representing the token usage for a completion request.
/// If tokens used are `0`, then the provider failed to supply token usage metrics.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// The number of input ("prompt") tokens used in a given request.
    pub input_tokens: u64,
    /// The number of output ("completion") tokens used in a given request.
    pub output_tokens: u64,
    /// We store this separately as some providers may only report one number
    pub total_tokens: u64,
    /// The number of input tokens read from a provider-managed cache
    pub cached_input_tokens: u64,
    /// The number of input tokens written to a provider-managed cache
    pub cache_creation_input_tokens: u64,
    /// The number of tool-use prompt tokens used in a given request.
    #[serde(default)]
    pub tool_use_prompt_tokens: u64,
    /// The number of tokens spent on internal reasoning / "thoughts" by reasoning-capable
    /// models (e.g. Gemini thinking, Anthropic extended thinking, OpenAI o-series).
    pub reasoning_tokens: u64,
}

impl Usage {
    /// Creates a new instance of `Usage`.
    pub fn new() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    /// Whether any usage values are set and non-zero.
    ///
    /// Zero-valued usage is this type's documented sentinel for "the provider
    /// supplied no usage metrics", so `false` means usage was not reported.
    pub fn has_values(&self) -> bool {
        *self != Self::new()
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.tool_use_prompt_tokens += other.tool_use_prompt_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

/// Provider behavior that affects how runtimes prepare completion requests.
///
/// Capabilities are immutable facts about a model implementation rather than
/// per-request state, so a runtime can snapshot this value when it erases a
/// concrete model instead of retaining a callback into the provider.
///
/// Prefer building from [`ProviderCapabilities::new`] or [`Default`] and
/// enabling flags with the `with_*` methods: that form keeps external
/// implementations compiling when new capabilities are added, where a struct
/// literal does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderCapabilities {
    /// Whether this provider's native structured output (`output_schema` ->
    /// `format`/`response_format`) composes with tool calls in the same
    /// multi-turn request without suppressing them.
    ///
    /// `false` is the safe assumption: the native constraint may make the model
    /// emit schema JSON instead of calling its tools — see issue #1928.
    /// Providers that enforce structured output *and* tool use together (e.g.
    /// OpenAI, Anthropic) set this to `true`, which lets runtimes keep
    /// guaranteed native structured output active when tools are present.
    pub composes_native_output_with_tools: bool,
}

impl ProviderCapabilities {
    /// Create the conservative capability set used by default.
    pub const fn new() -> Self {
        Self {
            composes_native_output_with_tools: false,
        }
    }

    /// Declare whether native structured output composes with tool calls.
    pub const fn with_native_output_tool_composition(mut self, supported: bool) -> Self {
        self.composes_native_output_with_tools = supported;
        self
    }
}

/// Trait defining a completion model that can be used to generate completion responses.
/// This trait is meant to be implemented by the user to define a custom completion model,
/// either from a third party provider (e.g.: OpenAI) or a local model.
///
/// Implementations return Rig's normalized [`CompletionResponse`] and
/// [`StreamingCompletionResponse`]; a provider's own wire types stay on the
/// provider's side of this boundary, reachable through its inherent
/// `raw_completion`/`raw_stream` methods. Model construction belongs to
/// [`crate::client::completion::CompletionClient`], not to this trait.
///
/// The trait demands only async service behavior — no `Clone` supertrait, in
/// the spirit of `tower::Service`: cloning or sharing a model is the caller's
/// concern (wrap it in an `Arc` if needed). The [`Self::completion_request`]
/// convenience gates on `Self: Clone` individually, which every built-in
/// provider model satisfies.
pub trait CompletionModel: WasmCompatSend + WasmCompatSync {
    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, CompletionError>> + WasmCompatSend;

    /// Streams a completion response for the given completion request.
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<StreamingCompletionResponse, CompletionError>>
    + WasmCompatSend;

    /// Generates a completion request builder for the given `prompt`.
    fn completion_request(&self, prompt: impl Into<Message>) -> CompletionRequestBuilder<Self>
    where
        Self: Sized + Clone,
    {
        CompletionRequestBuilder::new(self.clone(), prompt)
    }

    /// Provider behavior a runtime should account for when preparing requests.
    ///
    /// The default is conservative — see [`ProviderCapabilities`]. Override
    /// this to declare the capabilities a provider actually supports.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }
}

/// A shared model is a model: `Arc<M>` forwards every method to `M`, so the
/// "wrap it in an `Arc` if needed" guidance holds through the generic APIs
/// (`CompletionRequestBuilder`, agent construction), not just at direct call
/// sites. `Arc<M>: Clone` always holds, so [`CompletionModel::completion_request`]
/// clones the `Arc` — never the model.
impl<M: CompletionModel + ?Sized> CompletionModel for std::sync::Arc<M> {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, CompletionError>> + WasmCompatSend
    {
        (**self).completion(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<StreamingCompletionResponse, CompletionError>>
    + WasmCompatSend {
        (**self).stream(request)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        (**self).capabilities()
    }
}

/// Struct representing a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Optional model override for this request.
    pub model: Option<String>,
    /// Legacy preamble field preserved for backwards compatibility.
    ///
    /// New code should prefer a leading [`Message::System`]
    /// in `chat_history` as the canonical representation of system instructions.
    pub preamble: Option<String>,
    /// The chat history to be sent to the completion model provider.
    /// The very last message is the prompt.
    ///
    /// This used to be a non-empty container, so "there is always at least one"
    /// was a type guarantee. It is a `Vec` now and the field is public, so the
    /// guarantee is a *rule* instead: it is checked by
    /// [`CompletionRequest::validate_message_content`] at the request boundary.
    pub chat_history: Vec<Message>,
    /// The documents to be sent to the completion model provider
    pub documents: Vec<Document>,
    /// The tools to be sent to the completion model provider
    pub tools: Vec<ToolDefinition>,
    /// The temperature to be sent to the completion model provider
    pub temperature: Option<f64>,
    /// The max tokens to be sent to the completion model provider
    pub max_tokens: Option<u64>,
    /// Whether tools are required to be used by the model provider or not before providing a response.
    pub tool_choice: Option<ToolChoice>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    pub additional_params: Option<serde_json::Value>,
    /// Optional JSON Schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub output_schema: Option<schemars::Schema>,
    /// Whether to record sensitive request, response, and tool content on GenAI
    /// telemetry spans.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved context,
    /// tool results, model responses, and other sensitive or high-cardinality data
    /// through OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs. Only enable it when the caller has
    /// explicitly opted in to content telemetry.
    ///
    /// Higher-level agent drivers use this flag for portable input, output, and
    /// tool-content telemetry. Direct provider calls only forward the policy;
    /// the exact content fields available there are provider- and
    /// surface-dependent, especially for streaming responses that are consumed
    /// after the provider returns.
    ///
    /// This is local observability policy and is never serialized into provider
    /// request payloads.
    #[serde(skip)]
    pub record_telemetry_content: bool,
}

impl CompletionRequest {
    /// Reject a request with no messages, or a message that carries no content.
    ///
    /// Removing the non-empty container removed two guarantees at once, and this
    /// is where both are restated:
    ///
    /// - `chat_history` was non-empty by construction. As a `Vec` it is not, and
    ///   the field is public, so `CompletionRequest { chat_history: vec![], .. }`
    ///   is constructible and would reach a provider as `messages: []` — a remote
    ///   400 in place of a local error that names the problem.
    /// - Message content was likewise non-empty by construction, and every wire
    ///   rejects an empty content block.
    ///
    /// The rule also covers the block list *inside* a tool result. A user
    /// message carrying one `UserContent::ToolResult` is itself non-empty, but
    /// `ToolResult::content` was non-empty by construction under the removed
    /// container and is request-direction data just like the message content
    /// around it — so its check is relocated here rather than dropped. Only a
    /// tool result with *zero* blocks is rejected; a tool that legitimately
    /// returned an empty string produces one block and still sends.
    ///
    /// This is the **request** direction only, and the asymmetry is deliberate.
    /// Empty *assistant* content is a real provider outcome on the response path
    /// — a tool-call-only turn, a content-filtered turn, a truncated stream — and
    /// the agent layer drops such a turn rather than sending it, so it never
    /// reaches here. The response direction is guarded per-wire instead, by
    /// [`crate::message::require_non_empty`], because "this provider returned
    /// nothing where its protocol promises content" is a judgement only the
    /// provider's own conversion can make.
    ///
    /// `System` content is deliberately not checked. It is a `String` and always
    /// has been, so the removed container never constrained it; rejecting an
    /// empty one would be a new restriction rather than a relocated enforcement
    /// point, and would break a history carrying a conditionally built preamble
    /// that resolved to `""`.
    ///
    /// **Where this runs.** [`CompletionRequestBuilder::send`] and
    /// [`CompletionRequestBuilder::stream`] call it, which covers both agent
    /// surfaces too — the blocking and streaming turn drivers both issue their
    /// request through the builder. Handing a request straight to a
    /// [`CompletionModel`] bypasses it; call this yourself there.
    pub fn validate_message_content(&self) -> Result<(), CompletionError> {
        if self.chat_history.is_empty() {
            return Err(CompletionError::RequestError(
                "request has an empty chat history; providers require at least one message"
                    .to_owned()
                    .into(),
            ));
        }

        let empty_message = |role: &str, index: usize| {
            CompletionError::RequestError(
                format!(
                    "{role} message at index {index} has no content; \
                     providers reject empty content blocks"
                )
                .into(),
            )
        };

        // One match per message, with every per-variant rule in that
        // variant's arm, so extending validation means extending one arm.
        for (index, message) in self.chat_history.iter().enumerate() {
            match message {
                Message::System { .. } => {}
                Message::Assistant { content, .. } => {
                    if content.is_empty() {
                        return Err(empty_message("assistant", index));
                    }
                }
                Message::User { content } => {
                    if content.is_empty() {
                        return Err(empty_message("user", index));
                    }

                    for (position, item) in content.iter().enumerate() {
                        // Exhaustive on purpose: a future variant that carries
                        // its own request-direction block list must decide here
                        // whether its emptiness is checked, instead of slipping
                        // past a wildcard un-validated.
                        match item {
                            UserContent::ToolResult(result) if result.content.is_empty() => {
                                let name = &result.name;
                                return Err(CompletionError::RequestError(
                                    format!(
                                        "tool result for `{name}` at index {position} of the \
                                         user message at index {index} has no content; \
                                         providers reject empty content blocks"
                                    )
                                    .into(),
                                ));
                            }
                            UserContent::ToolResult(_)
                            | UserContent::Text(_)
                            | UserContent::Image(_)
                            | UserContent::Audio(_)
                            | UserContent::Video(_)
                            | UserContent::Document(_) => {}
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extracts a name from the output schema's `"title"` field, falling back to `"response_schema"`.
    /// Useful for providers that require a name alongside the JSON Schema (e.g., OpenAI).
    pub fn output_schema_name(&self) -> Option<String> {
        self.output_schema.as_ref().map(|schema| {
            schema
                .as_object()
                .and_then(|o| o.get("title"))
                .and_then(|v| v.as_str())
                .unwrap_or("response_schema")
                .to_string()
        })
    }

    /// Returns documents normalized into a message (if any).
    /// Most providers do not accept documents directly as input, so it needs to convert into a
    /// `Message` so that it can be incorporated into `chat_history`.
    pub fn normalized_documents(&self) -> Option<Message> {
        Self::normalized_documents_from(&self.documents)
    }

    fn normalized_documents_from(documents: &[Document]) -> Option<Message> {
        if documents.is_empty() {
            return None;
        }

        // Most providers will convert documents into a text unless it can handle document messages.
        // We use `UserContent::document` for those who handle it directly!
        let messages = documents
            .iter()
            .map(|doc| {
                UserContent::document(
                    doc.to_string(),
                    // In the future, we can customize `Document` to pass these extra types through.
                    // Most providers ditch these but they might want to use them.
                    Some(DocumentMediaType::TXT),
                )
            })
            .collect::<Vec<_>>();

        crate::message::non_empty(messages).map(|content| Message::User { content })
    }

    pub(crate) fn chat_history_with_documents(&self) -> Vec<Message> {
        let mut chat_history = self.chat_history.clone();
        if let Some(documents) = self.normalized_documents() {
            insert_after_leading_system(&mut chat_history, documents);
        }
        chat_history
    }
}

/// Insert `message` at the first non-system position so document context lands
/// after any leading system messages; telemetry and the sent request must
/// agree on this placement.
fn insert_after_leading_system(chat_history: &mut Vec<Message>, message: Message) {
    let insert_at = chat_history
        .iter()
        .position(|message| !matches!(message, Message::System { .. }))
        .unwrap_or(chat_history.len());
    chat_history.insert(insert_at, message);
}

fn merge_provider_tools_into_additional_params(
    additional_params: Option<serde_json::Value>,
    provider_tools: Vec<ProviderToolDefinition>,
) -> Option<serde_json::Value> {
    if provider_tools.is_empty() {
        return additional_params;
    }

    let mut provider_tools_json = provider_tools
        .into_iter()
        .map(|ProviderToolDefinition { kind, mut config }| {
            // Force the provider tool type from the strongly-typed field.
            config.insert("type".to_string(), serde_json::Value::String(kind));
            serde_json::Value::Object(config)
        })
        .collect::<Vec<_>>();

    let mut params_map = match additional_params {
        Some(serde_json::Value::Object(map)) => map,
        Some(serde_json::Value::Bool(stream)) => {
            let mut map = serde_json::Map::new();
            map.insert("stream".to_string(), serde_json::Value::Bool(stream));
            map
        }
        _ => serde_json::Map::new(),
    };

    let mut merged_tools = match params_map.remove("tools") {
        Some(serde_json::Value::Array(existing)) => existing,
        _ => Vec::new(),
    };
    merged_tools.append(&mut provider_tools_json);
    params_map.insert("tools".to_string(), serde_json::Value::Array(merged_tools));
    Some(serde_json::Value::Object(params_map))
}

/// Builder struct for constructing a completion request.
///
/// Example usage:
/// ```no_run
/// use rig_core::{
///     client::CompletionClient,
///     providers::openai::{Client, self},
///     completion::{CompletionModel, CompletionRequestBuilder},
/// };
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = Client::new("your-openai-api-key")?;
/// let model = openai.completion_model(openai::GPT_5_2);
///
/// // Create the completion request and execute it separately
/// let request = CompletionRequestBuilder::new(model.clone(), "Who are you?".to_string())
///     .preamble("You are Marvin from the Hitchhiker's Guide to the Galaxy.".to_string())
///     .temperature(0.5)
///     .build();
///
/// let response = model.completion(request).await?;
/// # Ok(())
/// # }
/// ```
///
/// Alternatively, you can execute the completion request directly from the builder:
/// ```no_run
/// use rig_core::{
///     client::CompletionClient,
///     providers::openai::{Client, self},
///     completion::CompletionRequestBuilder,
/// };
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = Client::new("your-openai-api-key")?;
/// let model = openai.completion_model(openai::GPT_5_2);
///
/// // Create the completion request and execute it directly
/// let response = CompletionRequestBuilder::new(model, "Who are you?".to_string())
///     .preamble("You are Marvin from the Hitchhiker's Guide to the Galaxy.".to_string())
///     .temperature(0.5)
///     .send()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// Note: It is usually unnecessary to create a completion request builder directly.
/// Instead, use the [CompletionModel::completion_request] method.
pub struct CompletionRequestBuilder<M: CompletionModel> {
    model: M,
    prompt: Message,
    request_model: Option<String>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    provider_tools: Vec<ProviderToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    tool_choice: Option<ToolChoice>,
    additional_params: Option<serde_json::Value>,
    output_schema: Option<schemars::Schema>,
    record_telemetry_content: bool,
}

impl<M: CompletionModel> CompletionRequestBuilder<M> {
    pub fn new(model: M, prompt: impl Into<Message>) -> Self {
        Self {
            model,
            prompt: prompt.into(),
            request_model: None,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            provider_tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    /// Sets the preamble for the completion request.
    pub fn preamble(mut self, preamble: String) -> Self {
        // Legacy public API: funnel preamble into canonical system messages at build-time.
        self.preamble = Some(preamble);
        self
    }

    /// Overrides the model used for this request.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.request_model = Some(model.into());
        self
    }

    /// Overrides the model used for this request.
    pub fn model_opt(mut self, model: Option<String>) -> Self {
        self.request_model = model;
        self
    }

    pub fn without_preamble(mut self) -> Self {
        self.preamble = None;
        self
    }

    /// Adds a message to the chat history for the completion request.
    pub fn message(mut self, message: Message) -> Self {
        self.chat_history.push(message);

        self
    }

    /// Adds a list of messages to the chat history for the completion request.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.chat_history.extend(messages);

        self
    }

    /// Adds a document to the completion request.
    pub fn document(mut self, document: Document) -> Self {
        self.documents.push(document);
        self
    }

    /// Adds a list of documents to the completion request.
    pub fn documents(self, documents: impl IntoIterator<Item = Document>) -> Self {
        documents
            .into_iter()
            .fold(self, |builder, doc| builder.document(doc))
    }

    /// Adds a tool to the completion request.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds a list of tools to the completion request.
    pub fn tools(self, tools: Vec<ToolDefinition>) -> Self {
        tools
            .into_iter()
            .fold(self, |builder, tool| builder.tool(tool))
    }

    /// Adds a provider-hosted tool to the completion request.
    pub fn provider_tool(mut self, tool: ProviderToolDefinition) -> Self {
        self.provider_tools.push(tool);
        self
    }

    /// Adds provider-hosted tools to the completion request.
    pub fn provider_tools(self, tools: Vec<ProviderToolDefinition>) -> Self {
        tools
            .into_iter()
            .fold(self, |builder, tool| builder.provider_tool(tool))
    }

    /// Adds additional parameters to the completion request.
    /// This can be used to set additional provider-specific parameters. For example,
    /// Cohere's completion models accept a `connectors` parameter that can be used to
    /// specify the data connectors used by Cohere when executing the completion
    /// (see `examples/cohere_connectors.rs`).
    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        match self.additional_params {
            Some(params) => {
                self.additional_params = Some(json_utils::merge(params, additional_params));
            }
            None => {
                self.additional_params = Some(additional_params);
            }
        }
        self
    }

    /// Sets the additional parameters for the completion request.
    /// This can be used to set additional provider-specific parameters. For example,
    /// Cohere's completion models accept a `connectors` parameter that can be used to
    /// specify the data connectors used by Cohere when executing the completion
    /// (see `examples/cohere_connectors.rs`).
    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }

    /// Sets the temperature for the completion request.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the temperature for the completion request.
    pub fn temperature_opt(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the max tokens for the completion request.
    /// Note: This is required if using Anthropic
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the max tokens for the completion request.
    /// Note: This is required if using Anthropic
    pub fn max_tokens_opt(mut self, max_tokens: Option<u64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the thing.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets the output schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    /// NOTE: For direct type conversion, you may want to use `Agent::prompt_typed()` - using this method
    /// with `Agent::prompt()` will still output a String at the end, it'll just be compatible with whatever
    /// type you want to use here. This method is primarily an escape hatch for agents being used as tools
    /// to still be able to leverage structured outputs.
    pub fn output_schema(mut self, schema: schemars::Schema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Sets the output schema for structured output from an optional value.
    /// NOTE: For direct type conversion, you may want to use `Agent::prompt_typed()` - using this method
    /// with `Agent::prompt()` will still output a String at the end, it'll just be compatible with whatever
    /// type you want to use here. This method is primarily an escape hatch for agents being used as tools
    /// to still be able to leverage structured outputs.
    pub fn output_schema_opt(mut self, schema: Option<schemars::Schema>) -> Self {
        self.output_schema = schema;
        self
    }

    /// Opt in or out of recording sensitive request, response, and tool content
    /// on GenAI telemetry spans for this request.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved context,
    /// tool results, model responses, and other sensitive or high-cardinality data
    /// through OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs. Only enable it when content telemetry is
    /// acceptable for this request. Structural metadata and token
    /// usage remain available when this is disabled.
    ///
    /// This low-level builder only stores the opt-in on the built request. It
    /// does not guarantee portable input/output message fields for direct model
    /// calls; exact coverage is provider- and surface-dependent. Agent APIs own
    /// normalized input/output recording and provide the consistent surface.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.record_telemetry_content = enabled;
        self
    }

    /// Returns the normalized input messages used by runtime telemetry.
    pub fn messages_for_telemetry(&self) -> Vec<Message> {
        let mut chat_history = self.chat_history.clone();
        if let Some(preamble) = &self.preamble {
            chat_history.insert(0, Message::system(preamble.clone()));
        }
        chat_history.push(self.prompt.clone());

        if let Some(documents) = CompletionRequest::normalized_documents_from(&self.documents) {
            insert_after_leading_system(&mut chat_history, documents);
        }

        chat_history
    }

    /// Builds the completion request.
    pub fn build(self) -> CompletionRequest {
        self.into_model_and_request().1
    }

    /// Moves the model out and builds the request from the remaining fields.
    ///
    /// `build`, `send`, and `stream` all funnel through this single
    /// destructuring, so the built request cannot drift between them and the
    /// terminal methods need no model clone.
    fn into_model_and_request(self) -> (M, CompletionRequest) {
        let model = self.model;
        // Build the final message list, prepending preamble if present
        let mut chat_history = self.chat_history;
        let prompt = self.prompt;
        if let Some(preamble) = self.preamble {
            chat_history.insert(0, Message::system(preamble));
        }

        // The push is what makes the history non-empty, so the fallback that
        // used to follow could never be taken — and it forced a clone of the
        // prompt to feed it.
        chat_history.push(prompt);
        let additional_params = merge_provider_tools_into_additional_params(
            self.additional_params,
            self.provider_tools,
        );

        let request = CompletionRequest {
            model: self.request_model,
            preamble: None,
            chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            tool_choice: self.tool_choice,
            additional_params,
            output_schema: self.output_schema,
            record_telemetry_content: self.record_telemetry_content,
        };
        (model, request)
    }

    /// Sends the completion request to the completion model provider and returns the completion response.
    pub async fn send(self) -> Result<CompletionResponse, CompletionError> {
        let (model, request) = self.into_model_and_request();
        request.validate_message_content()?;
        model.completion(request).await
    }

    /// Stream the completion request
    pub async fn stream(self) -> Result<StreamingCompletionResponse, CompletionError> {
        let (model, request) = self.into_model_and_request();
        request.validate_message_content()?;
        model.stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionResponse, FinishReason, ProviderCapabilities, Usage};
    use crate::message::AssistantContent;

    mod message_content_validation {
        use super::super::CompletionRequest;
        use crate::message::{AssistantContent, Message, UserContent};

        fn request(chat_history: Vec<Message>) -> CompletionRequest {
            CompletionRequest {
                model: None,
                preamble: None,
                chat_history,
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            }
        }

        #[test]
        fn a_populated_history_passes() {
            let request = request(vec![Message::user("hello")]);
            assert!(request.validate_message_content().is_ok());
        }

        #[test]
        fn an_empty_history_is_rejected() {
            // `chat_history` was non-empty by construction until the container
            // was removed; the field is public, so this is now constructible.
            let error = request(Vec::new())
                .validate_message_content()
                .expect_err("an empty history must not reach a provider");
            assert!(
                error.to_string().contains("empty chat history"),
                "unexpected error: {error}"
            );
        }

        #[test]
        fn an_empty_user_message_is_rejected_by_index_and_role() {
            let error = request(vec![
                Message::user("hello"),
                Message::User {
                    content: Vec::new(),
                },
            ])
            .validate_message_content()
            .expect_err("an empty user message must not reach a provider");
            let message = error.to_string();
            assert!(message.contains("user message at index 1"), "{message}");
        }

        #[test]
        fn an_empty_assistant_message_is_rejected_by_index_and_role() {
            let error = request(vec![Message::Assistant {
                id: None,
                content: Vec::new(),
            }])
            .validate_message_content()
            .expect_err("an empty assistant message must not reach a provider");
            let message = error.to_string();
            assert!(
                message.contains("assistant message at index 0"),
                "{message}"
            );
        }

        #[test]
        fn an_empty_system_message_is_not_rejected() {
            // System content is a `String` and always has been, so the removed
            // container never constrained it. Rejecting an empty one would be a
            // new restriction, and would break a history carrying a
            // conditionally built preamble that resolved to `""`.
            let request = request(vec![
                Message::System {
                    content: String::new(),
                },
                Message::user("hello"),
            ]);
            assert!(request.validate_message_content().is_ok());
        }

        #[test]
        fn a_block_less_tool_result_is_rejected_naming_the_tool() {
            use crate::message::{ToolCallId, ToolResult, ToolResultContent};
            // `ToolResult::content` was non-empty by construction until the
            // container was removed; the message around it has one item, so the
            // message-level check alone would let this reach the wire as
            // `"content": []`.
            let error = request(vec![
                Message::user("hello"),
                Message::User {
                    content: vec![UserContent::ToolResult(ToolResult {
                        call: ToolCallId::new_or_mint("call_1"),
                        provider: None,
                        name: "lookup".to_owned(),
                        content: Vec::<ToolResultContent>::new(),
                    })],
                },
            ])
            .validate_message_content()
            .expect_err("a block-less tool result must not reach a provider");
            let message = error.to_string();
            assert!(message.contains("`lookup`"), "{message}");
            assert!(message.contains("index 0"), "{message}");
            assert!(message.contains("user message at index 1"), "{message}");
        }

        #[test]
        fn a_tool_result_with_one_empty_string_block_is_accepted() {
            use crate::message::{ToolCallId, ToolResult, ToolResultContent};
            // The guard is on the cardinality of the block list, not on the
            // blocks' content. A tool that legitimately returned an empty
            // string produces one block and must still send — this pins the
            // "no blocks" / "no content" distinction so a future tightening
            // pass cannot collapse it.
            let request = request(vec![Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    call: ToolCallId::new_or_mint("call_1"),
                    provider: None,
                    name: "lookup".to_owned(),
                    content: vec![ToolResultContent::text("")],
                })],
            }]);
            assert!(request.validate_message_content().is_ok());
        }

        #[test]
        fn the_legacy_fabricated_sentinel_still_passes() {
            // Histories persisted before message content became a `Vec` encode a
            // content-less assistant turn as a single empty text part. That is a
            // one-element list, so it validates — the rule is block count, not
            // block content — and, being caller-supplied history, it is never
            // filtered: it goes to the wire as-is (where some providers reject
            // it). Callers migrating pre-`Vec` histories drop such turns
            // themselves; see MIGRATING.
            let request = request(vec![
                Message::user("hello"),
                Message::Assistant {
                    id: None,
                    content: vec![AssistantContent::text("")],
                },
                Message::User {
                    content: vec![UserContent::text("and again")],
                },
            ]);
            assert!(request.validate_message_content().is_ok());
        }
    }

    fn tool_call_choice() -> Vec<AssistantContent> {
        vec![AssistantContent::tool_call(
            "call_1",
            "lookup",
            serde_json::json!({"query": "rig"}),
        )]
    }

    #[test]
    fn normalized_response_round_trips_through_serde() {
        let response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage {
                input_tokens: 3,
                output_tokens: 2,
                total_tokens: 5,
                cached_input_tokens: 1,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 1,
            },
            "example",
        )
        .with_message_id("msg_123")
        .with_finish_reason(FinishReason::Stop)
        .with_model("provider-model-v2");

        let encoded = serde_json::to_value(&response).expect("serialize response");
        let decoded =
            serde_json::from_value::<CompletionResponse>(encoded.clone()).expect("deserialize");

        assert_eq!(
            serde_json::to_value(decoded).expect("re-serialize"),
            encoded
        );
    }

    /// Serde must not be a back door around `reconcile_with_output`: a
    /// persisted `"stop"` next to a tool-call choice deserializes as
    /// `ToolCalls`, exactly as if it had gone through the setter.
    #[test]
    fn deserializing_stop_with_a_tool_call_reconciles_to_tool_calls() {
        let mut encoded = serde_json::to_value(CompletionResponse::new(
            tool_call_choice(),
            Usage::new(),
            "example",
        ))
        .expect("serialize response");
        encoded["finish_reason"] = serde_json::json!("stop");

        let decoded =
            serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

        assert_eq!(decoded.finish_reason(), Some(FinishReason::ToolCalls));
    }

    /// Serde must not be a back door around the empty-string filtering either:
    /// a persisted `""` identifier deserializes as `None`.
    #[test]
    fn deserializing_empty_identifiers_yields_none() {
        let mut encoded = serde_json::to_value(CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage::new(),
            "example",
        ))
        .expect("serialize response");
        encoded["message_id"] = serde_json::json!("");
        encoded["response_id"] = serde_json::json!("");
        encoded["model"] = serde_json::json!("");

        let decoded =
            serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

        assert_eq!(decoded.message_id, None);
        assert_eq!(decoded.response_id, None);
        assert_eq!(decoded.model, None);
    }

    #[test]
    fn unknown_finish_reason_survives_a_serde_round_trip_verbatim() {
        let reason = FinishReason::Other("provider_specific_stop".to_owned());
        let encoded = serde_json::to_string(&reason).expect("serialize");
        let decoded = serde_json::from_str::<FinishReason>(&encoded).expect("deserialize");

        assert_eq!(decoded, reason);
    }

    #[test]
    fn stop_with_a_tool_call_reconciles_to_tool_calls() {
        let response = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
            .with_finish_reason(FinishReason::Stop);

        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    }

    /// The `Option` setter is what provider conversions actually reach for, so
    /// it must reconcile identically — a provider holding an `Option` must not
    /// have to choose between ergonomics and correctness.
    #[test]
    fn optional_setter_reconciles_exactly_like_the_plain_setter() {
        let via_option = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
            .with_optional_finish_reason(Some(FinishReason::Stop));
        let via_plain = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
            .with_finish_reason(FinishReason::Stop);

        assert_eq!(via_option.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(via_option.finish_reason, via_plain.finish_reason);
    }

    #[test]
    fn reconciliation_only_upgrades_a_natural_stop() {
        // A truncated tool call is still a truncation; a filtered one is still
        // filtered. Overriding either would lose why the turn actually ended.
        for reason in [
            FinishReason::Length,
            FinishReason::ContentFilter,
            FinishReason::Other("provider_specific".to_owned()),
        ] {
            let response = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
                .with_finish_reason(reason.clone());

            assert_eq!(response.finish_reason, Some(reason));
        }
    }

    #[test]
    fn reconciliation_leaves_a_stop_without_tool_calls_alone() {
        let response = CompletionResponse::new(
            vec![AssistantContent::text("done")],
            Usage::new(),
            "example",
        )
        .with_finish_reason(FinishReason::Stop);

        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn provider_capabilities_are_externally_configurable_from_default() {
        let capabilities =
            ProviderCapabilities::default().with_native_output_tool_composition(true);

        assert!(capabilities.composes_native_output_with_tools);
        assert!(!ProviderCapabilities::new().composes_native_output_with_tools);
        assert_eq!(ProviderCapabilities::new(), ProviderCapabilities::default());
    }

    #[test]
    fn usage_has_values_reflects_the_zero_sentinel() {
        use super::Usage;

        assert!(!Usage::new().has_values());

        let mut usage = Usage::new();
        usage.reasoning_tokens = 1;
        assert!(usage.has_values());
    }

    use super::*;
    use crate::test_utils::MockCompletionModel;

    #[test]
    fn completion_request_content_telemetry_is_opt_in_and_not_serialized() {
        let default_request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "completion prompt")
                .build();
        assert!(!default_request.record_telemetry_content);

        let default_json = serde_json::to_value(&default_request).expect("serialize request");
        assert!(
            default_json.get("record_telemetry_content").is_none(),
            "safe default should not serialize the telemetry opt-in field"
        );
        let default_roundtrip: CompletionRequest =
            serde_json::from_value(default_json).expect("deserialize default request");
        assert!(!default_roundtrip.record_telemetry_content);

        let opt_in_request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "completion prompt")
                .record_content_telemetry(true)
                .build();
        assert!(opt_in_request.record_telemetry_content);

        let opt_in_json = serde_json::to_value(&opt_in_request).expect("serialize opt-in request");
        assert!(
            opt_in_json.get("record_telemetry_content").is_none(),
            "local telemetry policy must not be serialized into provider requests"
        );
        let legacy_roundtrip: CompletionRequest =
            serde_json::from_value(opt_in_json).expect("deserialize legacy request");
        assert!(
            !legacy_roundtrip.record_telemetry_content,
            "missing field should deserialize to the safe default"
        );
    }

    /// The deserialization mirror carries `raw`: a response with a captured
    /// payload survives serialize → deserialize with the payload intact, a
    /// response serialized before the field existed still loads with `raw`
    /// unset, and an unset `raw` is not written.
    #[test]
    fn normalized_response_raw_round_trips_through_serde_mirror() {
        let payload = serde_json::json!({
            "id": "chatcmpl-1",
            "system_fingerprint": "fp_abc",
            "choices": [{"finish_reason": "stop"}]
        });
        let response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage::new(),
            "example",
        )
        .with_response_id("chatcmpl-1")
        .with_raw(payload.clone());

        let encoded = serde_json::to_value(&response).expect("serialize response");
        assert_eq!(encoded["raw"], payload);
        let decoded: CompletionResponse =
            serde_json::from_value(encoded.clone()).expect("deserialize response");
        assert_eq!(decoded.raw, payload);
        assert_eq!(decoded.response_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(
            serde_json::to_value(&decoded).expect("re-serialize"),
            encoded
        );

        let legacy = serde_json::json!({
            "choice": [{"type": "text", "text": "hello"}],
            "usage": serde_json::to_value(Usage::new()).unwrap(),
            "provider": "example"
        });
        let decoded: CompletionResponse = serde_json::from_value(legacy).expect("legacy loads");
        assert!(decoded.raw.is_null());

        let bare = serde_json::to_value(CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage::new(),
            "example",
        ))
        .unwrap();
        assert!(bare.get("raw").is_none());
    }

    fn test_document(id: &str, text: &str) -> Document {
        Document {
            id: id.to_string(),
            text: text.to_string(),
            additional_props: HashMap::new(),
        }
    }

    #[test]
    fn message_telemetry_includes_normalized_documents() {
        let builder = CompletionRequestBuilder::new(MockCompletionModel::default(), "prompt")
            .preamble("system".to_string())
            .message(Message::user("history"))
            .document(test_document("doc1", "static context secret"));

        let messages = builder.messages_for_telemetry();
        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[0], Message::System { .. }));
        assert!(is_document_message(&messages[1], "doc1"));
        assert!(matches!(
            &messages[2],
            Message::User { content }
                if matches!(content.first(), Some(UserContent::Text(text)) if text.text == "history")
        ));
        assert!(matches!(
            &messages[3],
            Message::User { content }
                if matches!(content.first(), Some(UserContent::Text(text)) if text.text == "prompt")
        ));

        let request = builder.build();
        assert_eq!(messages, request.chat_history_with_documents());
    }

    fn is_document_message(message: &Message, expected_id: &str) -> bool {
        let Message::User { content } = message else {
            return false;
        };

        content.iter().any(|content| {
            matches!(
                content,
                UserContent::Document(document)
                    if document.data.to_string().contains(&format!("<file id: {expected_id}>"))
            )
        })
    }

    #[test]
    fn test_document_display_without_metadata() {
        let doc = Document {
            id: "123".to_string(),
            text: "This is a test document.".to_string(),
            additional_props: HashMap::new(),
        };

        let expected = "<file id: 123>\nThis is a test document.\n</file>\n";
        assert_eq!(format!("{doc}"), expected);
    }

    #[test]
    fn test_document_display_with_metadata() {
        let mut additional_props = HashMap::new();
        additional_props.insert("author".to_string(), "John Doe".to_string());
        additional_props.insert("length".to_string(), "42".to_string());

        let doc = Document {
            id: "123".to_string(),
            text: "This is a test document.".to_string(),
            additional_props,
        };

        let expected = concat!(
            "<file id: 123>\n",
            "<metadata author: \"John Doe\" length: \"42\" />\n",
            "This is a test document.\n",
            "</file>\n"
        );
        assert_eq!(format!("{doc}"), expected);
    }

    #[test]
    fn test_normalize_documents_with_documents() {
        let doc1 = Document {
            id: "doc1".to_string(),
            text: "Document 1 text.".to_string(),
            additional_props: HashMap::new(),
        };

        let doc2 = Document {
            id: "doc2".to_string(),
            text: "Document 2 text.".to_string(),
            additional_props: HashMap::new(),
        };

        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec!["What is the capital of France?".into()],
            documents: vec![doc1, doc2],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let expected = Message::User {
            content: vec![
                UserContent::document(
                    "<file id: doc1>\nDocument 1 text.\n</file>\n".to_string(),
                    Some(DocumentMediaType::TXT),
                ),
                UserContent::document(
                    "<file id: doc2>\nDocument 2 text.\n</file>\n".to_string(),
                    Some(DocumentMediaType::TXT),
                ),
            ],
        };

        assert_eq!(request.normalized_documents(), Some(expected));
    }

    #[test]
    fn test_normalize_documents_without_documents() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec!["What is the capital of France?".into()],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        assert_eq!(request.normalized_documents(), None);
    }

    #[test]
    fn preamble_builder_funnels_to_system_message() {
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
                .preamble("System prompt".to_string())
                .message(Message::user("History"))
                .build();

        assert_eq!(request.preamble, None);

        let history = request.chat_history.into_iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 3);
        assert!(matches!(
            &history[0],
            Message::System { content } if content == "System prompt"
        ));
        assert!(matches!(&history[1], Message::User { .. }));
        assert!(matches!(&history[2], Message::User { .. }));
    }

    #[test]
    fn without_preamble_removes_legacy_preamble_injection() {
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
                .preamble("System prompt".to_string())
                .without_preamble()
                .build();

        assert_eq!(request.preamble, None);
        let history = request.chat_history.into_iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 1);
        assert!(matches!(&history[0], Message::User { .. }));
    }

    #[test]
    fn build_places_documents_after_preamble_system_message() {
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
                .preamble("System prompt".to_string())
                .document(test_document("doc1", "Document text."))
                .build();

        assert_eq!(request.documents.len(), 1);

        let history = request.chat_history_with_documents();
        let history = history.iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 3);
        assert!(matches!(
            history[0],
            Message::System { content } if content == "System prompt"
        ));
        assert!(is_document_message(history[1], "doc1"));
        assert!(matches!(history[2], Message::User { .. }));
    }

    #[test]
    fn build_places_documents_after_leading_system_messages_before_prior_history() {
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
                .message(Message::system("System one"))
                .message(Message::system("System two"))
                .message(Message::user("Earlier user turn"))
                .message(Message::assistant("Earlier assistant turn"))
                .document(test_document("doc1", "Document text."))
                .build();

        let history = request.chat_history_with_documents();
        let history = history.iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 6);
        assert!(matches!(
            history[0],
            Message::System { content } if content == "System one"
        ));
        assert!(matches!(
            history[1],
            Message::System { content } if content == "System two"
        ));
        assert!(is_document_message(history[2], "doc1"));
        assert!(matches!(history[3], Message::User { .. }));
        assert!(matches!(history[4], Message::Assistant { .. }));
        assert!(matches!(history[5], Message::User { .. }));
    }

    #[test]
    fn build_without_documents_keeps_message_order_unchanged() {
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
                .message(Message::system("System prompt"))
                .message(Message::user("Earlier user turn"))
                .build();

        let history = request.chat_history.iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 3);
        assert!(matches!(
            history[0],
            Message::System { content } if content == "System prompt"
        ));
        assert!(matches!(history[1], Message::User { .. }));
        assert!(matches!(history[2], Message::User { .. }));
    }

    #[test]
    fn chat_history_with_documents_places_documents_after_leading_system_messages() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![
                Message::system("System prompt"),
                Message::assistant("Earlier assistant turn"),
                Message::user("Earlier user turn"),
                Message::user("Prompt"),
            ],
            documents: vec![test_document("doc1", "Document text.")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        assert_eq!(request.documents.len(), 1);

        let history = request.chat_history_with_documents();
        let history = history.iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 5);
        assert!(matches!(history[0], Message::System { .. }));
        assert!(is_document_message(history[1], "doc1"));
        assert!(matches!(history[2], Message::Assistant { .. }));
        assert!(matches!(history[3], Message::User { .. }));
        assert!(matches!(history[4], Message::User { .. }));
    }

    #[test]
    fn chat_history_with_documents_places_documents_before_mid_conversation_system_messages() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![
                Message::system("Leading system prompt"),
                Message::assistant("Earlier assistant turn"),
                Message::system("Mid-conversation instruction"),
                Message::user("Prompt"),
            ],
            documents: vec![test_document("doc1", "Document text.")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let history = request.chat_history_with_documents();
        let history = history.iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 5);
        assert!(matches!(
            history[0],
            Message::System { content } if content == "Leading system prompt"
        ));
        assert!(is_document_message(history[1], "doc1"));
        assert!(matches!(history[2], Message::Assistant { .. }));
        assert!(matches!(
            history[3],
            Message::System { content } if content == "Mid-conversation instruction"
        ));
        assert!(matches!(history[4], Message::User { .. }));
    }

    #[test]
    fn chat_history_with_documents_does_not_duplicate_documents() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![
                Message::system("System prompt"),
                Message::user("Earlier user turn"),
                Message::assistant("Earlier assistant turn"),
                Message::user("Prompt"),
            ],
            documents: vec![test_document("doc1", "Document text.")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let history = request.chat_history_with_documents();
        let document_messages = history
            .iter()
            .filter(|message| is_document_message(message, "doc1"))
            .count();
        assert_eq!(document_messages, 1);
    }

    #[test]
    fn completion_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"code":"rate_limit","message":"slow down"}}"#;
        let error = CompletionError::ProviderResponse(
            provider_response::ProviderResponseError::without_status(body.to_string()),
        );

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error
                .provider_response_json()
                .expect("fixture body should parse as valid JSON"),
            Some(serde_json::json!({
                "error": {
                    "code": "rate_limit",
                    "message": "slow down"
                }
            }))
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_preserved_status() {
        let body = r#"{"error":{"message":"too many requests"}}"#;
        let error =
            CompletionError::ProviderResponse(provider_response::ProviderResponseError::new(
                http::StatusCode::TOO_MANY_REQUESTS,
                body.to_string(),
            ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_preserved_plain_text_body() {
        let error = CompletionError::ProviderResponse(
            provider_response::ProviderResponseError::without_status(
                "provider exploded".to_string(),
            ),
        );

        assert_eq!(error.provider_response_body(), Some("provider exploded"));
        assert_eq!(error.provider_response_status(), None);
        assert!(error.provider_response_json().is_err());
    }

    #[test]
    fn completion_error_provider_error_is_not_a_provider_response() {
        // `ProviderError` also carries Rig-generated diagnostics, so the helpers
        // must not report its string as a provider response body.
        let error = CompletionError::ProviderError("stream transport failed".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error
                .provider_response_json()
                .expect("no body is not an error"),
            None
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_http_non_success_body_and_status() {
        let body = r#"{"error":{"type":"invalid_request","message":"bad request"}}"#;
        let error = CompletionError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
            http::StatusCode::BAD_REQUEST,
            body.to_string(),
        ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            error.provider_response_json().expect("valid JSON body"),
            Some(serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": "bad request"
                }
            }))
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_unrelated_variant() {
        let error = CompletionError::ResponseError("failed to parse provider response".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error
                .provider_response_json()
                .expect("no body is not an error"),
            None
        );
    }

    #[test]
    fn provider_response_json_returns_none_for_empty_preserved_body() {
        let error = CompletionError::ProviderResponse(
            provider_response::ProviderResponseError::without_status(String::new()),
        );

        assert_eq!(error.provider_response_body(), Some(""));
        assert_eq!(
            error
                .provider_response_json()
                .expect("empty body is not a JSON parse error"),
            None
        );
    }
}

#[cfg(test)]
mod response_identity_tests {
    use super::*;

    /// Serde compatibility (rig#2265): responses persisted before
    /// `provider_request_id` existed still load, with the field `None`.
    #[test]
    fn completion_response_without_request_id_still_deserializes() {
        let response: CompletionResponse = serde_json::from_str(
            r#"{"choice": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "cached_input_tokens": 0, "cache_creation_input_tokens": 0,
                          "reasoning_tokens": 0},
                "provider": "test"}"#,
        )
        .expect("pre-identity CompletionResponse JSON should load");
        assert_eq!(response.provider_request_id, None);
        assert_eq!(response.identity(), ResponseIdentity::default());
    }

    /// The identity accessor mirrors the flat fields exactly.
    #[test]
    fn identity_accessor_mirrors_flat_fields() {
        let response = CompletionResponse::new(
            vec![crate::completion::AssistantContent::text("hi")],
            Usage::new(),
            "test",
        )
        .with_message_id("msg_1")
        .with_response_id("resp_1")
        .with_provider_request_id("req_1");
        assert_eq!(
            response.identity(),
            ResponseIdentity {
                message_id: Some("msg_1".into()),
                response_id: Some("resp_1".into()),
                provider_request_id: Some("req_1".into()),
            }
        );
    }
}
