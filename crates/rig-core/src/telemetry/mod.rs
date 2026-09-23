//! GenAI tracing spans, completion-parent adoption, and opt-in content recording.
//!
//! ```
//! use rig_core::telemetry::{GenAiOperation, SpanBuilder};
//!
//! let span = SpanBuilder::new("provider", "model", GenAiOperation::Chat).build();
//! ```
use crate::completion::{AssistantContent, Message, Usage};
use crate::message::{
    DocumentSourceKind, Image, MimeType, Reasoning, ReasoningContent, ToolResult,
    ToolResultContent, UserContent,
};
use base64::Engine;
use serde::Serialize;
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};
use tracing::callsite::Identifier;

/// Macro implementation dependency; public because exported macro expansions
/// must be able to resolve it from downstream crates.
#[doc(hidden)]
pub use tracing as __tracing;

/// Declares a span field without a value, including in [`completion_parent_span!`].
pub use tracing::field::Empty;

/// Declares caller-supplied header fields and canonical completion fields.
/// The header must end with a comma; nonempty extras must begin with one.
#[doc(hidden)]
#[macro_export]
macro_rules! __rig_canonical_completion_span {
    (
        target: $target:literal,
        $(parent: $parent:expr,)?
        name: $name:literal,
        // Both blocks are spliced verbatim into `info_span!`: the header block
        // must end with a trailing comma, the extras block must begin with one.
        // Violating either surfaces as an `info_span!` parse error at the call
        // site, not here.
        { $($header:tt)* }
        { $($extra:tt)* }
    ) => {
        $crate::telemetry::__tracing::info_span!(
            target: $target,
            $(parent: $parent,)?
            $name,
            $($header)*
            gen_ai.response.id = $crate::telemetry::__tracing::field::Empty,
            gen_ai.response.model = $crate::telemetry::__tracing::field::Empty,
            rig.provider_request_id = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.output_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.tool_use_prompt_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.reasoning_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.input.messages = $crate::telemetry::__tracing::field::Empty,
            gen_ai.output.messages = $crate::telemetry::__tracing::field::Empty
            $($extra)*
        )
    };
}

macro_rules! new_completion_span {
    ($name:literal, $provider:expr, $request_model:expr, $operation:expr, $system:expr) => {
        $crate::__rig_canonical_completion_span!(
            target: "rig::completions",
            name: $name,
            {
                gen_ai.operation.name = $operation,
                gen_ai.provider.name = $provider,
                gen_ai.request.model = $request_model,
                gen_ai.system_instructions = $system,
            }
            {}
        )
    };
}

/// A GenAI operation and its canonical span name. Completion operations carry
/// message content and may adopt a completion-parent span; the others record
/// only usage and identity on a fresh span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenAiOperation {
    /// A chat completion.
    Chat,
    /// A streaming chat completion.
    ChatStreaming,
    /// A Gemini generate-content request.
    GenerateContent,
    /// A Gemini Interactions API request.
    Interactions,
    /// A streaming Gemini Interactions API request.
    InteractionsStreaming,
    /// A text (or image) embedding request.
    Embeddings,
    /// A reranking request.
    Rerank,
    /// An audio transcription request.
    Transcription,
    /// An image generation request.
    ImageGeneration,
    /// An audio generation (text-to-speech) request.
    AudioGeneration,
}

impl GenAiOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::ChatStreaming => "chat_streaming",
            Self::GenerateContent => "generate_content",
            Self::Interactions => "interactions",
            Self::InteractionsStreaming => "interactions_streaming",
            Self::Embeddings => "embeddings",
            Self::Rerank => "rerank",
            Self::Transcription => "transcription",
            Self::ImageGeneration => "image_generation",
            Self::AudioGeneration => "audio_generation",
        }
    }

    pub(crate) fn is_completion(self) -> bool {
        matches!(
            self,
            Self::Chat
                | Self::ChatStreaming
                | Self::GenerateContent
                | Self::Interactions
                | Self::InteractionsStreaming
        )
    }
}

/// Span field holding the provider's transport request id (the request-id
/// response header), recorded on success and on provider errors. GenAI
/// semantic conventions define no attribute for it. Rig's own completion
/// spans declare it; it is not required of an adopted parent, which simply
/// does not record it when undeclared.
pub const PROVIDER_REQUEST_ID_FIELD: &str = "rig.provider_request_id";

/// Marker field for completion-parent adoption, independent of tracing target.
/// Its value is ignored. Adoption requires every field in
/// [`COMPLETION_PARENT_REQUIRED_FIELDS`]; otherwise the builder creates a child
/// span and warns once per incomplete callsite.
pub const COMPLETION_PARENT_MARKER_FIELD: &str = "rig.completion_parent";

/// Fields required alongside [`COMPLETION_PARENT_MARKER_FIELD`] for adoption.
/// Each must be statically declared with [`Empty`] or a value.
/// [`completion_parent_span!`] declares the complete set.
pub const COMPLETION_PARENT_REQUIRED_FIELDS: &[&str] = &[
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.system_instructions",
    "gen_ai.response.id",
    "gen_ai.response.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.tool_use_prompt_tokens",
    "gen_ai.usage.reasoning_tokens",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
];

/// Declares a completion-parent span with the marker and all required fields.
/// Defaults to the current span as parent. An optional `parent: <expr>` between
/// `target` and `name` accepts any parent supported by [`tracing::info_span!`],
/// including `None`.
///
/// Records the supplied operation and system instructions. Provider and model
/// fields are populated on adoption. Extra fields must not duplicate the marker
/// or required fields: recording a duplicate name updates only its first field.
/// Use [`Empty`] or `Option::<&str>::None` for unset values without a direct
/// `tracing` dependency.
///
/// ```
/// use rig_core::telemetry::completion_parent_span;
///
/// let span = completion_parent_span!(
///     target: "my_runtime",
///     name: "chat",
///     operation: "chat",
///     system_instructions: Option::<&str>::None,
///     gen_ai.agent.name = "assistant",
/// );
/// ```
#[macro_export]
macro_rules! completion_parent_span {
    (
        target: $target:literal,
        parent: $parent:expr,
        name: $name:literal,
        operation: $operation:expr,
        system_instructions: $system:expr
        $(, $($extra:tt)*)?
    ) => {
        $crate::__rig_canonical_completion_span!(
            target: $target,
            parent: $parent,
            name: $name,
            {
                rig.completion_parent = true,
                gen_ai.operation.name = $operation,
                gen_ai.system_instructions = $system,
                gen_ai.provider.name = $crate::telemetry::__tracing::field::Empty,
                gen_ai.request.model = $crate::telemetry::__tracing::field::Empty,
            }
            { $(, $($extra)*)? }
        )
    };
    // Default arm: delegates to the explicit-parent arm so the two cannot
    // drift in the fields they declare.
    (
        target: $target:literal,
        name: $name:literal,
        operation: $operation:expr,
        system_instructions: $system:expr
        $(, $($extra:tt)*)?
    ) => {
        $crate::completion_parent_span!(
            target: $target,
            parent: $crate::telemetry::__tracing::Span::current(),
            name: $name,
            operation: $operation,
            system_instructions: $system
            $(, $($extra)*)?
        )
    };
}

// `#[macro_export]` places the macro at the crate root; re-export it here so
// it is also reachable at its documented home alongside the contract
// constants it implements.
pub use crate::completion_parent_span;

/// Classification of the current span for completion-parent adoption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionParentVerdict {
    /// The marker and all required fields are present.
    Adopt,
    /// The marker, but the span omits at least one field in
    /// [`COMPLETION_PARENT_REQUIRED_FIELDS`]. The missing names are computed
    /// only if a warning is emitted.
    RejectMissingFields,
    /// No marker at all: an ordinary ambient span that becomes the parent of a
    /// fresh `rig::completions` child. Never warns.
    NotAParent,
}

/// Returns undeclared required fields for the warning diagnostic.
fn missing_required_fields(metadata: &tracing::Metadata<'_>) -> Vec<&'static str> {
    let fields = metadata.fields();
    COMPLETION_PARENT_REQUIRED_FIELDS
        .iter()
        .copied()
        .filter(|name| fields.field(name).is_none())
        .collect()
}

/// Classifies span metadata without logging or changing global state.
fn classify_completion_parent(metadata: &tracing::Metadata<'_>) -> CompletionParentVerdict {
    let fields = metadata.fields();
    // Exact match, never a prefix: a runtime field that merely starts with the
    // marker name (`rig.completion_parent.id`, say) is not the marker and must
    // not make its span a rejected parent.
    if fields.field(COMPLETION_PARENT_MARKER_FIELD).is_none() {
        return CompletionParentVerdict::NotAParent;
    }
    if COMPLETION_PARENT_REQUIRED_FIELDS
        .iter()
        .all(|name| fields.field(name).is_some())
    {
        CompletionParentVerdict::Adopt
    } else {
        CompletionParentVerdict::RejectMissingFields
    }
}

/// Incomplete parent callsites already reported, bounded by the program's
/// static callsites. Each callsite receives its own missing-field diagnostic.
static NEAR_MISS_WARNED: LazyLock<Mutex<HashSet<Identifier>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Clear the per-callsite warn budget.
///
/// The budget is process-global, so without this the warning tests are coupled:
/// whichever runs first consumes the budget for any callsite they share, and the
/// other sees silence. `cargo nextest` hides that (one process per test) while
/// `cargo test` exposes it, so the coupling would be green in CI and red
/// locally — the worst orientation for a latent test bug. Resetting makes each
/// test independent of callsite identity, ordering, and runner.
#[cfg(test)]
fn reset_near_miss_warnings() {
    NEAR_MISS_WARNED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

/// Warns once per incomplete parent callsite, recovering a poisoned dedup lock.
/// A subscriber panic must not prevent subsequent completions.
fn warn_once_on_completion_parent_verdict(
    verdict: CompletionParentVerdict,
    metadata: &tracing::Metadata<'_>,
) {
    match verdict {
        CompletionParentVerdict::Adopt | CompletionParentVerdict::NotAParent => {}
        CompletionParentVerdict::RejectMissingFields => {
            let first_sighting = {
                let mut warned = NEAR_MISS_WARNED
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                warned.insert(metadata.callsite())
            };
            // Release the lock before invoking subscribers, which may re-enter
            // the builder and deadlock if the guard is still held.
            if !first_sighting {
                return;
            }
            tracing::warn!(
                marker = COMPLETION_PARENT_MARKER_FIELD,
                missing_fields = ?missing_required_fields(metadata),
                "completion-parent span declares the marker but not every required field \
                 and is not adopted; provider telemetry lands on a fresh child span \
                 instead — declare the span with \
                 `rig_core::telemetry::completion_parent_span!`"
            );
        }
    }
}

macro_rules! new_modality_span {
    ($name:literal, $provider:expr, $request_model:expr, $operation:expr) => {
        $crate::telemetry::__tracing::info_span!(
            target: "rig::modalities",
            $name,
            gen_ai.operation.name = $operation,
            gen_ai.provider.name = $provider,
            gen_ai.request.model = $request_model,
            gen_ai.response.id = $crate::telemetry::__tracing::field::Empty,
            gen_ai.response.model = $crate::telemetry::__tracing::field::Empty,
            rig.provider_request_id = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.output_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.tool_use_prompt_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.reasoning_tokens = $crate::telemetry::__tracing::field::Empty,
        )
    };
}

/// Runs a call of the unary operation `Op` in its canonical span and records
/// the response with [`Operation::record`](crate::wire::Operation::record) on
/// success. Returns the call's result unchanged; errors leave only request
/// metadata on the span.
pub async fn instrument_modality<Op, E>(
    provider: &str,
    request_model: &str,
    call: impl Future<Output = Result<Op::Response, E>>,
) -> Result<Op::Response, E>
where
    Op: crate::wire::Operation<Telemetry = GenAiOperation>,
{
    debug_assert!(!Op::telemetry(false).is_completion());
    let span = SpanBuilder::new(provider, request_model, Op::telemetry(false)).build();
    let result = tracing::Instrument::instrument(call, span.clone()).await;
    if let Ok(response) = &result {
        Op::record(&span, response);
    }
    result
}

/// Builder for a canonical GenAI span.
///
/// A completion operation reuses the current span when it declares
/// [`COMPLETION_PARENT_MARKER_FIELD`] and every field in
/// [`COMPLETION_PARENT_REQUIRED_FIELDS`], and otherwise opens a
/// `rig::completions` child of the current span. Every other operation opens a
/// fresh `rig::modalities` span.
pub struct SpanBuilder<'a> {
    provider: &'a str,
    request_model: &'a str,
    operation: GenAiOperation,
    system_instructions: Option<String>,
}

impl<'a> SpanBuilder<'a> {
    /// Create a span builder for a provider request.
    pub fn new(provider: &'a str, request_model: &'a str, operation: GenAiOperation) -> Self {
        Self {
            provider,
            request_model,
            operation,
            system_instructions: None,
        }
    }

    /// Set the system instructions sent with the request when sensitive content
    /// telemetry has been explicitly enabled. Only completion spans record them.
    pub fn system_instructions(
        mut self,
        system_instructions: Option<&'a str>,
        record_content: bool,
    ) -> Self {
        self.system_instructions = system_instructions_json(system_instructions, record_content);
        self
    }

    /// Build the operation's canonical span, or enrich Rig's current
    /// completion-parent span for a completion operation.
    pub fn build(self) -> tracing::Span {
        if self.operation.is_completion()
            && let Some(parent) = self.adopt_completion_parent()
        {
            return parent;
        }

        let (provider, model) = (self.provider, self.request_model);
        let op = self.operation.as_str();
        let sys = self.system_instructions.as_deref();
        match self.operation {
            GenAiOperation::Chat => new_completion_span!("chat", provider, model, op, sys),
            GenAiOperation::ChatStreaming => {
                new_completion_span!("chat_streaming", provider, model, op, sys)
            }
            GenAiOperation::GenerateContent => {
                new_completion_span!("generate_content", provider, model, op, sys)
            }
            GenAiOperation::Interactions => {
                new_completion_span!("interactions", provider, model, op, sys)
            }
            GenAiOperation::InteractionsStreaming => {
                new_completion_span!("interactions_streaming", provider, model, op, sys)
            }
            GenAiOperation::Embeddings => new_modality_span!("embeddings", provider, model, op),
            GenAiOperation::Rerank => new_modality_span!("rerank", provider, model, op),
            GenAiOperation::Transcription => {
                new_modality_span!("transcription", provider, model, op)
            }
            GenAiOperation::ImageGeneration => {
                new_modality_span!("image_generation", provider, model, op)
            }
            GenAiOperation::AudioGeneration => {
                new_modality_span!("audio_generation", provider, model, op)
            }
        }
    }

    /// The current span, enriched with this request, when it is a conforming
    /// completion parent. Warns once per near-miss callsite.
    fn adopt_completion_parent(&self) -> Option<tracing::Span> {
        let current = tracing::Span::current();
        let metadata = current.metadata()?;
        let verdict = classify_completion_parent(metadata);
        warn_once_on_completion_parent_verdict(verdict, metadata);
        if verdict != CompletionParentVerdict::Adopt {
            return None;
        }
        current.record("gen_ai.operation.name", self.operation.as_str());
        current.record("gen_ai.provider.name", self.provider);
        current.record("gen_ai.request.model", self.request_model);
        if let Some(system_instructions) = self.system_instructions.as_deref() {
            current.record("gen_ai.system_instructions", system_instructions);
        }
        Some(current)
    }
}

#[derive(Serialize)]
struct TelemetryChatMessage {
    role: &'static str,
    parts: Vec<TelemetryPart>,
}

#[derive(Serialize)]
struct TelemetryOutputMessage {
    role: &'static str,
    parts: Vec<TelemetryPart>,
    finish_reason: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TelemetryPart {
    Text {
        content: String,
    },
    ToolCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        arguments: serde_json::Value,
    },
    ToolCallResponse {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        response: serde_json::Value,
    },
    Reasoning {
        content: String,
    },
    Uri {
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        modality: &'static str,
        uri: String,
    },
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        modality: &'static str,
        file_id: String,
    },
    Blob {
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        modality: &'static str,
        content: String,
    },
}

fn media_part<T>(
    data: &DocumentSourceKind,
    media_type: Option<&T>,
    modality: &'static str,
) -> Option<TelemetryPart>
where
    T: MimeType,
{
    let mime_type = media_type.map(|media_type| media_type.to_mime_type().to_string());
    match data {
        DocumentSourceKind::Url(uri) => Some(TelemetryPart::Uri {
            mime_type,
            modality,
            uri: uri.clone(),
        }),
        DocumentSourceKind::FileId(file_id) => Some(TelemetryPart::File {
            mime_type,
            modality,
            file_id: file_id.clone(),
        }),
        DocumentSourceKind::Base64(content) => Some(TelemetryPart::Blob {
            mime_type,
            modality,
            content: content.clone(),
        }),
        DocumentSourceKind::Raw(content) => Some(TelemetryPart::Blob {
            mime_type,
            modality,
            content: base64::engine::general_purpose::STANDARD.encode(content),
        }),
        DocumentSourceKind::String(content) => Some(TelemetryPart::Text {
            content: content.clone(),
        }),
        DocumentSourceKind::Unknown => None,
    }
}

fn image_part(image: &Image) -> Option<TelemetryPart> {
    media_part(&image.data, image.media_type.as_ref(), "image")
}

fn reasoning_parts(reasoning: &Reasoning) -> Vec<TelemetryPart> {
    reasoning
        .content
        .iter()
        .map(|content| {
            let content = match content {
                ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text) => text,
                ReasoningContent::Encrypted(content) => content,
                ReasoningContent::Redacted { data } => data,
            };
            TelemetryPart::Reasoning {
                content: content.clone(),
            }
        })
        .collect()
}

fn tool_result_response(result: &ToolResult) -> serde_json::Value {
    let mut content = result
        .content
        .iter()
        .filter_map(|content| match content {
            ToolResultContent::Text(text) => Some(serde_json::Value::String(text.text.clone())),
            ToolResultContent::Json { value } => Some(value.clone()),
            ToolResultContent::Image(image) => {
                image_part(image).and_then(|part| serde_json::to_value(part).ok())
            }
        })
        .collect::<Vec<_>>();

    if content.len() == 1 {
        content.pop().unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Array(content)
    }
}

fn user_parts(content: &[UserContent]) -> Vec<TelemetryPart> {
    content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) => Some(TelemetryPart::Text {
                content: text.text.clone(),
            }),
            UserContent::ToolResult(result) => Some(TelemetryPart::ToolCallResponse {
                id: Some(result.call.to_string()),
                response: tool_result_response(result),
            }),
            UserContent::Image(image) => image_part(image),
            UserContent::Audio(audio) => {
                media_part(&audio.data, audio.media_type.as_ref(), "audio")
            }
            UserContent::Video(video) => {
                media_part(&video.data, video.media_type.as_ref(), "video")
            }
            UserContent::Document(document) => {
                media_part(&document.data, document.media_type.as_ref(), "document")
            }
        })
        .collect()
}

fn assistant_parts(content: &[AssistantContent]) -> Vec<TelemetryPart> {
    content
        .iter()
        .flat_map(|content| match content {
            AssistantContent::Text(text) => vec![TelemetryPart::Text {
                content: text.text.clone(),
            }],
            AssistantContent::ToolCall(tool_call) => vec![TelemetryPart::ToolCall {
                id: Some(tool_call.id.to_string()),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone(),
            }],
            AssistantContent::Reasoning(reasoning) => reasoning_parts(reasoning),
            AssistantContent::Image(image) => image_part(image).into_iter().collect(),
        })
        .collect()
}

fn input_messages(messages: &[Message]) -> Vec<TelemetryChatMessage> {
    messages
        .iter()
        .map(|message| match message {
            Message::System { content } => TelemetryChatMessage {
                role: "system",
                parts: vec![TelemetryPart::Text {
                    content: content.clone(),
                }],
            },
            Message::User { content } => TelemetryChatMessage {
                role: "user",
                parts: user_parts(content),
            },
            Message::Assistant { content, .. } => TelemetryChatMessage {
                role: "assistant",
                parts: assistant_parts(content),
            },
        })
        .collect()
}

fn output_messages(content: &[AssistantContent]) -> Vec<TelemetryOutputMessage> {
    let finish_reason = if content
        .iter()
        .any(|content| matches!(content, AssistantContent::ToolCall(_)))
    {
        "tool_call"
    } else {
        // Rig's normalized assistant content does not retain provider finish
        // reasons such as length or content filtering. Avoid claiming a clean
        // stop when the actual reason is unavailable.
        "unknown"
    };
    vec![TelemetryOutputMessage {
        role: "assistant",
        parts: assistant_parts(content),
        finish_reason,
    }]
}

/// Serializes system instructions using the normalized GenAI telemetry shape.
pub fn system_instructions_json(instructions: Option<&str>, enabled: bool) -> Option<String> {
    if !enabled {
        return None;
    }

    instructions.and_then(|instructions| {
        serde_json::to_string(&vec![TelemetryPart::Text {
            content: instructions.to_string(),
        }])
        .ok()
    })
}

/// Records serialized model input messages on `gen_ai.input.messages` when
/// content telemetry is explicitly enabled.
///
/// Message content can contain prompts, retrieved context, tool results, and
/// other sensitive or high-cardinality data. Keep this disabled unless the
/// caller has explicitly opted in for debugging/observability.
pub fn record_model_input(span: &tracing::Span, messages: &[Message], enabled: bool) {
    if !enabled || span.is_disabled() {
        return;
    }

    if let Ok(messages) = serde_json::to_string(&input_messages(messages)) {
        span.record("gen_ai.input.messages", messages);
    }
}

/// Records serialized model output messages on `gen_ai.output.messages` when
/// content telemetry is explicitly enabled.
///
/// Message content can contain model responses, tool calls, and other sensitive
/// or high-cardinality data. Keep this disabled unless the caller has explicitly
/// opted in for debugging/observability.
pub fn record_model_output(span: &tracing::Span, content: &[AssistantContent], enabled: bool) {
    if !enabled || span.is_disabled() {
        return;
    }

    let messages = output_messages(content);
    if let Ok(messages) = serde_json::to_string(&messages) {
        span.record("gen_ai.output.messages", messages);
    }
}

/// Provider response metadata used to populate GenAI telemetry spans.
pub trait ProviderResponseExt {
    /// Provider-native usage type.
    type Usage: Serialize;

    /// Returns the provider response ID, if supplied.
    fn response_id(&self) -> Option<&str>;

    /// Returns the provider response model name, if supplied.
    fn response_model_name(&self) -> Option<&str>;

    /// Returns the primary text response, when available.
    fn text_response(&self) -> Option<String>;

    /// Returns provider-native usage metrics, if supplied.
    fn usage(&self) -> Option<Self::Usage>;
}

/// Records GenAI usage and response metadata on tracing spans.
pub trait SpanCombinator {
    /// Record Rig-normalized token usage fields on the span.
    fn record_token_usage(&self, usage: &Usage);

    /// Record a response's ID, model, and token usage on the span.
    fn record_response(&self, response_id: Option<&str>, model: Option<&str>, usage: &Usage);
}

impl SpanCombinator for tracing::Span {
    fn record_token_usage(&self, usage: &Usage) {
        if self.is_disabled() {
            return;
        }

        // A counter the provider did not report leaves its span field unset;
        // a reported zero is recorded as zero.
        let fields = [
            ("gen_ai.usage.input_tokens", usage.input_tokens),
            ("gen_ai.usage.output_tokens", usage.output_tokens),
            (
                "gen_ai.usage.cache_read.input_tokens",
                usage.cached_input_tokens,
            ),
            (
                "gen_ai.usage.cache_creation.input_tokens",
                usage.cache_creation_input_tokens,
            ),
            (
                "gen_ai.usage.tool_use_prompt_tokens",
                usage.tool_use_prompt_tokens,
            ),
            ("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens),
        ];
        for (field, value) in fields {
            if let Some(value) = value {
                self.record(field, value);
            }
        }
    }

    fn record_response(&self, response_id: Option<&str>, model: Option<&str>, usage: &Usage) {
        if self.is_disabled() {
            return;
        }
        if let Some(id) = response_id {
            self.record("gen_ai.response.id", id);
        }
        if let Some(model) = model {
            self.record("gen_ai.response.model", model);
        }
        self.record_token_usage(usage);
    }
}

#[cfg(test)]
mod equivalence_tests;
#[cfg(test)]
mod tests;
