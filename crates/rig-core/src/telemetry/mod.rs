//! This module primarily concerns being able to orchestrate telemetry across a given pipeline or workflow.
//! This includes tracing, being able to send traces to an OpenTelemetry collector, setting up your
//! agents with the correct tracing style so you can emit the right traces for platforms like Langfuse,
//! and more.

use crate::OneOrMany;
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

/// Marks a span field as declared but not yet valued.
///
/// Re-exported so a runtime can declare a contract field through
/// [`completion_parent_span!`](crate::completion_parent_span) without taking a
/// direct `tracing` dependency — the crate's own `__tracing` path is hidden,
/// and `Option::<&str>::None` is the only other portable spelling. `rig-core`
/// already exposes `tracing` types across this module's public API
/// ([`CompletionSpanBuilder::build`] returns a [`tracing::Span`]), so this adds
/// no new semver surface.
pub use tracing::field::Empty;

/// Implementation detail of [`new_completion_span!`] and
/// [`completion_parent_span!`](crate::completion_parent_span): declares a span
/// with the caller's header fields followed by the canonical completion
/// telemetry fields recorded over a completion's lifetime.
///
/// `tracing` bakes a span's field set into static metadata, so the canonical
/// list can only be single-sourced by a macro that owns the whole
/// `info_span!` invocation — a `const` list can never be spliced in. This is
/// the one copy; [`COMPLETION_PARENT_REQUIRED_FIELDS`] is the checklist form
/// of the same contract and a test asserts the two agree exactly.
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

/// A supported GenAI completion operation and its canonical span name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionOperation {
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
}

impl CompletionOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::ChatStreaming => "chat_streaming",
            Self::GenerateContent => "generate_content",
            Self::Interactions => "interactions",
            Self::InteractionsStreaming => "interactions_streaming",
        }
    }
}

/// Core-owned marker field for a runtime span that may absorb provider completion fields.
///
/// Runtimes declare this field on their completion-parent spans without having
/// to share a tracing target. The field's presence (its value is ignored) marks
/// the span as a *candidate* for adoption, but adoption also requires the span
/// to statically declare every field in [`COMPLETION_PARENT_REQUIRED_FIELDS`].
///
/// This second requirement exists because [`tracing::Span::record`] silently
/// no-ops for any field absent from a span's static metadata: a span carrying
/// only the marker would be adopted and then drop every recorded completion
/// field, losing telemetry with no error. A span missing any required field is
/// therefore *not* adopted — [`CompletionSpanBuilder::build`] creates a fresh
/// `rig::completions` child span instead, so telemetry is never silently lost.
///
/// The declarative source of the contract is the
/// [`completion_parent_span!`](crate::completion_parent_span) macro; declaring
/// a span through it is what guarantees the marker and the field set stay in
/// agreement. Changing [`COMPLETION_PARENT_REQUIRED_FIELDS`] therefore needs no
/// marker change to degrade safely: a hand-written span built against an older
/// field list simply fails the gate above and gets a fresh child span, with a
/// warning (once per offending callsite) naming what it is missing.
pub const COMPLETION_PARENT_MARKER_FIELD: &str = "rig.completion_parent";

/// Fields a completion-parent span must statically declare (as
/// [`tracing::field::Empty`] or a value) to be adopted by
/// [`CompletionSpanBuilder::build`], alongside [`COMPLETION_PARENT_MARKER_FIELD`].
///
/// These mirror the canonical fields the builder's own `rig::completions` span
/// declares, so an adopted span can absorb the request, response, usage, and
/// content telemetry recorded over a completion's lifetime without dropping any.
///
/// This constant is the *checklist* form of the contract used by the adoption
/// gate; the *declarative* form is the
/// [`completion_parent_span!`](crate::completion_parent_span) macro, which is
/// how a runtime declares a conforming span. A test asserts the two forms
/// agree exactly, so neither can drift from the other.
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

/// Declare a completion-parent span conforming to the adoption contract.
///
/// This macro is the single declarative source of the contract: it declares
/// [`COMPLETION_PARENT_MARKER_FIELD`] and every field in
/// [`COMPLETION_PARENT_REQUIRED_FIELDS`], so a span it builds is always
/// adoptable by [`CompletionSpanBuilder::build`] and can absorb every field
/// recorded over the completion's lifetime. Runtimes should declare their
/// completion-parent spans through it rather than hand-writing the marker and
/// field list — [`tracing::Span::record`] silently no-ops on undeclared
/// fields, so a hand-written span that omits one field loses that telemetry
/// with no error (and a span that omits enough to fail the adoption gate is
/// not adopted at all). A span declared through this macro tracks the contract
/// automatically; a hand-written one has to be maintained by hand.
///
/// By default the span is explicitly parented on
/// [`tracing::Span::current()`]; pass `parent: <expr>` between `target` and
/// `name` to override it with any parent expression [`tracing::info_span!`]
/// accepts (including `None`). `operation` and `system_instructions` are
/// declared with the given values; the provider records
/// `gen_ai.provider.name` and `gen_ai.request.model` at adoption time.
/// Additional runtime-specific fields may be appended after the named
/// arguments. Extra fields must not repeat the marker or a required contract
/// field: a duplicate compiles, but the span then declares two same-named
/// fields and [`tracing::Span::record`] targets only the first.
///
/// Invoking this macro does not require a direct `tracing` dependency. To
/// declare a field with no value yet, use [`Empty`] (re-exported here for
/// exactly this reason) or `Option::<&str>::None`; [`tracing::field::Empty`]
/// itself is the same type but only nameable by a crate that depends on
/// `tracing` directly.
///
/// # Examples
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

/// What [`CompletionSpanBuilder::build`] decided about the span that is
/// current when it runs.
///
/// This is the whole adoption decision table in one place, kept free of
/// side effects so every row can be asserted directly. Emitting the
/// diagnostics for it is [`warn_once_on_completion_parent_verdict`]'s job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionParentVerdict {
    /// The marker plus the complete contract — adopt and enrich.
    Adopt,
    /// The marker, but the span omits at least one field in
    /// [`COMPLETION_PARENT_REQUIRED_FIELDS`]. The missing names are computed
    /// only if a warning is emitted.
    RejectMissingFields,
    /// No marker at all: an ordinary ambient span that becomes the parent of a
    /// fresh `rig::completions` child. Never warns.
    NotAParent,
}

/// Fields in [`COMPLETION_PARENT_REQUIRED_FIELDS`] that `metadata` does not
/// statically declare. Only called on the warning path — the adoption gate
/// itself needs a yes/no, not the names, and allocating a `Vec` on every
/// completion would be waste.
fn missing_required_fields(metadata: &tracing::Metadata<'_>) -> Vec<&'static str> {
    let fields = metadata.fields();
    COMPLETION_PARENT_REQUIRED_FIELDS
        .iter()
        .copied()
        .filter(|name| fields.field(name).is_none())
        .collect()
}

/// Classify the span `metadata` as a completion parent. Pure: no logging, no
/// global state — see [`CompletionParentVerdict`] for the decision table.
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

/// Near-miss parent callsites already reported.
///
/// Keyed per callsite rather than by one process-wide flag: two runtimes can
/// each declare a near-miss parent, and a single flag would report whichever
/// won the race and stay silent about the other — while the `missing_fields`
/// list it printed describes only that one span. Callsites are static, so this
/// set is bounded by the number of distinct near-miss spans in the program
/// (normally zero).
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

/// Surface a verdict that a human should act on, once per offending callsite.
///
/// A rejected near-miss degrades safely — the fresh child span loses no
/// telemetry — but silently, so the operator's only symptom would be a
/// duplicated span layer in dashboards.
///
/// Neither a `Once` nor a poison-propagating lock: a subscriber panic must not
/// turn a diagnostic into a hard failure on every subsequent completion, and a
/// dedup set stays perfectly valid across an unrelated panic.
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
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                warned.insert(metadata.callsite())
            };
            // The guard is released above, before `warn!`, and that scoping is
            // load-bearing: `warn!` dispatches into arbitrary subscriber code,
            // and a subscriber that transitively reaches `build` would
            // deadlock re-entering this non-reentrant lock.
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

/// Builder for a canonical GenAI completion span.
///
/// Runtime spans declaring [`COMPLETION_PARENT_MARKER_FIELD`] and the
/// [`COMPLETION_PARENT_REQUIRED_FIELDS`] are enriched and reused so one model
/// turn has exactly one model span. Other ambient spans — and marker spans that
/// omit a required field — remain parents of a newly created `rig::completions`
/// span.
pub struct CompletionSpanBuilder<'a> {
    provider: &'a str,
    request_model: &'a str,
    operation: CompletionOperation,
    system_instructions: Option<String>,
}

impl<'a> CompletionSpanBuilder<'a> {
    /// Create a completion-span builder for a provider request.
    pub fn new(provider: &'a str, request_model: &'a str, operation: CompletionOperation) -> Self {
        Self {
            provider,
            request_model,
            operation,
            system_instructions: None,
        }
    }

    /// Set the system instructions sent with the request when sensitive content
    /// telemetry has been explicitly enabled.
    pub fn system_instructions(
        mut self,
        system_instructions: Option<&'a str>,
        record_content: bool,
    ) -> Self {
        self.system_instructions = system_instructions_json(system_instructions, record_content);
        self
    }

    /// Build a canonical completion span or enrich Rig's current completion-parent span.
    pub fn build(self) -> tracing::Span {
        let current = tracing::Span::current();
        if let Some(metadata) = current.metadata() {
            let verdict = classify_completion_parent(metadata);
            warn_once_on_completion_parent_verdict(verdict, metadata);
            if verdict == CompletionParentVerdict::Adopt {
                current.record("gen_ai.operation.name", self.operation.as_str());
                current.record("gen_ai.provider.name", self.provider);
                current.record("gen_ai.request.model", self.request_model);
                if let Some(system_instructions) = self.system_instructions.as_deref() {
                    current.record("gen_ai.system_instructions", system_instructions);
                }
                return current;
            }
        }

        let operation = self.operation.as_str();
        let system_instructions = self.system_instructions.as_deref();
        match self.operation {
            CompletionOperation::Chat => new_completion_span!(
                "chat",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::ChatStreaming => new_completion_span!(
                "chat_streaming",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::GenerateContent => new_completion_span!(
                "generate_content",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::Interactions => new_completion_span!(
                "interactions",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::InteractionsStreaming => new_completion_span!(
                "interactions_streaming",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
        }
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

fn user_parts(content: &OneOrMany<UserContent>) -> Vec<TelemetryPart> {
    content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) => Some(TelemetryPart::Text {
                content: text.text.clone(),
            }),
            UserContent::ToolResult(result) => Some(TelemetryPart::ToolCallResponse {
                id: Some(result.id.clone()),
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

fn assistant_parts(content: &OneOrMany<AssistantContent>) -> Vec<TelemetryPart> {
    content
        .iter()
        .flat_map(|content| match content {
            AssistantContent::Text(text) => vec![TelemetryPart::Text {
                content: text.text.clone(),
            }],
            AssistantContent::ToolCall(tool_call) => vec![TelemetryPart::ToolCall {
                id: Some(tool_call.id.clone()),
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

fn output_messages(content: &OneOrMany<AssistantContent>) -> Vec<TelemetryOutputMessage> {
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
pub fn record_model_output(
    span: &tracing::Span,
    content: &OneOrMany<AssistantContent>,
    enabled: bool,
) {
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
    /// Provider-native output message type.
    type OutputMessage: Serialize;
    /// Provider-native usage type.
    type Usage: Serialize;

    /// Returns the provider response ID, if supplied.
    fn get_response_id(&self) -> Option<String>;

    /// Returns the provider response model name, if supplied.
    fn get_response_model_name(&self) -> Option<String>;

    /// Returns serialized output messages produced by the provider.
    fn get_output_messages(&self) -> Vec<Self::OutputMessage>;

    /// Returns the primary text response, when available.
    fn get_text_response(&self) -> Option<String>;

    /// Returns provider-native usage metrics, if supplied.
    fn get_usage(&self) -> Option<Self::Usage>;
}

/// A trait designed specifically to be used with Spans for the purpose of recording telemetry.
/// Implemented for [`tracing::Span`] to record GenAI semantic convention fields.
pub trait SpanCombinator {
    /// Record Rig-normalized token usage fields on the span.
    fn record_token_usage(&self, usage: &Usage);

    /// Record provider response metadata such as response ID and model name.
    fn record_response_metadata<R>(&self, response: &R)
    where
        R: ProviderResponseExt;
}

impl SpanCombinator for tracing::Span {
    fn record_token_usage(&self, usage: &Usage) {
        if self.is_disabled() {
            return;
        }

        // Zero-valued usage is the documented sentinel for missing provider
        // usage metrics; leave the span fields unset.
        if usage.has_values() {
            self.record("gen_ai.usage.input_tokens", usage.input_tokens);
            self.record("gen_ai.usage.output_tokens", usage.output_tokens);
            self.record(
                "gen_ai.usage.cache_read.input_tokens",
                usage.cached_input_tokens,
            );
            self.record(
                "gen_ai.usage.cache_creation.input_tokens",
                usage.cache_creation_input_tokens,
            );
            self.record(
                "gen_ai.usage.tool_use_prompt_tokens",
                usage.tool_use_prompt_tokens,
            );
            self.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);
        }
    }

    fn record_response_metadata<R>(&self, response: &R)
    where
        R: ProviderResponseExt,
    {
        if self.is_disabled() {
            return;
        }

        if let Some(id) = response.get_response_id() {
            self.record("gen_ai.response.id", id);
        }

        if let Some(model_name) = response.get_response_model_name() {
            self.record("gen_ai.response.model", model_name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{AssistantContent, Message, Usage};
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::{Id, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

    #[test]
    fn content_attributes_follow_gen_ai_semantic_convention_json_shapes() {
        assert_eq!(
            system_instructions_json(Some("follow policy"), true).as_deref(),
            Some(r#"[{"type":"text","content":"follow policy"}]"#)
        );
        assert_eq!(system_instructions_json(Some("secret"), false), None);

        let input = input_messages(&[
            Message::system("follow policy"),
            Message::user("hello"),
            Message::tool_result("call_1", "sunny"),
        ]);
        assert_eq!(
            serde_json::to_value(input).expect("semantic-convention input DTOs serialize"),
            json!([
                {
                    "role": "system",
                    "parts": [{"type": "text", "content": "follow policy"}]
                },
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": "hello"}]
                },
                {
                    "role": "user",
                    "parts": [{
                        "type": "tool_call_response",
                        "id": "call_1",
                        "response": "sunny"
                    }]
                }
            ])
        );

        let output = OneOrMany::one(AssistantContent::tool_call(
            "call_1",
            "weather",
            json!({"city": "Paris"}),
        ));
        assert_eq!(
            serde_json::to_value(output_messages(&output))
                .expect("semantic-convention output DTOs serialize"),
            json!([{
                "role": "assistant",
                "parts": [{
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "weather",
                    "arguments": {"city": "Paris"}
                }],
                "finish_reason": "tool_call"
            }])
        );

        let text_output = OneOrMany::one(AssistantContent::text("done"));
        assert_eq!(
            serde_json::to_value(output_messages(&text_output))
                .expect("semantic-convention text output DTOs serialize"),
            json!([{
                "role": "assistant",
                "parts": [{"type": "text", "content": "done"}],
                "finish_reason": "unknown"
            }])
        );
    }

    #[derive(Clone, Default)]
    struct CapturedFields(Arc<Mutex<Vec<(String, u64)>>>);

    impl CapturedFields {
        fn push(&self, name: &str, value: u64) {
            if let Ok(mut fields) = self.0.lock() {
                fields.push((name.to_string(), value));
            }
        }

        fn contains(&self, name: &str, value: u64) -> bool {
            self.0.lock().is_ok_and(|fields| {
                fields
                    .iter()
                    .any(|field| field == &(name.to_string(), value))
            })
        }
    }

    struct FieldCaptureLayer {
        fields: CapturedFields,
    }

    impl<S> Layer<S> for FieldCaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_record(&self, _span: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
            values.record(&mut FieldCaptureVisitor {
                fields: self.fields.clone(),
            });
        }
    }

    struct FieldCaptureVisitor {
        fields: CapturedFields,
    }

    impl Visit for FieldCaptureVisitor {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.fields.push(field.name(), value);
        }

        fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
    }

    /// WARN-level events, rendered as `field=value` pairs joined with the
    /// event's message, in emission order.
    #[derive(Clone, Default)]
    struct CapturedWarnings(Arc<Mutex<Vec<String>>>);

    impl CapturedWarnings {
        fn push(&self, rendered: String) {
            if let Ok(mut events) = self.0.lock() {
                events.push(rendered);
            }
        }

        /// Drains, so a test can assert on one phase and then assert that a
        /// later phase added nothing. A cloning read would make the second
        /// assertion see the first phase's events and quietly fail.
        fn take(&self) -> Vec<String> {
            self.0
                .lock()
                .map(|mut events| std::mem::take(&mut *events))
                .unwrap_or_default()
        }
    }

    struct WarningCaptureLayer {
        warnings: CapturedWarnings,
    }

    impl<S> Layer<S> for WarningCaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            if *event.metadata().level() != tracing::Level::WARN {
                return;
            }
            let mut visitor = WarningCaptureVisitor::default();
            event.record(&mut visitor);
            self.warnings.push(visitor.rendered);
        }
    }

    #[derive(Default)]
    struct WarningCaptureVisitor {
        rendered: String,
    }

    impl Visit for WarningCaptureVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            use std::fmt::Write;

            // `missing_fields = ?vec` arrives here; the message itself arrives
            // as the reserved `message` field. Both matter to the assertions,
            // so render every field rather than special-casing.
            let _ = write!(&mut self.rendered, " {}={value:?}", field.name());
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            use std::fmt::Write;

            let _ = write!(&mut self.rendered, " {}={value}", field.name());
        }
    }

    #[derive(Clone, Default)]
    struct CapturedSpan(Arc<Mutex<Option<CapturedSpanData>>>);

    struct CapturedSpanData {
        name: String,
        target: String,
        parent_name: Option<String>,
        fields: Vec<String>,
        initial_values: Vec<(String, String)>,
        recorded_values: Vec<(String, String)>,
    }

    struct SpanCaptureLayer {
        span: CapturedSpan,
    }

    #[derive(Default)]
    struct StringFieldVisitor {
        values: Vec<(String, String)>,
    }

    impl Visit for StringFieldVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.values
                .push((field.name().to_owned(), value.to_owned()));
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.values
                .push((field.name().to_owned(), format!("{value:?}")));
        }
    }

    impl<S> Layer<S> for SpanCaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &Id,
            ctx: Context<'_, S>,
        ) {
            if let Ok(mut captured) = self.span.0.lock() {
                let mut visitor = StringFieldVisitor::default();
                attrs.record(&mut visitor);
                let parent_name = if let Some(parent) = attrs.parent() {
                    ctx.span(parent)
                        .map(|span| span.metadata().name().to_owned())
                } else if attrs.is_contextual() {
                    ctx.lookup_current()
                        .map(|span| span.metadata().name().to_owned())
                } else {
                    None
                };
                *captured = Some(CapturedSpanData {
                    name: attrs.metadata().name().to_owned(),
                    target: attrs.metadata().target().to_owned(),
                    parent_name,
                    fields: attrs
                        .metadata()
                        .fields()
                        .iter()
                        .map(|field| field.name().to_owned())
                        .collect(),
                    initial_values: visitor.values,
                    recorded_values: Vec::new(),
                });
            }
        }

        fn on_record(&self, _span: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
            if let Ok(mut captured) = self.span.0.lock()
                && let Some(captured) = captured.as_mut()
            {
                let mut visitor = StringFieldVisitor::default();
                values.record(&mut visitor);
                captured.recorded_values.extend(visitor.values);
            }
        }
    }

    fn contains_string(values: &[(String, String)], field: &str, value: &str) -> bool {
        values
            .iter()
            .any(|candidate| candidate == &(field.to_owned(), value.to_owned()))
    }

    #[test]
    fn completion_span_uses_canonical_names_fields_and_initial_attributes() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();

        for (operation, expected_name) in [
            (CompletionOperation::Chat, "chat"),
            (CompletionOperation::ChatStreaming, "chat_streaming"),
            (CompletionOperation::GenerateContent, "generate_content"),
            (CompletionOperation::Interactions, "interactions"),
            (
                CompletionOperation::InteractionsStreaming,
                "interactions_streaming",
            ),
        ] {
            let captured = CapturedSpan::default();
            let subscriber = Registry::default().with(SpanCaptureLayer {
                span: captured.clone(),
            });
            tracing::subscriber::with_default(subscriber, || {
                let span = CompletionSpanBuilder::new("openai", "gpt-5", operation)
                    .system_instructions(Some("system prompt"), true)
                    .build();
                assert!(!span.is_disabled());
            });

            let Ok(captured) = captured.0.lock() else {
                panic!("captured span lock poisoned");
            };
            let Some(span) = captured.as_ref() else {
                panic!("completion span was not created");
            };
            assert_eq!(span.name, expected_name);
            assert_eq!(span.target, "rig::completions");
            assert_eq!(span.parent_name, None);
            for (field, value) in [
                ("gen_ai.operation.name", expected_name),
                ("gen_ai.provider.name", "openai"),
                ("gen_ai.request.model", "gpt-5"),
                (
                    "gen_ai.system_instructions",
                    r#"[{"type":"text","content":"system prompt"}]"#,
                ),
            ] {
                assert!(
                    contains_string(&span.initial_values, field, value),
                    "missing initial {field}={value}"
                );
            }
            assert!(span.recorded_values.is_empty());
            assert!(
                !span
                    .initial_values
                    .iter()
                    .any(|(field, _)| field == "gen_ai.response.model")
            );
            for field in COMPLETION_PARENT_REQUIRED_FIELDS {
                assert!(
                    span.fields.iter().any(|candidate| candidate == field),
                    "missing {field}"
                );
            }
        }
    }

    /// The default arm parents on the ambient span, and an explicit `parent:`
    /// overrides it.
    ///
    /// A regression to `parent: None` in the default arm would root every
    /// completion-parent span, detaching it from the surrounding trace. No
    /// field-set assertion in this module can see that — the fields are
    /// identical either way — while an operator sees completion spans floating
    /// as roots instead of nesting under the agent span.
    #[test]
    fn completion_parent_span_macro_honours_its_parent_argument() {
        /// `SpanCaptureLayer` has no target filter and keeps only the most
        /// recent span, so read it immediately after the span under test is
        /// created, and confirm the target before trusting the parent.
        fn captured_parent(captured: &CapturedSpan) -> Option<String> {
            let Ok(captured) = captured.0.lock() else {
                panic!("captured span lock poisoned");
            };
            let Some(span) = captured.as_ref() else {
                panic!("completion-parent span was not captured");
            };
            assert_eq!(span.target, "third_party_runtime");
            span.parent_name.clone()
        }

        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let ambient = tracing::info_span!(target: "application", "ambient");

            // Default arm: nests under whatever span is current.
            ambient.in_scope(|| {
                let _default_arm = completion_parent_span!(
                    target: "third_party_runtime",
                    name: "chat",
                    operation: Empty,
                    system_instructions: Option::<&str>::None,
                );
            });
            assert_eq!(captured_parent(&captured).as_deref(), Some("ambient"));

            // Explicit arm: the caller's parent wins over the ambient span, so
            // this one is a root despite `ambient` being current.
            ambient.in_scope(|| {
                let _explicit_arm = completion_parent_span!(
                    target: "third_party_runtime",
                    parent: None,
                    name: "chat",
                    operation: Empty,
                    system_instructions: Option::<&str>::None,
                );
            });
            assert_eq!(captured_parent(&captured), None);
        });
    }

    #[test]
    fn unrelated_ambient_span_is_parent_not_adopted() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let ambient = tracing::info_span!(target: "application", "ambient");
            let _guard = ambient.enter();
            let span =
                CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            assert_ne!(span.id(), ambient.id());
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion span was not captured");
        };
        assert_eq!(span.target, "rig::completions");
        assert_eq!(span.parent_name.as_deref(), Some("ambient"));
    }

    #[test]
    fn marker_span_missing_required_fields_is_not_adopted() {
        // A span that carries the marker but omits required canonical fields
        // (here: everything past `gen_ai.request.model`) must NOT be adopted.
        // Adopting it would silently drop the response/usage/content telemetry
        // that `Span::record` no-ops on for undeclared fields. Instead the
        // builder creates a fresh `rig::completions` child so nothing is lost.
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            // Deliberately hand-written (not `completion_parent_span!`): the
            // point is a marker span that fails to declare required fields.
            let partial_marker = tracing::info_span!(
                target: "third_party_runtime",
                "chat",
                rig.completion_parent = true,
                gen_ai.operation.name = tracing::field::Empty,
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
            );
            // Premise check: the hand-written marker literal above must still
            // match the constant — if the marker is ever renamed this fails
            // first, pointing at the stale literal, so the test cannot keep
            // passing for the wrong reason (no marker at all, rather than the
            // marker with fields missing).
            let Some(metadata) = partial_marker.metadata() else {
                panic!("partial marker span was disabled");
            };
            assert!(
                metadata
                    .fields()
                    .field(COMPLETION_PARENT_MARKER_FIELD)
                    .is_some(),
                "hand-written marker literal is stale; update it to {COMPLETION_PARENT_MARKER_FIELD}"
            );
            let _guard = partial_marker.enter();
            let span =
                CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            assert_ne!(span.id(), partial_marker.id());
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion span was not captured");
        };
        // A canonical child span is created and parented under the marker span,
        // and it carries the completion fields the marker span could not absorb.
        assert_eq!(span.target, "rig::completions");
        assert_eq!(span.parent_name.as_deref(), Some("chat"));
        for (field, value) in [
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.provider.name", "openai"),
            ("gen_ai.request.model", "gpt-5"),
        ] {
            assert!(contains_string(&span.initial_values, field, value));
        }
    }

    #[test]
    fn completion_parent_span_macro_matches_the_contract_exactly() {
        use std::collections::HashSet;

        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(Registry::default(), || {
            let span = completion_parent_span!(
                target: "contract_test",
                name: "chat",
                operation: "chat",
                system_instructions: Option::<&str>::None,
            );
            let Some(metadata) = span.metadata() else {
                panic!("contract span was disabled");
            };
            let declared: HashSet<&str> =
                metadata.fields().iter().map(|field| field.name()).collect();
            let expected: HashSet<&str> = COMPLETION_PARENT_REQUIRED_FIELDS
                .iter()
                .copied()
                .chain([COMPLETION_PARENT_MARKER_FIELD])
                .collect();
            // Exact equality in both directions: a field added to the macro
            // but not the constant (or vice versa) is drift, not a superset.
            assert_eq!(declared, expected);
            // Duplicate field names collapse in a `HashSet`, so also pin the
            // count: set equality alone cannot catch a field declared twice.
            assert_eq!(metadata.fields().len(), expected.len());
            assert_eq!(
                classify_completion_parent(metadata),
                CompletionParentVerdict::Adopt
            );

            // The explicit-`parent:` arm declares the identical field set.
            let span = completion_parent_span!(
                target: "contract_test",
                parent: None,
                name: "chat",
                operation: "chat",
                system_instructions: Option::<&str>::None,
            );
            let Some(metadata) = span.metadata() else {
                panic!("contract span with explicit parent was disabled");
            };
            let declared: HashSet<&str> =
                metadata.fields().iter().map(|field| field.name()).collect();
            assert_eq!(declared, expected);
            assert_eq!(metadata.fields().len(), expected.len());
            assert_eq!(
                classify_completion_parent(metadata),
                CompletionParentVerdict::Adopt
            );

            // Runtime-specific extra fields are additive on top of the contract.
            let span = completion_parent_span!(
                target: "contract_test",
                name: "chat",
                operation: "chat",
                system_instructions: Option::<&str>::None,
                gen_ai.agent.name = "assistant",
            );
            let Some(metadata) = span.metadata() else {
                panic!("contract span with extras was disabled");
            };
            let declared: HashSet<&str> =
                metadata.fields().iter().map(|field| field.name()).collect();
            let expected: HashSet<&str> =
                expected.into_iter().chain(["gen_ai.agent.name"]).collect();
            assert_eq!(declared, expected);
            assert_eq!(metadata.fields().len(), expected.len());
            assert_eq!(
                classify_completion_parent(metadata),
                CompletionParentVerdict::Adopt
            );
        });
    }

    #[test]
    fn canonical_completion_span_declares_exactly_the_required_fields() {
        use std::collections::HashSet;

        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(Registry::default(), || {
            let span =
                CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            let Some(metadata) = span.metadata() else {
                panic!("completion span was disabled");
            };
            let declared: HashSet<&str> =
                metadata.fields().iter().map(|field| field.name()).collect();
            let expected: HashSet<&str> =
                COMPLETION_PARENT_REQUIRED_FIELDS.iter().copied().collect();
            // The adoption checklist and the span the builder itself creates
            // must be the same set, or an adopted parent could not absorb
            // every field the builder records.
            assert_eq!(declared, expected);
            // Duplicate field names collapse in a `HashSet`, so also pin the
            // count: set equality alone cannot catch a field declared twice.
            assert_eq!(metadata.fields().len(), expected.len());
        });
    }

    #[test]
    fn completion_parent_required_fields_are_pinned() {
        // Changing this list is a contract change. An adopted parent declares
        // its fields statically, so a runtime whose span was hand-written
        // against the old list stops being adopted once the list moves — it
        // degrades gracefully (fresh child span, one-time warning naming what
        // is missing), but it does degrade. Confirm that is intended, note it
        // in the CHANGELOG, then update this snapshot.
        //
        // This is the only test that notices. Every other contract test
        // compares the three forms of the contract to each other, so a
        // *coherent* change — a field added to both this constant and the
        // macro — leaves them all agreeing, and green.
        assert_eq!(COMPLETION_PARENT_MARKER_FIELD, "rig.completion_parent");
        assert_eq!(
            COMPLETION_PARENT_REQUIRED_FIELDS,
            &[
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
            ]
        );
    }

    /// Every row of the adoption decision table, asserted against the pure
    /// classifier so no global warn-once state is involved.
    #[test]
    fn classify_completion_parent_covers_the_decision_table() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(Registry::default(), || {
            let verdict = |span: &tracing::Span| {
                let Some(metadata) = span.metadata() else {
                    panic!("classifier fixture span was disabled");
                };
                classify_completion_parent(metadata)
            };

            // Marker + full contract.
            let conforming = completion_parent_span!(
                target: "classifier_test",
                name: "chat",
                operation: tracing::field::Empty,
                system_instructions: tracing::field::Empty,
            );
            assert_eq!(verdict(&conforming), CompletionParentVerdict::Adopt);

            // Marker, incomplete contract.
            let partial = tracing::info_span!(
                target: "classifier_test",
                "chat",
                rig.completion_parent = true,
                gen_ai.operation.name = tracing::field::Empty,
            );
            assert_eq!(
                verdict(&partial),
                CompletionParentVerdict::RejectMissingFields
            );
            let Some(partial_metadata) = partial.metadata() else {
                panic!("classifier fixture span was disabled");
            };
            // The names only get computed on the warning path, so pin them here.
            assert_eq!(
                missing_required_fields(partial_metadata),
                COMPLETION_PARENT_REQUIRED_FIELDS
                    .iter()
                    .copied()
                    .filter(|name| *name != "gen_ai.operation.name")
                    .collect::<Vec<_>>()
            );

            // An ordinary ambient span.
            let ambient = tracing::info_span!(target: "application", "ambient");
            assert_eq!(verdict(&ambient), CompletionParentVerdict::NotAParent);

            // Marker detection is an exact field-name match, never a prefix: a
            // runtime field that merely starts with the marker name must not
            // make its span a rejected parent and warn at a runtime that never
            // opted in.
            let lookalike = tracing::info_span!(
                target: "application",
                "ambient",
                rig.completion_parent.id = "abc",
                rig.completion_parent_id = "abc",
            );
            assert_eq!(verdict(&lookalike), CompletionParentVerdict::NotAParent);
        });
    }

    /// The near-miss diagnostic is the only thing that makes a rejected parent
    /// visible to an operator — otherwise the sole symptom is a duplicated span
    /// layer in dashboards — so its message, its `missing_fields` payload, and
    /// its once-per-callsite budget all need pinning.
    #[test]
    fn near_miss_completion_parent_warns_once_per_callsite() {
        let warnings = CapturedWarnings::default();
        let subscriber = Registry::default().with(WarningCaptureLayer {
            warnings: warnings.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        // The warn budget is process-global; claim a clean one rather than
        // relying on this test's fixture span owning a callsite no other test
        // touches.
        reset_near_miss_warnings();
        tracing::subscriber::with_default(subscriber, || {
            // The `warn!` callsite lives in `warn_once_on_completion_parent_verdict`
            // and is shared with every other near-miss test, so its interest may
            // already be cached as `never` from a run under a different
            // subscriber. Same hazard `test_utils::scoped_tracing_subscriber_guard`
            // documents; same fix used in
            // `agent::prompt_request::streaming`'s scoped-subscriber tests.
            tracing::callsite::rebuild_interest_cache();

            let near_miss = tracing::info_span!(
                target: "third_party_runtime",
                "chat",
                rig.completion_parent = true,
                gen_ai.operation.name = tracing::field::Empty,
            );
            let _guard = near_miss.enter();
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            // Second completion under the *same* span callsite: the budget is
            // per callsite, so this one must stay silent.
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });

        let captured = warnings.take();
        assert_eq!(
            captured.len(),
            1,
            "a near-miss callsite warns exactly once, got: {captured:?}"
        );
        let Some(message) = captured.first() else {
            panic!("near miss did not warn");
        };
        assert!(
            message.contains("gen_ai.provider.name"),
            "warning must name the missing fields, got: {message}"
        );
        assert!(
            message.contains("completion_parent_span!"),
            "warning must point at the supported fix, got: {message}"
        );
    }

    /// The property that justifies keying the budget on the callsite rather
    /// than a single process-wide flag: two runtimes each declaring a broken
    /// parent are both reported. A global flag would report whichever ran first
    /// and stay silent about the other — and every other test in this module
    /// passes under that behaviour, so this is the only one that pins it.
    #[test]
    fn distinct_near_miss_callsites_each_warn() {
        let warnings = CapturedWarnings::default();
        let subscriber = Registry::default().with(WarningCaptureLayer {
            warnings: warnings.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        reset_near_miss_warnings();
        tracing::subscriber::with_default(subscriber, || {
            tracing::callsite::rebuild_interest_cache();

            // Two separate `info_span!` invocations, and they must stay
            // separate: the callsite *is* the dedup key, so extracting these
            // into a shared helper collapses them into one and this test would
            // assert 1, not 2. `reset_near_miss_warnings` cannot protect this
            // test the way it protects the others — distinct callsites are the
            // thing under test.
            let first = tracing::info_span!(
                target: "runtime_a",
                "chat",
                rig.completion_parent = true,
                gen_ai.operation.name = tracing::field::Empty,
            );
            first.in_scope(|| {
                CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            });

            let second = tracing::info_span!(
                target: "runtime_b",
                "chat",
                rig.completion_parent = true,
                gen_ai.operation.name = tracing::field::Empty,
            );
            second.in_scope(|| {
                CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            });
        });

        let captured = warnings.take();
        assert_eq!(
            captured.len(),
            2,
            "each offending callsite warns once, got: {captured:?}"
        );
    }

    /// The happy path must stay quiet: a conforming parent is adopted silently,
    /// so the diagnostic above cannot become background noise on every
    /// completion.
    #[test]
    fn conforming_completion_parent_never_warns() {
        let warnings = CapturedWarnings::default();
        let subscriber = Registry::default().with(WarningCaptureLayer {
            warnings: warnings.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        // Claim a clean budget: the control below must be able to warn even if
        // another test already reported this fixture's callsite.
        reset_near_miss_warnings();
        tracing::subscriber::with_default(subscriber, || {
            tracing::callsite::rebuild_interest_cache();

            // Control. Asserting an absence proves nothing unless the warning
            // pipe is known live in *this* subscriber: without this, the
            // assertions below pass just as happily when the diagnostic has
            // been deleted, or when callsite interest is stuck at `never`.
            let near_miss = tracing::info_span!(
                target: "third_party_runtime",
                "chat",
                rig.completion_parent = true,
                gen_ai.operation.name = tracing::field::Empty,
            );
            near_miss.in_scope(|| {
                CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            });
            assert_eq!(
                warnings.take().len(),
                1,
                "control: a near miss must warn, or this test cannot detect silence"
            );

            let conforming = completion_parent_span!(
                target: "third_party_runtime",
                name: "chat",
                operation: Empty,
                system_instructions: Option::<&str>::None,
            );
            let _guard = conforming.enter();
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();

            // An ordinary ambient span is not a parent at all, and must not warn
            // either — a runtime that never opted in should never hear about
            // this contract.
            drop(_guard);
            let ambient = tracing::info_span!(target: "application", "ambient");
            let _ambient_guard = ambient.enter();
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });

        // The control drained the buffer, so anything here was emitted by the
        // conforming or ambient span.
        let captured = warnings.take();
        assert!(
            captured.is_empty(),
            "adoption and non-participation are both silent, got: {captured:?}"
        );
    }

    #[test]
    fn agent_chat_span_is_adopted_and_enriched() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let completion_parent = completion_parent_span!(
                target: "rig::agent_chat",
                name: "chat_streaming",
                operation: tracing::field::Empty,
                system_instructions: tracing::field::Empty,
            );
            let _guard = completion_parent.enter();
            let span = CompletionSpanBuilder::new(
                "anthropic",
                "claude-sonnet",
                CompletionOperation::ChatStreaming,
            )
            .system_instructions(Some("provider system"), true)
            .build();
            assert_eq!(span.id(), completion_parent.id());
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion-parent span was not captured");
        };
        assert_eq!(span.target, "rig::agent_chat");
        for (field, value) in [
            ("gen_ai.operation.name", "chat_streaming"),
            ("gen_ai.provider.name", "anthropic"),
            ("gen_ai.request.model", "claude-sonnet"),
            (
                "gen_ai.system_instructions",
                r#"[{"type":"text","content":"provider system"}]"#,
            ),
        ] {
            assert!(contains_string(&span.recorded_values, field, value));
        }
    }

    #[test]
    fn neutral_completion_parent_span_is_adopted_and_enriched() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let completion_parent = completion_parent_span!(
                target: "test_runtime",
                name: "chat",
                operation: tracing::field::Empty,
                system_instructions: tracing::field::Empty,
            );
            let _guard = completion_parent.enter();
            let span = CompletionSpanBuilder::new(
                "neutral-provider",
                "neutral-model",
                CompletionOperation::Chat,
            )
            .build();
            assert_eq!(span.id(), completion_parent.id());
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("neutral completion-parent span was not captured");
        };
        assert_eq!(span.target, "test_runtime");
        for (field, value) in [
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.provider.name", "neutral-provider"),
            ("gen_ai.request.model", "neutral-model"),
        ] {
            assert!(contains_string(&span.recorded_values, field, value));
        }
    }

    #[test]
    fn absent_provider_system_does_not_overwrite_agent_instructions() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let completion_parent = completion_parent_span!(
                target: "test_runtime",
                name: "chat",
                operation: tracing::field::Empty,
                system_instructions: "effective agent instructions",
            );
            let _guard = completion_parent.enter();
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion-parent span was not captured");
        };
        assert!(contains_string(
            &span.initial_values,
            "gen_ai.system_instructions",
            "effective agent instructions"
        ));
        assert!(
            !span
                .recorded_values
                .iter()
                .any(|(field, _)| field == "gen_ai.system_instructions")
        );
    }

    #[test]
    fn record_token_usage_records_tool_use_prompt_tokens() {
        let fields = CapturedFields::default();
        let subscriber = Registry::default().with(FieldCaptureLayer {
            fields: fields.clone(),
        });
        let usage = Usage {
            input_tokens: 1,
            output_tokens: 2,
            total_tokens: 15,
            cached_input_tokens: 3,
            cache_creation_input_tokens: 4,
            tool_use_prompt_tokens: 12,
            reasoning_tokens: 5,
        };

        // Scoped-subscriber tests must not run concurrently; see
        // `test_utils::scoped_tracing_subscriber_guard`.
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(
                "usage_recording",
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                gen_ai.usage.reasoning_tokens = tracing::field::Empty,
            );

            span.record_token_usage(&usage);
        });

        assert!(fields.contains("gen_ai.usage.tool_use_prompt_tokens", 12));
    }
}
