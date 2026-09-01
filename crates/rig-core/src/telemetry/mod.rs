//! This module primarily concerns being able to orchestrate telemetry across a given pipeline or workflow.
//! This includes tracing, being able to send traces to an OpenTelemetry collector, setting up your
//! agents with the correct tracing style so you can emit the right traces for platforms like Langfuse,
//! and more.
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
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
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

/// A supported non-completion GenAI operation and its canonical span name.
///
/// The completion operations stay on [`CompletionOperation`]: they carry
/// message content and participate in span adoption. These operations carry
/// only usage and identity metadata, and their spans are always fresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalityOperation {
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

impl ModalityOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings",
            Self::Rerank => "rerank",
            Self::Transcription => "transcription",
            Self::ImageGeneration => "image_generation",
            Self::AudioGeneration => "audio_generation",
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
            gen_ai.usage.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.output_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.tool_use_prompt_tokens = $crate::telemetry::__tracing::field::Empty,
            gen_ai.usage.reasoning_tokens = $crate::telemetry::__tracing::field::Empty,
        )
    };
}

/// Builder for a canonical GenAI span on a non-completion modality
/// (embeddings, rerank, transcription, image generation, audio generation).
///
/// Unlike [`CompletionSpanBuilder`], this never adopts an ambient span: the
/// adoption contract exists so one *model turn* has exactly one completion
/// span, and a modality call is not a model turn — an agent's RAG lookup
/// should appear as its own child span under the turn, not overwrite the
/// turn's fields. The span is created under whatever span is current, so
/// nesting comes for free.
pub struct ModalitySpanBuilder<'a> {
    provider: &'a str,
    request_model: &'a str,
    operation: ModalityOperation,
}

impl<'a> ModalitySpanBuilder<'a> {
    /// Create a modality-span builder for a provider request.
    pub fn new(provider: &'a str, request_model: &'a str, operation: ModalityOperation) -> Self {
        Self {
            provider,
            request_model,
            operation,
        }
    }

    /// Build a canonical modality span.
    pub fn build(self) -> tracing::Span {
        let operation = self.operation.as_str();
        match self.operation {
            ModalityOperation::Embeddings => {
                new_modality_span!("embeddings", self.provider, self.request_model, operation)
            }
            ModalityOperation::Rerank => {
                new_modality_span!("rerank", self.provider, self.request_model, operation)
            }
            ModalityOperation::Transcription => {
                new_modality_span!(
                    "transcription",
                    self.provider,
                    self.request_model,
                    operation
                )
            }
            ModalityOperation::ImageGeneration => new_modality_span!(
                "image_generation",
                self.provider,
                self.request_model,
                operation
            ),
            ModalityOperation::AudioGeneration => new_modality_span!(
                "audio_generation",
                self.provider,
                self.request_model,
                operation
            ),
        }
    }
}

/// The telemetry a normalized modality response can put on its span: the
/// usage and identity fields every normalized response carries since the
/// type-erasure sweep. Implemented for all six normalized response types.
pub trait ModalityResponseTelemetry {
    /// Rig-normalized token usage for the call.
    fn telemetry_usage(&self) -> &Usage;
    /// Provider-reported model identifier, when the wire named one.
    fn telemetry_model(&self) -> Option<&str>;
    /// Provider-assigned response-scoped identifier, when reported.
    fn telemetry_response_id(&self) -> Option<&str>;
}

macro_rules! impl_modality_response_telemetry {
    ($($ty:ty),+ $(,)?) => {
        $(impl ModalityResponseTelemetry for $ty {
            fn telemetry_usage(&self) -> &Usage {
                &self.usage
            }
            fn telemetry_model(&self) -> Option<&str> {
                self.model.as_deref()
            }
            fn telemetry_response_id(&self) -> Option<&str> {
                self.response_id.as_deref()
            }
        })+
    };
}

impl_modality_response_telemetry!(
    crate::embeddings::EmbeddingResponse,
    crate::embeddings::ImageEmbeddingResponse,
    crate::rerank::RerankResponse,
    crate::transcription::TranscriptionResponse,
);
#[cfg(feature = "image")]
impl_modality_response_telemetry!(crate::image_generation::ImageGenerationResponse);
#[cfg(feature = "audio")]
impl_modality_response_telemetry!(crate::audio_generation::AudioGenerationResponse);

/// Run a modality call inside its canonical span and record the normalized
/// response's usage and identity onto it on success.
///
/// This is the one seam every provider implementation calls, so the span
/// contract lives here rather than at thirty call sites. Errors close the
/// span with its request fields only; the provider error itself already
/// carries the preserved body and status.
pub async fn instrument_modality<F, T, E>(
    provider: &str,
    request_model: &str,
    operation: ModalityOperation,
    call: F,
) -> Result<T, E>
where
    F: Future<Output = Result<T, E>>,
    T: ModalityResponseTelemetry,
{
    let span = ModalitySpanBuilder::new(provider, request_model, operation).build();
    let result = tracing::Instrument::instrument(call, span.clone()).await;
    if let Ok(response) = &result {
        span.record_token_usage(response.telemetry_usage());
        if let Some(id) = response.telemetry_response_id() {
            span.record("gen_ai.response.id", id);
        }
        if let Some(model) = response.telemetry_model() {
            span.record("gen_ai.response.model", model);
        }
    }
    result
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

fn user_parts(content: &[UserContent]) -> Vec<TelemetryPart> {
    content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) => Some(TelemetryPart::Text {
                content: text.text.clone(),
            }),
            UserContent::ToolResult(result) => Some(TelemetryPart::ToolCallResponse {
                id: Some(result.call.as_str().to_owned()),
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
                id: Some(tool_call.id.as_str().to_owned()),
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

        if let Some(id) = response.response_id() {
            self.record("gen_ai.response.id", id);
        }

        if let Some(model_name) = response.response_model_name() {
            self.record("gen_ai.response.model", model_name);
        }
    }
}

#[cfg(test)]
mod tests;
