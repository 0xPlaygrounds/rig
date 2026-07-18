//! Owned asynchronous effect requests and validated completion ingress.

use std::{any::Any, fmt, sync::Arc};

use rig_core::{
    completion::{AssistantContent, CompletionRequest, Usage},
    streaming::ToolCallDeltaContent,
    tool::{ToolExecutionError, ToolOutput},
};

use crate::{
    CapabilityId, CorrelationId, Generation, GrantId, MemoryId, ModelId, OperationId, RunId,
    RuntimeId, TenantId,
};

/// Complete identity and authorization context carried by every effect message.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectHeader {
    /// Runtime world that issued the operation.
    pub runtime_id: RuntimeId,
    /// Run that owns the operation.
    pub run_id: RunId,
    /// Stable operation identity.
    pub operation_id: OperationId,
    /// Run generation at dispatch time.
    pub generation: Generation,
    /// Per-attempt correlation identity.
    pub correlation_id: CorrelationId,
    /// Tenant security boundary.
    pub tenant_id: TenantId,
    /// Capability used by a tool effect.
    pub capability_id: Option<CapabilityId>,
    /// Grant authorizing a tool effect.
    pub grant_id: Option<GrantId>,
    /// Exact advertised capability revision.
    pub capability_revision: Option<u64>,
}

impl fmt::Debug for EffectHeader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EffectHeader")
            .field("runtime_id", &self.runtime_id)
            .field("run_id", &self.run_id)
            .field("operation_id", &self.operation_id)
            .field("generation", &self.generation)
            .field("correlation_id", &"<redacted>")
            .field("tenant_id", &"<redacted>")
            .field("capability_id", &self.capability_id)
            .field("grant_id", &self.grant_id)
            .field("capability_revision", &self.capability_revision)
            .finish()
    }
}

/// Owned model request dispatched outside the ECS world.
#[derive(Clone)]
pub struct ModelEffectIntent {
    /// Correlation and ownership header.
    pub header: EffectHeader,
    /// Rebound model implementation identity.
    pub model_id: ModelId,
    /// Fully owned provider-neutral completion request.
    pub request: CompletionRequest,
    /// Whether to consume the provider streaming surface.
    pub streaming: bool,
}

impl fmt::Debug for ModelEffectIntent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelEffectIntent")
            .field("header", &self.header)
            .field("model_id", &self.model_id)
            .field("streaming", &self.streaming)
            .field("request", &"<redacted>")
            .finish()
    }
}

/// Owned portable tool request dispatched outside the ECS world.
#[derive(Clone)]
pub struct ToolEffectIntent {
    /// Correlation, authorization, and ownership header.
    pub header: EffectHeader,
    /// Model tool-call identity used for transcript pairing.
    pub tool_call_id: String,
    /// Provider-specific call identity.
    pub provider_call_id: Option<String>,
    /// Exact provider-facing tool name advertised for the turn.
    pub name: String,
    /// Fully owned JSON arguments.
    pub arguments: serde_json::Value,
    /// Stable model-call order used for deterministic commit.
    pub order: usize,
}

impl fmt::Debug for ToolEffectIntent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolEffectIntent")
            .field("header", &self.header)
            .field("tool_call_id", &self.tool_call_id)
            .field(
                "provider_call_id_configured",
                &self.provider_call_id.is_some(),
            )
            .field("name", &self.name)
            .field("arguments", &"<redacted>")
            .field("order", &self.order)
            .finish()
    }
}

/// Owned memory request dispatched outside the ECS world.
#[derive(Clone)]
pub enum MemoryEffectIntent {
    /// Load history before the first model call.
    Load {
        /// Correlation and ownership header.
        header: EffectHeader,
        /// Rebound memory implementation identity.
        memory_id: MemoryId,
        /// Conversation key.
        conversation_id: String,
    },
    /// Append only newly committed canonical messages.
    Append {
        /// Correlation and ownership header.
        header: EffectHeader,
        /// Rebound memory implementation identity.
        memory_id: MemoryId,
        /// Conversation key.
        conversation_id: String,
        /// Canonical messages committed by this successful run.
        messages: Vec<rig_core::completion::Message>,
    },
}

impl fmt::Debug for MemoryEffectIntent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load {
                header, memory_id, ..
            } => formatter
                .debug_struct("MemoryEffectIntent::Load")
                .field("header", header)
                .field("memory_id", memory_id)
                .field("conversation_id", &"<redacted>")
                .finish(),
            Self::Append {
                header,
                memory_id,
                messages,
                ..
            } => formatter
                .debug_struct("MemoryEffectIntent::Append")
                .field("header", header)
                .field("memory_id", memory_id)
                .field("conversation_id", &"<redacted>")
                .field("message_count", &messages.len())
                .finish(),
        }
    }
}

impl MemoryEffectIntent {
    /// Borrow the shared effect header.
    #[must_use]
    pub const fn header(&self) -> &EffectHeader {
        match self {
            Self::Load { header, .. } | Self::Append { header, .. } => header,
        }
    }
}

/// One owned effect waiting for bounded execution.
#[derive(Clone, Debug)]
pub enum EffectIntent {
    /// Provider model operation.
    Model(Box<ModelEffectIntent>),
    /// Portable tool operation.
    Tool(ToolEffectIntent),
    /// Conversation-memory operation.
    Memory(MemoryEffectIntent),
}

impl EffectIntent {
    /// Borrow the identity header for scheduling and cancellation.
    #[must_use]
    pub const fn header(&self) -> &EffectHeader {
        match self {
            Self::Model(intent) => &intent.header,
            Self::Tool(intent) => &intent.header,
            Self::Memory(intent) => intent.header(),
        }
    }
}

/// Provider-stream item observable before the containing model turn commits.
#[derive(Clone, serde::Deserialize, PartialEq, serde::Serialize)]
#[non_exhaustive]
pub enum ProvisionalDelta {
    /// Text delta.
    Text(String),
    /// Complete tool call normalized by the provider stream.
    ToolCall(rig_core::message::ToolCall),
    /// Partial tool-call data.
    ToolCallDelta {
        /// Provider tool-call identity.
        id: String,
        /// Rig-internal stream correlation identity.
        internal_call_id: String,
        /// Name or argument delta.
        content: ToolCallDeltaContent,
    },
    /// Complete reasoning block.
    Reasoning(rig_core::message::Reasoning),
    /// Partial reasoning text.
    ReasoningDelta {
        /// Provider reasoning identity.
        id: Option<String>,
        /// Partial text.
        reasoning: String,
    },
    /// Provider-native item that is not part of canonical transcript state.
    Unknown(serde_json::Value),
}

impl ProvisionalDelta {
    pub(crate) const fn kind(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::ToolCall(_) => "tool_call",
            Self::ToolCallDelta { .. } => "tool_call_delta",
            Self::Reasoning(_) => "reasoning",
            Self::ReasoningDelta { .. } => "reasoning_delta",
            Self::Unknown(_) => "unknown",
        }
    }
}

impl fmt::Debug for ProvisionalDelta {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProvisionalDelta")
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}

/// Non-persisted type-erased provider final retained only on local surfaces.
#[derive(Clone)]
pub struct ErasedRawFinal {
    value: Arc<dyn Any + Send + Sync>,
    type_name: &'static str,
}

impl ErasedRawFinal {
    pub(crate) fn new<T>(value: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            value: Arc::new(value),
            type_name: std::any::type_name::<T>(),
        }
    }

    /// Concrete Rust type name for diagnostics without exposing content.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// Downcast the local side channel to the concrete provider response.
    #[must_use]
    pub fn downcast<T>(&self) -> Option<Arc<T>>
    where
        T: Any + Send + Sync,
    {
        Arc::downcast::<T>(Arc::clone(&self.value)).ok()
    }
}

impl fmt::Debug for ErasedRawFinal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ErasedRawFinal")
            .field("type_name", &self.type_name)
            .finish_non_exhaustive()
    }
}

/// Canonical result of one completed model effect.
pub struct ModelEffectOutput {
    /// Aggregated assistant choice after a complete blocking response or stream.
    pub choice: Vec<AssistantContent>,
    /// Provider-reported usage for this billed operation.
    pub usage: Usage,
    /// Provider-assigned assistant message identity.
    pub message_id: Option<String>,
    /// Concrete non-persisted provider final when the surface produced one.
    pub raw_final: Option<ErasedRawFinal>,
}

impl fmt::Debug for ModelEffectOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelEffectOutput")
            .field("choice_items", &self.choice.len())
            .field("usage", &self.usage)
            .field("message_id_configured", &self.message_id.is_some())
            .field(
                "raw_final_type",
                &self.raw_final.as_ref().map(ErasedRawFinal::type_name),
            )
            .finish()
    }
}

/// Canonical result of one portable tool effect.
pub enum ToolEffectOutput {
    /// Successful canonical tool output.
    Success(ToolOutput),
    /// Normalized tool failure with safe model-visible output.
    Failure(ToolExecutionError),
}

impl fmt::Debug for ToolEffectOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Success(_) => "ToolEffectOutput::Success(<redacted>)",
            Self::Failure(_) => "ToolEffectOutput::Failure(<redacted>)",
        })
    }
}

/// Canonical result of one memory effect.
pub enum MemoryEffectOutput {
    /// Loaded conversation history.
    Loaded(Vec<rig_core::completion::Message>),
    /// Append completed successfully.
    Appended,
}

impl fmt::Debug for MemoryEffectOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loaded(messages) => formatter
                .debug_tuple("MemoryEffectOutput::Loaded")
                .field(&format_args!("{} messages", messages.len()))
                .finish(),
            Self::Appended => formatter.write_str("MemoryEffectOutput::Appended"),
        }
    }
}

/// Typed failures produced while executing an owned model effect.
#[derive(thiserror::Error)]
#[non_exhaustive]
pub enum ModelEffectError {
    /// The provider rejected or failed the completion operation.
    #[error("provider completion failed")]
    Provider(#[source] rig_core::completion::CompletionError),
    /// A provisional stream item could not cross the bounded ingress channel.
    #[error("runtime effect ingress closed before model streaming completed")]
    IngressClosed,
    /// The configured effect deadline elapsed.
    #[error("model effect timed out")]
    TimedOut,
    /// Restoration or retirement left the stable model identity unbound.
    #[error("model binding `{model_id}` is unavailable")]
    MissingBinding {
        /// Stable model identity that could not be resolved.
        model_id: ModelId,
    },
}

impl ModelEffectError {
    /// Inspect a preserved provider response body without exposing it in diagnostics.
    #[must_use]
    pub fn provider_response_body(&self) -> Option<&str> {
        match self {
            Self::Provider(error) => error.provider_response_body(),
            _ => None,
        }
    }

    /// Inspect a preserved provider response HTTP status.
    #[must_use]
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        match self {
            Self::Provider(error) => error.provider_response_status(),
            _ => None,
        }
    }

    /// Parse a preserved provider response body as JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        match self {
            Self::Provider(error) => error.provider_response_json(),
            _ => Ok(None),
        }
    }
}

impl fmt::Debug for ModelEffectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple(match self {
                Self::Provider(_) => "Provider",
                Self::IngressClosed => "IngressClosed",
                Self::TimedOut => "TimedOut",
                Self::MissingBinding { .. } => "MissingBinding",
            })
            .field(&self.to_string())
            .finish()
    }
}

/// Typed failures produced while executing an owned conversation-memory effect.
#[derive(thiserror::Error)]
#[non_exhaustive]
pub enum MemoryEffectError {
    /// The portable memory backend failed.
    #[error("conversation memory backend failed")]
    Backend(#[source] rig_core::memory::MemoryError),
    /// The configured effect deadline elapsed.
    #[error("conversation memory effect timed out")]
    TimedOut,
    /// Restoration or retirement left the stable memory identity unbound.
    #[error("memory binding `{memory_id}` is unavailable")]
    MissingBinding {
        /// Stable memory identity that could not be resolved.
        memory_id: MemoryId,
    },
}

impl fmt::Debug for MemoryEffectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple(match self {
                Self::Backend(_) => "Backend",
                Self::TimedOut => "TimedOut",
                Self::MissingBinding { .. } => "MissingBinding",
            })
            .field(&self.to_string())
            .finish()
    }
}

/// Terminal payload returned by one completed effect future.
#[derive(Debug)]
pub enum EffectCompletion {
    /// Model operation result.
    Model {
        /// Fully correlated effect header.
        header: EffectHeader,
        /// Success or normalized provider failure.
        result: Result<ModelEffectOutput, ModelEffectError>,
    },
    /// Tool operation result.
    Tool {
        /// Fully correlated effect header.
        header: EffectHeader,
        /// Model tool-call identity.
        tool_call_id: String,
        /// Stable model-call order.
        order: usize,
        /// Success or normalized tool failure.
        result: ToolEffectOutput,
    },
    /// Memory operation result.
    Memory {
        /// Fully correlated effect header.
        header: EffectHeader,
        /// Success or normalized backend failure.
        result: Result<MemoryEffectOutput, MemoryEffectError>,
    },
}

impl EffectCompletion {
    /// Borrow the shared correlation header.
    #[must_use]
    pub const fn header(&self) -> &EffectHeader {
        match self {
            Self::Model { header, .. }
            | Self::Tool { header, .. }
            | Self::Memory { header, .. } => header,
        }
    }
}

/// One message crossing the bounded effect ingress boundary.
#[derive(Debug)]
pub enum EffectIngress {
    /// Provisional streaming content that cannot mutate canonical history.
    Delta {
        /// Fully correlated model effect header.
        header: EffectHeader,
        /// Monotonic provider-stream item sequence within this operation.
        sequence: u64,
        /// Provisional provider-normalized content.
        delta: ProvisionalDelta,
    },
    /// Final completion of an effect.
    Completion(EffectCompletion),
}

/// Stable reason an ingress message was rejected without mutating state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum EffectRejectionReason {
    /// Header names another runtime world.
    ForeignRuntime,
    /// Run is missing, cleaned, or foreign.
    UnknownRun,
    /// Header tenant differs from the authoritative run tenant.
    WrongTenant,
    /// Header generation is stale or superseded.
    WrongGeneration,
    /// Operation is no longer active.
    UnknownOperation,
    /// Correlation identity differs from the active attempt.
    WrongCorrelation,
    /// Capability, grant, or revision differs from the immutable turn snapshot.
    WrongAuthorization,
    /// Completion was already accepted.
    Duplicate,
    /// A provisional stream sequence skipped ahead of the next expected item.
    OutOfOrder,
    /// Completion payload does not match the dispatched operation identity.
    WrongPayload,
    /// Run is terminal or canceled.
    Late,
}

/// Auditable record of rejected effect ingress.
#[derive(Clone, Debug)]
pub struct EffectRejection {
    /// Rejected header.
    pub header: EffectHeader,
    /// Validation rule that rejected it.
    pub reason: EffectRejectionReason,
}

/// Redacted hosted/erased provider-final envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostedProviderDiagnostic {
    /// Operation that produced the final.
    pub operation_id: OperationId,
    /// Concrete type name, never serialized provider content.
    pub provider_type: &'static str,
    /// Whether a local concrete value existed before erasure.
    pub available: bool,
}
