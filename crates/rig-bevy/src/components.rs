//! Authoritative ECS topology and run-state components.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    sync::Arc,
};

use bevy_ecs::prelude::{Component, Entity, Resource};
use rig_core::{
    completion::{AssistantContent, Message, ToolDefinition, Usage},
    message::ToolCall,
    tool::ToolOutput,
};
use serde::{Deserialize, Serialize};

use crate::{
    AgentId, AgentSpec, CapabilityId, EffectHeader, EffectIntent, EffectRejection, Generation,
    GrantId, MemoryId, ModelId, OperationId, RunId, RuntimeId, StreamingMode, TenantId,
};

/// Agent specification entity stored in the authoritative world.
#[derive(Clone, Component, Debug)]
pub struct AgentNode {
    /// Stable agent identity.
    pub id: AgentId,
    /// Immutable construction policy.
    pub spec: AgentSpec,
    /// Provider capability fact used by `OutputMode::Auto` policy.
    pub composes_native_output_with_tools: bool,
}

/// Executable capability category.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CapabilityKind {
    /// Context-free portable tool.
    Tool,
    /// Portable store or vector retrieval adapter.
    Store,
}

/// Versioned executable capability entity.
#[derive(Clone, Component, Deserialize, Serialize)]
pub struct CapabilityNode {
    /// Stable version identity.
    pub id: CapabilityId,
    /// Owning tenant.
    pub tenant_id: TenantId,
    /// Capability category.
    pub kind: CapabilityKind,
    /// Provider-facing definition when this is a tool.
    pub definition: Option<ToolDefinition>,
    /// Monotonic revision stored in immutable turn snapshots.
    pub revision: u64,
    /// Retired versions remain available to already-dispatched snapshots only.
    pub retired: bool,
}

impl fmt::Debug for CapabilityNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CapabilityNode")
            .field("id", &self.id)
            .field("tenant_id", &"<redacted>")
            .field("kind", &self.kind)
            .field("definition_configured", &self.definition.is_some())
            .field("revision", &self.revision)
            .field("retired", &self.retired)
            .finish()
    }
}

/// Explicit authorization relationship between an agent and a capability version.
#[derive(Clone, Component, Deserialize, Serialize)]
pub struct GrantNode {
    /// Stable grant identity.
    pub id: GrantId,
    /// Authorized agent.
    pub agent_id: AgentId,
    /// Exact capability version.
    pub capability_id: CapabilityId,
    /// Tenant security boundary shared by both endpoints.
    pub tenant_id: TenantId,
    /// Revoked grants cannot enter new turn snapshots.
    pub revoked: bool,
}

impl fmt::Debug for GrantNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrantNode")
            .field("id", &self.id)
            .field("agent_id", &self.agent_id)
            .field("capability_id", &self.capability_id)
            .field("tenant_id", &"<redacted>")
            .field("revoked", &self.revoked)
            .finish()
    }
}

/// Current lifecycle phase for one run.
#[derive(Clone, Copy, Component, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RunPhase {
    /// Waiting to dispatch or receive memory history.
    LoadingMemory,
    /// Ready to construct an owned model request.
    ReadyModel,
    /// Waiting for a model effect.
    WaitingModel,
    /// A model outcome is waiting for policy interpretation.
    EvaluatingModel,
    /// Waiting for one deterministic tool batch.
    WaitingTools,
    /// Waiting to append newly committed messages to memory.
    AppendingMemory,
    /// Terminal and observable until explicit retention cleanup.
    Terminal,
}

/// Stable run topology and progression component.
#[derive(Clone, Component)]
pub struct RunNode {
    /// Stable run identity.
    pub id: RunId,
    /// Owning agent.
    pub agent_id: AgentId,
    /// Selected model implementation.
    pub model_id: ModelId,
    /// Owning tenant.
    pub tenant_id: TenantId,
    /// Current generation.
    pub generation: Generation,
    /// Current lifecycle phase.
    pub phase: RunPhase,
    /// Model execution surface.
    pub streaming: StreamingMode,
    /// Runtime tick at creation.
    pub created_tick: u64,
}

impl fmt::Debug for RunNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunNode")
            .field("id", &self.id)
            .field("agent_id", &self.agent_id)
            .field("model_id", &self.model_id)
            .field("tenant_id", &"<redacted>")
            .field("generation", &self.generation)
            .field("phase", &self.phase)
            .field("streaming", &self.streaming)
            .field("created_tick", &self.created_tick)
            .finish()
    }
}

/// Canonical committed transcript; provisional and rejected content never enters it.
#[derive(Clone, Component, Default, Deserialize, Serialize)]
pub struct CanonicalTranscript {
    /// Ordered canonical messages.
    pub messages: Vec<Message>,
    /// First message belonging to this run rather than loaded memory.
    pub new_messages_start: usize,
}

impl fmt::Debug for CanonicalTranscript {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CanonicalTranscript")
            .field("message_count", &self.messages.len())
            .field("new_messages_start", &self.new_messages_start)
            .finish()
    }
}

/// One billed completed model operation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCallRecord {
    /// Stable operation identity.
    pub operation_id: OperationId,
    /// Usage reported for this operation.
    pub usage: Usage,
    /// Whether policy accepted the response into canonical state.
    pub accepted: bool,
}

/// Run-level usage and model-operation accounting.
#[derive(Clone, Component, Debug, Default, Deserialize, Serialize)]
pub struct RunAccounting {
    /// Number of model operations dispatched, including retries and continuations.
    pub model_calls_dispatched: usize,
    /// Aggregated usage for completed billed operations.
    pub usage: Usage,
    /// Exactly one record per accepted completion ingress.
    pub model_calls: Vec<ModelCallRecord>,
    /// Number of invalid effect messages rejected without mutation.
    pub rejected_effects: usize,
}

/// Stable terminal outcome category.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub enum TerminalReason {
    /// Canonical assistant response completed successfully.
    Completed,
    /// Explicit cancellation won terminal arbitration.
    Cancelled,
    /// ECS policy stopped the run.
    Stopped,
    /// Total model-call budget was exhausted.
    ModelCallBudgetExhausted,
    /// Provider, tool, memory, validation, or schedule failure.
    Failed {
        /// Stable failure category safe for snapshots and redacted diagnostics.
        code: String,
    },
}

/// Terminal state remains separate from observation and cleanup.
#[derive(Clone, Component, Debug, Deserialize, Serialize)]
pub struct TerminalState {
    /// Terminal outcome.
    pub reason: TerminalReason,
    /// Tick at which terminal arbitration completed.
    pub terminal_tick: u64,
}

/// Non-persisted operator diagnostic for a failed terminal run.
#[derive(Component, Debug)]
pub(crate) struct TerminalDiagnostic {
    pub(crate) message: String,
}

/// Non-persisted typed cause retained for local error inspection.
#[derive(Clone, Component)]
pub(crate) enum TerminalCause {
    Model(Arc<crate::ModelEffectError>),
    Memory(Arc<crate::MemoryEffectError>),
}

/// Observation marker required before retention cleanup may remove a run.
#[derive(Clone, Component, Debug, Deserialize, Serialize)]
pub struct TerminalObservation {
    /// Tick at which a handle observed the terminal result.
    pub observed_tick: u64,
}

/// Explicit cancellation request processed before ingress and dispatch.
#[derive(Clone, Copy, Component, Debug)]
pub(crate) struct CancellationRequest;

/// Exact capability authorization advertised to one model turn.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AdvertisedCapability {
    /// Exact capability version.
    pub capability_id: CapabilityId,
    /// Exact active grant.
    pub grant_id: GrantId,
    /// Capability revision.
    pub revision: u64,
    /// Provider-facing tool definition.
    pub definition: ToolDefinition,
}

/// Immutable tool and grant snapshot for the currently active model turn.
#[derive(Clone, Component, Debug, Default, Deserialize, Serialize)]
pub(crate) struct TurnCapabilitySnapshot {
    pub(crate) entries: Vec<AdvertisedCapability>,
    pub(crate) output_tool_name: Option<String>,
}

/// Active operation validation state.
#[derive(Clone, Debug)]
pub(crate) struct ActiveOperation {
    pub(crate) header: EffectHeader,
    pub(crate) kind: ActiveOperationKind,
    pub(crate) expected_tool_call_id: Option<String>,
    pub(crate) expected_tool_order: Option<usize>,
    pub(crate) completed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ActiveOperationKind {
    Model,
    Tool,
    MemoryLoad,
    MemoryAppend,
}

/// All operations that may still produce ingress for a run.
#[derive(Component, Debug, Default)]
pub(crate) struct ActiveOperations(pub(crate) BTreeMap<OperationId, ActiveOperation>);

/// Next provisional stream sequence expected for each active model operation.
#[derive(Component, Debug, Default)]
pub(crate) struct AcceptedDeltas(pub(crate) BTreeMap<OperationId, u64>);

/// One accepted model effect awaiting policy interpretation.
#[derive(Component)]
pub(crate) struct PendingModelOutcome {
    pub(crate) operation_id: OperationId,
    pub(crate) choice: Vec<AssistantContent>,
    pub(crate) message_id: Option<String>,
}

impl fmt::Debug for PendingModelOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingModelOutcome")
            .field("operation_id", &self.operation_id)
            .field("choice_count", &self.choice.len())
            .field("message_id_configured", &self.message_id.is_some())
            .finish()
    }
}

/// One tool invocation in a deterministic batch.
pub(crate) struct PlannedToolCall {
    pub(crate) call: ToolCall,
    pub(crate) capability_id: CapabilityId,
    pub(crate) grant_id: GrantId,
    pub(crate) revision: u64,
    pub(crate) order: usize,
    pub(crate) operation_id: Option<OperationId>,
    pub(crate) result: Option<ToolOutput>,
    pub(crate) suppressed: bool,
}

impl fmt::Debug for PlannedToolCall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PlannedToolCall")
            .field("tool_name", &self.call.function.name)
            .field("capability_id", &self.capability_id)
            .field("grant_id", &self.grant_id)
            .field("revision", &self.revision)
            .field("order", &self.order)
            .field("operation_id", &self.operation_id)
            .field("result_configured", &self.result.is_some())
            .field("suppressed", &self.suppressed)
            .field("arguments", &"<redacted>")
            .finish()
    }
}

/// Tool batch committed only after every non-suppressed sibling completes.
#[derive(Component)]
pub(crate) struct PendingToolBatch {
    pub(crate) assistant_content: Vec<AssistantContent>,
    pub(crate) message_id: Option<String>,
    pub(crate) calls: Vec<PlannedToolCall>,
}

impl fmt::Debug for PendingToolBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingToolBatch")
            .field("assistant_content_count", &self.assistant_content.len())
            .field("message_id_configured", &self.message_id.is_some())
            .field("call_count", &self.calls.len())
            .finish()
    }
}

/// Non-persisted local provider final for the latest accepted model operation.
#[derive(Component, Debug)]
pub(crate) struct RawFinalRecord {
    pub(crate) operation_id: OperationId,
    pub(crate) raw: crate::effects::ErasedRawFinal,
}

/// Memory lifecycle and committed-only append bookkeeping.
#[derive(Component)]
pub(crate) struct MemoryProgress {
    pub(crate) memory_id: MemoryId,
    pub(crate) conversation_id: String,
    pub(crate) loaded: bool,
    pub(crate) appended: bool,
}

impl fmt::Debug for MemoryProgress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryProgress")
            .field("memory_id", &self.memory_id)
            .field("conversation_id", &"<redacted>")
            .field("loaded", &self.loaded)
            .field("appended", &self.appended)
            .finish()
    }
}

/// Structured output recovery state held as ECS data.
#[derive(Component)]
pub(crate) struct StructuredOutputState {
    pub(crate) schema: schemars::Schema,
    pub(crate) policy: crate::StructuredOutputPolicy,
    pub(crate) resolved_mode: crate::OutputMode,
    pub(crate) output_tool_name: Option<String>,
    pub(crate) retries: usize,
    pub(crate) value: Option<serde_json::Value>,
}

impl fmt::Debug for StructuredOutputState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StructuredOutputState")
            .field("policy", &self.policy)
            .field("resolved_mode", &self.resolved_mode)
            .field("output_tool_configured", &self.output_tool_name.is_some())
            .field("retries", &self.retries)
            .field("value_configured", &self.value.is_some())
            .field("schema", &"<redacted>")
            .finish()
    }
}

/// Per-run request-only corrective feedback excluded from committed history.
#[derive(Component, Default)]
pub(crate) struct RecoveryFeedback(pub(crate) Vec<String>);

impl fmt::Debug for RecoveryFeedback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RecoveryFeedback")
            .field("message_count", &self.0.len())
            .finish()
    }
}

/// Number of response retry decisions already taken.
#[derive(Component, Debug, Default)]
pub(crate) struct ResponseRetryState(pub(crate) usize);

/// Number of invalid-call retry decisions already taken.
#[derive(Component, Debug, Default)]
pub(crate) struct InvalidToolRetryState(pub(crate) usize);

/// Public lifecycle events emitted from authoritative ECS transitions.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub enum RunEvent {
    /// An owned model operation was dispatched.
    ModelDispatched(OperationId),
    /// Provider streaming content is provisional and not canonical history.
    Provisional {
        /// Model operation that emitted the delta.
        operation_id: OperationId,
        /// Provider-normalized delta.
        delta: Box<crate::ProvisionalDelta>,
    },
    /// A complete provider final became available after stream success.
    ProviderFinal {
        /// Model operation that produced the final.
        operation_id: OperationId,
        /// Concrete local type name without response content.
        provider_type: &'static str,
    },
    /// A portable tool operation was dispatched.
    ToolDispatched {
        /// Tool operation identity.
        operation_id: OperationId,
        /// Model tool-call identity.
        tool_call_id: String,
    },
    /// A tool result committed in stable model-call order.
    ToolCommitted {
        /// Tool operation identity.
        operation_id: OperationId,
        /// Model tool-call identity.
        tool_call_id: String,
    },
    /// A response was rejected and rolled back before a fresh model request.
    ResponseRetried,
    /// An invalid tool call was suppressed without execution.
    ToolSuppressed {
        /// Model tool-call identity.
        tool_call_id: String,
    },
    /// The run reached an externally observable terminal state.
    Terminal(TerminalReason),
    /// Invalid ingress was rejected without mutating authoritative state.
    EffectRejected(crate::EffectRejectionReason),
}

impl fmt::Debug for RunEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelDispatched(operation_id) => formatter
                .debug_tuple("ModelDispatched")
                .field(operation_id)
                .finish(),
            Self::Provisional {
                operation_id,
                delta,
            } => formatter
                .debug_struct("Provisional")
                .field("operation_id", operation_id)
                .field("delta_kind", &delta.kind())
                .finish(),
            Self::ProviderFinal {
                operation_id,
                provider_type,
            } => formatter
                .debug_struct("ProviderFinal")
                .field("operation_id", operation_id)
                .field("provider_type", provider_type)
                .finish(),
            Self::ToolDispatched {
                operation_id,
                tool_call_id,
            } => formatter
                .debug_struct("ToolDispatched")
                .field("operation_id", operation_id)
                .field("tool_call_id", tool_call_id)
                .finish(),
            Self::ToolCommitted {
                operation_id,
                tool_call_id,
            } => formatter
                .debug_struct("ToolCommitted")
                .field("operation_id", operation_id)
                .field("tool_call_id", tool_call_id)
                .finish(),
            Self::ResponseRetried => formatter.write_str("ResponseRetried"),
            Self::ToolSuppressed { tool_call_id } => formatter
                .debug_struct("ToolSuppressed")
                .field("tool_call_id", tool_call_id)
                .finish(),
            Self::Terminal(reason) => formatter.debug_tuple("Terminal").field(reason).finish(),
            Self::EffectRejected(reason) => formatter
                .debug_tuple("EffectRejected")
                .field(reason)
                .finish(),
        }
    }
}

/// Bounded event log and lag-aware broadcast channel for one run.
#[derive(Component, Debug)]
pub(crate) struct RunEvents {
    pub(crate) events: VecDeque<RunEvent>,
    capacity: usize,
    sender: tokio::sync::broadcast::Sender<RunEvent>,
}

impl RunEvents {
    pub(crate) fn new(capacity: usize) -> Self {
        let (sender, _) = tokio::sync::broadcast::channel(capacity);
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            sender,
        }
    }

    pub(crate) fn subscribe(&self) -> tokio::sync::broadcast::Receiver<RunEvent> {
        self.sender.subscribe()
    }

    pub(crate) fn publish(&mut self, event: RunEvent) {
        if self.events.len() == self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event.clone());
        let _ = self.sender.send(event);
    }
}

/// Resource populated by the driver before each schedule pass.
#[derive(Resource, Default)]
pub(crate) struct PendingIngress(pub(crate) Vec<crate::EffectIngress>);

/// Owned effects emitted by dispatch systems and drained by the driver.
#[derive(Resource, Default)]
pub(crate) struct PendingEffects(pub(crate) Vec<EffectIntent>);

/// Rejected ingress audit records.
#[derive(Resource, Default)]
pub(crate) struct RejectionLog(pub(crate) Vec<EffectRejection>);

/// Monotonic schedule tick.
#[derive(Resource, Default)]
pub(crate) struct RuntimeTick(pub(crate) u64);

/// Monotonic count bumped by authoritative mutations to one run.
#[derive(Component, Debug, Default)]
pub(crate) struct RunProgress(pub(crate) u64);

/// Marker for a dispatchable run waiting for room in the bounded effect queue.
///
/// This is authoritative waiting state rather than synthetic progress: drivers
/// can await global effect-capacity changes without treating another run's load
/// as quiescence or consuming the livelock-pass budget.
#[derive(Component, Debug, Default)]
pub(crate) struct EffectQueueWait;

/// Non-persisted parent span for one runtime-owned run lifecycle.
#[derive(Component)]
pub(crate) struct RunTelemetrySpan(pub(crate) tracing::Span);

/// Retired, unreferenced capability implementations ready for registry cleanup.
#[derive(Resource, Default)]
pub(crate) struct CapabilitiesToDrop(pub(crate) Vec<CapabilityId>);

/// Runtime identity resource used by ingress validation.
#[derive(Resource)]
pub(crate) struct RuntimeIdentity(pub(crate) RuntimeId);

/// Stable lookup from domain IDs to Bevy entities.
#[derive(Resource, Default)]
pub(crate) struct TopologyIndex {
    pub(crate) agents: BTreeMap<AgentId, Entity>,
    pub(crate) runs: BTreeMap<RunId, Entity>,
    pub(crate) capabilities: BTreeMap<CapabilityId, Entity>,
    pub(crate) grants: BTreeMap<GrantId, Entity>,
}

/// Capability versions referenced by active turn snapshots.
#[derive(Resource, Default)]
pub(crate) struct CapabilityReferences(pub(crate) BTreeMap<CapabilityId, BTreeSet<RunId>>);

#[cfg(test)]
mod debug_tests {
    use rig_core::message::ToolFunction;

    use super::*;

    #[test]
    fn private_ecs_state_debug_is_content_redacted() {
        let secret = "secret-ecs-content-734563";
        let call = ToolCall::new(
            "call".to_string(),
            ToolFunction::new(
                "safe_tool".to_string(),
                serde_json::json!({"token": secret}),
            ),
        );
        let pending_model = PendingModelOutcome {
            operation_id: OperationId::new(),
            choice: vec![AssistantContent::text(secret)],
            message_id: Some(secret.to_string()),
        };
        let planned = PlannedToolCall {
            call,
            capability_id: CapabilityId::new(),
            grant_id: GrantId::new(),
            revision: 1,
            order: 0,
            operation_id: None,
            result: Some(ToolOutput::text(secret)),
            suppressed: false,
        };
        let pending_batch = PendingToolBatch {
            assistant_content: vec![AssistantContent::text(secret)],
            message_id: Some(secret.to_string()),
            calls: vec![planned],
        };
        let memory = MemoryProgress {
            memory_id: MemoryId::new(),
            conversation_id: secret.to_string(),
            loaded: false,
            appended: false,
        };
        let structured = StructuredOutputState {
            schema: schemars::schema_for!(String),
            policy: crate::StructuredOutputPolicy::default(),
            resolved_mode: crate::OutputMode::Native,
            output_tool_name: Some(secret.to_string()),
            retries: 0,
            value: Some(serde_json::json!({"secret": secret})),
        };
        let recovery = RecoveryFeedback(vec![secret.to_string()]);

        for debug in [
            format!("{pending_model:?}"),
            format!("{pending_batch:?}"),
            format!("{memory:?}"),
            format!("{structured:?}"),
            format!("{recovery:?}"),
        ] {
            assert!(!debug.contains(secret));
        }
    }
}
