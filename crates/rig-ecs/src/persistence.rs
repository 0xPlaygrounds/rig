//! Protected, versioned stable-domain snapshots with exact implementation rebinding.

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{AssistantContent, CompletionModel, Message},
    memory::ConversationMemory,
    message::UserContent,
    tool::{PortableDynamicTool, PortableTool},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    AgentId, AgentNode, AgentSpec, BindingIdentity, CapabilityId, CapabilityNode, Generation,
    GrantNode, MemoryId, ModelId, RunAccounting, RunId, RunPhase, RuntimeConfig, RuntimeError,
    SnapshotError, TenantId,
    components::{
        AcceptedDeltas, ActiveOperations, AdvertisedCapability, CanonicalTranscript,
        CapabilityReferences, EffectQueueWait, InvalidToolRetryState, MemoryProgress,
        RecoveryFeedback, ResponseRetryState, RunEvents, RunNode, RunProgress, RunTelemetrySpan,
        RuntimeTick, StructuredOutputState, TerminalObservation, TerminalState, TopologyIndex,
        TurnCapabilitySnapshot,
    },
    runtime::{
        DynamicToolBinding, LocalRuntime, MemoryBinding, ModelBinding, ToolBinding,
        TypedMemoryBinding, TypedModelBinding, TypedToolBinding,
    },
    schedule::checked_usage_sum,
};

const SNAPSHOT_VERSION: u32 = 2;

#[derive(Debug, thiserror::Error)]
pub(crate) enum CanonicalTranscriptError {
    #[error("consecutive assistant messages are not canonical")]
    ConsecutiveAssistant,
    #[error("assistant tool-call identities must be unique within a turn")]
    DuplicateToolCall,
    #[error("tool results must immediately and exactly answer the preceding assistant calls")]
    InvalidToolPairing,
    #[error("tool results cannot appear without a preceding assistant tool call")]
    OrphanToolResult,
    #[error("assistant tool calls cannot remain unanswered")]
    OrphanToolCall,
}

/// Resumable canonical-transcript checker.
///
/// One validator observes a message stream incrementally; the run's live state
/// is kept in [`TranscriptValidation`] so commits validate only the appended
/// tail while restore, snapshot, and memory-load acceptance scan whole slices.
#[derive(Clone, Debug, Default)]
pub(crate) struct TranscriptValidator {
    previous_assistant: bool,
    pending_tool_calls: Option<BTreeSet<String>>,
}

impl TranscriptValidator {
    pub(crate) fn observe(&mut self, message: &Message) -> Result<(), CanonicalTranscriptError> {
        if let Some(expected) = self.pending_tool_calls.take() {
            let Message::User { content } = message else {
                return Err(CanonicalTranscriptError::InvalidToolPairing);
            };
            let results = content
                .iter()
                .filter_map(|item| match item {
                    UserContent::ToolResult(result) => Some(result.id.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let unique_results = results.iter().cloned().collect::<BTreeSet<_>>();
            // Pairing is keyed strictly on identities — each expected id
            // exactly once, nothing unexpected — but the pairing message may
            // also carry ordinary user content: rig's message model allows
            // mixed content and classic-runtime histories rely on it.
            if results.len() != unique_results.len() || unique_results != expected {
                return Err(CanonicalTranscriptError::InvalidToolPairing);
            }
            self.previous_assistant = false;
            return Ok(());
        }
        match message {
            Message::Assistant { content, .. } => {
                if self.previous_assistant {
                    return Err(CanonicalTranscriptError::ConsecutiveAssistant);
                }
                self.previous_assistant = true;
                let calls = content
                    .iter()
                    .filter_map(|item| match item {
                        AssistantContent::ToolCall(call) => Some(call.id.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let unique_calls = calls.iter().cloned().collect::<BTreeSet<_>>();
                if calls.len() != unique_calls.len() {
                    return Err(CanonicalTranscriptError::DuplicateToolCall);
                }
                if !unique_calls.is_empty() {
                    self.pending_tool_calls = Some(unique_calls);
                }
            }
            Message::User { content } => {
                self.previous_assistant = false;
                if content
                    .iter()
                    .any(|item| matches!(item, UserContent::ToolResult(_)))
                {
                    return Err(CanonicalTranscriptError::OrphanToolResult);
                }
            }
            Message::System { .. } => self.previous_assistant = false,
        }
        Ok(())
    }

    /// Check the pairing-completeness required at every rest point (each
    /// committed turn, snapshot, and restore).
    pub(crate) fn finish(&self) -> Result<(), CanonicalTranscriptError> {
        if self.pending_tool_calls.is_some() {
            return Err(CanonicalTranscriptError::OrphanToolCall);
        }
        Ok(())
    }

    /// Validate a whole slice and return the resulting resumable state.
    pub(crate) fn over(
        messages: &[Message],
    ) -> Result<TranscriptValidator, CanonicalTranscriptError> {
        let mut validator = TranscriptValidator::default();
        for message in messages {
            validator.observe(message)?;
        }
        Ok(validator)
    }
}

/// Live transcript-validator state for one run; never serialized — restore
/// reseeds it by scanning the restored transcript.
#[derive(Component, Clone, Debug, Default)]
pub(crate) struct TranscriptValidation(pub(crate) TranscriptValidator);

pub(crate) fn validate_canonical_transcript(
    messages: &[Message],
) -> Result<(), CanonicalTranscriptError> {
    TranscriptValidator::over(messages)?.finish()
}

/// Content retained in a protected snapshot.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum SnapshotContentPolicy {
    /// Persist topology and binding metadata only; omit prompts, transcripts,
    /// memory keys, recovery feedback, structured values, and provider params.
    #[default]
    MetadataOnly,
    /// Explicitly persist canonical run and agent configuration content.
    CanonicalRunState,
}

/// Caller-supplied authenticated protection for a serialized stable snapshot payload.
pub trait SnapshotProtector: Send + Sync {
    /// Stable algorithm/key identifier recorded outside the protected payload.
    fn protector_id(&self) -> &str;

    /// Encrypt and authenticate plaintext snapshot bytes.
    fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, SnapshotError>;

    /// Authenticate and decrypt protected snapshot bytes.
    fn unprotect(&self, protected: &[u8]) -> Result<Vec<u8>, SnapshotError>;
}

/// Opaque protected snapshot envelope; its payload is never interpreted without a protector.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProtectedSnapshot {
    /// Stable schema version used for early compatibility checks.
    pub version: u32,
    /// Protection implementation/key identifier.
    pub protector_id: String,
    /// Protected serialized stable-domain payload.
    pub payload: Vec<u8>,
    /// SHA-256 digest of the protected bytes for corruption detection before unprotection.
    pub protected_digest: [u8; 32],
}

/// Exact implementation bindings required to restore stable domain records.
#[derive(Default)]
pub struct RebindRegistry {
    pub(crate) models: HashMap<ModelId, Arc<dyn ModelBinding>>,
    pub(crate) tools: HashMap<CapabilityId, Arc<dyn ToolBinding>>,
    pub(crate) memories: HashMap<MemoryId, Arc<dyn MemoryBinding>>,
    pub(crate) model_tenants: HashMap<ModelId, TenantId>,
    pub(crate) memory_tenants: HashMap<MemoryId, TenantId>,
    pub(crate) model_binding_identities: HashMap<ModelId, BindingIdentity>,
    pub(crate) tool_binding_identities: HashMap<CapabilityId, BindingIdentity>,
    pub(crate) tool_kinds: HashMap<CapabilityId, crate::CapabilityKind>,
    pub(crate) memory_binding_identities: HashMap<MemoryId, BindingIdentity>,
}

impl RebindRegistry {
    /// Construct an empty exact-rebinding registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind one exact stable model identity to a concrete implementation.
    pub fn bind_model<M>(
        &mut self,
        id: ModelId,
        tenant_id: TenantId,
        identity: BindingIdentity,
        model: M,
    ) -> &mut Self
    where
        M: CompletionModel + Send + Sync + 'static,
        M::Response: Any + Send + Sync,
        M::StreamingResponse: Any + Send + Sync,
    {
        self.models.insert(id, Arc::new(TypedModelBinding(model)));
        self.model_tenants.insert(id, tenant_id);
        self.model_binding_identities.insert(id, identity);
        self
    }

    /// Bind one exact capability version to a concrete portable tool.
    pub fn bind_tool<T>(
        &mut self,
        id: CapabilityId,
        identity: BindingIdentity,
        tool: T,
    ) -> &mut Self
    where
        T: PortableTool + Send + Sync + 'static,
        T::Args: Send + Sync + 'static,
        T::Output: Send + 'static,
    {
        self.tools
            .insert(id, Arc::new(TypedToolBinding(Arc::new(tool))));
        self.tool_binding_identities.insert(id, identity);
        self.tool_kinds.insert(id, crate::CapabilityKind::Tool);
        self
    }

    /// Bind one exact vector-store capability to a concrete portable index.
    pub fn bind_vector_store<I>(
        &mut self,
        id: CapabilityId,
        identity: BindingIdentity,
        index: I,
    ) -> &mut Self
    where
        I: rig_core::vector_store::VectorStoreIndex + PortableTool + Send + Sync + 'static,
        I::Args: Send + Sync + 'static,
        I::Output: Send + 'static,
    {
        self.tools
            .insert(id, Arc::new(TypedToolBinding(Arc::new(index))));
        self.tool_binding_identities.insert(id, identity);
        self.tool_kinds.insert(id, crate::CapabilityKind::Store);
        self
    }

    /// Bind one exact capability version to a context-free dynamic tool.
    pub fn bind_dynamic_tool(
        &mut self,
        id: CapabilityId,
        identity: BindingIdentity,
        tool: PortableDynamicTool,
    ) -> &mut Self {
        self.tools.insert(id, Arc::new(DynamicToolBinding(tool)));
        self.tool_binding_identities.insert(id, identity);
        self.tool_kinds.insert(id, crate::CapabilityKind::Tool);
        self
    }

    /// Bind one exact memory identity to a concrete backend.
    pub fn bind_memory<M>(
        &mut self,
        id: MemoryId,
        tenant_id: TenantId,
        identity: BindingIdentity,
        memory: M,
    ) -> &mut Self
    where
        M: ConversationMemory + Send + Sync + 'static,
    {
        self.memories
            .insert(id, Arc::new(TypedMemoryBinding(Arc::new(memory))));
        self.memory_tenants.insert(id, tenant_id);
        self.memory_binding_identities.insert(id, identity);
        self
    }
}

#[derive(Deserialize, Serialize)]
struct DomainSnapshot {
    version: u32,
    content_policy: SnapshotContentPolicy,
    runtime_tick: u64,
    agents: Vec<AgentRecord>,
    capabilities: Vec<CapabilityRecord>,
    grants: Vec<GrantNode>,
    runs: Vec<RunRecord>,
    model_bindings: Vec<BindingRecord<ModelId>>,
    memory_bindings: Vec<BindingRecord<MemoryId>>,
}

#[derive(Deserialize, Serialize)]
struct BindingRecord<I> {
    id: I,
    tenant_id: TenantId,
    binding_identity: BindingIdentity,
}

#[derive(Deserialize, Serialize)]
struct AgentRecord {
    id: AgentId,
    spec: AgentSpec,
    composes_native_output_with_tools: bool,
}

#[derive(Deserialize, Serialize)]
struct CapabilityRecord {
    node: CapabilityNode,
    binding_identity: Option<BindingIdentity>,
}

#[derive(Deserialize, Serialize)]
struct RunRecord {
    id: RunId,
    agent_id: AgentId,
    model_id: ModelId,
    tenant_id: TenantId,
    generation: Generation,
    phase: RunPhase,
    streaming: crate::StreamingMode,
    created_tick: u64,
    transcript: CanonicalTranscript,
    accounting: RunAccounting,
    terminal: Option<TerminalState>,
    observation: Option<TerminalObservation>,
    memory: Option<MemoryRecord>,
    structured: Option<StructuredRecord>,
    feedback: Vec<String>,
    response_retries: usize,
    invalid_tool_retries: usize,
    turn_snapshot: Option<TurnSnapshotRecord>,
}

#[derive(Deserialize, Serialize)]
struct MemoryRecord {
    memory_id: MemoryId,
    conversation_id: String,
    loaded: bool,
    appended: bool,
}

#[derive(Deserialize, Serialize)]
struct StructuredRecord {
    schema: schemars::Schema,
    policy: crate::StructuredOutputPolicy,
    resolved_mode: crate::OutputMode,
    output_tool_name: Option<String>,
    retries: usize,
    value: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize)]
struct TurnSnapshotRecord {
    entries: Vec<AdvertisedCapability>,
    output_tool_name: Option<String>,
}

impl LocalRuntime {
    /// Serialize metadata-only stable domain records with caller-supplied protection.
    pub fn protected_snapshot(
        &self,
        protector: &dyn SnapshotProtector,
    ) -> Result<ProtectedSnapshot, SnapshotError> {
        self.protected_snapshot_with_policy(protector, SnapshotContentPolicy::MetadataOnly)
    }

    /// Serialize stable domain records under an explicit content-retention policy.
    pub fn protected_snapshot_with_policy(
        &self,
        protector: &dyn SnapshotProtector,
        content_policy: SnapshotContentPolicy,
    ) -> Result<ProtectedSnapshot, SnapshotError> {
        let mut runs = Vec::new();
        let run_entities = if matches!(content_policy, SnapshotContentPolicy::CanonicalRunState) {
            self.world
                .resource::<TopologyIndex>()
                .runs
                .values()
                .copied()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        for entity in run_entities {
            let run = self.world.get::<RunNode>(entity).ok_or_else(|| {
                SnapshotError::InvalidRelationship("missing run node".to_string())
            })?;
            if !matches!(run.phase, RunPhase::ReadyModel | RunPhase::Terminal)
                || self
                    .world
                    .get::<ActiveOperations>(entity)
                    .is_some_and(|operations| operations.0.values().any(|op| !op.completed))
                || self.world.entity(entity).contains::<EffectQueueWait>()
            {
                return Err(SnapshotError::NotQuiescent);
            }
            let transcript = self
                .world
                .get::<CanonicalTranscript>(entity)
                .cloned()
                .ok_or_else(|| {
                    SnapshotError::InvalidRelationship("missing transcript".to_string())
                })?;
            // Persist-side twin of the restore-side check: a snapshot that
            // restore would refuse must fail here, at save time.
            if let Err(error) = validate_canonical_transcript(&transcript.messages) {
                return Err(SnapshotError::InvalidRelationship(format!(
                    "run `{}` has a non-canonical transcript: {error}",
                    run.id
                )));
            }
            let accounting = self
                .world
                .get::<RunAccounting>(entity)
                .cloned()
                .ok_or_else(|| {
                    SnapshotError::InvalidRelationship("missing accounting".to_string())
                })?;
            let memory = self
                .world
                .get::<MemoryProgress>(entity)
                .map(|memory| MemoryRecord {
                    memory_id: memory.memory_id,
                    conversation_id: memory.conversation_id.clone(),
                    loaded: memory.loaded,
                    appended: memory.appended,
                });
            let structured = self
                .world
                .get::<StructuredOutputState>(entity)
                .map(|state| StructuredRecord {
                    schema: state.schema.clone(),
                    policy: state.policy.clone(),
                    resolved_mode: state.resolved_mode,
                    output_tool_name: state.output_tool_name.clone(),
                    retries: state.retries,
                    value: state.value.clone(),
                });
            runs.push(RunRecord {
                id: run.id,
                agent_id: run.agent_id,
                model_id: run.model_id,
                tenant_id: run.tenant_id,
                generation: run.generation,
                phase: run.phase,
                streaming: run.streaming,
                created_tick: run.created_tick,
                transcript,
                accounting,
                terminal: self.world.get::<TerminalState>(entity).cloned(),
                observation: self.world.get::<TerminalObservation>(entity).cloned(),
                memory,
                structured,
                feedback: self
                    .world
                    .get::<RecoveryFeedback>(entity)
                    .map(|feedback| feedback.0.clone())
                    .unwrap_or_default(),
                response_retries: self
                    .world
                    .get::<ResponseRetryState>(entity)
                    .map_or(0, |state| state.0),
                invalid_tool_retries: self
                    .world
                    .get::<InvalidToolRetryState>(entity)
                    .map_or(0, |state| state.0),
                turn_snapshot: self
                    .world
                    .get::<TurnCapabilitySnapshot>(entity)
                    .map(|snapshot| TurnSnapshotRecord {
                        entries: snapshot.entries.clone(),
                        output_tool_name: snapshot.output_tool_name.clone(),
                    }),
            });
        }
        runs.sort_by_key(|run| run.id);

        let mut agents = self
            .world
            .resource::<TopologyIndex>()
            .agents
            .values()
            .filter_map(|entity| self.world.get::<AgentNode>(*entity))
            .map(|agent| {
                if agent.spec.additional_params.is_some() {
                    return Err(SnapshotError::NonPersistableProviderParameters(agent.id));
                }
                let mut spec = agent.spec.clone();
                spec.additional_params = None;
                if matches!(content_policy, SnapshotContentPolicy::MetadataOnly) {
                    if spec.memory_id.is_some() {
                        return Err(SnapshotError::MetadataOnlyMemory(agent.id));
                    }
                    if spec.preamble.is_some() {
                        return Err(SnapshotError::MetadataOnlyPreamble(agent.id));
                    }
                    spec.name = None;
                    spec.preamble = None;
                    spec.memory_id = None;
                    spec.conversation_id = None;
                    spec.record_telemetry_content = false;
                }
                Ok(AgentRecord {
                    id: agent.id,
                    spec,
                    composes_native_output_with_tools: agent.composes_native_output_with_tools,
                })
            })
            .collect::<Result<Vec<_>, SnapshotError>>()?;
        agents.sort_by_key(|agent| agent.id);

        let mut capabilities = self
            .world
            .resource::<TopologyIndex>()
            .capabilities
            .values()
            .filter_map(|entity| self.world.get::<CapabilityNode>(*entity))
            .map(|node| {
                let binding_identity = if self.tools.contains_key(&node.id) {
                    Some(*self.tool_binding_identities.get(&node.id).ok_or_else(|| {
                        SnapshotError::MissingBindingIdentity {
                            id: node.id.to_string(),
                            kind: "capability",
                        }
                    })?)
                } else {
                    None
                };
                Ok(CapabilityRecord {
                    node: node.clone(),
                    binding_identity,
                })
            })
            .collect::<Result<Vec<_>, SnapshotError>>()?;
        capabilities.sort_by_key(|record| record.node.id);

        let mut grants = self
            .world
            .resource::<TopologyIndex>()
            .grants
            .values()
            .filter_map(|entity| self.world.get::<GrantNode>(*entity).cloned())
            .collect::<Vec<_>>();
        grants.sort_by_key(|grant| grant.id);

        let referenced_models = agents
            .iter()
            .map(|agent| agent.spec.model_id)
            .chain(runs.iter().map(|run| run.model_id))
            .collect::<BTreeSet<_>>();
        let referenced_memories = agents
            .iter()
            .filter_map(|agent| agent.spec.memory_id)
            .chain(
                runs.iter()
                    .filter_map(|run| run.memory.as_ref().map(|memory| memory.memory_id)),
            )
            .collect::<BTreeSet<_>>();
        let mut model_bindings = referenced_models
            .iter()
            .map(|id| {
                if !self.models.contains_key(id) {
                    return Err(SnapshotError::MissingModel(*id));
                }
                Ok(BindingRecord {
                    id: *id,
                    tenant_id: *self.model_tenants.get(id).ok_or_else(|| {
                        SnapshotError::InvalidRelationship(format!(
                            "model binding `{id}` has no tenant owner"
                        ))
                    })?,
                    binding_identity: *self.model_binding_identities.get(id).ok_or_else(|| {
                        SnapshotError::MissingBindingIdentity {
                            id: id.to_string(),
                            kind: "model",
                        }
                    })?,
                })
            })
            .collect::<Result<Vec<_>, SnapshotError>>()?;
        model_bindings.sort_by_key(|binding| binding.id);
        let mut memory_bindings = referenced_memories
            .iter()
            .map(|id| {
                if !self.memories.contains_key(id) {
                    return Err(SnapshotError::MissingMemory(*id));
                }
                Ok(BindingRecord {
                    id: *id,
                    tenant_id: *self.memory_tenants.get(id).ok_or_else(|| {
                        SnapshotError::InvalidRelationship(format!(
                            "memory binding `{id}` has no tenant owner"
                        ))
                    })?,
                    binding_identity: *self.memory_binding_identities.get(id).ok_or_else(|| {
                        SnapshotError::MissingBindingIdentity {
                            id: id.to_string(),
                            kind: "memory",
                        }
                    })?,
                })
            })
            .collect::<Result<Vec<_>, SnapshotError>>()?;
        memory_bindings.sort_by_key(|binding| binding.id);

        let plaintext = serde_json::to_vec(&DomainSnapshot {
            version: SNAPSHOT_VERSION,
            content_policy,
            runtime_tick: self.world.resource::<RuntimeTick>().0,
            agents,
            capabilities,
            grants,
            runs,
            model_bindings,
            memory_bindings,
        })?;
        let payload = protector.protect(&plaintext)?;
        let protected_digest = Sha256::digest(&payload).into();
        Ok(ProtectedSnapshot {
            version: SNAPSHOT_VERSION,
            protector_id: protector.protector_id().to_string(),
            payload,
            protected_digest,
        })
    }

    /// Restore stable ECS domain state into a new runtime identity after exact rebinding.
    pub fn restore(
        config: RuntimeConfig,
        protected: &ProtectedSnapshot,
        protector: &dyn SnapshotProtector,
        mut bindings: RebindRegistry,
    ) -> Result<Self, RuntimeError> {
        if protected.version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion(protected.version).into());
        }
        if protected.protector_id != protector.protector_id() {
            return Err(SnapshotError::Unprotect(
                "snapshot protector identifier does not match".to_string(),
            )
            .into());
        }
        let digest: [u8; 32] = Sha256::digest(&protected.payload).into();
        if digest != protected.protected_digest {
            return Err(SnapshotError::Integrity.into());
        }
        let plaintext = protector.unprotect(&protected.payload)?;
        let snapshot: DomainSnapshot =
            serde_json::from_slice(&plaintext).map_err(SnapshotError::Serialization)?;
        if snapshot.version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion(snapshot.version).into());
        }
        validate_snapshot(&snapshot, &bindings)?;
        let mut runtime = Self::with_config(config)?;
        runtime.world.resource_mut::<RuntimeTick>().0 = snapshot.runtime_tick;
        runtime.models = std::mem::take(&mut bindings.models);
        runtime.tools = std::mem::take(&mut bindings.tools);
        runtime.memories = std::mem::take(&mut bindings.memories);
        runtime.model_tenants = std::mem::take(&mut bindings.model_tenants);
        runtime.memory_tenants = std::mem::take(&mut bindings.memory_tenants);
        runtime.model_binding_identities = std::mem::take(&mut bindings.model_binding_identities);
        runtime.tool_binding_identities = std::mem::take(&mut bindings.tool_binding_identities);
        runtime.memory_binding_identities = std::mem::take(&mut bindings.memory_binding_identities);

        for agent in snapshot.agents {
            let entity = runtime
                .world
                .spawn(AgentNode {
                    id: agent.id,
                    spec: agent.spec,
                    composes_native_output_with_tools: agent.composes_native_output_with_tools,
                })
                .id();
            runtime
                .world
                .resource_mut::<TopologyIndex>()
                .agents
                .insert(agent.id, entity);
        }
        for capability in snapshot.capabilities {
            let id = capability.node.id;
            let entity = runtime.world.spawn(capability.node).id();
            runtime
                .world
                .resource_mut::<TopologyIndex>()
                .capabilities
                .insert(id, entity);
        }
        for grant in snapshot.grants {
            let id = grant.id;
            let entity = runtime.world.spawn(grant).id();
            runtime
                .world
                .resource_mut::<TopologyIndex>()
                .grants
                .insert(id, entity);
        }
        for run in snapshot.runs {
            let run_id = run.id;
            if !runtime
                .world
                .resource::<TopologyIndex>()
                .agents
                .contains_key(&run.agent_id)
            {
                return Err(SnapshotError::InvalidRelationship(format!(
                    "run `{}` references missing agent `{}`",
                    run.id, run.agent_id
                ))
                .into());
            }
            let generation = run.generation.next();
            let run_span = tracing::info_span!(
                target: "rig::ecs",
                "rig.ecs.run",
                run.id = %run.id,
                run.generation = generation.0,
                run.streaming = ?run.streaming,
                run.tenant = "<redacted>",
                restored = true,
            );
            let validation = TranscriptValidator::over(&run.transcript.messages)
                .map(TranscriptValidation)
                .map_err(|error| {
                    SnapshotError::InvalidRelationship(format!(
                        "run `{}` has a non-canonical transcript: {error}",
                        run.id
                    ))
                })?;
            let mut entity = runtime.world.spawn((
                RunNode {
                    id: run.id,
                    agent_id: run.agent_id,
                    model_id: run.model_id,
                    tenant_id: run.tenant_id,
                    generation,
                    phase: run.phase,
                    streaming: run.streaming,
                    created_tick: run.created_tick,
                },
                run.transcript,
                validation,
                run.accounting,
                ActiveOperations::default(),
                AcceptedDeltas::default(),
                RecoveryFeedback(run.feedback),
                ResponseRetryState(run.response_retries),
                InvalidToolRetryState(run.invalid_tool_retries),
                RunEvents::new(runtime.config.event_capacity),
                RunProgress::default(),
                RunTelemetrySpan(run_span),
            ));
            if let Some(mut terminal) = run.terminal {
                if run.observation.is_none() {
                    // Restoring renews the retention lease for runs nobody has
                    // observed yet — the snapshot was taken to preserve them.
                    terminal.terminal_tick = snapshot.runtime_tick;
                }
                entity.insert(terminal);
            }
            if let Some(observation) = run.observation {
                entity.insert(observation);
            }
            if let Some(memory) = run.memory {
                entity.insert(MemoryProgress {
                    memory_id: memory.memory_id,
                    conversation_id: memory.conversation_id,
                    loaded: memory.loaded,
                    appended: memory.appended,
                });
            }
            if let Some(structured) = run.structured {
                entity.insert(StructuredOutputState {
                    schema: structured.schema,
                    policy: structured.policy,
                    resolved_mode: structured.resolved_mode,
                    output_tool_name: structured.output_tool_name,
                    retries: structured.retries,
                    value: structured.value,
                });
            }
            let referenced_capabilities = run
                .turn_snapshot
                .as_ref()
                .map(|snapshot| {
                    snapshot
                        .entries
                        .iter()
                        .map(|entry| entry.capability_id)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if let Some(snapshot) = run.turn_snapshot {
                entity.insert(TurnCapabilitySnapshot {
                    entries: snapshot.entries,
                    output_tool_name: snapshot.output_tool_name,
                });
            }
            let run_entity = entity.id();
            for capability_id in referenced_capabilities {
                runtime
                    .world
                    .resource_mut::<CapabilityReferences>()
                    .0
                    .entry(capability_id)
                    .or_default()
                    .insert(run_id);
            }
            runtime
                .world
                .resource_mut::<TopologyIndex>()
                .runs
                .insert(run_id, run_entity);
        }
        Ok(runtime)
    }
}

fn validate_snapshot(
    snapshot: &DomainSnapshot,
    bindings: &RebindRegistry,
) -> Result<(), SnapshotError> {
    if snapshot
        .agents
        .iter()
        .any(|agent| agent.spec.additional_params.is_some())
    {
        return Err(SnapshotError::InvalidRelationship(
            "snapshot contains prohibited provider parameters".to_string(),
        ));
    }
    if matches!(snapshot.content_policy, SnapshotContentPolicy::MetadataOnly)
        && (!snapshot.runs.is_empty()
            || snapshot.agents.iter().any(|agent| {
                agent.spec.name.is_some()
                    || agent.spec.preamble.is_some()
                    || agent.spec.memory_id.is_some()
                    || agent.spec.conversation_id.is_some()
                    || agent.spec.additional_params.is_some()
                    || agent.spec.record_telemetry_content
            }))
    {
        return Err(SnapshotError::InvalidRelationship(
            "metadata-only snapshot contains prohibited run or agent content".to_string(),
        ));
    }
    let mut model_records = BTreeMap::new();
    for expected in &snapshot.model_bindings {
        if model_records.insert(expected.id, expected).is_some() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "duplicate model binding record `{}`",
                expected.id
            )));
        }
        if !bindings.models.contains_key(&expected.id) {
            return Err(SnapshotError::MissingModel(expected.id));
        }
        if bindings.model_tenants.get(&expected.id) != Some(&expected.tenant_id)
            || bindings.model_binding_identities.get(&expected.id)
                != Some(&expected.binding_identity)
        {
            return Err(SnapshotError::BindingMismatch {
                id: expected.id.to_string(),
                kind: "model",
            });
        }
    }

    let mut memory_records = BTreeMap::new();
    for expected in &snapshot.memory_bindings {
        if memory_records.insert(expected.id, expected).is_some() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "duplicate memory binding record `{}`",
                expected.id
            )));
        }
        if !bindings.memories.contains_key(&expected.id) {
            return Err(SnapshotError::MissingMemory(expected.id));
        }
        if bindings.memory_tenants.get(&expected.id) != Some(&expected.tenant_id)
            || bindings.memory_binding_identities.get(&expected.id)
                != Some(&expected.binding_identity)
        {
            return Err(SnapshotError::BindingMismatch {
                id: expected.id.to_string(),
                kind: "memory",
            });
        }
    }

    let mut agents = BTreeMap::new();
    for agent in &snapshot.agents {
        if agents.insert(agent.id, agent).is_some() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "duplicate agent `{}`",
                agent.id
            )));
        }
        if !model_records.contains_key(&agent.spec.model_id) {
            return Err(SnapshotError::MissingModel(agent.spec.model_id));
        }
        if model_records
            .get(&agent.spec.model_id)
            .is_none_or(|record| record.tenant_id != agent.spec.tenant_id)
        {
            return Err(SnapshotError::InvalidRelationship(format!(
                "agent `{}` crosses its model binding tenant boundary",
                agent.id
            )));
        }
        let model = bindings
            .models
            .get(&agent.spec.model_id)
            .ok_or(SnapshotError::MissingModel(agent.spec.model_id))?;
        if model.composes_native_output_with_tools() != agent.composes_native_output_with_tools {
            return Err(SnapshotError::BindingMismatch {
                id: agent.spec.model_id.to_string(),
                kind: "model capability",
            });
        }
        if let Some(memory_id) = agent.spec.memory_id {
            if !memory_records.contains_key(&memory_id) {
                return Err(SnapshotError::MissingMemory(memory_id));
            }
            if memory_records
                .get(&memory_id)
                .is_none_or(|record| record.tenant_id != agent.spec.tenant_id)
            {
                return Err(SnapshotError::InvalidRelationship(format!(
                    "agent `{}` crosses its memory binding tenant boundary",
                    agent.id
                )));
            }
            if agent.spec.conversation_id.is_none() {
                return Err(SnapshotError::InvalidRelationship(format!(
                    "agent `{}` configures memory without a conversation identifier",
                    agent.id
                )));
            }
        } else if agent.spec.conversation_id.is_some() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "agent `{}` configures a conversation identifier without memory",
                agent.id
            )));
        }
    }

    let mut capabilities = BTreeMap::new();
    for capability in &snapshot.capabilities {
        if capabilities
            .insert(capability.node.id, capability)
            .is_some()
        {
            return Err(SnapshotError::InvalidRelationship(format!(
                "duplicate capability `{}`",
                capability.node.id
            )));
        }
        if let Some(expected) = capability.binding_identity {
            let actual = bindings
                .tools
                .get(&capability.node.id)
                .ok_or(SnapshotError::MissingCapability(capability.node.id))?;
            if bindings.tool_binding_identities.get(&capability.node.id) != Some(&expected)
                || bindings.tool_kinds.get(&capability.node.id) != Some(&capability.node.kind)
                || capability.node.definition.as_ref() != Some(&actual.definition())
            {
                return Err(SnapshotError::BindingMismatch {
                    id: capability.node.id.to_string(),
                    kind: "capability",
                });
            }
        } else if matches!(
            capability.node.kind,
            crate::CapabilityKind::Tool | crate::CapabilityKind::Store
        ) {
            return Err(SnapshotError::MissingCapability(capability.node.id));
        }
    }

    let mut grants = BTreeMap::new();
    for grant in &snapshot.grants {
        if grants.insert(grant.id, grant).is_some() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "duplicate grant `{}`",
                grant.id
            )));
        }
        let agent = agents.get(&grant.agent_id).ok_or_else(|| {
            SnapshotError::InvalidRelationship(format!(
                "grant `{}` references missing agent `{}`",
                grant.id, grant.agent_id
            ))
        })?;
        let capability = capabilities.get(&grant.capability_id).ok_or_else(|| {
            SnapshotError::InvalidRelationship(format!(
                "grant `{}` references missing capability `{}`",
                grant.id, grant.capability_id
            ))
        })?;
        if grant.tenant_id != agent.spec.tenant_id || grant.tenant_id != capability.node.tenant_id {
            return Err(SnapshotError::InvalidRelationship(format!(
                "grant `{}` crosses a tenant ownership boundary",
                grant.id
            )));
        }
    }

    let mut active_tool_names = BTreeSet::new();
    for grant in grants.values().copied().filter(|grant| !grant.revoked) {
        let Some(capability) = capabilities.get(&grant.capability_id) else {
            continue;
        };
        if capability.node.retired {
            continue;
        }
        let Some(definition) = capability.node.definition.as_ref() else {
            continue;
        };
        if !active_tool_names.insert((grant.agent_id, definition.name.clone())) {
            return Err(SnapshotError::InvalidRelationship(format!(
                "agent `{}` has duplicate active capability name `{}`",
                grant.agent_id, definition.name
            )));
        }
    }

    let mut runs = BTreeSet::new();
    for run in &snapshot.runs {
        if !runs.insert(run.id) {
            return Err(SnapshotError::InvalidRelationship(format!(
                "duplicate run `{}`",
                run.id
            )));
        }
        if !matches!(run.phase, RunPhase::ReadyModel | RunPhase::Terminal) {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` is not at a quiescent persistence phase",
                run.id
            )));
        }
        if run.transcript.new_messages_start > run.transcript.messages.len() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` has an out-of-range new-message boundary",
                run.id
            )));
        }
        if let Err(error) = validate_canonical_transcript(&run.transcript.messages) {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` has a non-canonical transcript: {error}",
                run.id
            )));
        }
        if run.accounting.model_calls_dispatched < run.accounting.model_calls.len() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` records more completed model calls than dispatches",
                run.id
            )));
        }
        let mut operation_ids = BTreeSet::new();
        let mut recorded_usage = rig_core::completion::Usage::new();
        for call in &run.accounting.model_calls {
            if !operation_ids.insert(call.operation_id) {
                return Err(SnapshotError::InvalidRelationship(format!(
                    "run `{}` has duplicate model-call operation identities",
                    run.id
                )));
            }
            recorded_usage = checked_usage_sum(recorded_usage, call.usage).ok_or_else(|| {
                SnapshotError::InvalidRelationship(format!(
                    "run `{}` has overflowing model-call usage",
                    run.id
                ))
            })?;
        }
        if recorded_usage != run.accounting.usage {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` aggregate usage does not match its model-call records",
                run.id
            )));
        }
        let agent = agents.get(&run.agent_id).ok_or_else(|| {
            SnapshotError::InvalidRelationship(format!(
                "run `{}` references missing agent `{}`",
                run.id, run.agent_id
            ))
        })?;
        if run.model_id != agent.spec.model_id || !model_records.contains_key(&run.model_id) {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` does not use its agent's rebound model",
                run.id
            )));
        }
        if run.tenant_id != agent.spec.tenant_id {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` crosses its agent's tenant boundary",
                run.id
            )));
        }
        if run.accounting.model_calls_dispatched > agent.spec.max_model_calls {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` exceeds its agent's model-call budget",
                run.id
            )));
        }
        match (&run.structured, &agent.spec.structured_output) {
            (Some(structured), Some((schema, policy)))
                if structured.schema == *schema && structured.policy == *policy =>
            {
                if structured.retries > policy.max_retries {
                    return Err(SnapshotError::InvalidRelationship(format!(
                        "run `{}` exceeds its structured-output retry policy",
                        run.id
                    )));
                }
                let output_tool_consistent =
                    matches!(structured.resolved_mode, crate::OutputMode::Tool)
                        == structured.output_tool_name.is_some();
                let resolution_consistent = match policy.mode {
                    crate::OutputMode::Auto => matches!(
                        structured.resolved_mode,
                        crate::OutputMode::Auto
                            | crate::OutputMode::Native
                            | crate::OutputMode::Tool
                    ),
                    crate::OutputMode::Native => matches!(
                        structured.resolved_mode,
                        crate::OutputMode::Auto | crate::OutputMode::Native
                    ),
                    crate::OutputMode::Prompted => matches!(
                        structured.resolved_mode,
                        crate::OutputMode::Auto | crate::OutputMode::Prompted
                    ),
                    crate::OutputMode::Tool => matches!(
                        structured.resolved_mode,
                        crate::OutputMode::Auto
                            | crate::OutputMode::Native
                            | crate::OutputMode::Tool
                    ),
                };
                if !output_tool_consistent || !resolution_consistent {
                    return Err(SnapshotError::InvalidRelationship(format!(
                        "run `{}` has inconsistent structured-output resolution state",
                        run.id
                    )));
                }
                if let Some(value) = &structured.value {
                    let schema_valid = jsonschema::validator_for(structured.schema.as_value())
                        .is_ok_and(|validator| validator.is_valid(value));
                    let completed = run.terminal.as_ref().is_some_and(|terminal| {
                        matches!(terminal.reason, crate::TerminalReason::Completed)
                    });
                    if !matches!(run.phase, RunPhase::Terminal) || !completed || !schema_valid {
                        return Err(SnapshotError::InvalidRelationship(format!(
                            "run `{}` has an invalid structured-output value",
                            run.id
                        )));
                    }
                }
            }
            (None, None) => {}
            _ => {
                return Err(SnapshotError::InvalidRelationship(format!(
                    "run `{}` does not preserve its agent's structured-output policy",
                    run.id
                )));
            }
        }
        let response_retry_limit = match agent.spec.response_retry_policy {
            crate::ResponseRetryPolicy::Accept => 0,
            crate::ResponseRetryPolicy::RejectEmpty { max_retries } => max_retries,
        };
        if run.response_retries > response_retry_limit {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` exceeds its response retry policy",
                run.id
            )));
        }
        let invalid_tool_retry_limit = match agent.spec.invalid_tool_policy {
            crate::InvalidToolPolicy::Retry { max_retries } => max_retries,
            crate::InvalidToolPolicy::Fail
            | crate::InvalidToolPolicy::Repair
            | crate::InvalidToolPolicy::Skip
            | crate::InvalidToolPolicy::Stop => 0,
        };
        if run.invalid_tool_retries > invalid_tool_retry_limit {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` exceeds its invalid-tool retry policy",
                run.id
            )));
        }
        if matches!(run.phase, RunPhase::Terminal) != run.terminal.is_some() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` has inconsistent terminal state",
                run.id
            )));
        }
        if run.observation.is_some() && run.terminal.is_none() {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` was observed before becoming terminal",
                run.id
            )));
        }
        if run
            .terminal
            .as_ref()
            .is_some_and(|terminal| terminal.terminal_tick < run.created_tick)
            || run.observation.as_ref().is_some_and(|observation| {
                run.terminal
                    .as_ref()
                    .is_some_and(|terminal| observation.observed_tick < terminal.terminal_tick)
            })
        {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` has out-of-order lifecycle ticks",
                run.id
            )));
        }
        if run.created_tick > snapshot.runtime_tick
            || run
                .terminal
                .as_ref()
                .is_some_and(|terminal| terminal.terminal_tick > snapshot.runtime_tick)
            || run
                .observation
                .as_ref()
                .is_some_and(|observation| observation.observed_tick > snapshot.runtime_tick)
        {
            return Err(SnapshotError::InvalidRelationship(format!(
                "run `{}` has lifecycle ticks ahead of the snapshot runtime tick",
                run.id
            )));
        }
        match (
            &run.memory,
            agent.spec.memory_id,
            &agent.spec.conversation_id,
        ) {
            (Some(memory), Some(memory_id), Some(conversation_id))
                if memory.memory_id == memory_id
                    && memory.conversation_id == *conversation_id
                    && memory_records.contains_key(&memory.memory_id) =>
            {
                // A run may legitimately fail or be cancelled before its
                // memory load succeeds; only those terminals may be unloaded.
                let must_have_loaded = memory.appended
                    || run.terminal.as_ref().is_none_or(|terminal| {
                        matches!(terminal.reason, crate::TerminalReason::Completed)
                    });
                if (!memory.loaded && must_have_loaded)
                    || (!matches!(run.phase, RunPhase::Terminal) && memory.appended)
                    || (memory.appended
                        && run.terminal.as_ref().is_none_or(|terminal| {
                            !matches!(terminal.reason, crate::TerminalReason::Completed)
                        }))
                {
                    return Err(SnapshotError::InvalidRelationship(format!(
                        "run `{}` has inconsistent conversation memory lifecycle flags",
                        run.id
                    )));
                }
            }
            (None, None, None) => {}
            _ => {
                return Err(SnapshotError::InvalidRelationship(format!(
                    "run `{}` has inconsistent conversation memory topology",
                    run.id
                )));
            }
        }
        if let Some(turn) = &run.turn_snapshot {
            let mut entries = BTreeSet::new();
            for entry in &turn.entries {
                if !entries.insert((entry.capability_id, entry.grant_id, entry.revision)) {
                    return Err(SnapshotError::InvalidRelationship(format!(
                        "run `{}` has a duplicate advertised capability",
                        run.id
                    )));
                }
                let capability = capabilities.get(&entry.capability_id).ok_or_else(|| {
                    SnapshotError::InvalidRelationship(format!(
                        "run `{}` snapshots missing capability `{}`",
                        run.id, entry.capability_id
                    ))
                })?;
                let grant = grants.get(&entry.grant_id).ok_or_else(|| {
                    SnapshotError::InvalidRelationship(format!(
                        "run `{}` snapshots missing grant `{}`",
                        run.id, entry.grant_id
                    ))
                })?;
                if capability.node.tenant_id != run.tenant_id
                    || capability.node.revision != entry.revision
                    || capability.node.definition.as_ref() != Some(&entry.definition)
                    || grant.agent_id != run.agent_id
                    || grant.capability_id != entry.capability_id
                    || grant.tenant_id != run.tenant_id
                {
                    return Err(SnapshotError::InvalidRelationship(format!(
                        "run `{}` has a mismatched advertised capability snapshot",
                        run.id
                    )));
                }
            }
        }
    }
    Ok(())
}
