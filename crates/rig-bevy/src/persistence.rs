//! Versioned stable-domain snapshots and explicit implementation rebinding.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::world::World;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    adapters::{AgentModelBindings, MemoryBindings},
    components::{
        AgentNode, CallBudget, CapabilitySnapshot, CommittedTranscript, ModelBinding,
        ModelOperation, ProgressState, ResponseRecovery, RetainUntil, RetentionWindow, RunNode,
        RunPhase, StoreBinding, StoreOperation, TerminalReason, TerminalState, ToolCallNode,
        UsageLedger,
    },
    policy::{AgentPolicy, ToolCatalog, ToolGrant},
    runtime::BevyRuntime,
    schedule::initialize_world,
    topology::{
        AgentId, CapabilityId, EffectIdentity, Generation, OperationId, OwnedByAgent, RunId,
        StoreOperationId, TenantId, ToolCallId, WorldId, allocate_after_restore,
    },
};
use rig_core::{
    OneOrMany,
    completion::AssistantContent,
    memory::ConversationMemory,
    message::{Message, ToolResult as MessageToolResult, ToolResultContent, UserContent},
    tool::{DynamicTool, Tool},
};

/// Current stable domain snapshot schema.
pub const SNAPSHOT_VERSION: u32 = 4;

/// Whether canonical message content is included in a snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SnapshotContent {
    /// Omit prompt, response, tool-result, and memory content.
    #[default]
    Omit,
    /// Persist canonical messages as plaintext after an explicit host opt-in.
    ///
    /// The host is responsible for protecting the resulting snapshot at rest.
    Plaintext,
}

/// Snapshot content policy. The safe default excludes canonical messages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SnapshotOptions {
    /// Message-content persistence policy.
    pub content: SnapshotContent,
}

/// Explicit serializable runtime snapshot. It contains no Bevy entities,
/// tasks, clients, channels, implementation pointers, or provider finals.
///
/// The default snapshot omits preambles, provider-specific request parameters,
/// retry feedback, memory conversation keys, and canonical messages. Plaintext
/// content is available only through an explicit [`SnapshotOptions`] opt-in.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    /// Snapshot schema version.
    pub version: u32,
    /// Source world used only for diagnostics; restoration allocates a new world.
    pub source_world: WorldId,
    /// Whether sensitive canonical content was explicitly persisted.
    pub content: SnapshotContent,
    /// Stable agent records sorted by ID.
    pub agents: Vec<AgentRecord>,
    /// Stable run records sorted by ID.
    pub runs: Vec<RunRecord>,
    /// Immutable advertised capability snapshots sorted by stable ID.
    pub capabilities: Vec<CapabilitySnapshotRecord>,
    /// Model-operation domain records sorted by stable ID.
    pub model_operations: Vec<ModelOperationRecord>,
    /// Tool-call domain records sorted by stable ID.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Store-operation domain records sorted by stable ID.
    pub store_operations: Vec<StoreOperationRecord>,
}

impl std::fmt::Debug for RuntimeSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeSnapshot")
            .field("version", &self.version)
            .field("source_world", &self.source_world)
            .field("content", &self.content)
            .field("agents", &self.agents.len())
            .field("runs", &self.runs.len())
            .field("capabilities", &self.capabilities.len())
            .field("model_operations", &self.model_operations.len())
            .field("tool_calls", &self.tool_calls.len())
            .field("store_operations", &self.store_operations.len())
            .finish()
    }
}

/// Stable agent persistence record.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRecord {
    /// Stable ID.
    pub id: AgentId,
    /// Tenant boundary.
    pub tenant: TenantId,
    /// Optional public name.
    pub name: Option<String>,
    /// Optional system instructions. Hosts should avoid secrets in preambles.
    pub preamble: Option<String>,
    /// Explicit model implementation binding name.
    pub model_binding: String,
    /// ECS-native policy copied without implementation pointers.
    pub policy: AgentPolicy,
    /// Exact tenant-scoped tool grants for this agent.
    pub tool_grants: Vec<ToolGrant>,
    /// Optional explicit memory/store binding.
    pub store_binding: Option<StoreBinding>,
}

impl std::fmt::Debug for AgentRecord {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AgentRecord")
            .field("id", &self.id)
            .field("name", &self.name.as_ref().map(|_| "<redacted>"))
            .field("preamble", &self.preamble.as_ref().map(|_| "<redacted>"))
            .field("model_binding", &self.model_binding)
            .field("policy", &self.policy)
            .field("tool_grants", &self.tool_grants.len())
            .field(
                "store_binding",
                &self.store_binding.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Stable run persistence record.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    /// Stable ID.
    pub id: RunId,
    /// Stable owning agent.
    pub agent: AgentId,
    /// Tenant boundary.
    pub tenant: TenantId,
    /// Generation at snapshot time.
    pub generation: Generation,
    /// Current lifecycle phase.
    pub phase: RunPhase,
    /// Total model-call budget.
    pub budget: CallBudget,
    /// Canonical committed transcript only.
    pub transcript: CommittedTranscript,
    /// Committed usage only.
    pub usage: UsageLedger,
    /// Progress/livelock state.
    pub progress: ProgressState,
    /// Optional externally observable terminal fact.
    pub terminal: Option<TerminalState>,
    /// Retention duration applied to future terminal commits.
    pub retention_window: RetentionWindow,
    /// Current terminal observation deadline.
    pub retain_until: RetainUntil,
}

impl std::fmt::Debug for RunRecord {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunRecord")
            .field("id", &self.id)
            .field("agent", &self.agent)
            .field("generation", &self.generation)
            .field("phase", &self.phase)
            .field("budget", &self.budget)
            .field(
                "transcript",
                &format_args!("<redacted:{}>", self.transcript.0.len()),
            )
            .field("usage", &self.usage)
            .field("terminal", &self.terminal.as_ref().map(|_| "<redacted>"))
            .field("retention_window", &self.retention_window)
            .field("retain_until", &self.retain_until)
            .finish()
    }
}

/// Persisted immutable capability snapshot without a Bevy entity.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySnapshotRecord {
    /// Stable snapshot identity.
    pub snapshot: CapabilitySnapshot,
}

/// Persisted model-operation state. Owned requests and provider finals are excluded.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOperationRecord {
    /// Stable operation identity.
    pub id: OperationId,
    /// Full dispatch correlation.
    pub effect: EffectIdentity,
    /// Whether canonical ingress already committed.
    pub committed: bool,
    /// Whether this operation can no longer commit.
    pub retired: bool,
}

/// Persisted tool-call state without arguments, outputs, or implementation pointers.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Stable call identity.
    pub id: ToolCallId,
    /// Exact model-operation correlation that produced the call.
    pub effect: EffectIdentity,
    /// Owning run.
    pub run: RunId,
    /// Immutable authorizing capability snapshot.
    pub capability: CapabilityId,
    /// Provider-facing tool name.
    pub name: String,
    /// Exact implementation revision.
    pub revision: u64,
    /// Model-call order.
    pub ordinal: usize,
    /// Whether policy prevented execution.
    pub suppressed: bool,
    /// Whether the result committed.
    pub committed: bool,
}

/// Persisted store-operation state without messages or backend pointers.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreOperationRecord {
    /// Stable store-operation identity.
    pub id: StoreOperationId,
    /// Full dispatch correlation.
    pub effect: EffectIdentity,
    /// Whether canonical ingress already committed.
    pub committed: bool,
    /// Whether this operation can no longer commit.
    pub retired: bool,
}

/// Concrete implementations explicitly made available during restoration.
#[derive(Clone, Default)]
pub struct BindingManifest {
    models: AgentModelBindings,
    tools: ToolCatalog,
    memories: MemoryBindings,
    additional_params: BTreeMap<AgentId, serde_json::Value>,
}

impl std::fmt::Debug for BindingManifest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BindingManifest")
            .field("model_agents", &self.models.ids().collect::<Vec<_>>())
            .field("tools", &"<bound implementations>")
            .field("memories", &"<bound implementations>")
            .field("additional_params", &self.additional_params.len())
            .finish()
    }
}

impl BindingManifest {
    /// Bind one concrete model implementation to its exact persisted agent ID.
    ///
    /// Binding by identity keeps agents with the same model type or diagnostic
    /// binding name from being collapsed onto one host implementation.
    pub fn bind_model<M>(&mut self, agent: AgentId, model: M) -> Result<(), PersistenceError>
    where
        M: rig_core::completion::CompletionModel + 'static,
    {
        if self.models.bind(agent, model) {
            Ok(())
        } else {
            Err(PersistenceError::DuplicateBinding {
                kind: "model",
                name: format!("agent {}", agent.0),
                revision: 0,
            })
        }
    }

    /// Bind an exact portable typed-tool revision for one tenant.
    pub fn bind_tool<T>(
        &mut self,
        tenant: TenantId,
        revision: u64,
        tool: T,
    ) -> Result<(), PersistenceError>
    where
        T: Tool + 'static,
    {
        if self.tools.bind(tenant, revision, tool) {
            Ok(())
        } else {
            Err(PersistenceError::DuplicateBinding {
                kind: "tool",
                name: T::NAME.to_string(),
                revision,
            })
        }
    }

    /// Bind an exact portable dynamic-tool revision for one tenant.
    pub fn bind_dynamic_tool(
        &mut self,
        tenant: TenantId,
        revision: u64,
        tool: DynamicTool,
    ) -> Result<(), PersistenceError> {
        let name = tool.name().to_string();
        if self.tools.bind_dynamic(tenant, revision, tool) {
            Ok(())
        } else {
            Err(PersistenceError::DuplicateBinding {
                kind: "tool",
                name,
                revision,
            })
        }
    }

    /// Bind a concrete memory/store implementation by its persisted name.
    pub fn bind_memory<M>(&mut self, name: impl Into<String>, memory: M)
    where
        M: ConversationMemory + 'static,
    {
        self.memories.bind(name, memory);
    }

    /// Supply non-persisted provider-specific request parameters for one agent.
    ///
    /// Snapshots always omit these values. A restored run that requires them
    /// must receive them again through this explicit host-owned binding.
    pub fn bind_additional_params(&mut self, agent: AgentId, params: serde_json::Value) {
        self.additional_params.insert(agent, params);
    }

    fn model_bound(&self, agent: AgentId) -> bool {
        self.models.contains(agent)
    }

    fn tool_bound(&self, tenant: TenantId, name: &str, revision: u64) -> bool {
        self.tools.contains(tenant, name, revision)
    }

    fn memory_bound(&self, name: &str) -> bool {
        self.memories.contains(name)
    }

    fn additional_params(&self, agent: AgentId) -> Option<&serde_json::Value> {
        self.additional_params.get(&agent)
    }
}

/// Typed snapshot or restoration failure.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum PersistenceError {
    /// Snapshot schema is not supported.
    #[error("unsupported runtime snapshot version {found}; expected {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
    /// A stable identifier appears more than once.
    #[error("duplicate stable {kind} identifier {id}")]
    DuplicateId { kind: &'static str, id: u64 },
    /// A run references an unknown agent.
    #[error("run {run} references missing agent {agent}")]
    MissingAgent { run: u64, agent: u64 },
    /// A persisted child record references an unknown run.
    #[error("{kind} {id} references missing run {run}")]
    MissingRun {
        kind: &'static str,
        id: u64,
        run: u64,
    },
    /// A tool call references an unknown capability snapshot.
    #[error("tool call {call} references missing capability {capability}")]
    MissingCapability { call: u64, capability: u64 },
    /// Stable records disagree about ownership, tenant, generation, or correlation.
    #[error("{kind} {id} violates snapshot integrity: {detail}")]
    Integrity {
        /// Stable record kind.
        kind: &'static str,
        /// Stable record ID.
        id: u64,
        /// Redacted invariant description.
        detail: &'static str,
    },
    /// A manifest attempts to bind the same stable implementation twice.
    #[error("duplicate {kind} binding `{name}` revision {revision}")]
    DuplicateBinding {
        /// Binding kind.
        kind: &'static str,
        /// Stable implementation name.
        name: String,
        /// Exact revision, or zero for unversioned implementations.
        revision: u64,
    },
    /// A required implementation was not explicitly rebound.
    #[error("missing model implementation binding for agent {agent} (`{name}`)")]
    MissingModelBinding { agent: u64, name: String },
    /// An exact portable tool implementation was not explicitly rebound.
    #[error("missing tool implementation binding `{name}` revision {revision}")]
    MissingToolBinding { name: String, revision: u64 },
    /// A memory/store implementation was not explicitly rebound.
    #[error("missing store implementation binding `{0}`")]
    MissingStoreBinding(String),
    /// JSON encoding/decoding failed.
    #[error("snapshot JSON error: {0}")]
    Json(String),
    /// A restored stable or world ID cannot advance without wrapping.
    #[error("snapshot exhausts the stable identity space")]
    IdentityExhausted,
    /// A drivable restore requires an explicit plaintext-content opt-in.
    #[error("snapshot omits canonical content required for a drivable restore")]
    ContentOmitted,
}

/// Capture a deterministic stable-domain snapshot.
pub fn snapshot(world: &mut World, source_world: WorldId) -> RuntimeSnapshot {
    snapshot_with_options(world, source_world, SnapshotOptions::default())
}

/// Capture a deterministic stable-domain snapshot with explicit content policy.
pub fn snapshot_with_options(
    world: &mut World,
    source_world: WorldId,
    options: SnapshotOptions,
) -> RuntimeSnapshot {
    let mut grants_by_agent = std::collections::BTreeMap::<AgentId, Vec<ToolGrant>>::new();
    for (owner, grant) in world.query::<(&OwnedByAgent, &ToolGrant)>().iter(world) {
        grants_by_agent
            .entry(owner.0)
            .or_default()
            .push(grant.clone());
    }
    for grants in grants_by_agent.values_mut() {
        grants.sort_by(|left, right| {
            (&left.name, left.revision, left.tenant).cmp(&(
                &right.name,
                right.revision,
                right.tenant,
            ))
        });
    }
    let mut agents = world
        .query::<(
            &AgentNode,
            &ModelBinding,
            &AgentPolicy,
            Option<&StoreBinding>,
        )>()
        .iter(world)
        .map(|(agent, binding, policy, store_binding)| {
            let mut policy = policy.clone();
            // Provider parameters may contain authentication material, so they
            // are excluded even when canonical content is explicitly enabled.
            policy.additional_params = None;
            let mut store_binding = store_binding.cloned();
            let (name, preamble) = if options.content == SnapshotContent::Plaintext {
                (agent.name.clone(), agent.preamble.clone())
            } else {
                policy.response_retry.feedback.clear();
                policy.invalid_tool = match policy.invalid_tool {
                    crate::policy::InvalidToolPolicy::Fail => {
                        crate::policy::InvalidToolPolicy::Fail
                    }
                    crate::policy::InvalidToolPolicy::Retry { .. } => {
                        crate::policy::InvalidToolPolicy::Retry {
                            feedback: String::new(),
                        }
                    }
                    crate::policy::InvalidToolPolicy::Repair { name, .. } => {
                        crate::policy::InvalidToolPolicy::Repair {
                            name,
                            arguments: String::new(),
                        }
                    }
                    crate::policy::InvalidToolPolicy::Skip { .. } => {
                        crate::policy::InvalidToolPolicy::Skip {
                            reason: String::new(),
                        }
                    }
                    crate::policy::InvalidToolPolicy::Stop { .. } => {
                        crate::policy::InvalidToolPolicy::Stop {
                            reason: String::new(),
                        }
                    }
                };
                if let Some(store) = &mut store_binding {
                    store.conversation.clear();
                }
                (None, None)
            };
            AgentRecord {
                id: agent.id,
                tenant: agent.tenant,
                name,
                preamble,
                model_binding: binding.0.clone(),
                policy,
                tool_grants: grants_by_agent.remove(&agent.id).unwrap_or_default(),
                store_binding,
            }
        })
        .collect::<Vec<_>>();
    agents.sort_by_key(|agent| agent.id);

    let mut runs = world
        .query::<(
            &RunNode,
            &CallBudget,
            &CommittedTranscript,
            &UsageLedger,
            &ProgressState,
            Option<&TerminalState>,
            &RetentionWindow,
            &RetainUntil,
        )>()
        .iter(world)
        .map(
            |(
                run,
                budget,
                transcript,
                usage,
                progress,
                terminal,
                retention_window,
                retain_until,
            )| RunRecord {
                id: run.id,
                agent: run.agent,
                tenant: run.tenant,
                generation: run.generation,
                phase: run.phase,
                budget: *budget,
                transcript: if options.content == SnapshotContent::Plaintext {
                    transcript.clone()
                } else {
                    CommittedTranscript::default()
                },
                usage: *usage,
                progress: *progress,
                terminal: terminal.map(sanitized_terminal),
                retention_window: *retention_window,
                retain_until: *retain_until,
            },
        )
        .collect::<Vec<_>>();
    runs.sort_by_key(|run| run.id);

    let mut capabilities = world
        .query::<&CapabilitySnapshot>()
        .iter(world)
        .cloned()
        .map(|snapshot| CapabilitySnapshotRecord { snapshot })
        .collect::<Vec<_>>();
    capabilities.sort_by_key(|record| record.snapshot.id);

    let mut model_operations = world
        .query::<&ModelOperation>()
        .iter(world)
        .map(|operation| ModelOperationRecord {
            id: operation.id,
            effect: operation.effect,
            committed: operation.committed,
            retired: operation.retired,
        })
        .collect::<Vec<_>>();
    model_operations.sort_by_key(|operation| operation.id);

    let mut tool_calls = world
        .query::<&ToolCallNode>()
        .iter(world)
        .map(|call| ToolCallRecord {
            id: call.id,
            effect: call.effect,
            run: call.run,
            capability: call.capability,
            name: call.name.clone(),
            revision: call.revision,
            ordinal: call.ordinal,
            suppressed: call.suppressed,
            committed: call.committed,
        })
        .collect::<Vec<_>>();
    tool_calls.sort_by_key(|call| call.id);

    let mut store_operations = world
        .query::<&StoreOperation>()
        .iter(world)
        .map(|operation| StoreOperationRecord {
            id: operation.id,
            effect: operation.effect,
            committed: operation.committed,
            retired: operation.retired,
        })
        .collect::<Vec<_>>();
    store_operations.sort_by_key(|operation| operation.id);

    RuntimeSnapshot {
        version: SNAPSHOT_VERSION,
        source_world,
        content: options.content,
        agents,
        runs,
        capabilities,
        model_operations,
        tool_calls,
        store_operations,
    }
}

fn sanitized_terminal(terminal: &TerminalState) -> TerminalState {
    let reason = match &terminal.reason {
        TerminalReason::Completed => TerminalReason::Completed,
        TerminalReason::BudgetExhausted => TerminalReason::BudgetExhausted,
        TerminalReason::Cancelled(_) => TerminalReason::Cancelled("cancelled".into()),
        TerminalReason::Stopped(_) => TerminalReason::Stopped("stopped by policy".into()),
        TerminalReason::ProviderFailure(_) => {
            TerminalReason::ProviderFailure("provider completion failed".into())
        }
        TerminalReason::ToolFailure(_) => {
            TerminalReason::ToolFailure("tool processing failed".into())
        }
        TerminalReason::OutputFailure(_) => {
            TerminalReason::OutputFailure("structured output failed".into())
        }
        TerminalReason::StoreFailure(_) => {
            TerminalReason::StoreFailure("store effect failed".into())
        }
        TerminalReason::Livelock => TerminalReason::Livelock,
    };
    TerminalState {
        reason,
        committed_tick: terminal.committed_tick,
    }
}

/// Serialize deterministically after stable-ID sorting.
pub fn to_json(snapshot: &RuntimeSnapshot) -> Result<String, PersistenceError> {
    serde_json::to_string(snapshot).map_err(|error| PersistenceError::Json(error.to_string()))
}

/// Parse and validate a snapshot before restoration.
pub fn from_json(input: &str) -> Result<RuntimeSnapshot, PersistenceError> {
    let snapshot: RuntimeSnapshot =
        serde_json::from_str(input).map_err(|error| PersistenceError::Json(error.to_string()))?;
    validate(&snapshot)?;
    Ok(snapshot)
}

/// Validate schema and stable topology integrity.
pub fn validate(snapshot: &RuntimeSnapshot) -> Result<(), PersistenceError> {
    if snapshot.version != SNAPSHOT_VERSION {
        return Err(PersistenceError::UnsupportedVersion {
            found: snapshot.version,
            expected: SNAPSHOT_VERSION,
        });
    }
    let mut agents = BTreeMap::new();
    for agent in &snapshot.agents {
        if agents.insert(agent.id, agent).is_some() {
            return Err(PersistenceError::DuplicateId {
                kind: "agent",
                id: agent.id.0,
            });
        }
        if agent.policy.agent != agent.id {
            return Err(PersistenceError::Integrity {
                kind: "agent",
                id: agent.id.0,
                detail: "policy owner does not match the agent",
            });
        }
        if agent.policy.max_tool_concurrency == 0
            || agent
                .tool_grants
                .iter()
                .any(|grant| grant.tenant != agent.tenant)
        {
            return Err(PersistenceError::Integrity {
                kind: "agent",
                id: agent.id.0,
                detail: "policy or grant crosses an ownership boundary",
            });
        }
        if let Some(schema) = &agent.policy.output_schema
            && serde_json::from_value::<schemars::Schema>(schema.clone()).is_err()
        {
            return Err(PersistenceError::Integrity {
                kind: "agent",
                id: agent.id.0,
                detail: "output schema is malformed",
            });
        }
    }
    let mut runs = BTreeMap::new();
    for run in &snapshot.runs {
        if runs.insert(run.id, run).is_some() {
            return Err(PersistenceError::DuplicateId {
                kind: "run",
                id: run.id.0,
            });
        }
        let Some(agent) = agents.get(&run.agent) else {
            return Err(PersistenceError::MissingAgent {
                run: run.id.0,
                agent: run.agent.0,
            });
        };
        if run.tenant != agent.tenant || run.budget.dispatched > run.budget.limit {
            return Err(PersistenceError::Integrity {
                kind: "run",
                id: run.id.0,
                detail: "tenant or call accounting disagrees with its owner",
            });
        }
    }
    let mut operations = BTreeMap::new();
    let mut operation_ids = BTreeSet::new();
    for operation in &snapshot.model_operations {
        if operations.insert(operation.id, operation).is_some() {
            return Err(PersistenceError::DuplicateId {
                kind: "model operation",
                id: operation.id.0,
            });
        }
        let Some(run) = runs.get(&operation.effect.run) else {
            return Err(PersistenceError::MissingRun {
                kind: "model operation",
                id: operation.id.0,
                run: operation.effect.run.0,
            });
        };
        if operation.effect.operation != operation.id
            || operation.effect.world != snapshot.source_world
            || operation.effect.tenant != run.tenant
            || operation.effect.generation != run.generation
        {
            return Err(PersistenceError::Integrity {
                kind: "model operation",
                id: operation.id.0,
                detail: "effect correlation disagrees with its run",
            });
        }
        operation_ids.insert(operation.effect.operation);
    }
    let mut capabilities = BTreeMap::new();
    let mut capability_effects = BTreeSet::new();
    for record in &snapshot.capabilities {
        let capability = &record.snapshot;
        if capabilities.insert(capability.id, capability).is_some() {
            return Err(PersistenceError::DuplicateId {
                kind: "capability",
                id: capability.id.0,
            });
        }
        let Some(run) = runs.get(&capability.run) else {
            return Err(PersistenceError::MissingRun {
                kind: "capability",
                id: capability.id.0,
                run: capability.run.0,
            });
        };
        if capability.effect.run != capability.run
            || capability.effect.world != snapshot.source_world
            || capability.effect.tenant != capability.tenant
            || capability.tenant != run.tenant
            || capability.effect.generation != run.generation
            || operations
                .get(&capability.effect.operation)
                .is_none_or(|operation| operation.effect != capability.effect)
            || !capability_effects.insert(capability.effect)
        {
            return Err(PersistenceError::Integrity {
                kind: "capability",
                id: capability.id.0,
                detail: "advertisement correlation disagrees with its operation or run",
            });
        }
    }
    let mut call_ids = BTreeSet::new();
    let mut call_ordinals = BTreeSet::new();
    for call in &snapshot.tool_calls {
        if !call_ids.insert(call.id) {
            return Err(PersistenceError::DuplicateId {
                kind: "tool call",
                id: call.id.0,
            });
        }
        let Some(run) = runs.get(&call.run) else {
            return Err(PersistenceError::MissingRun {
                kind: "tool call",
                id: call.id.0,
                run: call.run.0,
            });
        };
        let Some(capability) = capabilities.get(&call.capability) else {
            return Err(PersistenceError::MissingCapability {
                call: call.id.0,
                capability: call.capability.0,
            });
        };
        let advertised = capability
            .tools
            .iter()
            .any(|tool| tool.name == call.name && tool.revision == call.revision);
        if call.effect != capability.effect
            || call.effect.run != call.run
            || call.effect.tenant != run.tenant
            || call.effect.generation != run.generation
            || !advertised
            || !call_ordinals.insert((call.effect, call.ordinal))
        {
            return Err(PersistenceError::Integrity {
                kind: "tool call",
                id: call.id.0,
                detail: "call disagrees with its immutable capability snapshot",
            });
        }
    }
    let mut store_operation_ids = BTreeSet::new();
    for operation in &snapshot.store_operations {
        if !store_operation_ids.insert(operation.id) {
            return Err(PersistenceError::DuplicateId {
                kind: "store operation",
                id: operation.id.0,
            });
        }
        let Some(run) = runs.get(&operation.effect.run) else {
            return Err(PersistenceError::MissingRun {
                kind: "store operation",
                id: operation.id.0,
                run: operation.effect.run.0,
            });
        };
        if operation.effect.world != snapshot.source_world
            || operation.effect.tenant != run.tenant
            || operation.effect.generation != run.generation
            || operation.effect.correlation != operation.id.0
            || !operation_ids.insert(operation.effect.operation)
        {
            return Err(PersistenceError::Integrity {
                kind: "store operation",
                id: operation.id.0,
                detail: "effect correlation disagrees with its run",
            });
        }
    }
    Ok(())
}

/// Restore stable state into a fresh, drivable runtime after explicit rebinding.
pub fn restore(
    snapshot: &RuntimeSnapshot,
    bindings: &BindingManifest,
) -> Result<BevyRuntime, PersistenceError> {
    validate(snapshot)?;
    if snapshot.content != SnapshotContent::Plaintext {
        return Err(PersistenceError::ContentOmitted);
    }
    for agent in &snapshot.agents {
        if !bindings.model_bound(agent.id) {
            return Err(PersistenceError::MissingModelBinding {
                agent: agent.id.0,
                name: agent.model_binding.clone(),
            });
        }
        for grant in &agent.tool_grants {
            if !bindings.tool_bound(agent.tenant, &grant.name, grant.revision) {
                return Err(PersistenceError::MissingToolBinding {
                    name: grant.name.clone(),
                    revision: grant.revision,
                });
            }
        }
        if let Some(store) = &agent.store_binding
            && !bindings.memory_bound(&store.implementation)
        {
            return Err(PersistenceError::MissingStoreBinding(
                store.implementation.clone(),
            ));
        }
    }
    for capability in &snapshot.capabilities {
        for tool in &capability.snapshot.tools {
            if !bindings.tool_bound(capability.snapshot.tenant, &tool.name, tool.revision) {
                return Err(PersistenceError::MissingToolBinding {
                    name: tool.name.clone(),
                    revision: tool.revision,
                });
            }
        }
    }

    let mut models = AgentModelBindings::default();
    let mut world = World::new();
    initialize_world(&mut world);
    for agent in &snapshot.agents {
        let model =
            bindings
                .models
                .get(agent.id)
                .ok_or_else(|| PersistenceError::MissingModelBinding {
                    agent: agent.id.0,
                    name: agent.model_binding.clone(),
                })?;
        debug_assert!(models.bind_erased(agent.id, model));
        let mut policy = agent.policy.clone();
        policy.additional_params = bindings.additional_params(agent.id).cloned();
        {
            let mut entity = world.spawn((
                AgentNode {
                    id: agent.id,
                    tenant: agent.tenant,
                    name: agent.name.clone(),
                    preamble: agent.preamble.clone(),
                },
                ModelBinding(agent.model_binding.clone()),
                policy,
            ));
            if let Some(store) = &agent.store_binding {
                entity.insert(store.clone());
            }
        }
        for grant in &agent.tool_grants {
            world.spawn((OwnedByAgent(agent.id), grant.clone()));
        }
    }
    for run in &snapshot.runs {
        let interrupted = run.terminal.is_none();
        let generation = if interrupted {
            run.generation.next()
        } else {
            run.generation
        };
        let phase = if interrupted {
            RunPhase::Ready
        } else {
            run.phase
        };
        let mut progress = run.progress;
        if interrupted {
            progress.idle_passes = 0;
        }
        let mut transcript = run.transcript.clone();
        if interrupted {
            repair_interrupted_tool_turn(&mut transcript, run.id, &snapshot.tool_calls);
        }
        let mut entity = world.spawn((
            RunNode {
                id: run.id,
                agent: run.agent,
                tenant: run.tenant,
                generation,
                phase,
            },
            run.budget,
            transcript,
            run.usage,
            ResponseRecovery::default(),
            progress,
            run.retention_window,
            run.retain_until,
        ));
        if let Some(terminal) = &run.terminal {
            entity.insert(terminal.clone());
        }
    }
    for capability in &snapshot.capabilities {
        world.spawn(capability.snapshot.clone());
    }
    for operation in &snapshot.model_operations {
        world.spawn(ModelOperation {
            id: operation.id,
            effect: operation.effect,
            committed: operation.committed,
            // Executor tasks are intentionally not persisted. Any pre-crash
            // completion is therefore diagnostic-only after restoration.
            retired: true,
        });
    }
    for call in &snapshot.tool_calls {
        let interrupted = snapshot
            .runs
            .iter()
            .any(|run| run.id == call.run && run.terminal.is_none());
        world.spawn(ToolCallNode {
            id: call.id,
            effect: call.effect,
            run: call.run,
            capability: call.capability,
            name: call.name.clone(),
            revision: call.revision,
            ordinal: call.ordinal,
            suppressed: call.suppressed || interrupted,
            committed: call.committed || interrupted,
        });
    }
    for operation in &snapshot.store_operations {
        world.spawn(StoreOperation {
            id: operation.id,
            effect: operation.effect,
            committed: operation.committed,
            retired: true,
        });
    }
    let max_stable = snapshot
        .agents
        .iter()
        .map(|record| record.id.0)
        .chain(snapshot.runs.iter().map(|record| record.id.0))
        .chain(
            snapshot
                .capabilities
                .iter()
                .map(|record| record.snapshot.id.0),
        )
        .chain(
            snapshot
                .model_operations
                .iter()
                .flat_map(|record| [record.id.0, record.effect.operation.0]),
        )
        .chain(snapshot.tool_calls.iter().map(|record| record.id.0))
        .chain(
            snapshot
                .store_operations
                .iter()
                .flat_map(|record| [record.id.0, record.effect.operation.0]),
        )
        .max()
        .unwrap_or(0);
    let world_id = allocate_after_restore(max_stable, snapshot.source_world.0)
        .ok_or(PersistenceError::IdentityExhausted)?;
    Ok(BevyRuntime::from_restored(
        world,
        world_id,
        models,
        bindings.tools.clone(),
        bindings.memories.clone(),
    ))
}

fn repair_interrupted_tool_turn(
    transcript: &mut CommittedTranscript,
    run: RunId,
    records: &[ToolCallRecord],
) {
    let Some(latest_effect) = records
        .iter()
        .filter(|record| record.run == run)
        .map(|record| record.effect)
        .max_by_key(|effect| effect.correlation)
    else {
        return;
    };
    let mut interrupted = records
        .iter()
        .filter(|record| record.run == run && record.effect == latest_effect)
        .collect::<Vec<_>>();
    interrupted.sort_by_key(|record| record.ordinal);
    if interrupted.is_empty() {
        return;
    }
    let Some(Message::Assistant { content, .. }) = transcript.0.last() else {
        return;
    };
    let calls = content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    let results = interrupted
        .into_iter()
        .filter_map(|record| calls.get(record.ordinal))
        .map(|call| {
            UserContent::ToolResult(MessageToolResult {
                id: call.id.clone(),
                call_id: call.call_id.clone(),
                content: OneOrMany::one(ToolResultContent::text(
                    "tool execution was interrupted and safely suppressed during restoration",
                )),
            })
        })
        .collect::<Vec<_>>();
    if let Some(content) = OneOrMany::from_iter_optional(results) {
        transcript.0.push(Message::User { content });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{AgentSpec, BevyRunError, BevyRuntime};
    use rig_core::{
        completion::Usage,
        test_utils::{CountingMemory, MockCompletionModel},
        tool::Tool,
    };
    use serde::Deserialize;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::sync::Notify;

    #[derive(Deserialize)]
    struct NoopArgs {}

    #[derive(Debug, thiserror::Error)]
    #[error("noop failed")]
    struct NoopError;

    struct NoopTool;

    impl Tool for NoopTool {
        const NAME: &'static str = "noop";
        type Args = NoopArgs;
        type Output = String;
        type Error = NoopError;

        fn description(&self) -> String {
            "No-op".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn call(&self, _: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("ok".into())
        }
    }

    struct BlockingTool {
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl Tool for BlockingTool {
        const NAME: &'static str = "wait";
        type Args = NoopArgs;
        type Output = String;
        type Error = NoopError;

        fn description(&self) -> String {
            "Wait until released".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn call(&self, _: Self::Args) -> Result<Self::Output, Self::Error> {
            self.started.notify_one();
            self.release.notified().await;
            Ok("released".into())
        }
    }

    struct CountingRestoredTool(Arc<AtomicUsize>);

    impl Tool for CountingRestoredTool {
        const NAME: &'static str = "wait";
        type Args = NoopArgs;
        type Output = String;
        type Error = NoopError;

        fn description(&self) -> String {
            "Restored wait binding".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn call(&self, _: Self::Args) -> Result<Self::Output, Self::Error> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok("unexpected re-execution".into())
        }
    }

    fn fixture() -> (World, WorldId) {
        let mut world = World::new();
        initialize_world(&mut world);
        let agent = AgentId::allocate();
        world.spawn((
            AgentNode {
                id: agent,
                tenant: TenantId(4),
                name: Some("researcher".into()),
                preamble: Some("Answer concisely".into()),
            },
            ModelBinding("scripted".into()),
            AgentPolicy {
                agent,
                max_calls: 3,
                max_tool_concurrency: 1,
                invalid_tool: crate::policy::InvalidToolPolicy::Fail,
                output_mode: crate::policy::OutputMode::Auto,
                response_retry: crate::policy::ResponseRetryPolicy {
                    max_retries: 1,
                    retries: 0,
                    feedback: "valid JSON required".into(),
                    best_effort: true,
                },
                output_schema: None,
                additional_params: Some(serde_json::json!({"service_tier": "batch"})),
                temperature: Some(0.25),
                max_tokens: Some(512),
            },
        ));
        world.spawn((
            RunNode {
                id: RunId::allocate(),
                agent,
                tenant: TenantId(4),
                generation: Generation(2),
                phase: RunPhase::Terminal,
            },
            CallBudget {
                limit: 3,
                dispatched: 1,
            },
            CommittedTranscript::default(),
            UsageLedger(Usage::new()),
            ResponseRecovery::default(),
            ProgressState {
                changes: 1,
                idle_passes: 0,
                max_idle_passes: 8,
            },
            TerminalState {
                reason: crate::components::TerminalReason::Completed,
                committed_tick: 5,
            },
            RetentionWindow(64),
            RetainUntil(69),
        ));
        (world, WorldId::allocate())
    }

    #[test]
    fn round_trip_is_deterministic_and_requires_rebinding() {
        let (mut world, world_id) = fixture();
        let first = snapshot_with_options(
            &mut world,
            world_id,
            SnapshotOptions {
                content: SnapshotContent::Plaintext,
            },
        );
        let json = to_json(&first).expect("serialize snapshot");
        let parsed = from_json(&json).expect("parse snapshot");
        assert_eq!(first, parsed);
        let agent_id = parsed.agents.first().expect("fixture agent").id;

        let error = restore(&parsed, &BindingManifest::default()).expect_err("binding required");
        assert_eq!(
            error,
            PersistenceError::MissingModelBinding {
                agent: agent_id.0,
                name: "scripted".into(),
            }
        );

        let mut bindings = BindingManifest::default();
        bindings
            .bind_model(agent_id, MockCompletionModel::text("unused"))
            .expect("bind concrete model");
        let rebound_params = serde_json::json!({"service_tier": "priority"});
        bindings.bind_additional_params(agent_id, rebound_params.clone());
        let restored = restore(&parsed, &bindings).expect("restore");
        assert_ne!(world_id, restored.world_id());
        assert!(matches!(
            restored.rebind_agent::<rig_core::providers::openai::CompletionModel>(agent_id),
            Err(BevyRunError::MismatchedModelBinding { binding, .. }) if binding == "scripted"
        ));
        let restored_params = restored.inspect(|world| {
            world
                .query::<&AgentPolicy>()
                .iter(world)
                .find(|policy| policy.agent == agent_id)
                .and_then(|policy| policy.additional_params.clone())
        });
        assert_eq!(restored_params, Some(rebound_params));
        let second = restored.inspect(|world| {
            snapshot_with_options(
                world,
                restored.world_id(),
                SnapshotOptions {
                    content: SnapshotContent::Plaintext,
                },
            )
        });
        assert_eq!(first.agents, second.agents);
        assert_eq!(first.runs, second.runs);
    }

    #[tokio::test]
    async fn model_rebinding_is_scoped_to_exact_agent_identity() {
        let runtime = BevyRuntime::default();
        let _first = runtime.spawn_agent(AgentSpec::new(MockCompletionModel::text("live-first")));
        let _second = runtime.spawn_agent(AgentSpec::new(MockCompletionModel::text("live-second")));
        let snapshot = runtime.inspect(|world| {
            snapshot_with_options(
                world,
                runtime.world_id(),
                SnapshotOptions {
                    content: SnapshotContent::Plaintext,
                },
            )
        });
        let [first, second] = snapshot.agents.as_slice() else {
            panic!("expected two persisted agents");
        };
        assert_eq!(first.model_binding, "default");
        assert_eq!(second.model_binding, "default");

        let mut bindings = BindingManifest::default();
        bindings
            .bind_model(first.id, MockCompletionModel::text("restored-first"))
            .expect("bind first agent model");
        bindings
            .bind_model(second.id, MockCompletionModel::text("restored-second"))
            .expect("bind second agent model");
        let restored = restore(&snapshot, &bindings).expect("restore both agents");

        let first_outcome = restored
            .rebind_agent::<MockCompletionModel>(first.id)
            .expect("resolve first exact binding")
            .prompt("first")
            .await
            .expect("run first agent");
        let second_outcome = restored
            .rebind_agent::<MockCompletionModel>(second.id)
            .expect("resolve second exact binding")
            .prompt("second")
            .await
            .expect("run second agent");
        assert!(matches!(
            first_outcome.choice.first(),
            AssistantContent::Text(text) if text.text == "restored-first"
        ));
        assert!(matches!(
            second_outcome.choice.first(),
            AssistantContent::Text(text) if text.text == "restored-second"
        ));
    }

    #[test]
    fn restoration_requires_exact_tool_and_store_bindings() {
        let runtime = BevyRuntime::default();
        let revision = runtime.register_tool(TenantId(9), NoopTool);
        runtime.bind_memory("memory", CountingMemory::default());
        let _agent = runtime.spawn_agent(
            AgentSpec::new(MockCompletionModel::text("unused"))
                .tenant(TenantId(9))
                .model_binding("scripted")
                .grant_tool("noop", revision)
                .memory("memory", "conversation"),
        );
        let snapshot = runtime.inspect(|world| {
            snapshot_with_options(
                world,
                runtime.world_id(),
                SnapshotOptions {
                    content: SnapshotContent::Plaintext,
                },
            )
        });
        let agent_id = snapshot.agents.first().expect("snapshot agent").id;

        let mut bindings = BindingManifest::default();
        bindings
            .bind_model(agent_id, MockCompletionModel::text("unused"))
            .expect("bind concrete model");
        assert_eq!(
            restore(&snapshot, &bindings).expect_err("tool binding required"),
            PersistenceError::MissingToolBinding {
                name: "noop".into(),
                revision,
            }
        );

        bindings
            .bind_tool(TenantId(9), revision, NoopTool)
            .expect("bind exact tool revision");
        assert_eq!(
            restore(&snapshot, &bindings).expect_err("store binding required"),
            PersistenceError::MissingStoreBinding("memory".into())
        );

        bindings.bind_memory("memory", CountingMemory::default());
        restore(&snapshot, &bindings).expect("all implementations explicitly rebound");
    }

    #[test]
    fn safe_snapshot_omits_credentials_and_tenant_content() {
        let (mut world, world_id) = fixture();
        let secret_prompt = "prompt-secret-9e4b";
        let secret_preamble = "preamble-secret-9e4b";
        let secret_parameter = "credential-secret-9e4b";
        for (mut agent, mut policy) in world
            .query::<(&mut AgentNode, &mut AgentPolicy)>()
            .iter_mut(&mut world)
        {
            agent.preamble = Some(secret_preamble.into());
            policy.additional_params = Some(serde_json::json!({"api_key": secret_parameter}));
            policy.response_retry.feedback = "retry-secret-9e4b".into();
        }
        for mut transcript in world
            .query::<&mut CommittedTranscript>()
            .iter_mut(&mut world)
        {
            transcript.0.push(Message::user(secret_prompt));
        }

        let safe = to_json(&snapshot(&mut world, world_id)).expect("safe snapshot JSON");
        for secret in [
            secret_prompt,
            secret_preamble,
            secret_parameter,
            "retry-secret-9e4b",
        ] {
            assert!(!safe.contains(secret), "safe snapshot leaked {secret}");
        }

        let plaintext = to_json(&snapshot_with_options(
            &mut world,
            world_id,
            SnapshotOptions {
                content: SnapshotContent::Plaintext,
            },
        ))
        .expect("plaintext snapshot JSON");
        assert!(plaintext.contains(secret_prompt));
        assert!(plaintext.contains(secret_preamble));
        assert!(!plaintext.contains(secret_parameter));
    }

    #[test]
    fn tampered_cross_record_correlations_are_rejected() {
        let (mut world, world_id) = fixture();
        let snapshot = snapshot_with_options(
            &mut world,
            world_id,
            SnapshotOptions {
                content: SnapshotContent::Plaintext,
            },
        );

        let mut wrong_policy_owner = snapshot.clone();
        wrong_policy_owner
            .agents
            .first_mut()
            .expect("fixture agent")
            .policy
            .agent = AgentId(u64::MAX - 1);
        assert!(matches!(
            validate(&wrong_policy_owner),
            Err(PersistenceError::Integrity { kind: "agent", .. })
        ));

        let mut wrong_run_tenant = snapshot;
        wrong_run_tenant
            .runs
            .first_mut()
            .expect("fixture run")
            .tenant = TenantId(999);
        assert!(matches!(
            validate(&wrong_run_tenant),
            Err(PersistenceError::Integrity { kind: "run", .. })
        ));
    }

    #[tokio::test]
    async fn restoration_suppresses_an_interrupted_tool_without_reexecution() {
        let runtime = BevyRuntime::default();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let revision = runtime.register_tool(
            TenantId::default(),
            BlockingTool {
                started: Arc::clone(&started),
                release,
            },
        );
        let pending = runtime
            .spawn_agent(
                AgentSpec::new(MockCompletionModel::new([
                    rig_core::test_utils::MockTurn::tool_call(
                        "wait-call",
                        "wait",
                        serde_json::json!({}),
                    ),
                ]))
                .model_binding("scripted")
                .max_calls(2)
                .grant_tool("wait", revision),
            )
            .begin_prompt("wait durably")
            .expect("begin tool run");
        let run = pending.handle().identity().run;
        let driving = tokio::spawn(pending.run());
        started.notified().await;

        let snapshot = runtime.inspect(|world| {
            snapshot_with_options(
                world,
                runtime.world_id(),
                SnapshotOptions {
                    content: SnapshotContent::Plaintext,
                },
            )
        });
        assert_eq!(snapshot.tool_calls.len(), 1);
        driving.abort();
        let _ = driving.await;

        let mut wrong_effect = snapshot.clone();
        wrong_effect
            .capabilities
            .first_mut()
            .expect("tool capability")
            .snapshot
            .effect
            .correlation += 1;
        assert!(matches!(
            validate(&wrong_effect),
            Err(PersistenceError::Integrity {
                kind: "capability",
                ..
            })
        ));
        let mut wrong_capability = snapshot.clone();
        wrong_capability
            .tool_calls
            .first_mut()
            .expect("tool call")
            .capability = CapabilityId(u64::MAX - 2);
        assert!(matches!(
            validate(&wrong_capability),
            Err(PersistenceError::MissingCapability { .. })
        ));

        let restored_calls = Arc::new(AtomicUsize::new(0));
        let agent_id = snapshot.agents.first().expect("snapshot agent").id;
        let mut bindings = BindingManifest::default();
        bindings
            .bind_model(agent_id, MockCompletionModel::text("resumed"))
            .expect("bind concrete restored model");
        bindings
            .bind_tool(
                TenantId::default(),
                revision,
                CountingRestoredTool(Arc::clone(&restored_calls)),
            )
            .expect("bind restored tool revision");
        let restored = restore(&snapshot, &bindings).expect("restore interrupted tool run");
        let restored_transcript = restored.inspect(|world| {
            world
                .query::<(&RunNode, &CommittedTranscript)>()
                .iter(world)
                .find(|(node, _)| node.id == run)
                .map(|(_, transcript)| transcript.0.clone())
                .expect("restored transcript")
        });
        assert!(
            matches!(restored_transcript.last(), Some(Message::User { content })
                if content.iter().all(|item| matches!(item, UserContent::ToolResult(_))))
        );

        assert!(matches!(
            restored.resume_run::<MockCompletionModel>(run, "must not be discarded"),
            Err(BevyRunError::ToolTurnContinuationPending)
        ));
        let pending = restored
            .resume_tool_turn::<MockCompletionModel>(run)
            .expect("resume restored tool turn");
        assert!(matches!(
            restored.resume_tool_turn::<MockCompletionModel>(run),
            Err(BevyRunError::RunAlreadyClaimed(claimed)) if claimed == run
        ));
        let claimed_transcript = restored.inspect(|world| {
            world
                .query::<(&RunNode, &CommittedTranscript)>()
                .iter(world)
                .find(|(node, _)| node.id == run)
                .map(|(_, transcript)| transcript.0.clone())
                .expect("claimed transcript")
        });
        assert!(!claimed_transcript.last().is_some_and(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().all(|item| matches!(item, UserContent::ToolResult(_)))
            )
        }));
        let outcome = pending.run().await.expect("restored tool turn completes");
        assert_eq!(outcome.terminal, TerminalReason::Completed);
        assert_eq!(restored_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn restored_nonterminal_run_is_drivable_and_advances_ids() {
        let runtime = BevyRuntime::default();
        let pending = runtime
            .spawn_agent(
                AgentSpec::new(MockCompletionModel::text("not-dispatched"))
                    .model_binding("scripted")
                    .max_calls(1),
            )
            .begin_prompt("durable prompt")
            .expect("begin nonterminal run");
        let run = pending.handle().identity().run;
        let before_max = runtime.inspect(|world| {
            snapshot_with_options(
                world,
                runtime.world_id(),
                SnapshotOptions {
                    content: SnapshotContent::Plaintext,
                },
            )
        });
        drop(pending);
        let agent_id = before_max.agents.first().expect("snapshot agent").id;

        let mut bindings = BindingManifest::default();
        bindings
            .bind_model(agent_id, MockCompletionModel::text("resumed"))
            .expect("bind concrete restored model");
        let restored = restore(&before_max, &bindings).expect("restore nonterminal run");
        let pending = restored
            .resume_run::<MockCompletionModel>(run, "durable prompt")
            .expect("resume handle");
        assert!(matches!(
            restored.resume_run::<MockCompletionModel>(run, "duplicate prompt"),
            Err(BevyRunError::RunAlreadyClaimed(claimed)) if claimed == run
        ));
        let outcome = pending.run().await.expect("restored run completes");
        assert_eq!(outcome.terminal, TerminalReason::Completed);

        let _new_agent = restored.spawn_agent(AgentSpec::new(MockCompletionModel::text("new")));
        let after = restored.inspect(|world| snapshot(world, restored.world_id()));
        let new_agent = after
            .agents
            .iter()
            .map(|agent| agent.id)
            .max()
            .expect("new agent persisted");
        let previous_max = before_max
            .agents
            .iter()
            .map(|agent| agent.id.0)
            .chain(before_max.runs.iter().map(|run| run.id.0))
            .max()
            .unwrap_or(0);
        assert!(new_agent.0 > previous_max);
    }
}
