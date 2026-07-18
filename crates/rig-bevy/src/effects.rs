//! Owned effect requests and correlated ingress completions.

use std::collections::VecDeque;

use bevy_ecs::{component::Component, resource::Resource, world::World};
use rig_core::{
    OneOrMany,
    completion::{AssistantContent, CompletionRequest, Usage},
    message::{Message, ToolCall},
    tool::ToolResult,
};

use crate::{
    components::{
        ModelOperation, RetainUntil, RetentionWindow, RunNode, RunPhase, TerminalReason,
        TerminalState, UsageLedger,
    },
    topology::{AgentId, EffectIdentity, RunId, StoreOperationId, TenantId, ToolCallId, WorldId},
};

pub(crate) struct ModelDispatchIntent {
    pub agent: AgentId,
    pub run: RunId,
    pub tenant: TenantId,
    pub request: CompletionRequest,
    pub prompt: rig_core::message::Message,
    pub transcript: Vec<rig_core::message::Message>,
    pub streaming: bool,
    pub composes_native_output_with_tools: bool,
}

pub(crate) struct PreparedModelEffect {
    pub identity: EffectIdentity,
    pub request: CompletionRequest,
    pub output_tool: Option<String>,
}

pub(crate) enum ModelDispatchFailure {
    MissingAgent,
    MissingRun,
    InvalidPolicy,
    BudgetExhausted,
}

pub(crate) enum ModelDispatchOutcome {
    Prepared(Box<PreparedModelEffect>),
    Failed {
        run: RunId,
        failure: ModelDispatchFailure,
    },
}

pub(crate) enum ModelCommitAction {
    Evaluate {
        prompt: Option<Message>,
        choice: OneOrMany<AssistantContent>,
        message_id: Option<String>,
    },
}

pub(crate) struct ModelIngressCommand {
    pub identity: EffectIdentity,
    pub usage: Usage,
    pub action: ModelCommitAction,
}

pub(crate) enum ModelIngressApplied {
    Accepted {
        transcript: Vec<Message>,
        choice: Box<OneOrMany<AssistantContent>>,
        tools: ToolPlan,
    },
    Retry {
        feedback: String,
    },
    PolicyFailure(ModelPolicyFailure),
    Rejected(IngressDecision),
}

pub(crate) enum ModelPolicyFailure {
    UnknownTool(String),
    Stopped(String),
    InvalidRepair,
    StructuredOutput,
}

pub(crate) enum ToolPlan {
    None,
    Dispatch {
        calls: Vec<ToolCall>,
        requests: Vec<ToolEffectRequest>,
        concurrency: usize,
    },
    Suppressed {
        calls: Vec<ToolCall>,
        reason: String,
    },
}

pub(crate) struct ModelIngressOutcome {
    pub identity: EffectIdentity,
    pub applied: ModelIngressApplied,
}

pub(crate) struct ToolIngressCommand {
    pub completion: ToolEffectCompletion,
}

pub(crate) struct ToolIngressOutcome {
    pub call: ToolCallId,
    pub decision: IngressDecision,
}

#[derive(Resource)]
pub(crate) struct ToolIngressQueue(pub IngressQueue<ToolIngressCommand>);

#[derive(Resource)]
pub(crate) struct ValidatedToolIngressQueue(pub IngressQueue<ToolIngressCommand>);

#[derive(Resource)]
pub(crate) struct ToolIngressOutcomeQueue(pub IngressQueue<ToolIngressOutcome>);

pub(crate) struct StoreDispatchIntent {
    pub agent: AgentId,
    pub run: RunId,
    pub kind: StoreEffectKind,
}

pub(crate) struct PreparedStoreEffect {
    pub binding: String,
    pub request: StoreEffectRequest,
}

pub(crate) enum StoreDispatchOutcome {
    Prepared(PreparedStoreEffect),
    NoBinding {
        run: RunId,
    },
    Rejected {
        run: RunId,
        decision: IngressDecision,
    },
}

pub(crate) enum StoreEffectCompletion {
    Loaded(Vec<Message>),
    Appended,
    Failed(String),
}

pub(crate) struct StoreIngressCommand {
    pub identity: EffectIdentity,
    pub store_operation: StoreOperationId,
    pub completion: StoreEffectCompletion,
}

pub(crate) enum StoreIngressApplied {
    Loaded(Vec<Message>),
    Appended,
    Failed(String),
    Rejected(IngressDecision),
}

pub(crate) struct StoreIngressOutcome {
    pub store_operation: StoreOperationId,
    pub applied: StoreIngressApplied,
}

#[derive(Resource)]
pub(crate) struct StoreDispatchQueue(pub IngressQueue<StoreDispatchIntent>);

#[derive(Resource)]
pub(crate) struct PreparedStoreQueue(pub IngressQueue<StoreDispatchOutcome>);

#[derive(Resource)]
pub(crate) struct StoreIngressQueue(pub IngressQueue<StoreIngressCommand>);

#[derive(Resource)]
pub(crate) struct ValidatedStoreIngressQueue(pub IngressQueue<StoreIngressCommand>);

#[derive(Resource)]
pub(crate) struct StoreIngressOutcomeQueue(pub IngressQueue<StoreIngressOutcome>);

#[derive(Resource)]
pub(crate) struct ModelDispatchQueue(pub IngressQueue<ModelDispatchIntent>);

#[derive(Resource)]
pub(crate) struct PreparedModelQueue(pub IngressQueue<ModelDispatchOutcome>);

#[derive(Resource)]
pub(crate) struct ModelIngressQueue(pub IngressQueue<ModelIngressCommand>);

#[derive(Resource)]
pub(crate) struct ValidatedModelIngressQueue(pub IngressQueue<ModelIngressCommand>);

#[derive(Resource)]
pub(crate) struct ModelIngressOutcomeQueue(pub IngressQueue<ModelIngressOutcome>);

macro_rules! bounded_queue_default {
    ($($name:ty),+ $(,)?) => {
        $(
            impl Default for $name {
                fn default() -> Self {
                    Self(IngressQueue::new(1024))
                }
            }
        )+
    };
}

bounded_queue_default!(
    ModelDispatchQueue,
    PreparedModelQueue,
    ModelIngressQueue,
    ValidatedModelIngressQueue,
    ModelIngressOutcomeQueue,
    ToolIngressQueue,
    ValidatedToolIngressQueue,
    ToolIngressOutcomeQueue,
    StoreDispatchQueue,
    PreparedStoreQueue,
    StoreIngressQueue,
    ValidatedStoreIngressQueue,
    StoreIngressOutcomeQueue,
);

/// Owned model input. It contains no ECS borrow, entity reference, lock guard, or client.
#[derive(Component, Clone)]
pub struct ModelEffectRequest {
    /// Correlation validated by ingress.
    pub identity: EffectIdentity,
    /// Fully owned canonical portable request.
    pub request: CompletionRequest,
    /// Whether provider deltas should be published provisionally.
    pub streaming: bool,
}

impl std::fmt::Debug for ModelEffectRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelEffectRequest")
            .field("identity", &self.identity)
            .field("request", &"<redacted>")
            .field("streaming", &self.streaming)
            .finish()
    }
}

/// Canonical portion of a completed model effect.
#[derive(Clone)]
pub struct ModelEffectCompletion {
    /// Correlation copied from the request.
    pub identity: EffectIdentity,
    /// Provider-normalized assistant content.
    pub choice: OneOrMany<AssistantContent>,
    /// Billed usage reported by the provider.
    pub usage: Usage,
    /// Non-persisted, redacted provider-final diagnostics for erased/hosted mode.
    pub hosted_final: Option<HostedFinalEnvelope>,
}

impl std::fmt::Debug for ModelEffectCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelEffectCompletion")
            .field("identity", &self.identity)
            .field("choice", &"<redacted>")
            .field("usage", &self.usage)
            .field("hosted_final", &self.hosted_final)
            .finish()
    }
}

/// Provider failure returned through the same correlated ingress boundary.
#[derive(Clone)]
pub struct ModelEffectFailure {
    /// Correlation copied from the request.
    pub identity: EffectIdentity,
    /// Operator-safe error text. Raw credentials and request bodies must not enter this value.
    pub error: String,
}

impl std::fmt::Debug for ModelEffectFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelEffectFailure")
            .field("identity", &self.identity)
            .field("error", &"<redacted>")
            .finish()
    }
}

/// Hosted/erased final diagnostics; never persisted and never claims a concrete provider type.
#[derive(Clone, PartialEq, Eq)]
pub struct HostedFinalEnvelope {
    /// Provider surface name when known.
    pub provider: Option<String>,
    /// Provider response identifier when known.
    pub response_id: Option<String>,
    /// Whether a concrete typed final was observed by the local executor.
    pub typed_final_observed: bool,
}

impl std::fmt::Debug for HostedFinalEnvelope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostedFinalEnvelope")
            .field("provider", &self.provider)
            .field(
                "response_id",
                &self.response_id.as_ref().map(|_| "<redacted>"),
            )
            .field("typed_final_observed", &self.typed_final_observed)
            .finish()
    }
}

/// Provisional stream event. Deltas are observations, not transcript mutations.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub enum SubscriptionEvent<R> {
    /// Provider text delta; provisional until accepted completion.
    ProvisionalText(String),
    /// Provider reasoning delta; provisional until accepted completion.
    ProvisionalReasoning(String),
    /// Provider tool call became observable before terminal acceptance.
    ProvisionalToolCall(rig_core::message::ToolCall),
    /// Concrete typed provider final from a local runtime.
    ProviderFinal(R),
    /// Provisional data was rejected or superseded.
    RolledBack(String),
    /// Accepted canonical content committed.
    Accepted(OneOrMany<AssistantContent>),
    /// Provider failed after zero or more provisional events.
    ProviderFailure(String),
    /// Run was cancelled.
    Cancelled(String),
    /// Terminal reason is externally observable.
    Terminal(TerminalReason),
}

/// One owned portable-tool effect request.
#[derive(Component, Clone)]
pub struct ToolEffectRequest {
    /// Correlation identity.
    pub identity: EffectIdentity,
    /// Stable tool-call identity.
    pub call: ToolCallId,
    /// Capability snapshot selected for the model turn.
    pub capability: crate::topology::CapabilityId,
    /// Exact provider-facing name.
    pub name: String,
    /// Exact implementation revision.
    pub revision: u64,
    /// Owned serialized arguments.
    pub arguments: String,
    /// Model-call order for deterministic commit.
    pub ordinal: usize,
}

impl std::fmt::Debug for ToolEffectRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolEffectRequest")
            .field("identity", &self.identity)
            .field("call", &self.call)
            .field("capability", &self.capability)
            .field("name", &self.name)
            .field("revision", &self.revision)
            .field("arguments", &"<redacted>")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

/// Completed portable-tool effect.
#[derive(Clone)]
pub struct ToolEffectCompletion {
    /// Correlation copied from the request.
    pub identity: EffectIdentity,
    /// Stable call identity.
    pub call: ToolCallId,
    /// Immutable capability snapshot copied from the request.
    pub capability: crate::topology::CapabilityId,
    /// Exact revision that executed.
    pub revision: u64,
    /// Canonical tool result.
    pub result: ToolResult,
    /// Model-call order for deterministic commit.
    pub ordinal: usize,
}

impl std::fmt::Debug for ToolEffectCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolEffectCompletion")
            .field("identity", &self.identity)
            .field("call", &self.call)
            .field("capability", &self.capability)
            .field("revision", &self.revision)
            .field("result", &"<redacted>")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

/// Owned memory/store effect kind.
#[derive(Clone)]
#[non_exhaustive]
pub enum StoreEffectKind {
    /// Load canonical history before a first model operation.
    Load { conversation: String },
    /// Append only newly committed canonical messages.
    Append {
        conversation: String,
        messages: Vec<rig_core::message::Message>,
    },
}

impl std::fmt::Debug for StoreEffectKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load { .. } => formatter.debug_struct("Load").finish_non_exhaustive(),
            Self::Append { messages, .. } => formatter
                .debug_struct("Append")
                .field("messages", &format_args!("<redacted:{}>", messages.len()))
                .finish_non_exhaustive(),
        }
    }
}

/// Owned correlated memory/store effect.
#[derive(Component, Clone)]
pub struct StoreEffectRequest {
    /// Correlation identity.
    pub identity: EffectIdentity,
    /// Stable store operation.
    pub store_operation: StoreOperationId,
    /// Owned operation kind.
    pub kind: StoreEffectKind,
}

impl std::fmt::Debug for StoreEffectRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StoreEffectRequest")
            .field("identity", &self.identity)
            .field("store_operation", &self.store_operation)
            .field("kind", &self.kind)
            .finish()
    }
}

/// Bounded ingress queue resource. Backpressure is explicit at insertion.
#[derive(Resource)]
pub struct IngressQueue<T> {
    capacity: usize,
    items: VecDeque<T>,
}

impl<T> IngressQueue<T> {
    /// Construct a queue with a non-zero bounded capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            items: VecDeque::new(),
        }
    }

    /// Push or recover ownership of the rejected item when full.
    pub fn push(&mut self, item: T) -> Result<(), T> {
        if self.items.len() >= self.capacity {
            return Err(item);
        }
        self.items.push_back(item);
        Ok(())
    }

    /// Pop the oldest ingress item.
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    pub(crate) fn take_where(&mut self, predicate: impl FnMut(&T) -> bool) -> Option<T> {
        self.items
            .iter()
            .position(predicate)
            .and_then(|position| self.items.remove(position))
    }

    pub(crate) fn drain(&mut self) -> Vec<T> {
        self.items.drain(..).collect()
    }

    /// Current queue length.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the queue has no pending items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Classification for every completion that reaches ingress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IngressDecision {
    /// Completion may commit exactly once.
    Accept,
    /// Operation already committed this correlation.
    Duplicate,
    /// Run generation moved on.
    StaleGeneration,
    /// Runtime world differs.
    ForeignWorld,
    /// Tenant differs.
    WrongTenant,
    /// Correlation differs from the active operation.
    WrongCorrelation,
    /// Operation was retired by cancellation or supersession.
    Retired,
    /// Run is already terminal.
    Late,
    /// No matching authoritative operation exists.
    UnknownOperation,
}

/// Diagnostic fact retained for rejected ingress without mutating authoritative state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RejectedIngress {
    /// Rejected correlation.
    pub identity: EffectIdentity,
    /// Classification.
    pub decision: IngressDecision,
}

/// Runtime diagnostics for rejected, duplicate, and late results.
#[derive(Resource, Default)]
pub struct IngressDiagnostics(pub Vec<RejectedIngress>);

/// Validate a completion against authoritative operation and run facts.
pub fn classify_model_ingress(
    world: &mut World,
    runtime_world: WorldId,
    identity: EffectIdentity,
) -> IngressDecision {
    if identity.world != runtime_world {
        return IngressDecision::ForeignWorld;
    }

    let operation = world
        .query::<&ModelOperation>()
        .iter(world)
        .find(|operation| operation.id == identity.operation)
        .copied();
    let Some(operation) = operation else {
        return IngressDecision::UnknownOperation;
    };
    if operation.effect.world != identity.world {
        return IngressDecision::ForeignWorld;
    }
    if operation.effect.tenant != identity.tenant {
        return IngressDecision::WrongTenant;
    }
    if operation.effect.generation != identity.generation {
        return IngressDecision::StaleGeneration;
    }
    if operation.effect.correlation != identity.correlation || operation.effect.run != identity.run
    {
        return IngressDecision::WrongCorrelation;
    }
    if operation.retired {
        return IngressDecision::Retired;
    }
    if operation.committed {
        return IngressDecision::Duplicate;
    }

    let run = world
        .query::<&RunNode>()
        .iter(world)
        .find(|run| run.id == identity.run)
        .cloned();
    let Some(run) = run else {
        return IngressDecision::UnknownOperation;
    };
    if run.tenant != identity.tenant {
        return IngressDecision::WrongTenant;
    }
    if run.generation != identity.generation {
        return IngressDecision::StaleGeneration;
    }
    if run.phase == RunPhase::Terminal || run.phase == RunPhase::CleanupEligible {
        return IngressDecision::Late;
    }
    IngressDecision::Accept
}

/// Commit usage and mark an accepted model operation idempotently.
pub fn commit_model_accounting(world: &mut World, identity: EffectIdentity, usage: Usage) -> bool {
    let mut operation_query = world.query::<&mut ModelOperation>();
    let Some(mut operation) = operation_query
        .iter_mut(world)
        .find(|operation| operation.id == identity.operation)
    else {
        return false;
    };
    if operation.committed || operation.retired {
        return false;
    }
    operation.committed = true;

    let mut ledger_query = world.query::<(&RunNode, &mut UsageLedger)>();
    if let Some((_run, mut ledger)) = ledger_query
        .iter_mut(world)
        .find(|(run, _)| run.id == identity.run)
    {
        ledger.0 += usage;
    }
    true
}

/// Commit terminal state once, replacing only with a higher-priority racing reason.
pub fn commit_terminal(world: &mut World, run_id: RunId, reason: TerminalReason, tick: u64) {
    let entity = world
        .query::<(
            bevy_ecs::entity::Entity,
            &RunNode,
            Option<&TerminalState>,
            Option<&RetentionWindow>,
        )>()
        .iter(world)
        .find(|(_, run, _, _)| run.id == run_id)
        .map(|(entity, _, terminal, retention)| (entity, terminal.cloned(), retention.copied()));
    let Some((entity, current, retention)) = entity else {
        return;
    };
    if current
        .as_ref()
        .is_some_and(|current| current.reason.priority() >= reason.priority())
    {
        return;
    }
    let mut entity_mut = world.entity_mut(entity);
    entity_mut.insert(TerminalState {
        reason,
        committed_tick: tick,
    });
    if let Some(retention) = retention {
        entity_mut.insert(RetainUntil(tick.saturating_add(retention.0)));
    }
    if let Some(mut run) = entity_mut.get_mut::<RunNode>() {
        run.phase = RunPhase::Terminal;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        components::{CallBudget, CommittedTranscript, ProgressState},
        topology::{AgentId, Generation, OperationId, TenantId, WorldId},
    };

    fn fixture() -> (World, WorldId, EffectIdentity) {
        let mut world = World::new();
        let world_id = WorldId::allocate();
        let run_id = RunId::allocate();
        let operation_id = OperationId::allocate();
        let identity = EffectIdentity {
            world: world_id,
            tenant: TenantId(7),
            run: run_id,
            operation: operation_id,
            generation: Generation(2),
            correlation: 11,
        };
        world.spawn((
            RunNode {
                id: run_id,
                agent: AgentId::allocate(),
                tenant: TenantId(7),
                generation: Generation(2),
                phase: RunPhase::Waiting,
            },
            CallBudget {
                limit: 1,
                dispatched: 1,
            },
            CommittedTranscript::default(),
            UsageLedger(Usage::new()),
            ProgressState {
                changes: 0,
                idle_passes: 0,
                max_idle_passes: 4,
            },
        ));
        world.spawn((ModelOperation {
            id: operation_id,
            effect: identity,
            committed: false,
            retired: false,
        },));
        (world, world_id, identity)
    }

    #[test]
    fn ingress_rejects_foreign_tenant_generation_and_duplicates() {
        let (mut world, world_id, identity) = fixture();
        assert_eq!(
            classify_model_ingress(&mut world, world_id, identity),
            IngressDecision::Accept
        );

        let mut wrong = identity;
        wrong.world = WorldId(world_id.0 + 1);
        assert_eq!(
            classify_model_ingress(&mut world, world_id, wrong),
            IngressDecision::ForeignWorld
        );
        wrong = identity;
        wrong.tenant = crate::topology::TenantId(99);
        assert_eq!(
            classify_model_ingress(&mut world, world_id, wrong),
            IngressDecision::WrongTenant
        );
        wrong = identity;
        wrong.generation = crate::topology::Generation(3);
        assert_eq!(
            classify_model_ingress(&mut world, world_id, wrong),
            IngressDecision::StaleGeneration
        );

        assert!(commit_model_accounting(&mut world, identity, Usage::new()));
        assert_eq!(
            classify_model_ingress(&mut world, world_id, identity),
            IngressDecision::Duplicate
        );
        assert!(!commit_model_accounting(&mut world, identity, Usage::new()));
    }

    #[test]
    fn ingress_queue_has_explicit_backpressure() {
        let mut queue = IngressQueue::new(1);
        assert!(queue.push(1).is_ok());
        assert_eq!(queue.push(2), Err(2));
        assert_eq!(queue.pop(), Some(1));
    }
}
