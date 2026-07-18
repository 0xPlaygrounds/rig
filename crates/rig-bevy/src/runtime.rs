//! Local and hosted runtime construction and handles.

use std::sync::{Arc, Mutex, MutexGuard};

use bevy_ecs::{entity::Entity, world::World};
use futures::{StreamExt, TryStreamExt};
use rig_core::{
    OneOrMany,
    client::completion::CompletionClient,
    completion::{
        AssistantContent, CompletionError, CompletionModel, CompletionRequest, GetTokenUsage,
        Message, Usage,
    },
    message::{ToolResult as MessageToolResult, UserContent},
    streaming::StreamedAssistantContent,
    tool::{DynamicTool, Tool, ToolOutput},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    adapters::{AgentModelBindings, MemoryBindings},
    components::{
        AgentNode, CallBudget, CancellationRequested, CapabilitySnapshot, CommittedTranscript,
        ModelBinding, ModelOperation, ProgressState, ResponseRecovery, RetainUntil,
        RetentionWindow, RunNode, RunPhase, StoreBinding, StoreOperation, TerminalReason,
        TerminalState, ToolCallNode, UsageLedger,
    },
    debug::{RunExplanation, explain},
    effects::{
        HostedFinalEnvelope, IngressDecision, ModelCommitAction, ModelDispatchFailure,
        ModelDispatchIntent, ModelDispatchOutcome, ModelDispatchQueue, ModelIngressApplied,
        ModelIngressCommand, ModelIngressOutcomeQueue, ModelPolicyFailure, PreparedModelQueue,
        PreparedStoreQueue, StoreDispatchIntent, StoreDispatchOutcome, StoreDispatchQueue,
        StoreEffectCompletion, StoreEffectKind, StoreIngressApplied, StoreIngressCommand,
        StoreIngressOutcomeQueue, StoreIngressQueue, SubscriptionEvent, ToolEffectCompletion,
        ToolIngressCommand, ToolIngressOutcomeQueue, ToolIngressQueue, ToolPlan, commit_terminal,
    },
    policy::{
        AgentPolicy, InvalidToolPolicy, OutputMode, ResponseRetryPolicy, ToolCatalog, ToolGrant,
    },
    schedule::{RigSchedule, RuntimeTick, ScheduleProgress, initialize_world, set_runtime_world},
    topology::{
        AgentId, EffectIdentity, Generation, HandleIdentity, OwnedByAgent, RunId, TenantId, WorldId,
    },
};

/// Immutable construction specification for an ECS-native agent.
pub struct AgentSpec<M>
where
    M: CompletionModel,
{
    /// Portable model implementation rebound outside persisted ECS data.
    pub model: M,
    name: Option<String>,
    preamble: Option<String>,
    model_binding: String,
    tenant: TenantId,
    max_calls: usize,
    tool_grants: Vec<ToolGrant>,
    invalid_tool_policy: InvalidToolPolicy,
    max_tool_concurrency: usize,
    output_schema: Option<schemars::Schema>,
    output_mode: OutputMode,
    response_retry_policy: ResponseRetryPolicy,
    additional_params: Option<serde_json::Value>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    memory: Option<MemorySpec>,
}

#[derive(Clone)]
struct MemorySpec {
    binding: String,
    conversation: String,
}

#[derive(Clone, Copy)]
enum ResumeKind {
    Prompt,
    ToolTurn,
}

struct ClaimedRun {
    tenant: TenantId,
    generation: Generation,
    transcript: Vec<Message>,
    tool_prompt: Option<Message>,
}

impl<M> AgentSpec<M>
where
    M: CompletionModel,
{
    /// Construct a specification from a portable completion model.
    pub fn new(model: M) -> Self {
        Self {
            model,
            name: None,
            preamble: None,
            model_binding: "default".into(),
            tenant: TenantId::default(),
            max_calls: 1,
            tool_grants: Vec::new(),
            invalid_tool_policy: InvalidToolPolicy::Fail,
            max_tool_concurrency: 1,
            output_schema: None,
            output_mode: OutputMode::Auto,
            response_retry_policy: ResponseRetryPolicy {
                max_retries: 1,
                retries: 0,
                feedback: "Return a valid JSON object matching the required schema.".into(),
                best_effort: true,
            },
            additional_params: None,
            temperature: None,
            max_tokens: None,
            memory: None,
        }
    }

    /// Set a public agent name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set system instructions.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Set the explicit implementation binding name persisted for restoration.
    pub fn model_binding(mut self, name: impl Into<String>) -> Self {
        self.model_binding = name.into();
        self
    }

    /// Set the owning tenant.
    pub fn tenant(mut self, tenant: TenantId) -> Self {
        self.tenant = tenant;
        for grant in &mut self.tool_grants {
            grant.tenant = tenant;
        }
        self
    }

    /// Set the total model-call budget, including retries and tool continuations.
    pub fn max_calls(mut self, max_calls: usize) -> Self {
        self.max_calls = max_calls;
        self
    }

    /// Grant one exact portable-tool revision.
    pub fn grant_tool(mut self, name: impl Into<String>, revision: u64) -> Self {
        self.tool_grants.push(ToolGrant {
            tenant: self.tenant,
            name: name.into(),
            revision,
            active: true,
        });
        self
    }

    /// Set ECS-native invalid-tool recovery policy.
    pub fn invalid_tool_policy(mut self, policy: InvalidToolPolicy) -> Self {
        self.invalid_tool_policy = policy;
        self
    }

    /// Set bounded parallel tool execution.
    pub fn max_tool_concurrency(mut self, limit: usize) -> Self {
        self.max_tool_concurrency = limit.max(1);
        self
    }

    /// Require structured output described by `T`'s JSON Schema.
    pub fn output_schema<T>(mut self) -> Self
    where
        T: schemars::JsonSchema,
    {
        self.output_schema = Some(schemars::schema_for!(T));
        self
    }

    /// Set the structured-output enforcement strategy.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.output_mode = mode;
        self
    }

    /// Set bounded structured-output recovery and exhaustion behavior.
    pub fn response_retry_policy(mut self, policy: ResponseRetryPolicy) -> Self {
        self.response_retry_policy = policy;
        self
    }

    /// Set provider-specific request parameters for every model effect.
    ///
    /// The value is redacted from debug output and omitted from snapshots. A
    /// restored agent must receive it again through `BindingManifest`.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set the sampling temperature for every model effect.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the output-token limit for every model effect.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Bind conversation memory by explicit implementation and conversation names.
    pub fn memory(mut self, binding: impl Into<String>, conversation: impl Into<String>) -> Self {
        self.memory = Some(MemorySpec {
            binding: binding.into(),
            conversation: conversation.into(),
        });
        self
    }
}

/// Adds ECS-runtime construction without colliding with classic `agent()`.
pub trait BevyCompletionClientExt: CompletionClient {
    /// Create an ECS-native agent specification for `model`.
    fn bevy_agent(&self, model: impl Into<String>) -> AgentSpec<Self::CompletionModel> {
        AgentSpec::new(self.completion_model(model))
    }
}

impl<C> BevyCompletionClientExt for C where C: CompletionClient {}

struct RuntimeState {
    world: World,
    models: AgentModelBindings,
    memories: MemoryBindings,
    retention_ticks: u64,
}

#[derive(Clone)]
struct AgentFacts {
    policy: AgentPolicy,
    memory: Option<StoreBinding>,
}

fn lock_state(state: &Mutex<RuntimeState>) -> MutexGuard<'_, RuntimeState> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn is_tool_result_continuation(message: &Message) -> bool {
    matches!(
        message,
        Message::User { content }
            if content.iter().all(|item| matches!(item, UserContent::ToolResult(_)))
    )
}

/// Owns the authoritative Bevy world and deterministic Rig schedule.
#[derive(Clone)]
pub struct BevyRuntime {
    state: Arc<Mutex<RuntimeState>>,
    world_id: WorldId,
}

impl std::fmt::Debug for BevyRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BevyRuntime")
            .field("world_id", &self.world_id)
            .finish_non_exhaustive()
    }
}

impl Default for BevyRuntime {
    fn default() -> Self {
        let world_id = WorldId::allocate();
        let mut world = World::new();
        initialize_world(&mut world);
        set_runtime_world(&mut world, world_id);
        world.insert_resource(ToolCatalog::default());
        Self {
            state: Arc::new(Mutex::new(RuntimeState {
                world,
                models: AgentModelBindings::default(),
                memories: MemoryBindings::default(),
                retention_ticks: 64,
            })),
            world_id,
        }
    }
}

impl BevyRuntime {
    pub(crate) fn from_restored(
        mut world: World,
        world_id: WorldId,
        models: AgentModelBindings,
        tools: ToolCatalog,
        memories: MemoryBindings,
    ) -> Self {
        set_runtime_world(&mut world, world_id);
        world.insert_resource(tools);
        Self {
            state: Arc::new(Mutex::new(RuntimeState {
                world,
                models,
                memories,
                retention_ticks: 64,
            })),
            world_id,
        }
    }

    /// Runtime world identity used for foreign-handle and ingress rejection.
    pub fn world_id(&self) -> WorldId {
        self.world_id
    }

    /// Set terminal-result retention in schedule ticks.
    pub fn set_retention_ticks(&self, ticks: u64) {
        lock_state(&self.state).retention_ticks = ticks;
    }

    /// Register a portable typed tool and return its stable revision.
    pub fn register_tool<T>(&self, tenant: TenantId, tool: T) -> u64
    where
        T: Tool + 'static,
    {
        lock_state(&self.state)
            .world
            .resource_mut::<ToolCatalog>()
            .register(tenant, tool)
    }

    /// Register a portable dynamic tool.
    pub fn register_dynamic_tool(&self, tenant: TenantId, tool: DynamicTool) -> u64 {
        lock_state(&self.state)
            .world
            .resource_mut::<ToolCatalog>()
            .register_dynamic(tenant, tool)
    }

    /// Bind a portable memory implementation outside persisted domain state.
    pub fn bind_memory<M>(&self, name: impl Into<String>, memory: M)
    where
        M: rig_core::memory::ConversationMemory + 'static,
    {
        lock_state(&self.state).memories.bind(name, memory);
    }

    /// Spawn an agent entity and return a local typed handle.
    pub fn spawn_agent<M>(&self, spec: AgentSpec<M>) -> AgentHandle<M>
    where
        M: CompletionModel + 'static,
    {
        let agent = AgentId::allocate();
        let mut state = lock_state(&self.state);
        debug_assert!(state.models.bind(agent, spec.model.clone()));
        {
            let mut entity = state.world.spawn((
                AgentNode {
                    id: agent,
                    tenant: spec.tenant,
                    name: spec.name.clone(),
                    preamble: spec.preamble.clone(),
                },
                ModelBinding(spec.model_binding.clone()),
                AgentPolicy {
                    agent,
                    max_calls: spec.max_calls,
                    max_tool_concurrency: spec.max_tool_concurrency,
                    invalid_tool: spec.invalid_tool_policy.clone(),
                    output_mode: spec.output_mode,
                    response_retry: spec.response_retry_policy.clone(),
                    output_schema: spec
                        .output_schema
                        .as_ref()
                        .map(|schema| schema.as_value().clone()),
                    additional_params: spec.additional_params.clone(),
                    temperature: spec.temperature,
                    max_tokens: spec.max_tokens,
                },
            ));
            if let Some(memory) = &spec.memory {
                entity.insert(StoreBinding {
                    implementation: memory.binding.clone(),
                    conversation: memory.conversation.clone(),
                });
            }
        }
        for grant in &spec.tool_grants {
            state.world.spawn((OwnedByAgent(agent), grant.clone()));
        }
        drop(state);
        AgentHandle {
            state: Arc::clone(&self.state),
            world_id: self.world_id,
            agent,
            tenant: spec.tenant,
            model: spec.model,
        }
    }

    /// Resolve a restored agent's exact host-bound typed model implementation.
    pub fn rebind_agent<M>(&self, agent: AgentId) -> Result<AgentHandle<M>, BevyRunError>
    where
        M: CompletionModel + 'static,
    {
        let mut state = lock_state(&self.state);
        let (tenant, binding) = state
            .world
            .query::<(&AgentNode, &ModelBinding)>()
            .iter(&state.world)
            .find(|(node, _)| node.id == agent)
            .map(|(node, binding)| (node.tenant, binding.0.clone()))
            .ok_or(BevyRunError::MissingAgent(agent))?;
        let model = match state.models.resolve::<M>(agent) {
            Some(model) => model,
            None if state.models.contains(agent) => {
                return Err(BevyRunError::MismatchedModelBinding {
                    binding,
                    requested: std::any::type_name::<M>(),
                });
            }
            None => return Err(BevyRunError::MissingModelBinding(binding)),
        };
        drop(state);
        Ok(AgentHandle {
            state: Arc::clone(&self.state),
            world_id: self.world_id,
            agent,
            tenant,
            model,
        })
    }

    /// Resume a nonterminal restored run with its host-retained pending prompt.
    ///
    /// Executor tasks and provisional prompts are intentionally not persisted.
    /// Restoration advances the run generation and retires old effects, so the
    /// host resubmits the pending prompt through this method. If restoration
    /// left a tool-result continuation, this method rejects the prompt instead
    /// of silently discarding it; use [`Self::resume_tool_turn`] for that case.
    pub fn resume_run<M>(
        &self,
        run: RunId,
        prompt: impl Into<Message>,
    ) -> Result<PendingRun<M>, BevyRunError>
    where
        M: CompletionModel + 'static,
    {
        let agent_id = self.restored_run_agent(run)?;
        let agent = self.rebind_agent::<M>(agent_id)?;
        let prompt = prompt.into();
        let claimed = self.claim_restored_run(run, ResumeKind::Prompt)?;
        Ok(PendingRun::new(
            agent,
            StartedRun {
                handle: RunHandle {
                    state: Arc::clone(&self.state),
                    identity: HandleIdentity {
                        world: self.world_id,
                        tenant: claimed.tenant,
                        run,
                        generation: claimed.generation,
                    },
                },
                prompt,
                transcript: claimed.transcript,
                memory_loaded: true,
            },
        ))
    }

    /// Resume a restored tool-result continuation without accepting a new prompt.
    ///
    /// This is the only resume surface that consumes the synthetic tool-result
    /// message inserted while repairing an interrupted tool turn. Callers can
    /// start a new prompt after this continuation reaches a terminal outcome.
    pub fn resume_tool_turn<M>(&self, run: RunId) -> Result<PendingRun<M>, BevyRunError>
    where
        M: CompletionModel + 'static,
    {
        let agent_id = self.restored_run_agent(run)?;
        let agent = self.rebind_agent::<M>(agent_id)?;
        let claimed = self.claim_restored_run(run, ResumeKind::ToolTurn)?;
        let prompt = claimed
            .tool_prompt
            .ok_or(BevyRunError::NoToolTurnContinuation)?;
        Ok(PendingRun::new(
            agent,
            StartedRun {
                handle: RunHandle {
                    state: Arc::clone(&self.state),
                    identity: HandleIdentity {
                        world: self.world_id,
                        tenant: claimed.tenant,
                        run,
                        generation: claimed.generation,
                    },
                },
                prompt,
                transcript: claimed.transcript,
                memory_loaded: true,
            },
        ))
    }

    fn restored_run_agent(&self, run: RunId) -> Result<AgentId, BevyRunError> {
        let mut state = lock_state(&self.state);
        state
            .world
            .query::<(&RunNode, Option<&TerminalState>)>()
            .iter(&state.world)
            .find(|(node, _)| node.id == run)
            .and_then(|(node, terminal)| terminal.is_none().then_some(node.agent))
            .ok_or(BevyRunError::MissingRun)
    }

    fn claim_restored_run(&self, run: RunId, kind: ResumeKind) -> Result<ClaimedRun, BevyRunError> {
        let mut state = lock_state(&self.state);
        let claimed = {
            let mut query = state.world.query::<(
                &mut RunNode,
                &mut CommittedTranscript,
                Option<&TerminalState>,
            )>();
            let (mut node, mut transcript, terminal) = query
                .iter_mut(&mut state.world)
                .find(|(node, _, _)| node.id == run)
                .ok_or(BevyRunError::MissingRun)?;
            if terminal.is_some() {
                return Err(BevyRunError::MissingRun);
            }
            if node.phase != RunPhase::Ready {
                return Err(BevyRunError::RunAlreadyClaimed(run));
            }

            let has_tool_turn = transcript.0.last().is_some_and(is_tool_result_continuation);
            let tool_prompt = match kind {
                ResumeKind::Prompt if has_tool_turn => {
                    return Err(BevyRunError::ToolTurnContinuationPending);
                }
                ResumeKind::Prompt => None,
                ResumeKind::ToolTurn if !has_tool_turn => {
                    return Err(BevyRunError::NoToolTurnContinuation);
                }
                ResumeKind::ToolTurn => transcript.0.pop(),
            };

            node.phase = RunPhase::Waiting;
            ClaimedRun {
                tenant: node.tenant,
                generation: node.generation,
                transcript: transcript.0.clone(),
                tool_prompt,
            }
        };
        state.world.resource_mut::<ScheduleProgress>().record(run);
        Ok(claimed)
    }

    /// Inspect the world without exposing a lock guard to asynchronous work.
    pub fn inspect<T>(&self, inspect: impl FnOnce(&mut World) -> T) -> T {
        inspect(&mut lock_state(&self.state).world)
    }

    /// Run one deterministic ECS schedule pass.
    pub fn tick(&self) {
        lock_state(&self.state).world.run_schedule(RigSchedule);
    }

    /// Despawn supporting run/operation entities only after they become cleanup eligible.
    pub fn cleanup(&self) -> usize {
        let mut state = lock_state(&self.state);
        let run_ids = state
            .world
            .query::<&RunNode>()
            .iter(&state.world)
            .filter(|run| run.phase == RunPhase::CleanupEligible)
            .map(|run| run.id)
            .collect::<Vec<_>>();
        let run_ids = run_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        let entities = state
            .world
            .query::<(
                Entity,
                Option<&RunNode>,
                Option<&ModelOperation>,
                Option<&ToolCallNode>,
                Option<&StoreOperation>,
                Option<&CapabilitySnapshot>,
            )>()
            .iter(&state.world)
            .filter_map(|(entity, run, model, tool, store, capability)| {
                let remove = run.is_some_and(|run| run_ids.contains(&run.id))
                    || model.is_some_and(|operation| run_ids.contains(&operation.effect.run))
                    || tool.is_some_and(|call| run_ids.contains(&call.run))
                    || store.is_some_and(|operation| run_ids.contains(&operation.effect.run))
                    || capability.is_some_and(|snapshot| run_ids.contains(&snapshot.run));
                remove.then_some(entity)
            })
            .collect::<Vec<_>>();
        let count = entities.len();
        for entity in entities {
            let _ = state.world.despawn(entity);
        }
        count
    }
}

/// Local typed agent handle. It owns a model clone but no ECS entity handle.
#[derive(Clone)]
pub struct AgentHandle<M>
where
    M: CompletionModel,
{
    state: Arc<Mutex<RuntimeState>>,
    world_id: WorldId,
    agent: AgentId,
    tenant: TenantId,
    model: M,
}

struct StartedRun {
    handle: RunHandle,
    prompt: Message,
    transcript: Vec<Message>,
    memory_loaded: bool,
}

struct AcceptedModelTurn {
    identity: EffectIdentity,
    prompt: Option<Message>,
    choice: OneOrMany<AssistantContent>,
    message_id: Option<String>,
    usage: Usage,
}

enum EvaluatedModelTurn {
    Accepted {
        transcript: Vec<Message>,
        choice: Box<OneOrMany<AssistantContent>>,
        tools: ToolPlan,
    },
    Retry {
        feedback: String,
    },
}

impl<M> AgentHandle<M>
where
    M: CompletionModel + 'static,
{
    fn facts(&self) -> Result<AgentFacts, BevyRunError> {
        let mut state = lock_state(&self.state);
        let (node, policy, memory) = state
            .world
            .query::<(&AgentNode, &AgentPolicy, Option<&StoreBinding>)>()
            .iter(&state.world)
            .find(|(node, _, _)| node.id == self.agent)
            .map(|(node, policy, memory)| (node.clone(), policy.clone(), memory.cloned()))
            .ok_or(BevyRunError::MissingAgent(self.agent))?;
        if node.tenant != self.tenant || policy.agent != self.agent {
            return Err(BevyRunError::InvalidPolicy(
                "agent ownership facts are inconsistent".into(),
            ));
        }
        let mut grants = state
            .world
            .query::<(&OwnedByAgent, &ToolGrant)>()
            .iter(&state.world)
            .filter(|(owner, _)| owner.0 == self.agent)
            .map(|(_, grant)| grant.clone())
            .collect::<Vec<_>>();
        grants.sort_by(|left, right| {
            (&left.name, left.revision, left.tenant).cmp(&(
                &right.name,
                right.revision,
                right.tenant,
            ))
        });
        if grants.iter().any(|grant| grant.tenant != self.tenant) {
            return Err(BevyRunError::InvalidPolicy(
                "tool grant crosses the owning tenant boundary".into(),
            ));
        }
        Ok(AgentFacts { policy, memory })
    }

    fn start_run(&self, prompt: Message) -> Result<StartedRun, BevyRunError> {
        let facts = self.facts()?;
        let run = RunId::allocate();
        let generation = Generation(0);
        let retention = lock_state(&self.state).retention_ticks;
        {
            let mut state = lock_state(&self.state);
            let tick = state.world.resource::<RuntimeTick>().0;
            let terminal = (facts.policy.max_calls == 0).then_some(TerminalState {
                reason: TerminalReason::BudgetExhausted,
                committed_tick: tick,
            });
            let mut entity = state.world.spawn((
                RunNode {
                    id: run,
                    agent: self.agent,
                    tenant: self.tenant,
                    generation,
                    phase: if terminal.is_some() {
                        RunPhase::Terminal
                    } else if facts.memory.is_some() {
                        RunPhase::LoadingMemory
                    } else {
                        RunPhase::Ready
                    },
                },
                CallBudget {
                    limit: facts.policy.max_calls,
                    dispatched: 0,
                },
                CommittedTranscript::default(),
                UsageLedger(Usage::new()),
                ResponseRecovery::default(),
                ProgressState {
                    changes: 1,
                    idle_passes: 0,
                    max_idle_passes: 128,
                },
                RetainUntil(u64::MAX),
                RetentionWindow(retention),
            ));
            if let Some(terminal) = terminal {
                entity.insert(terminal);
                entity.insert(RetainUntil(tick.saturating_add(retention)));
            }
        }

        let handle = RunHandle {
            state: Arc::clone(&self.state),
            identity: HandleIdentity {
                world: self.world_id,
                tenant: self.tenant,
                run,
                generation,
            },
        };
        if facts.policy.max_calls == 0 {
            return Err(BevyRunError::BudgetExhausted { handle });
        }

        Ok(StartedRun {
            handle,
            prompt,
            transcript: Vec::new(),
            memory_loaded: false,
        })
    }

    async fn load_memory(&self, started: &mut StartedRun) -> Result<(), BevyRunError> {
        self.ensure_active(&started.handle)?;
        if started.memory_loaded {
            return Ok(());
        }
        if let Some(memory) = self.facts()?.memory {
            let prepared = self.prepare_store(StoreDispatchIntent {
                agent: self.agent,
                run: started.handle.identity.run,
                kind: StoreEffectKind::Load {
                    conversation: memory.conversation.clone(),
                },
            })?;
            let memories = lock_state(&self.state).memories.clone();
            let completion = match memories.load(&prepared.binding, &memory.conversation).await {
                Ok(messages) => StoreEffectCompletion::Loaded(messages),
                Err(error) => StoreEffectCompletion::Failed(error.to_string()),
            };
            self.ensure_active(&started.handle)?;
            match self.commit_store(prepared.request, completion)? {
                StoreIngressApplied::Loaded(messages) => started.transcript = messages,
                StoreIngressApplied::Failed(error) => return Err(BevyRunError::Memory(error)),
                StoreIngressApplied::Rejected(decision) => {
                    return Err(BevyRunError::RejectedIngress(decision));
                }
                StoreIngressApplied::Appended => {
                    return Err(BevyRunError::RejectedIngress(
                        IngressDecision::WrongCorrelation,
                    ));
                }
            }
        }
        // Memory is committed history; the new prompt remains provisional until
        // a model operation is accepted.
        started.transcript.shrink_to_fit();
        started.memory_loaded = true;
        Ok(())
    }

    fn prepare_store(
        &self,
        intent: StoreDispatchIntent,
    ) -> Result<crate::effects::PreparedStoreEffect, BevyRunError> {
        let run = intent.run;
        let mut state = lock_state(&self.state);
        state
            .world
            .resource_mut::<StoreDispatchQueue>()
            .0
            .push(intent)
            .map_err(|_| BevyRunError::Backpressure("store dispatch"))?;
        state.world.run_schedule(RigSchedule);
        match state
            .world
            .resource_mut::<PreparedStoreQueue>()
            .0
            .take_where(|outcome| match outcome {
                StoreDispatchOutcome::Prepared(prepared) => prepared.request.identity.run == run,
                StoreDispatchOutcome::NoBinding { run: outcome }
                | StoreDispatchOutcome::Rejected { run: outcome, .. } => *outcome == run,
            })
            .ok_or(BevyRunError::Backpressure("store dispatch result"))?
        {
            StoreDispatchOutcome::Prepared(prepared) => Ok(prepared),
            StoreDispatchOutcome::NoBinding { .. } => Err(BevyRunError::InvalidPolicy(
                "agent store binding disappeared during dispatch".into(),
            )),
            StoreDispatchOutcome::Rejected { decision, .. } => {
                Err(BevyRunError::RejectedIngress(decision))
            }
        }
    }

    fn commit_store(
        &self,
        request: crate::effects::StoreEffectRequest,
        completion: StoreEffectCompletion,
    ) -> Result<StoreIngressApplied, BevyRunError> {
        let operation = request.store_operation;
        let mut state = lock_state(&self.state);
        state
            .world
            .resource_mut::<StoreIngressQueue>()
            .0
            .push(StoreIngressCommand {
                identity: request.identity,
                store_operation: operation,
                completion,
            })
            .map_err(|_| BevyRunError::Backpressure("store ingress"))?;
        state.world.run_schedule(RigSchedule);
        state
            .world
            .resource_mut::<StoreIngressOutcomeQueue>()
            .0
            .take_where(|outcome| outcome.store_operation == operation)
            .map(|outcome| outcome.applied)
            .ok_or(BevyRunError::Backpressure("store ingress result"))
    }

    fn ensure_active(&self, handle: &RunHandle) -> Result<(), BevyRunError> {
        match handle.terminal() {
            Ok(Some(TerminalReason::Cancelled(reason))) => Err(BevyRunError::Cancelled {
                reason,
                handle: handle.clone(),
            }),
            Ok(Some(_)) => Err(BevyRunError::RejectedIngress(IngressDecision::Late)),
            Ok(None) => Ok(()),
            Err(HandleError::MissingRun) => Err(BevyRunError::MissingRun),
            Err(_) => Err(BevyRunError::RejectedIngress(
                IngressDecision::WrongCorrelation,
            )),
        }
    }

    fn prepare_operation(
        &self,
        started: &StartedRun,
        transcript: &[Message],
        prompt: Message,
        streaming: bool,
    ) -> Result<(EffectIdentity, CompletionRequest, Option<String>), BevyRunError> {
        let request = self.model.completion_request(prompt.clone()).build();
        let mut state = lock_state(&self.state);
        state
            .world
            .resource_mut::<ModelDispatchQueue>()
            .0
            .push(ModelDispatchIntent {
                agent: self.agent,
                run: started.handle.identity.run,
                tenant: self.tenant,
                request,
                prompt,
                transcript: transcript.to_vec(),
                streaming,
                composes_native_output_with_tools: self.model.composes_native_output_with_tools(),
            })
            .map_err(|_| BevyRunError::Backpressure("model dispatch"))?;
        state.world.run_schedule(RigSchedule);
        let outcome = state
            .world
            .resource_mut::<PreparedModelQueue>()
            .0
            .take_where(|outcome| match outcome {
                ModelDispatchOutcome::Prepared(prepared) => {
                    prepared.identity.run == started.handle.identity.run
                }
                ModelDispatchOutcome::Failed { run, .. } => *run == started.handle.identity.run,
            })
            .ok_or(BevyRunError::Backpressure("model dispatch result"))?;
        match outcome {
            ModelDispatchOutcome::Prepared(prepared) => {
                Ok((prepared.identity, prepared.request, prepared.output_tool))
            }
            ModelDispatchOutcome::Failed {
                failure: ModelDispatchFailure::BudgetExhausted,
                ..
            } => Err(BevyRunError::BudgetExhausted {
                handle: started.handle.clone(),
            }),
            ModelDispatchOutcome::Failed {
                failure: ModelDispatchFailure::MissingRun,
                ..
            } => Err(BevyRunError::MissingRun),
            ModelDispatchOutcome::Failed {
                failure: ModelDispatchFailure::MissingAgent,
                ..
            } => Err(BevyRunError::MissingAgent(self.agent)),
            ModelDispatchOutcome::Failed {
                failure: ModelDispatchFailure::InvalidPolicy,
                ..
            } => Err(BevyRunError::InvalidPolicy(
                "model dispatch facts are inconsistent".into(),
            )),
        }
    }

    fn evaluate_choice(&self, turn: AcceptedModelTurn) -> Result<EvaluatedModelTurn, BevyRunError> {
        match self.apply_model_ingress(ModelIngressCommand {
            identity: turn.identity,
            usage: turn.usage,
            action: ModelCommitAction::Evaluate {
                prompt: turn.prompt,
                choice: turn.choice,
                message_id: turn.message_id,
            },
        })? {
            ModelIngressApplied::Accepted {
                transcript,
                choice,
                tools,
            } => Ok(EvaluatedModelTurn::Accepted {
                transcript,
                choice,
                tools,
            }),
            ModelIngressApplied::Retry { feedback } => Ok(EvaluatedModelTurn::Retry { feedback }),
            ModelIngressApplied::PolicyFailure(ModelPolicyFailure::UnknownTool(name)) => {
                Err(BevyRunError::UnknownTool(name))
            }
            ModelIngressApplied::PolicyFailure(ModelPolicyFailure::Stopped(reason)) => {
                Err(BevyRunError::Stopped(reason))
            }
            ModelIngressApplied::PolicyFailure(ModelPolicyFailure::InvalidRepair) => Err(
                BevyRunError::InvalidPolicy("tool repair target or arguments are invalid".into()),
            ),
            ModelIngressApplied::PolicyFailure(ModelPolicyFailure::StructuredOutput) => {
                Err(BevyRunError::StructuredOutput {
                    message: "structured output did not satisfy the required schema".into(),
                    handle: RunHandle {
                        state: Arc::clone(&self.state),
                        identity: HandleIdentity {
                            world: self.world_id,
                            tenant: self.tenant,
                            run: turn.identity.run,
                            generation: turn.identity.generation,
                        },
                    },
                })
            }
            ModelIngressApplied::Rejected(decision) => Err(BevyRunError::RejectedIngress(decision)),
        }
    }

    fn apply_model_ingress(
        &self,
        command: ModelIngressCommand,
    ) -> Result<ModelIngressApplied, BevyRunError> {
        let identity = command.identity;
        let mut state = lock_state(&self.state);
        state
            .world
            .resource_mut::<crate::effects::ModelIngressQueue>()
            .0
            .push(command)
            .map_err(|_| BevyRunError::Backpressure("model ingress"))?;
        state.world.run_schedule(RigSchedule);
        state
            .world
            .resource_mut::<ModelIngressOutcomeQueue>()
            .0
            .take_where(|outcome| outcome.identity == identity)
            .map(|outcome| outcome.applied)
            .ok_or(BevyRunError::Backpressure("model ingress result"))
    }

    fn commit_completed(&self, handle: &RunHandle) -> Result<(), BevyRunError> {
        let mut state = lock_state(&self.state);
        let tick = state.world.resource::<RuntimeTick>().0;
        commit_terminal(
            &mut state.world,
            handle.identity.run,
            TerminalReason::Completed,
            tick,
        );
        let terminal = state
            .world
            .query::<(&RunNode, &TerminalState)>()
            .iter(&state.world)
            .find(|(run, _)| run.id == handle.identity.run)
            .map(|(_, terminal)| terminal.reason.clone());
        match terminal {
            Some(TerminalReason::Completed) => Ok(()),
            Some(_) => Err(BevyRunError::RejectedIngress(IngressDecision::Late)),
            None => Err(BevyRunError::MissingRun),
        }
    }

    fn commit_tool_results(
        &self,
        handle: &RunHandle,
        results: Message,
    ) -> Result<Vec<Message>, BevyRunError> {
        let mut state = lock_state(&self.state);
        let transcript = state
            .world
            .query::<(&mut RunNode, &mut CommittedTranscript)>()
            .iter_mut(&mut state.world)
            .find(|(run, _)| run.id == handle.identity.run)
            .map(|(mut run, mut transcript)| {
                if matches!(run.phase, RunPhase::Terminal | RunPhase::CleanupEligible) {
                    return None;
                }
                transcript.0.push(results);
                run.phase = RunPhase::Ready;
                Some(transcript.0.clone())
            })
            .ok_or(BevyRunError::MissingRun)?;
        if transcript.is_some() {
            state
                .world
                .resource_mut::<ScheduleProgress>()
                .record(handle.identity.run);
        }
        transcript.ok_or(BevyRunError::RejectedIngress(IngressDecision::Late))
    }

    fn commit_tool_failure(&self, handle: &RunHandle) {
        let mut state = lock_state(&self.state);
        let tick = state.world.resource::<RuntimeTick>().0;
        commit_terminal(
            &mut state.world,
            handle.identity.run,
            TerminalReason::ToolFailure("tool processing failed".into()),
            tick,
        );
    }

    fn commit_provider_failure(
        &self,
        handle: &RunHandle,
        identity: Option<EffectIdentity>,
        _error: &CompletionError,
    ) {
        let mut state = lock_state(&self.state);
        if let Some(identity) = identity
            && let Some(mut operation) = state
                .world
                .query::<&mut ModelOperation>()
                .iter_mut(&mut state.world)
                .find(|operation| operation.id == identity.operation)
        {
            operation.retired = true;
        }
        let tick = state.world.resource::<RuntimeTick>().0;
        commit_terminal(
            &mut state.world,
            handle.identity.run,
            TerminalReason::ProviderFailure("provider completion failed".into()),
            tick,
        );
    }

    async fn append_memory(
        &self,
        handle: &RunHandle,
        new_messages: Vec<Message>,
    ) -> Result<(), BevyRunError> {
        let Some(memory) = self.facts()?.memory else {
            return Ok(());
        };
        let prepared = self.prepare_store(StoreDispatchIntent {
            agent: self.agent,
            run: handle.identity.run,
            kind: StoreEffectKind::Append {
                conversation: memory.conversation.clone(),
                messages: new_messages.clone(),
            },
        })?;
        let memories = lock_state(&self.state).memories.clone();
        let completion = match memories
            .append(&prepared.binding, &memory.conversation, new_messages)
            .await
        {
            Ok(()) => StoreEffectCompletion::Appended,
            Err(error) => StoreEffectCompletion::Failed(error.to_string()),
        };
        match self.commit_store(prepared.request, completion)? {
            StoreIngressApplied::Appended => Ok(()),
            StoreIngressApplied::Failed(error) => Err(BevyRunError::Memory(error)),
            StoreIngressApplied::Rejected(decision) => Err(BevyRunError::RejectedIngress(decision)),
            StoreIngressApplied::Loaded(_) => Err(BevyRunError::RejectedIngress(
                IngressDecision::WrongCorrelation,
            )),
        }
    }

    /// Create authoritative run state synchronously and return a separately
    /// drivable run. Cloning its handle before polling enables in-flight cancellation.
    pub fn begin_prompt(&self, prompt: impl Into<Message>) -> Result<PendingRun<M>, BevyRunError> {
        Ok(PendingRun::new(
            self.clone(),
            self.start_run(prompt.into())?,
        ))
    }

    /// Execute a blocking run through owned model effects and ingress commits.
    pub async fn prompt(
        &self,
        prompt: impl Into<Message>,
    ) -> Result<LocalRunOutcome<M::Response>, BevyRunError> {
        self.begin_prompt(prompt)?.run().await
    }

    async fn drive_prompt(
        &self,
        mut started: StartedRun,
    ) -> Result<LocalRunOutcome<M::Response>, BevyRunError> {
        self.load_memory(&mut started).await?;
        let initial_len = started.transcript.len();
        let mut transcript = started.transcript.clone();
        let mut next_prompt = started.prompt.clone();
        let mut commit_prompt = Some(started.prompt.clone());
        let mut provider_history = transcript.clone();
        loop {
            let (identity, request, output_tool) =
                self.prepare_operation(&started, &provider_history, next_prompt.clone(), false)?;
            let response = self.model.completion(request).await;
            self.ensure_active(&started.handle)?;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    self.commit_provider_failure(&started.handle, Some(identity), &error);
                    return Err(BevyRunError::Completion(error));
                }
            };

            let mut choice = canonicalize_output_tool(&response.choice, output_tool.as_deref())
                .unwrap_or_else(|| response.choice.clone());
            let evaluated = self.evaluate_choice(AcceptedModelTurn {
                identity,
                prompt: commit_prompt.clone(),
                choice: choice.clone(),
                message_id: response.message_id.clone(),
                usage: response.usage,
            })?;
            let tools = match evaluated {
                EvaluatedModelTurn::Retry { feedback } => {
                    provider_history = transcript.clone();
                    if let Some(prompt) = &commit_prompt {
                        provider_history.push(prompt.clone());
                    }
                    provider_history.push(Message::Assistant {
                        id: response.message_id.clone(),
                        content: choice,
                    });
                    next_prompt = Message::user(feedback);
                    continue;
                }
                EvaluatedModelTurn::Accepted {
                    transcript: accepted,
                    choice: accepted_choice,
                    tools,
                } => {
                    transcript = accepted;
                    choice = *accepted_choice;
                    tools
                }
            };
            let terminal = matches!(tools, ToolPlan::None);

            if terminal {
                let new_messages = transcript
                    .get(initial_len..)
                    .ok_or(BevyRunError::MissingRun)?
                    .to_vec();
                self.append_memory(&started.handle, new_messages).await?;
                self.commit_completed(&started.handle)?;
                let usage = self.usage_for(started.handle.identity.run);
                return Ok(LocalRunOutcome {
                    handle: started.handle,
                    choice,
                    usage,
                    transcript,
                    raw_response: response.raw_response,
                    terminal: TerminalReason::Completed,
                });
            }

            let tool_message = match self.execute_tool_plan(&started.handle, tools).await {
                Ok(message) => message,
                Err(error) => {
                    self.commit_tool_failure(&started.handle);
                    return Err(error);
                }
            };
            transcript = self.commit_tool_results(&started.handle, tool_message.clone())?;
            provider_history = transcript
                .get(..transcript.len().saturating_sub(1))
                .ok_or(BevyRunError::MissingRun)?
                .to_vec();
            next_prompt = tool_message.clone();
            commit_prompt = None;
        }
    }

    async fn execute_tool_plan(
        &self,
        handle: &RunHandle,
        plan: ToolPlan,
    ) -> Result<Message, BevyRunError> {
        let (calls, requests, limit) = match plan {
            ToolPlan::Suppressed { calls, reason } => {
                return tool_results_message(
                    calls
                        .iter()
                        .map(|call| tool_result_content(call, ToolResultValue::Skipped(&reason))),
                );
            }
            ToolPlan::Dispatch {
                calls,
                requests,
                concurrency,
            } => (calls, requests, concurrency),
            ToolPlan::None => {
                return Err(BevyRunError::Tool(
                    "tool execution requested for a terminal model turn".into(),
                ));
            }
        };
        let catalog = lock_state(&self.state)
            .world
            .resource::<ToolCatalog>()
            .clone();
        let mut completed =
            futures::stream::iter(calls.into_iter().zip(requests).map(|(call, request)| {
                let catalog = catalog.clone();
                let agent = self.clone();
                let handle = handle.clone();
                async move {
                    agent.ensure_active(&handle)?;
                    let result = catalog
                        .execute(
                            request.identity.tenant,
                            &request.name,
                            request.revision,
                            request.arguments.clone(),
                        )
                        .await;
                    Ok::<_, BevyRunError>((
                        call,
                        ToolEffectCompletion {
                            identity: request.identity,
                            call: request.call,
                            capability: request.capability,
                            revision: request.revision,
                            result,
                            ordinal: request.ordinal,
                        },
                    ))
                }
            }))
            .buffer_unordered(limit)
            .try_collect::<Vec<_>>()
            .await?;
        completed.sort_by_key(|(_, completion)| completion.ordinal);
        self.ensure_active(handle)?;
        let mut contents = Vec::with_capacity(completed.len());
        for (call, completion) in completed {
            let result = completion.result.clone();
            self.commit_tool_completion(handle, completion)?;
            contents.push(tool_result_content(
                &call,
                ToolResultValue::Executed(result),
            ));
        }
        tool_results_message(contents)
    }

    fn commit_tool_completion(
        &self,
        handle: &RunHandle,
        completion: ToolEffectCompletion,
    ) -> Result<(), BevyRunError> {
        let mut state = lock_state(&self.state);
        if completion.identity.world != handle.identity.world
            || completion.identity.tenant != handle.identity.tenant
            || completion.identity.generation != handle.identity.generation
            || completion.identity.run != handle.identity.run
        {
            return Err(BevyRunError::RejectedIngress(
                IngressDecision::WrongCorrelation,
            ));
        }
        let call = completion.call;
        state
            .world
            .resource_mut::<ToolIngressQueue>()
            .0
            .push(ToolIngressCommand { completion })
            .map_err(|_| BevyRunError::Backpressure("tool ingress"))?;
        state.world.run_schedule(RigSchedule);
        let decision = state
            .world
            .resource_mut::<ToolIngressOutcomeQueue>()
            .0
            .take_where(|outcome| outcome.call == call)
            .map(|outcome| outcome.decision)
            .ok_or(BevyRunError::Backpressure("tool ingress result"))?;
        if decision == IngressDecision::Accept {
            Ok(())
        } else {
            Err(BevyRunError::RejectedIngress(decision))
        }
    }

    fn usage_for(&self, run_id: RunId) -> Usage {
        let mut state = lock_state(&self.state);
        state
            .world
            .query::<(&RunNode, &UsageLedger)>()
            .iter(&state.world)
            .find(|(run, _)| run.id == run_id)
            .map(|(_, usage)| usage.0)
            .unwrap_or_else(Usage::new)
    }

    /// Drive the same ECS state through provider streaming, publishing provisional events.
    pub async fn stream_prompt<F>(
        &self,
        prompt: impl Into<Message>,
        observer: F,
    ) -> Result<StreamingRunOutcome<M::StreamingResponse>, BevyRunError>
    where
        F: FnMut(SubscriptionEvent<M::StreamingResponse>),
    {
        self.begin_prompt(prompt)?.stream(observer).await
    }

    async fn drive_stream<F>(
        &self,
        mut started: StartedRun,
        mut observer: F,
    ) -> Result<StreamingRunOutcome<M::StreamingResponse>, BevyRunError>
    where
        F: FnMut(SubscriptionEvent<M::StreamingResponse>),
    {
        self.load_memory(&mut started).await?;
        let initial_len = started.transcript.len();
        let mut transcript = started.transcript.clone();
        let mut next_prompt = started.prompt.clone();
        let mut commit_prompt = Some(started.prompt.clone());
        let mut provider_history = transcript.clone();
        loop {
            let (identity, request, output_tool) =
                self.prepare_operation(&started, &provider_history, next_prompt.clone(), true)?;
            let stream = self.model.stream(request).await;
            if let Err(BevyRunError::Cancelled { reason, handle }) =
                self.ensure_active(&started.handle)
            {
                observer(SubscriptionEvent::Cancelled(reason.clone()));
                observer(SubscriptionEvent::Terminal(TerminalReason::Cancelled(
                    reason.clone(),
                )));
                return Err(BevyRunError::Cancelled { reason, handle });
            }
            let mut stream = match stream {
                Ok(stream) => stream,
                Err(error) => {
                    observer(SubscriptionEvent::ProviderFailure(error.to_string()));
                    self.commit_provider_failure(&started.handle, Some(identity), &error);
                    observer(SubscriptionEvent::Terminal(
                        TerminalReason::ProviderFailure(error.to_string()),
                    ));
                    return Err(BevyRunError::Completion(error));
                }
            };
            let mut provider_final = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamedAssistantContent::Text(text)) => {
                        observer(SubscriptionEvent::ProvisionalText(text.text));
                    }
                    Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                        observer(SubscriptionEvent::ProvisionalReasoning(reasoning));
                    }
                    Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                        observer(SubscriptionEvent::ProvisionalToolCall(tool_call));
                    }
                    Ok(StreamedAssistantContent::Final(final_response)) => {
                        provider_final = Some(final_response.clone());
                    }
                    Ok(
                        StreamedAssistantContent::Reasoning(_)
                        | StreamedAssistantContent::ToolCallDelta { .. }
                        | StreamedAssistantContent::Unknown(_),
                    ) => {}
                    Err(error) => {
                        if let Err(BevyRunError::Cancelled { reason, handle }) =
                            self.ensure_active(&started.handle)
                        {
                            observer(SubscriptionEvent::Cancelled(reason.clone()));
                            observer(SubscriptionEvent::Terminal(TerminalReason::Cancelled(
                                reason.clone(),
                            )));
                            return Err(BevyRunError::Cancelled { reason, handle });
                        }
                        observer(SubscriptionEvent::RolledBack(
                            "provider stream failed; provisional events were not committed".into(),
                        ));
                        observer(SubscriptionEvent::ProviderFailure(error.to_string()));
                        self.commit_provider_failure(&started.handle, Some(identity), &error);
                        observer(SubscriptionEvent::Terminal(
                            TerminalReason::ProviderFailure(error.to_string()),
                        ));
                        return Err(BevyRunError::Completion(error));
                    }
                }
            }
            let Some(provider_final) = provider_final else {
                let error = CompletionError::ResponseError(
                    "provider stream ended without a typed final response".into(),
                );
                observer(SubscriptionEvent::RolledBack(
                    "stream ended without a final; provisional events were not committed".into(),
                ));
                self.commit_provider_failure(&started.handle, Some(identity), &error);
                observer(SubscriptionEvent::Terminal(
                    TerminalReason::ProviderFailure(error.to_string()),
                ));
                return Err(BevyRunError::Completion(error));
            };
            if let Err(BevyRunError::Cancelled { reason, handle }) =
                self.ensure_active(&started.handle)
            {
                observer(SubscriptionEvent::Cancelled(reason.clone()));
                observer(SubscriptionEvent::Terminal(TerminalReason::Cancelled(
                    reason.clone(),
                )));
                return Err(BevyRunError::Cancelled { reason, handle });
            }
            let mut choice = canonicalize_output_tool(&stream.choice, output_tool.as_deref())
                .unwrap_or_else(|| stream.choice.clone());
            let usage = provider_final.token_usage();
            let evaluated = match self.evaluate_choice(AcceptedModelTurn {
                identity,
                prompt: commit_prompt.clone(),
                choice: choice.clone(),
                message_id: stream.message_id.clone(),
                usage,
            }) {
                Ok(evaluated) => evaluated,
                Err(error) => {
                    observer(SubscriptionEvent::RolledBack(
                        "ECS response policy rejected the streamed turn".into(),
                    ));
                    if let Ok(Some(reason)) = started.handle.terminal() {
                        observer(SubscriptionEvent::Terminal(reason));
                    }
                    return Err(error);
                }
            };
            let tools = match evaluated {
                EvaluatedModelTurn::Retry { feedback } => {
                    observer(SubscriptionEvent::RolledBack(
                        "response policy requested a fresh model preparation; provisional events \
                         were not committed"
                            .into(),
                    ));
                    provider_history = transcript.clone();
                    if let Some(prompt) = &commit_prompt {
                        provider_history.push(prompt.clone());
                    }
                    provider_history.push(Message::Assistant {
                        id: stream.message_id.clone(),
                        content: choice,
                    });
                    next_prompt = Message::user(feedback);
                    continue;
                }
                EvaluatedModelTurn::Accepted {
                    transcript: accepted,
                    choice: accepted_choice,
                    tools,
                } => {
                    transcript = accepted;
                    choice = *accepted_choice;
                    tools
                }
            };
            let terminal = matches!(tools, ToolPlan::None);
            if terminal {
                let new_messages = transcript
                    .get(initial_len..)
                    .ok_or(BevyRunError::MissingRun)?
                    .to_vec();
                if let Err(error) = self.append_memory(&started.handle, new_messages).await {
                    if let Ok(Some(reason)) = started.handle.terminal() {
                        observer(SubscriptionEvent::Terminal(reason));
                    }
                    return Err(error);
                }
                self.commit_completed(&started.handle)?;
                observer(SubscriptionEvent::Accepted(choice.clone()));
                observer(SubscriptionEvent::ProviderFinal(provider_final.clone()));
                observer(SubscriptionEvent::Terminal(TerminalReason::Completed));
                return Ok(StreamingRunOutcome {
                    handle: started.handle.clone(),
                    choice,
                    usage: self.usage_for(started.handle.identity.run),
                    transcript,
                    provider_final,
                    terminal: TerminalReason::Completed,
                });
            }

            let tool_message = match self.execute_tool_plan(&started.handle, tools).await {
                Ok(message) => message,
                Err(error) => {
                    self.commit_tool_failure(&started.handle);
                    return Err(error);
                }
            };
            transcript = self.commit_tool_results(&started.handle, tool_message.clone())?;
            observer(SubscriptionEvent::Accepted(choice.clone()));
            provider_history = transcript
                .get(..transcript.len().saturating_sub(1))
                .ok_or(BevyRunError::MissingRun)?
                .to_vec();
            next_prompt = tool_message.clone();
            commit_prompt = None;
        }
    }

    /// Erase concrete provider finals while preserving an honest diagnostics envelope.
    pub fn hosted(self) -> HostedAgentHandle<M> {
        HostedAgentHandle(self)
    }
}

fn canonicalize_output_tool(
    choice: &OneOrMany<AssistantContent>,
    output_tool: Option<&str>,
) -> Option<OneOrMany<AssistantContent>> {
    let name = output_tool?;
    choice.iter().find_map(|content| match content {
        AssistantContent::ToolCall(call) if call.function.name == name => Some(OneOrMany::one(
            AssistantContent::text(call.function.arguments.to_string()),
        )),
        _ => None,
    })
}

enum ToolResultValue<'a> {
    Executed(rig_core::tool::ToolResult),
    Skipped(&'a str),
}

fn tool_result_content(
    call: &rig_core::message::ToolCall,
    result: ToolResultValue<'_>,
) -> UserContent {
    let output = match result {
        ToolResultValue::Executed(result) => result.output().clone(),
        ToolResultValue::Skipped(reason) => ToolOutput::text(reason),
    };
    UserContent::ToolResult(MessageToolResult {
        id: call.id.clone(),
        call_id: call.call_id.clone(),
        content: output.into_content(),
    })
}

fn tool_results_message(
    results: impl IntoIterator<Item = UserContent>,
) -> Result<Message, BevyRunError> {
    let content = OneOrMany::from_iter_optional(results)
        .ok_or_else(|| BevyRunError::Tool("tool batch produced no results".into()))?;
    Ok(Message::User { content })
}

/// Authoritative run state created before any memory, model, or tool effect is polled.
///
/// Keep a clone of [`RunHandle`] and drive this value on another task when the
/// caller needs to cancel or inspect an in-flight run. Dropping this value, its
/// unpolled driver future, or an in-flight driver future cancels a nonterminal
/// run; ownership is released only after the driver observes a terminal state.
pub struct PendingRun<M>
where
    M: CompletionModel,
{
    agent: AgentHandle<M>,
    started: StartedRun,
    lease: RunLease,
}

struct RunLease {
    handle: RunHandle,
    active: bool,
}

impl RunLease {
    fn new(handle: &RunHandle) -> Self {
        Self {
            handle: handle.clone(),
            active: true,
        }
    }

    fn release_if_terminal(&mut self) {
        if !matches!(self.handle.terminal(), Ok(None)) {
            self.active = false;
        }
    }
}

impl Drop for RunLease {
    fn drop(&mut self) {
        if self.active {
            let _ = self.handle.cancel("run driver dropped");
        }
    }
}

impl<M> PendingRun<M>
where
    M: CompletionModel + 'static,
{
    fn new(agent: AgentHandle<M>, started: StartedRun) -> Self {
        let lease = RunLease::new(&started.handle);
        Self {
            agent,
            started,
            lease,
        }
    }

    /// Stable handle available before the first owned effect is dispatched.
    pub fn handle(&self) -> &RunHandle {
        &self.started.handle
    }

    /// Drive this run through the blocking provider surface.
    pub async fn run(mut self) -> Result<LocalRunOutcome<M::Response>, BevyRunError> {
        let result = self.agent.drive_prompt(self.started).await;
        self.lease.release_if_terminal();
        result
    }

    /// Drive this run through the streaming provider surface.
    pub async fn stream<F>(
        mut self,
        observer: F,
    ) -> Result<StreamingRunOutcome<M::StreamingResponse>, BevyRunError>
    where
        F: FnMut(SubscriptionEvent<M::StreamingResponse>),
    {
        let result = self.agent.drive_stream(self.started, observer).await;
        self.lease.release_if_terminal();
        result
    }
}

/// Stable run handle with no raw Bevy entity.
#[derive(Clone)]
pub struct RunHandle {
    state: Arc<Mutex<RuntimeState>>,
    identity: HandleIdentity,
}

impl std::fmt::Debug for RunHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunHandle")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl RunHandle {
    /// Stable handle identity.
    pub fn identity(&self) -> HandleIdentity {
        self.identity
    }

    /// Request cancellation; terminal state remains observable until retention expires.
    pub fn cancel(&self, reason: impl Into<String>) -> Result<(), HandleError> {
        let mut state = lock_state(&self.state);
        let runtime_world = state
            .world
            .query::<&ModelOperation>()
            .iter(&state.world)
            .find(|operation| operation.effect.run == self.identity.run)
            .map(|operation| operation.effect.world)
            .unwrap_or(self.identity.world);
        if runtime_world != self.identity.world {
            return Err(HandleError::ForeignWorld);
        }
        let entity = state
            .world
            .query::<(Entity, &RunNode, Option<&TerminalState>)>()
            .iter(&state.world)
            .find(|(_, run, _)| run.id == self.identity.run)
            .map(|(entity, run, terminal)| (entity, run.clone(), terminal.is_some()))
            .ok_or(HandleError::MissingRun)?;
        if entity.1.tenant != self.identity.tenant {
            return Err(HandleError::WrongTenant);
        }
        if entity.1.generation != self.identity.generation {
            return Err(HandleError::StaleGeneration);
        }
        if entity.2 {
            return Ok(());
        }
        state
            .world
            .entity_mut(entity.0)
            .insert(CancellationRequested(reason.into()));
        for mut operation in state
            .world
            .query::<&mut ModelOperation>()
            .iter_mut(&mut state.world)
            .filter(|operation| operation.effect.run == self.identity.run)
        {
            operation.retired = true;
        }
        for mut operation in state
            .world
            .query::<&mut StoreOperation>()
            .iter_mut(&mut state.world)
            .filter(|operation| operation.effect.run == self.identity.run)
        {
            operation.retired = true;
        }
        for mut call in state
            .world
            .query::<&mut ToolCallNode>()
            .iter_mut(&mut state.world)
            .filter(|call| call.run == self.identity.run)
        {
            call.suppressed = true;
        }
        state.world.run_schedule(RigSchedule);
        Ok(())
    }

    /// Read the retained terminal reason.
    pub fn terminal(&self) -> Result<Option<TerminalReason>, HandleError> {
        let mut state = lock_state(&self.state);
        state
            .world
            .query::<(&RunNode, Option<&TerminalState>)>()
            .iter(&state.world)
            .find(|(run, _)| run.id == self.identity.run)
            .map(|(run, terminal)| {
                if run.tenant != self.identity.tenant {
                    Err(HandleError::WrongTenant)
                } else if run.generation != self.identity.generation {
                    Err(HandleError::StaleGeneration)
                } else {
                    Ok(terminal.map(|terminal| terminal.reason.clone()))
                }
            })
            .unwrap_or(Err(HandleError::MissingRun))
    }

    /// Produce a safe-by-default explanation.
    pub fn explain(&self) -> Result<RunExplanation, HandleError> {
        explain(&mut lock_state(&self.state).world, self.identity.run)
            .ok_or(HandleError::MissingRun)
    }
}

/// Successful local blocking outcome with a concrete typed provider final.
pub struct LocalRunOutcome<R> {
    /// Stable retained handle.
    pub handle: RunHandle,
    /// Accepted provider-normalized content.
    pub choice: OneOrMany<AssistantContent>,
    /// Aggregate usage across completed billed operations.
    pub usage: Usage,
    /// Canonical committed transcript.
    pub transcript: Vec<Message>,
    /// Concrete typed provider response from the terminal operation.
    pub raw_response: R,
    /// Terminal reason.
    pub terminal: TerminalReason,
}

/// Successful local streaming outcome with a concrete typed provider final.
pub struct StreamingRunOutcome<R> {
    /// Stable retained handle.
    pub handle: RunHandle,
    /// Accepted provider-normalized content.
    pub choice: OneOrMany<AssistantContent>,
    /// Usage reported by the terminal streaming response.
    pub usage: Usage,
    /// Canonical committed transcript.
    pub transcript: Vec<Message>,
    /// Concrete typed provider final observed after provisional events.
    pub provider_final: R,
    /// Terminal reason.
    pub terminal: TerminalReason,
}

/// Hosted handle that intentionally erases concrete provider finals.
pub struct HostedAgentHandle<M>(AgentHandle<M>)
where
    M: CompletionModel;

impl<M> HostedAgentHandle<M>
where
    M: CompletionModel + 'static,
{
    /// Run in hosted mode, returning canonical output and a non-persisted diagnostics envelope.
    pub async fn prompt(
        &self,
        prompt: impl Into<Message>,
    ) -> Result<HostedRunOutcome, BevyRunError> {
        let outcome = self.0.prompt(prompt).await?;
        Ok(HostedRunOutcome {
            handle: outcome.handle,
            choice: outcome.choice,
            usage: outcome.usage,
            transcript: outcome.transcript,
            final_diagnostics: HostedFinalEnvelope {
                provider: None,
                response_id: None,
                typed_final_observed: true,
            },
            terminal: outcome.terminal,
        })
    }
}

/// Hosted/erased outcome with no concrete provider-type claim.
pub struct HostedRunOutcome {
    /// Stable retained handle.
    pub handle: RunHandle,
    /// Accepted provider-normalized content.
    pub choice: OneOrMany<AssistantContent>,
    /// Aggregate usage.
    pub usage: Usage,
    /// Canonical committed transcript.
    pub transcript: Vec<Message>,
    /// Non-persisted diagnostics envelope.
    pub final_diagnostics: HostedFinalEnvelope,
    /// Terminal reason.
    pub terminal: TerminalReason,
}

/// Run failure. Budget exhaustion includes a retained terminal handle.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BevyRunError {
    /// Portable model failure.
    #[error(transparent)]
    Completion(#[from] CompletionError),
    /// Zero or exhausted call budget.
    #[error("model-call budget exhausted")]
    BudgetExhausted { handle: RunHandle },
    /// A retained handle cancelled the run while an owned effect was in flight.
    #[error("run cancelled: {reason}")]
    Cancelled {
        /// Caller-supplied cancellation detail.
        reason: String,
        /// Retained run handle exposing the winning terminal fact.
        handle: RunHandle,
    },
    /// Stable run no longer exists.
    #[error("run state is unavailable")]
    MissingRun,
    /// Stable agent no longer exists.
    #[error("agent {0:?} is unavailable")]
    MissingAgent(AgentId),
    /// A restored agent has no concrete host model binding.
    #[error("model implementation binding `{0}` is unavailable")]
    MissingModelBinding(String),
    /// The requested typed handle disagrees with the concrete restored binding.
    #[error("model binding `{binding}` is not compatible with requested type `{requested}`")]
    MismatchedModelBinding {
        /// Persisted binding name.
        binding: String,
        /// Requested Rust model type.
        requested: &'static str,
    },
    /// A repaired tool-result continuation must be resumed without a new prompt.
    #[error("a tool-result continuation is pending; use resume_tool_turn")]
    ToolTurnContinuationPending,
    /// No repaired tool-result continuation exists for this run.
    #[error("no tool-result continuation is pending")]
    NoToolTurnContinuation,
    /// Another caller already owns the restored run continuation.
    #[error("restored run {0:?} has already been claimed")]
    RunAlreadyClaimed(RunId),
    /// Authoritative ECS policy facts are malformed or inconsistent.
    #[error("invalid ECS agent policy: {0}")]
    InvalidPolicy(String),
    /// Ingress rejected a stale, duplicate, foreign, or late result.
    #[error("model completion rejected at ingress: {0:?}")]
    RejectedIngress(IngressDecision),
    /// A bounded ECS effect or ingress queue could not accept work.
    #[error("bounded ECS queue is full: {0}")]
    Backpressure(&'static str),
    /// Tool policy or execution failed.
    #[error("tool runtime error: {0}")]
    Tool(String),
    /// Model called a tool outside the advertised snapshot.
    #[error("unknown or disallowed tool `{0}`")]
    UnknownTool(String),
    /// ECS policy stopped the run.
    #[error("run stopped: {0}")]
    Stopped(String),
    /// Structured output exhausted its bounded recovery policy.
    #[error("structured output error: {message}")]
    StructuredOutput {
        /// Validation or exhaustion detail safe for callers.
        message: String,
        /// Retained run handle exposing the terminal reason.
        handle: RunHandle,
    },
    /// Memory/store operation failed.
    #[error("memory/store error: {0}")]
    Memory(String),
}

/// Stable-handle validation failure.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum HandleError {
    /// Handle belongs to another runtime world.
    #[error("handle belongs to another runtime world")]
    ForeignWorld,
    /// Handle tenant does not own the run.
    #[error("handle tenant does not own the run")]
    WrongTenant,
    /// Handle generation is stale.
    #[error("handle generation is stale")]
    StaleGeneration,
    /// Retention elapsed or run never existed.
    #[error("run is unavailable")]
    MissingRun,
}

/// Serializable terminal summary excluding raw finals, clients, and handles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalSummary {
    /// Stable run.
    pub run: RunId,
    /// Terminal reason.
    pub reason: TerminalReason,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::{
        completion::CompletionResponse,
        streaming::StreamingCompletionResponse,
        test_utils::{
            AppendFailingMemory, CountingMemory, MockCompletionModel, MockResponse,
            MockStreamEvent, MockTurn,
        },
    };
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };
    use tokio::sync::Notify;

    #[derive(Deserialize)]
    struct AddArgs {
        value: i32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("add-one failed")]
    struct AddError;

    struct AddOne;

    impl Tool for AddOne {
        const NAME: &'static str = "add_one";
        type Args = AddArgs;
        type Output = i32;
        type Error = AddError;

        fn description(&self) -> String {
            "Add one to a number".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"]
            })
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args.value + 1)
        }
    }

    #[derive(Clone)]
    struct ConcurrencyProbe {
        active: Arc<AtomicUsize>,
        maximum: Arc<AtomicUsize>,
    }

    impl Tool for ConcurrencyProbe {
        const NAME: &'static str = "concurrency_probe";
        type Args = AddArgs;
        type Output = i32;
        type Error = AddError;

        fn description(&self) -> String {
            "Measure bounded tool execution".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.maximum.fetch_max(active, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(10)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(args.value)
        }
    }

    #[derive(Clone)]
    struct BlockingFirstTool {
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl Tool for BlockingFirstTool {
        const NAME: &'static str = "blocking_first";
        type Args = AddArgs;
        type Output = i32;
        type Error = AddError;

        fn description(&self) -> String {
            "Block the first tool in a cancellation test".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(args.value)
        }
    }

    #[derive(Clone)]
    struct CountingSecondTool(Arc<AtomicUsize>);

    impl Tool for CountingSecondTool {
        const NAME: &'static str = "counting_second";
        type Args = AddArgs;
        type Output = i32;
        type Error = AddError;

        fn description(&self) -> String {
            "Count starts of a queued sibling tool".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(args.value)
        }
    }

    #[derive(Clone)]
    struct ControllableModel {
        inner: MockCompletionModel,
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl ControllableModel {
        fn new() -> Self {
            Self {
                inner: MockCompletionModel::text("too late"),
                started: Arc::new(Notify::new()),
                release: Arc::new(Notify::new()),
            }
        }
    }

    impl CompletionModel for ControllableModel {
        type Response = MockResponse;
        type StreamingResponse = MockResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self::new()
        }

        async fn completion(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            self.started.notify_one();
            self.release.notified().await;
            self.inner.completion(request).await
        }

        async fn stream(
            &self,
            request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            self.inner.stream(request).await
        }
    }

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Answer {
        answer: i32,
    }

    fn tool_result_count(message: &Message) -> usize {
        match message {
            Message::User { content } => content
                .iter()
                .filter(|content| matches!(content, UserContent::ToolResult(_)))
                .count(),
            _ => 0,
        }
    }

    fn invalid_tool_turn() -> MockTurn {
        MockTurn::tool_call(
            "invalid-call",
            "not_advertised",
            serde_json::json!({"value": 4}),
        )
    }

    #[test]
    fn tenant_builder_order_does_not_change_tool_authorization() {
        let tenant = TenantId(9);
        let spec = AgentSpec::new(MockCompletionModel::text("done"))
            .grant_tool("add_one", 1)
            .tenant(tenant);

        assert_eq!(spec.tenant, tenant);
        assert_eq!(spec.tool_grants[0].tenant, tenant);
    }

    #[tokio::test]
    async fn blocking_tools_commit_one_ordered_result_message() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call-1", "add_one", serde_json::json!({"value": 2})),
            MockTurn::text("3"),
        ]);
        let runtime = BevyRuntime::default();
        let revision = runtime.register_tool(TenantId::default(), AddOne);
        let agent = runtime.spawn_agent(
            AgentSpec::new(model.clone())
                .max_calls(2)
                .grant_tool("add_one", revision),
        );

        let outcome = agent.prompt("calculate").await.expect("tool run succeeds");
        assert_eq!(model.request_count(), 2);
        assert_eq!(outcome.transcript.len(), 4);
        assert_eq!(tool_result_count(&outcome.transcript[2]), 1);
        assert_eq!(
            outcome.handle.terminal(),
            Ok(Some(TerminalReason::Completed))
        );
        assert!(matches!(outcome.choice.first(), AssistantContent::Text(text) if text.text == "3"));
    }

    #[tokio::test]
    async fn streaming_tools_use_the_same_multi_turn_commit_path() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("call-1", "add_one", serde_json::json!({"value": 2})),
                MockStreamEvent::final_response_with_total_tokens(2),
            ],
            vec![
                MockStreamEvent::text("3"),
                MockStreamEvent::final_response_with_total_tokens(1),
            ],
        ]);
        let runtime = BevyRuntime::default();
        let revision = runtime.register_tool(TenantId::default(), AddOne);
        let agent = runtime.spawn_agent(
            AgentSpec::new(model.clone())
                .max_calls(2)
                .grant_tool("add_one", revision),
        );
        let mut events = Vec::new();

        let outcome = agent
            .stream_prompt("calculate", |event| events.push(event))
            .await
            .expect("streamed tool run succeeds");
        assert_eq!(model.request_count(), 2);
        assert_eq!(outcome.transcript.len(), 4);
        assert_eq!(tool_result_count(&outcome.transcript[2]), 1);
        assert_eq!(outcome.usage.total_tokens, 3);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, SubscriptionEvent::Accepted(_)))
                .count(),
            2
        );
        assert!(matches!(
            events.last(),
            Some(SubscriptionEvent::Terminal(TerminalReason::Completed))
        ));
    }

    #[tokio::test]
    async fn parallel_tool_bodies_are_bounded_and_commit_in_model_order() {
        let response = MockTurn::from_contents([
            AssistantContent::ToolCall(rig_core::message::ToolCall::new(
                "first".into(),
                rig_core::message::ToolFunction::new(
                    "concurrency_probe".into(),
                    serde_json::json!({"value": 1}),
                ),
            )),
            AssistantContent::ToolCall(rig_core::message::ToolCall::new(
                "second".into(),
                rig_core::message::ToolFunction::new(
                    "concurrency_probe".into(),
                    serde_json::json!({"value": 2}),
                ),
            )),
        ])
        .expect("non-empty tool turn");
        let model = MockCompletionModel::new([response, MockTurn::text("done")]);
        let runtime = BevyRuntime::default();
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let revision = runtime.register_tool(
            TenantId::default(),
            ConcurrencyProbe {
                active,
                maximum: Arc::clone(&maximum),
            },
        );
        let agent = runtime.spawn_agent(
            AgentSpec::new(model)
                .max_calls(2)
                .max_tool_concurrency(2)
                .grant_tool("concurrency_probe", revision),
        );

        let outcome = agent.prompt("run both").await.expect("parallel tool run");
        assert_eq!(maximum.load(Ordering::SeqCst), 2);
        let ids = match outcome.transcript.get(2) {
            Some(Message::User { content }) => content
                .iter()
                .filter_map(|content| match content {
                    UserContent::ToolResult(result) => Some(result.id.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        assert_eq!(ids, ["first", "second"]);
    }

    #[tokio::test]
    async fn cancellation_does_not_start_queued_sibling_tools() {
        let response = MockTurn::from_contents([
            AssistantContent::ToolCall(rig_core::message::ToolCall::new(
                "first".into(),
                rig_core::message::ToolFunction::new(
                    "blocking_first".into(),
                    serde_json::json!({"value": 1}),
                ),
            )),
            AssistantContent::ToolCall(rig_core::message::ToolCall::new(
                "second".into(),
                rig_core::message::ToolFunction::new(
                    "counting_second".into(),
                    serde_json::json!({"value": 2}),
                ),
            )),
        ])
        .expect("non-empty tool turn");
        let model = MockCompletionModel::new([response, MockTurn::text("must not run")]);
        let runtime = BevyRuntime::default();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let second_starts = Arc::new(AtomicUsize::new(0));
        let first_revision = runtime.register_tool(
            TenantId::default(),
            BlockingFirstTool {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            },
        );
        let second_revision = runtime.register_tool(
            TenantId::default(),
            CountingSecondTool(Arc::clone(&second_starts)),
        );
        let pending = runtime
            .spawn_agent(
                AgentSpec::new(model)
                    .max_calls(2)
                    .max_tool_concurrency(1)
                    .grant_tool("blocking_first", first_revision)
                    .grant_tool("counting_second", second_revision),
            )
            .begin_prompt("run both")
            .expect("begin cancellable tool run");
        let handle = pending.handle().clone();
        let driving = tokio::spawn(pending.run());

        started.notified().await;
        handle.cancel("stop the batch").expect("cancel tool run");
        release.notify_one();
        let result = driving.await.expect("driver task should join");

        assert!(matches!(
            result,
            Err(BevyRunError::Cancelled { reason, .. }) if reason == "stop the batch"
        ));
        assert_eq!(second_starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn invalid_tool_fail_and_stop_are_terminal_without_dispatch() {
        let fail_runtime = BevyRuntime::default();
        let fail_agent = fail_runtime.spawn_agent(AgentSpec::new(MockCompletionModel::new([
            invalid_tool_turn(),
        ])));
        let pending = fail_agent.begin_prompt("fail").expect("begin fail run");
        let fail_handle = pending.handle().clone();
        assert!(matches!(
            pending.run().await,
            Err(BevyRunError::UnknownTool(name)) if name == "not_advertised"
        ));
        assert!(matches!(
            fail_handle.terminal(),
            Ok(Some(TerminalReason::ToolFailure(reason)))
                if reason == "invalid tool call"
        ));

        let stop_runtime = BevyRuntime::default();
        let stop_agent = stop_runtime.spawn_agent(
            AgentSpec::new(MockCompletionModel::new([invalid_tool_turn()])).invalid_tool_policy(
                InvalidToolPolicy::Stop {
                    reason: "policy stop".into(),
                },
            ),
        );
        let pending = stop_agent.begin_prompt("stop").expect("begin stop run");
        let stop_handle = pending.handle().clone();
        assert!(matches!(
            pending.run().await,
            Err(BevyRunError::Stopped(reason)) if reason == "policy stop"
        ));
        assert_eq!(
            stop_handle.terminal(),
            Ok(Some(TerminalReason::Stopped("policy stop".into())))
        );
    }

    #[tokio::test]
    async fn invalid_tool_retry_skip_and_repair_follow_the_selected_policy() {
        let retry_model = MockCompletionModel::new([invalid_tool_turn(), MockTurn::text("done")]);
        let retry_runtime = BevyRuntime::default();
        let retry_outcome = retry_runtime
            .spawn_agent(
                AgentSpec::new(retry_model.clone())
                    .max_calls(2)
                    .invalid_tool_policy(InvalidToolPolicy::Retry {
                        feedback: "choose an advertised tool".into(),
                    }),
            )
            .prompt("recover")
            .await
            .expect("retry policy recovers");
        assert_eq!(retry_model.request_count(), 2);
        assert_eq!(retry_outcome.transcript.len(), 2);

        let skip_model = MockCompletionModel::new([invalid_tool_turn(), MockTurn::text("done")]);
        let skip_runtime = BevyRuntime::default();
        let skip_outcome = skip_runtime
            .spawn_agent(
                AgentSpec::new(skip_model.clone())
                    .max_calls(2)
                    .invalid_tool_policy(InvalidToolPolicy::Skip {
                        reason: "skipped by policy".into(),
                    }),
            )
            .prompt("recover")
            .await
            .expect("skip policy recovers");
        assert_eq!(skip_model.request_count(), 2);
        assert_eq!(tool_result_count(&skip_outcome.transcript[2]), 1);

        let model = MockCompletionModel::new([invalid_tool_turn(), MockTurn::text("done")]);
        let runtime = BevyRuntime::default();
        let revision = runtime.register_tool(TenantId::default(), AddOne);
        let agent = runtime.spawn_agent(
            AgentSpec::new(model.clone())
                .max_calls(2)
                .grant_tool("add_one", revision)
                .invalid_tool_policy(InvalidToolPolicy::Repair {
                    name: "add_one".into(),
                    arguments: r#"{"value": 4}"#.into(),
                }),
        );
        let outcome = agent.prompt("repair").await.expect("repair recovers");
        assert_eq!(model.request_count(), 2);
        assert_eq!(tool_result_count(&outcome.transcript[2]), 1);
        assert!(format!("{:?}", outcome.transcript[2]).contains('5'));
    }

    #[tokio::test]
    async fn invalid_repair_configuration_is_terminal() {
        for policy in [
            InvalidToolPolicy::Repair {
                name: "missing".into(),
                arguments: "{}".into(),
            },
            InvalidToolPolicy::Repair {
                name: "add_one".into(),
                arguments: "not-json".into(),
            },
        ] {
            let runtime = BevyRuntime::default();
            let revision = runtime.register_tool(TenantId::default(), AddOne);
            let pending = runtime
                .spawn_agent(
                    AgentSpec::new(MockCompletionModel::new([invalid_tool_turn()]))
                        .grant_tool("add_one", revision)
                        .invalid_tool_policy(policy),
                )
                .begin_prompt("repair")
                .expect("begin repaired run");
            let handle = pending.handle().clone();
            assert!(matches!(
                pending.run().await,
                Err(BevyRunError::InvalidPolicy(_))
            ));
            assert!(matches!(
                handle.terminal(),
                Ok(Some(TerminalReason::ToolFailure(reason)))
                    if reason == "invalid tool repair policy"
            ));
        }
    }

    #[tokio::test]
    async fn tool_results_remain_paired_before_budget_or_provider_failure() {
        for (turns, max_calls) in [
            (
                vec![
                    MockTurn::tool_call("call", "add_one", serde_json::json!({"value": 1})),
                    MockTurn::error("provider failed"),
                ],
                2,
            ),
            (
                vec![MockTurn::tool_call(
                    "call",
                    "add_one",
                    serde_json::json!({"value": 1}),
                )],
                1,
            ),
        ] {
            let runtime = BevyRuntime::default();
            let revision = runtime.register_tool(TenantId::default(), AddOne);
            let pending = runtime
                .spawn_agent(
                    AgentSpec::new(MockCompletionModel::new(turns))
                        .max_calls(max_calls)
                        .grant_tool("add_one", revision),
                )
                .begin_prompt("pair")
                .expect("begin tool run");
            let run = pending.handle().identity().run;
            assert!(pending.run().await.is_err());
            let transcript = runtime.inspect(|world| {
                world
                    .query::<(&RunNode, &CommittedTranscript)>()
                    .iter(world)
                    .find(|(node, _)| node.id == run)
                    .map(|(_, transcript)| transcript.0.clone())
                    .unwrap_or_default()
            });
            assert_eq!(transcript.len(), 3);
            assert_eq!(tool_result_count(&transcript[2]), 1);
        }
    }

    #[tokio::test]
    async fn structured_retry_excludes_rejected_content_from_committed_history() {
        let model = MockCompletionModel::new([
            MockTurn::text("not json"),
            MockTurn::text(r#"{"answer": 42}"#),
        ]);
        let runtime = BevyRuntime::default();
        let agent = runtime.spawn_agent(
            AgentSpec::new(model.clone())
                .max_calls(2)
                .output_schema::<Answer>()
                .output_mode(OutputMode::Native)
                .response_retry_policy(ResponseRetryPolicy {
                    max_retries: 1,
                    retries: 0,
                    feedback: "valid JSON required".into(),
                    best_effort: false,
                }),
        );

        let outcome = agent.prompt("answer").await.expect("retry recovers");
        assert_eq!(model.request_count(), 2);
        assert_eq!(outcome.transcript.len(), 2);
        assert!(!format!("{:?}", outcome.transcript).contains("not json"));
        assert!(
            model
                .requests()
                .iter()
                .all(|request| request.output_schema.is_some())
        );
    }

    #[tokio::test]
    async fn structured_output_enforces_types_not_only_required_keys() {
        let model = MockCompletionModel::new([
            MockTurn::text(r#"{"answer":"wrong type"}"#),
            MockTurn::text(r#"{"answer":42}"#),
        ]);
        let runtime = BevyRuntime::default();
        let outcome = runtime
            .spawn_agent(
                AgentSpec::new(model.clone())
                    .max_calls(2)
                    .output_schema::<Answer>()
                    .output_mode(OutputMode::Native)
                    .response_retry_policy(ResponseRetryPolicy {
                        max_retries: 1,
                        retries: 0,
                        feedback: "valid typed JSON required".into(),
                        best_effort: false,
                    }),
            )
            .prompt("typed")
            .await
            .expect("typed retry succeeds");
        assert_eq!(model.request_count(), 2);
        assert_eq!(outcome.transcript.len(), 2);
    }

    #[tokio::test]
    async fn streaming_never_publishes_success_after_late_failures() {
        let runtime = BevyRuntime::default();
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("provisional"),
            MockStreamEvent::final_response_with_total_tokens(1),
            MockStreamEvent::error("late stream error"),
        ]]);
        let mut events = Vec::new();
        assert!(
            runtime
                .spawn_agent(AgentSpec::new(model))
                .stream_prompt("fail", |event| events.push(event))
                .await
                .is_err()
        );
        assert!(!events.iter().any(|event| matches!(
            event,
            SubscriptionEvent::ProviderFinal(_) | SubscriptionEvent::Accepted(_)
        )));

        let store_runtime = BevyRuntime::default();
        store_runtime.bind_memory("failing", AppendFailingMemory::default());
        let store_model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("accepted only after append"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ]]);
        let mut store_events = Vec::new();
        assert!(
            store_runtime
                .spawn_agent(
                    AgentSpec::new(store_model)
                        .memory("failing", "conversation")
                        .max_calls(1),
                )
                .stream_prompt("fail append", |event| store_events.push(event))
                .await
                .is_err()
        );
        assert!(!store_events.iter().any(|event| matches!(
            event,
            SubscriptionEvent::ProviderFinal(_) | SubscriptionEvent::Accepted(_)
        )));
        assert!(store_events.iter().any(|event| matches!(
            event,
            SubscriptionEvent::Terminal(TerminalReason::StoreFailure(reason))
                if reason == "store effect failed"
        )));
    }

    #[tokio::test]
    async fn synthetic_output_tool_suppresses_peer_execution_and_commits_text() {
        let response = MockTurn::from_contents([
            AssistantContent::ToolCall(rig_core::message::ToolCall::new(
                "output".into(),
                rig_core::message::ToolFunction::new(
                    "__rig_submit".into(),
                    serde_json::json!({"answer": 7}),
                ),
            )),
            AssistantContent::ToolCall(rig_core::message::ToolCall::new(
                "peer".into(),
                rig_core::message::ToolFunction::new(
                    "add_one".into(),
                    serde_json::json!({"value": 1}),
                ),
            )),
        ])
        .expect("non-empty response");
        let model = MockCompletionModel::new([response]);
        let runtime = BevyRuntime::default();
        let revision = runtime.register_tool(TenantId::default(), AddOne);
        let agent = runtime.spawn_agent(
            AgentSpec::new(model)
                .max_calls(1)
                .grant_tool("add_one", revision)
                .output_schema::<Answer>()
                .output_mode(OutputMode::Tool),
        );

        let outcome = agent.prompt("answer").await.expect("output tool finalizes");
        assert_eq!(outcome.transcript.len(), 2);
        assert!(
            matches!(outcome.choice.first(), AssistantContent::Text(text) if text.text.contains("\"answer\":7"))
        );
        let tool_calls =
            runtime.inspect(|world| world.query::<&ToolCallNode>().iter(world).count());
        assert_eq!(tool_calls, 0, "suppressed peer must not create an effect");
    }

    #[tokio::test]
    async fn memory_loads_before_dispatch_and_appends_only_committed_messages() {
        let memory = CountingMemory::default();
        let model = MockCompletionModel::text("done");
        let runtime = BevyRuntime::default();
        runtime.bind_memory("memory", memory.clone());
        let agent = runtime.spawn_agent(
            AgentSpec::new(model)
                .memory("memory", "conversation")
                .max_calls(1),
        );

        let outcome = agent.prompt("hello").await.expect("memory run succeeds");
        assert_eq!(memory.load_count(), 1);
        assert_eq!(memory.append_count(), 1);
        assert_eq!(outcome.transcript.len(), 2);
        let store_operations = runtime.inspect(|world| {
            world
                .query::<&StoreOperation>()
                .iter(world)
                .filter(|operation| operation.committed)
                .count()
        });
        assert_eq!(store_operations, 2);
    }

    #[tokio::test]
    async fn retained_handle_cancels_an_in_flight_model_effect() {
        let model = ControllableModel::new();
        let runtime = BevyRuntime::default();
        let agent = runtime.spawn_agent(AgentSpec::new(model.clone()).max_calls(1));
        let pending = agent.begin_prompt("hello").expect("run state is created");
        let handle = pending.handle().clone();
        let cancel_handle = handle.clone();
        let started = Arc::clone(&model.started);
        let release = Arc::clone(&model.release);

        let drive = pending.run();
        let cancel = async move {
            started.notified().await;
            cancel_handle.cancel("caller left").expect("cancel run");
            release.notify_one();
        };
        let (result, ()) = tokio::join!(drive, cancel);

        let Err(BevyRunError::Cancelled {
            reason,
            handle: error_handle,
        }) = result
        else {
            panic!("cancelled effect must not report provider success");
        };
        assert_eq!(reason, "caller left");
        assert_eq!(
            error_handle.terminal(),
            Ok(Some(TerminalReason::Cancelled("caller left".into())))
        );
        let (usage, transcript_len, retired) = runtime.inspect(|world| {
            let (usage, transcript) = world
                .query::<(&RunNode, &UsageLedger, &CommittedTranscript)>()
                .iter(world)
                .find(|(run, _, _)| run.id == handle.identity.run)
                .map(|(_, usage, transcript)| (usage.0, transcript.0.len()))
                .expect("retained run");
            let retired = world
                .query::<&ModelOperation>()
                .iter(world)
                .all(|operation| operation.effect.run != handle.identity.run || operation.retired);
            (usage, transcript, retired)
        });
        assert_eq!(usage, Usage::new());
        assert_eq!(transcript_len, 0);
        assert!(retired);
    }

    #[test]
    fn dropping_an_unpolled_pending_run_cancels_it() {
        let runtime = BevyRuntime::default();
        let pending = runtime
            .spawn_agent(AgentSpec::new(MockCompletionModel::text("unused")))
            .begin_prompt("hello")
            .expect("create pending run");
        let handle = pending.handle().clone();

        drop(pending);

        assert_eq!(
            handle.terminal(),
            Ok(Some(TerminalReason::Cancelled("run driver dropped".into())))
        );
    }

    #[test]
    fn dropping_an_unpolled_driver_future_cancels_it() {
        let runtime = BevyRuntime::default();
        let pending = runtime
            .spawn_agent(AgentSpec::new(MockCompletionModel::text("unused")))
            .begin_prompt("hello")
            .expect("create pending run");
        let handle = pending.handle().clone();
        let driving = pending.run();

        drop(driving);

        assert_eq!(
            handle.terminal(),
            Ok(Some(TerminalReason::Cancelled("run driver dropped".into())))
        );
    }

    #[tokio::test]
    async fn aborting_an_in_flight_driver_cancels_it() {
        let model = ControllableModel::new();
        let runtime = BevyRuntime::default();
        let pending = runtime
            .spawn_agent(AgentSpec::new(model.clone()).max_calls(1))
            .begin_prompt("hello")
            .expect("create pending run");
        let handle = pending.handle().clone();
        let driving = tokio::spawn(pending.run());

        model.started.notified().await;
        driving.abort();
        let _ = driving.await;

        assert_eq!(
            handle.terminal(),
            Ok(Some(TerminalReason::Cancelled("run driver dropped".into())))
        );
    }

    #[tokio::test]
    async fn retention_starts_at_terminal_commit_and_cleanup_removes_children() {
        let runtime = BevyRuntime::default();
        runtime.set_retention_ticks(2);
        let agent = runtime.spawn_agent(AgentSpec::new(MockCompletionModel::text("done")));
        let pending = agent.begin_prompt("hello").expect("begin run");
        let handle = pending.handle().clone();

        for _ in 0..5 {
            runtime.tick();
        }
        assert_eq!(runtime.cleanup(), 0, "active run has no cleanup deadline");
        let outcome = pending.run().await.expect("run succeeds");
        assert_eq!(
            outcome.handle.terminal(),
            Ok(Some(TerminalReason::Completed))
        );

        runtime.tick();
        assert_eq!(runtime.cleanup(), 0, "one retained tick remains");
        runtime.tick();
        let removed = runtime.cleanup();
        assert!(
            removed >= 3,
            "run, model operation, and capability are removed"
        );
        assert_eq!(handle.terminal(), Err(HandleError::MissingRun));
        let supporting = runtime.inspect(|world| {
            let models = world
                .query::<&ModelOperation>()
                .iter(world)
                .filter(|operation| operation.effect.run == handle.identity.run)
                .count();
            let capabilities = world
                .query::<&CapabilitySnapshot>()
                .iter(world)
                .filter(|snapshot| snapshot.run == handle.identity.run)
                .count();
            models + capabilities
        });
        assert_eq!(supporting, 0);
    }

    #[tokio::test]
    async fn zero_budget_is_terminal_without_model_or_memory_dispatch() {
        let memory = CountingMemory::default();
        let model = MockCompletionModel::text("unused");
        let runtime = BevyRuntime::default();
        runtime.bind_memory("memory", memory.clone());
        let agent = runtime.spawn_agent(
            AgentSpec::new(model.clone())
                .memory("memory", "conversation")
                .max_calls(0),
        );

        let error = match agent.prompt("hello").await {
            Ok(_) => panic!("zero budget should reject"),
            Err(error) => error,
        };
        let BevyRunError::BudgetExhausted { handle } = error else {
            panic!("expected budget exhaustion");
        };
        assert_eq!(model.request_count(), 0);
        assert_eq!(memory.load_count(), 0);
        assert_eq!(handle.terminal(), Ok(Some(TerminalReason::BudgetExhausted)));
    }
}
