//! Ordered schedule labels, system sets, and deferred-command boundaries.

use std::collections::BTreeMap;

use bevy_ecs::{
    entity::Entity,
    query::With,
    resource::Resource,
    schedule::{ApplyDeferred, IntoScheduleConfigs, Schedule, ScheduleLabel, SystemSet},
    system::{Commands, Query, Res, ResMut},
    world::World,
};
use rig_core::{
    OneOrMany,
    completion::{AssistantContent, ToolDefinition},
    message::{Message, ToolCall},
};

use crate::{
    components::{
        AgentNode, CallBudget, CancellationRequested, CapabilitySnapshot, CommittedTranscript,
        ModelOperation, ProgressState, ResponseRecovery, RetainUntil, RetentionWindow, RunNode,
        RunPhase, StoreBinding, StoreOperation, TerminalReason, TerminalState, ToolCallNode,
    },
    effects::{
        IngressDecision, IngressDiagnostics, ModelCommitAction, ModelDispatchFailure,
        ModelDispatchOutcome, ModelDispatchQueue, ModelEffectRequest, ModelIngressApplied,
        ModelIngressOutcome, ModelIngressOutcomeQueue, ModelIngressQueue, ModelPolicyFailure,
        PreparedModelEffect, PreparedModelQueue, PreparedStoreEffect, PreparedStoreQueue,
        RejectedIngress, StoreDispatchOutcome, StoreDispatchQueue, StoreEffectCompletion,
        StoreEffectRequest, StoreIngressApplied, StoreIngressOutcome, StoreIngressOutcomeQueue,
        StoreIngressQueue, ToolEffectRequest, ToolIngressOutcome, ToolIngressOutcomeQueue,
        ToolIngressQueue, ToolPlan, ValidatedModelIngressQueue, ValidatedStoreIngressQueue,
        ValidatedToolIngressQueue, classify_model_ingress, commit_model_accounting,
        commit_terminal,
    },
    policy::{AgentPolicy, OutputMode, ToolCatalog, ToolGrant, synthetic_output_tool_name},
    topology::{
        EffectIdentity, OperationId, OwnedByAgent, RunId, StoreOperationId, ToolCallId, WorldId,
    },
};

/// Deterministic schedule used by local and hosted drivers.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RigSchedule;

/// Ordered runtime phases. Ingress is deliberately ahead of policy and dispatch.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RigSet {
    /// Validate owned effect completions.
    Ingress,
    /// Resolve cancellation, invalid calls, retry, output, and stop policy.
    Policy,
    /// Make commands from ingress/policy visible before dispatch.
    ApplyPolicy,
    /// Snapshot owned effect inputs and queue work.
    Dispatch,
    /// Commit accepted canonical facts.
    Commit,
    /// Make terminal facts externally observable.
    ApplyCommit,
    /// Detect progress, quiescence, and livelock.
    Progress,
    /// Mark retention-expired runs for cleanup.
    Cleanup,
}

/// Monotonic runtime schedule tick.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeTick(pub u64);

/// Runtime-world identity consulted by ingress and dispatch systems.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeWorld(pub WorldId);

impl Default for RuntimeWorld {
    fn default() -> Self {
        Self(WorldId(0))
    }
}

/// Resource describing whether the latest schedule pass changed authoritative state.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct ScheduleProgress {
    changes: BTreeMap<RunId, u64>,
}

impl ScheduleProgress {
    pub(crate) fn record(&mut self, run: RunId) {
        let changes = self.changes.entry(run).or_default();
        *changes = changes.saturating_add(1);
    }

    fn take(&mut self, run: RunId) -> u64 {
        self.changes.remove(&run).unwrap_or(0)
    }
}

/// Build the runtime schedule with explicit deferred boundaries.
pub fn runtime_schedule() -> Schedule {
    let mut schedule = Schedule::new(RigSchedule);
    schedule.configure_sets(
        (
            RigSet::Ingress,
            RigSet::Policy,
            RigSet::ApplyPolicy,
            RigSet::Dispatch,
            RigSet::Commit,
            RigSet::ApplyCommit,
            RigSet::Progress,
            RigSet::Cleanup,
        )
            .chain(),
    );
    schedule.add_systems((
        increment_tick.in_set(RigSet::Ingress),
        validate_model_ingress.in_set(RigSet::Ingress),
        validate_tool_ingress.in_set(RigSet::Ingress),
        validate_store_ingress.in_set(RigSet::Ingress),
        resolve_cancellation.in_set(RigSet::Policy),
        ApplyDeferred.in_set(RigSet::ApplyPolicy),
        prepare_model_effects.in_set(RigSet::Dispatch),
        prepare_store_effects.in_set(RigSet::Dispatch),
        commit_model_ingress.in_set(RigSet::Commit),
        commit_tool_ingress.in_set(RigSet::Commit),
        commit_store_ingress.in_set(RigSet::Commit),
        publish_terminal_phase.in_set(RigSet::Commit),
        ApplyDeferred.in_set(RigSet::ApplyCommit),
        track_progress.in_set(RigSet::Progress),
        mark_cleanup_eligible.in_set(RigSet::Cleanup),
    ));
    schedule
}

/// Insert required resources and the deterministic schedule into `world`.
pub fn initialize_world(world: &mut World) {
    world.init_resource::<RuntimeTick>();
    world.init_resource::<ScheduleProgress>();
    world.init_resource::<RuntimeWorld>();
    world.init_resource::<IngressDiagnostics>();
    world.init_resource::<ModelDispatchQueue>();
    world.init_resource::<PreparedModelQueue>();
    world.init_resource::<ModelIngressQueue>();
    world.init_resource::<ValidatedModelIngressQueue>();
    world.init_resource::<ModelIngressOutcomeQueue>();
    world.init_resource::<ToolIngressQueue>();
    world.init_resource::<ValidatedToolIngressQueue>();
    world.init_resource::<ToolIngressOutcomeQueue>();
    world.init_resource::<StoreDispatchQueue>();
    world.init_resource::<PreparedStoreQueue>();
    world.init_resource::<StoreIngressQueue>();
    world.init_resource::<ValidatedStoreIngressQueue>();
    world.init_resource::<StoreIngressOutcomeQueue>();
    let mut schedules = world
        .remove_resource::<bevy_ecs::schedule::Schedules>()
        .unwrap_or_default();
    schedules.insert(runtime_schedule());
    world.insert_resource(schedules);
}

pub(crate) fn set_runtime_world(world: &mut World, identity: WorldId) {
    world.insert_resource(RuntimeWorld(identity));
}

fn record_progress(world: &mut World, run: RunId) {
    world.resource_mut::<ScheduleProgress>().record(run);
}

fn prepare_model_effects(world: &mut World) {
    let mut intents = world.resource_mut::<ModelDispatchQueue>().0.drain();
    intents.sort_by_key(|intent| intent.run);
    for intent in intents {
        let run = intent.run;
        let outcome = prepare_model_effect(world, intent);
        if world
            .resource_mut::<PreparedModelQueue>()
            .0
            .push(outcome)
            .is_err()
        {
            let tick = world.resource::<RuntimeTick>().0;
            commit_terminal(world, run, TerminalReason::Livelock, tick);
        }
    }
}

fn prepare_model_effect(
    world: &mut World,
    mut intent: crate::effects::ModelDispatchIntent,
) -> ModelDispatchOutcome {
    let runtime_world = world.resource::<RuntimeWorld>().0;
    let facts = world
        .query::<(&AgentNode, &AgentPolicy)>()
        .iter(world)
        .find(|(agent, _)| agent.id == intent.agent)
        .map(|(agent, policy)| (agent.clone(), policy.clone()));
    let Some((agent, policy)) = facts else {
        return ModelDispatchOutcome::Failed {
            run: intent.run,
            failure: ModelDispatchFailure::MissingAgent,
        };
    };
    if agent.tenant != intent.tenant || policy.agent != intent.agent {
        return ModelDispatchOutcome::Failed {
            run: intent.run,
            failure: ModelDispatchFailure::InvalidPolicy,
        };
    }
    let grants = world
        .query::<(&OwnedByAgent, &ToolGrant)>()
        .iter(world)
        .filter(|(owner, grant)| owner.0 == intent.agent && grant.tenant == intent.tenant)
        .map(|(_, grant)| grant.clone())
        .collect::<Vec<_>>();
    let output_schema = match policy
        .output_schema
        .clone()
        .map(serde_json::from_value::<schemars::Schema>)
        .transpose()
    {
        Ok(schema) => schema,
        Err(_) => {
            return ModelDispatchOutcome::Failed {
                run: intent.run,
                failure: ModelDispatchFailure::InvalidPolicy,
            };
        }
    };
    let run = world
        .query::<(Entity, &RunNode, &mut CallBudget)>()
        .iter_mut(world)
        .find(|(_, run, _)| run.id == intent.run)
        .map(|(entity, run, mut budget)| {
            let consumed = budget.consume();
            (entity, run.clone(), consumed, budget.dispatched)
        });
    let Some((run_entity, run, consumed, correlation)) = run else {
        return ModelDispatchOutcome::Failed {
            run: intent.run,
            failure: ModelDispatchFailure::MissingRun,
        };
    };
    if run.agent != intent.agent || run.tenant != intent.tenant {
        return ModelDispatchOutcome::Failed {
            run: intent.run,
            failure: ModelDispatchFailure::InvalidPolicy,
        };
    }
    if !consumed {
        let tick = world.resource::<RuntimeTick>().0;
        commit_terminal(world, intent.run, TerminalReason::BudgetExhausted, tick);
        return ModelDispatchOutcome::Failed {
            run: intent.run,
            failure: ModelDispatchFailure::BudgetExhausted,
        };
    }
    let operation = OperationId::allocate();
    let identity = EffectIdentity {
        world: runtime_world,
        tenant: intent.tenant,
        run: intent.run,
        operation,
        generation: run.generation,
        correlation: correlation as u64,
    };
    let snapshot = world.resource::<ToolCatalog>().snapshot(identity, &grants);
    let mut tools = snapshot
        .tools
        .iter()
        .map(|tool| tool.definition.clone())
        .collect::<Vec<_>>();
    let mut preamble = agent.preamble;
    let mut native_schema = None;
    let output_tool = output_schema.as_ref().and_then(|schema| {
        let mode = match policy.output_mode {
            OutputMode::Auto if !tools.is_empty() && !intent.composes_native_output_with_tools => {
                OutputMode::Tool
            }
            OutputMode::Auto => OutputMode::Native,
            mode => mode,
        };
        match mode {
            OutputMode::Native => {
                native_schema = Some(schema.clone());
                None
            }
            OutputMode::Tool => {
                let name = synthetic_output_tool_name(
                    tools.iter().map(|definition| definition.name.clone()),
                );
                tools.push(ToolDefinition {
                    name: name.clone(),
                    description: "Submit the final structured answer exactly once.".into(),
                    parameters: schema.clone().to_value(),
                });
                append_instruction(
                    &mut preamble,
                    format!(
                        "When the answer is ready, call `{name}` exactly once with the final \
                         structured result. Do not call ordinary tools after finalizing."
                    ),
                );
                Some(name)
            }
            OutputMode::Prompted => {
                let encoded = serde_json::to_string(schema.as_value()).unwrap_or_default();
                append_instruction(
                    &mut preamble,
                    format!(
                        "Return only one JSON object matching this JSON Schema, without prose \
                         or markdown fences:\n{encoded}"
                    ),
                );
                None
            }
            OutputMode::Auto => None,
        }
    });
    let mut messages = intent.transcript;
    if let Some(preamble) = preamble {
        messages.insert(0, Message::system(preamble));
    }
    messages.push(intent.prompt.clone());
    intent.request.chat_history =
        OneOrMany::from_iter_optional(messages).unwrap_or_else(|| OneOrMany::one(intent.prompt));
    intent.request.tools.extend(tools);
    intent.request.additional_params =
        match (intent.request.additional_params, policy.additional_params) {
            (Some(base), Some(policy)) => Some(rig_core::json_utils::merge(base, policy)),
            (base, policy) => policy.or(base),
        };
    intent.request.temperature = policy.temperature.or(intent.request.temperature);
    intent.request.max_tokens = policy.max_tokens.or(intent.request.max_tokens);
    intent.request.output_schema = native_schema.or(intent.request.output_schema);
    if let Some(mut run) = world.entity_mut(run_entity).get_mut::<RunNode>() {
        run.phase = RunPhase::Waiting;
    }
    world.spawn((
        ModelOperation {
            id: operation,
            effect: identity,
            committed: false,
            retired: false,
        },
        ModelEffectRequest {
            identity,
            request: intent.request.clone(),
            streaming: intent.streaming,
        },
    ));
    world.spawn(snapshot);
    record_progress(world, identity.run);
    ModelDispatchOutcome::Prepared(Box::new(PreparedModelEffect {
        identity,
        request: intent.request,
        output_tool,
    }))
}

fn validate_model_ingress(world: &mut World) {
    let mut commands = world.resource_mut::<ModelIngressQueue>().0.drain();
    commands.sort_by_key(|command| command.identity);
    let runtime_world = world.resource::<RuntimeWorld>().0;
    for command in commands {
        let decision = classify_model_ingress(world, runtime_world, command.identity);
        if decision == IngressDecision::Accept {
            if let Err(command) = world
                .resource_mut::<ValidatedModelIngressQueue>()
                .0
                .push(command)
            {
                let tick = world.resource::<RuntimeTick>().0;
                commit_terminal(world, command.identity.run, TerminalReason::Livelock, tick);
            }
        } else {
            world
                .resource_mut::<IngressDiagnostics>()
                .0
                .push(RejectedIngress {
                    identity: command.identity,
                    decision,
                });
            let _ = world
                .resource_mut::<ModelIngressOutcomeQueue>()
                .0
                .push(ModelIngressOutcome {
                    identity: command.identity,
                    applied: ModelIngressApplied::Rejected(decision),
                });
        }
    }
}

fn commit_model_ingress(world: &mut World) {
    let mut commands = world.resource_mut::<ValidatedModelIngressQueue>().0.drain();
    commands.sort_by_key(|command| command.identity);
    for command in commands {
        let identity = command.identity;
        if !commit_model_accounting(world, identity, command.usage) {
            let _ = world
                .resource_mut::<ModelIngressOutcomeQueue>()
                .0
                .push(ModelIngressOutcome {
                    identity,
                    applied: ModelIngressApplied::Rejected(IngressDecision::Duplicate),
                });
            continue;
        }
        record_progress(world, identity.run);
        let applied = match command.action {
            ModelCommitAction::Evaluate {
                prompt,
                choice,
                message_id,
            } => apply_evaluated_turn(world, identity, prompt, choice, message_id),
        };
        let _ = world
            .resource_mut::<ModelIngressOutcomeQueue>()
            .0
            .push(ModelIngressOutcome { identity, applied });
    }
}

fn apply_evaluated_turn(
    world: &mut World,
    identity: EffectIdentity,
    prompt: Option<Message>,
    mut choice: OneOrMany<AssistantContent>,
    message_id: Option<String>,
) -> ModelIngressApplied {
    let agent = world
        .query::<&RunNode>()
        .iter(world)
        .find(|run| run.id == identity.run)
        .map(|run| run.agent);
    let Some(agent) = agent else {
        return ModelIngressApplied::Rejected(IngressDecision::UnknownOperation);
    };
    let policy = world
        .query::<&AgentPolicy>()
        .iter(world)
        .find(|policy| policy.agent == agent)
        .cloned();
    let Some(policy) = policy else {
        return ModelIngressApplied::Rejected(IngressDecision::UnknownOperation);
    };

    let mut calls = tool_calls(&choice);
    if calls.is_empty()
        && let Some(schema) = policy.output_schema.as_ref()
        && !choice_satisfies_schema(&choice, schema)
    {
        let can_retry = world
            .query::<(&RunNode, &mut ResponseRecovery)>()
            .iter_mut(world)
            .find(|(run, _)| run.id == identity.run)
            .is_some_and(|(_, mut recovery)| {
                if recovery.retries >= policy.response_retry.max_retries {
                    false
                } else {
                    recovery.retries += 1;
                    true
                }
            });
        if can_retry {
            set_run_ready(world, identity.run);
            return ModelIngressApplied::Retry {
                feedback: policy.response_retry.feedback,
            };
        }
        if !policy.response_retry.best_effort {
            let tick = world.resource::<RuntimeTick>().0;
            commit_terminal(
                world,
                identity.run,
                TerminalReason::OutputFailure("structured output validation failed".into()),
                tick,
            );
            return ModelIngressApplied::PolicyFailure(ModelPolicyFailure::StructuredOutput);
        }
    }

    let snapshot = world
        .query::<&CapabilitySnapshot>()
        .iter(world)
        .find(|snapshot| snapshot.effect == identity)
        .cloned();
    let Some(snapshot) = snapshot else {
        return ModelIngressApplied::Rejected(IngressDecision::WrongCorrelation);
    };
    let invalid_name = calls
        .iter()
        .find(|call| {
            !snapshot
                .tools
                .iter()
                .any(|tool| tool.name == call.function.name)
        })
        .map(|call| call.function.name.clone());
    let mut suppressed_reason = None;
    if let Some(invalid_name) = invalid_name {
        match policy.invalid_tool {
            crate::policy::InvalidToolPolicy::Fail => {
                let tick = world.resource::<RuntimeTick>().0;
                commit_terminal(
                    world,
                    identity.run,
                    TerminalReason::ToolFailure("invalid tool call".into()),
                    tick,
                );
                return ModelIngressApplied::PolicyFailure(ModelPolicyFailure::UnknownTool(
                    invalid_name,
                ));
            }
            crate::policy::InvalidToolPolicy::Stop { reason } => {
                let tick = world.resource::<RuntimeTick>().0;
                commit_terminal(
                    world,
                    identity.run,
                    TerminalReason::Stopped(reason.clone()),
                    tick,
                );
                return ModelIngressApplied::PolicyFailure(ModelPolicyFailure::Stopped(reason));
            }
            crate::policy::InvalidToolPolicy::Retry { feedback } => {
                set_run_ready(world, identity.run);
                return ModelIngressApplied::Retry { feedback };
            }
            crate::policy::InvalidToolPolicy::Repair { name, arguments } => {
                let repaired_arguments = serde_json::from_str::<serde_json::Value>(&arguments);
                let target = snapshot.tools.iter().find(|tool| tool.name == name);
                let (Some(_), Ok(repaired_arguments)) = (target, repaired_arguments) else {
                    let tick = world.resource::<RuntimeTick>().0;
                    commit_terminal(
                        world,
                        identity.run,
                        TerminalReason::ToolFailure("invalid tool repair policy".into()),
                        tick,
                    );
                    return ModelIngressApplied::PolicyFailure(ModelPolicyFailure::InvalidRepair);
                };
                for content in choice.iter_mut() {
                    if let AssistantContent::ToolCall(call) = content
                        && !snapshot
                            .tools
                            .iter()
                            .any(|tool| tool.name == call.function.name)
                    {
                        let original_name = call.function.name.clone();
                        call.function.name = name.clone();
                        call.function.arguments = repaired_arguments.clone();
                        // Gemini uses the function name as its provider-side
                        // correlation ID. Keep opaque IDs from other providers,
                        // but repair name-shaped IDs so the assistant call and
                        // subsequent function response remain coherent.
                        if call.call_id.is_none() && call.id == original_name {
                            call.id = name.clone();
                        }
                    }
                }
                calls = tool_calls(&choice);
            }
            crate::policy::InvalidToolPolicy::Skip { reason } => {
                suppressed_reason = Some(reason);
            }
        }
    }

    let tools = if let Some(reason) = suppressed_reason {
        for (ordinal, call) in calls.iter().enumerate() {
            world.spawn(ToolCallNode {
                id: ToolCallId::allocate(),
                effect: identity,
                run: identity.run,
                capability: snapshot.id,
                name: call.function.name.clone(),
                revision: 0,
                ordinal,
                suppressed: true,
                committed: true,
            });
        }
        ToolPlan::Suppressed { calls, reason }
    } else if calls.is_empty() {
        ToolPlan::None
    } else {
        let mut requests = Vec::with_capacity(calls.len());
        for (ordinal, call) in calls.iter().enumerate() {
            let Some(tool) = snapshot
                .tools
                .iter()
                .find(|tool| tool.name == call.function.name)
            else {
                return ModelIngressApplied::Rejected(IngressDecision::WrongCorrelation);
            };
            let call_id = ToolCallId::allocate();
            let request = ToolEffectRequest {
                identity,
                call: call_id,
                capability: snapshot.id,
                name: tool.name.clone(),
                revision: tool.revision,
                arguments: call.function.arguments.to_string(),
                ordinal,
            };
            world.spawn((
                ToolCallNode {
                    id: call_id,
                    effect: identity,
                    run: identity.run,
                    capability: snapshot.id,
                    name: tool.name.clone(),
                    revision: tool.revision,
                    ordinal,
                    suppressed: false,
                    committed: false,
                },
                request.clone(),
            ));
            requests.push(request);
        }
        ToolPlan::Dispatch {
            calls,
            requests,
            concurrency: policy.max_tool_concurrency.max(1),
        }
    };

    let transcript = world
        .query::<(&mut RunNode, &mut CommittedTranscript)>()
        .iter_mut(world)
        .find(|(run, _)| run.id == identity.run)
        .map(|(mut run, mut transcript)| {
            if let Some(prompt) = prompt {
                transcript.0.push(prompt);
            }
            transcript.0.push(Message::Assistant {
                id: message_id,
                content: choice.clone(),
            });
            run.phase = if matches!(tools, ToolPlan::Dispatch { .. }) {
                RunPhase::Waiting
            } else {
                RunPhase::Ready
            };
            transcript.0.clone()
        });
    if transcript.is_some() {
        record_progress(world, identity.run);
    }
    match transcript {
        Some(transcript) => ModelIngressApplied::Accepted {
            transcript,
            choice: Box::new(choice),
            tools,
        },
        None => ModelIngressApplied::Rejected(IngressDecision::UnknownOperation),
    }
}

fn tool_calls(choice: &OneOrMany<AssistantContent>) -> Vec<ToolCall> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

fn choice_satisfies_schema(
    choice: &OneOrMany<AssistantContent>,
    schema: &serde_json::Value,
) -> bool {
    let text = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<String>();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text.trim()) else {
        return false;
    };
    jsonschema::validator_for(schema).is_ok_and(|validator| validator.is_valid(&value))
}

fn set_run_ready(world: &mut World, run_id: crate::topology::RunId) {
    let changed = world
        .query::<&mut RunNode>()
        .iter_mut(world)
        .find(|run| run.id == run_id)
        .is_some_and(|mut run| {
            let changed = run.phase != RunPhase::Ready;
            run.phase = RunPhase::Ready;
            changed
        });
    if changed {
        record_progress(world, run_id);
    }
}

fn validate_tool_ingress(world: &mut World) {
    let mut commands = world.resource_mut::<ToolIngressQueue>().0.drain();
    commands.sort_by_key(|command| command.completion.ordinal);
    for command in commands {
        let completion = &command.completion;
        let decision = classify_tool_ingress(world, completion);
        if decision == IngressDecision::Accept {
            if let Err(command) = world
                .resource_mut::<ValidatedToolIngressQueue>()
                .0
                .push(command)
            {
                let tick = world.resource::<RuntimeTick>().0;
                commit_terminal(
                    world,
                    command.completion.identity.run,
                    TerminalReason::Livelock,
                    tick,
                );
            }
        } else {
            world
                .resource_mut::<IngressDiagnostics>()
                .0
                .push(RejectedIngress {
                    identity: completion.identity,
                    decision,
                });
            let _ = world
                .resource_mut::<ToolIngressOutcomeQueue>()
                .0
                .push(ToolIngressOutcome {
                    call: completion.call,
                    decision,
                });
        }
    }
}

fn classify_tool_ingress(
    world: &mut World,
    completion: &crate::effects::ToolEffectCompletion,
) -> IngressDecision {
    let runtime_world = world.resource::<RuntimeWorld>().0;
    if completion.identity.world != runtime_world {
        return IngressDecision::ForeignWorld;
    }
    let Some(run) = world
        .query::<&RunNode>()
        .iter(world)
        .find(|run| run.id == completion.identity.run)
        .cloned()
    else {
        return IngressDecision::UnknownOperation;
    };
    if run.tenant != completion.identity.tenant {
        return IngressDecision::WrongTenant;
    }
    if run.generation != completion.identity.generation {
        return IngressDecision::StaleGeneration;
    }
    if matches!(run.phase, RunPhase::Terminal | RunPhase::CleanupEligible) {
        return IngressDecision::Late;
    }
    let Some(call) = world
        .query::<&ToolCallNode>()
        .iter(world)
        .find(|call| call.id == completion.call)
    else {
        return IngressDecision::UnknownOperation;
    };
    if call.effect != completion.identity
        || call.run != completion.identity.run
        || call.capability != completion.capability
        || call.revision != completion.revision
        || call.ordinal != completion.ordinal
    {
        return IngressDecision::WrongCorrelation;
    }
    if call.suppressed {
        return IngressDecision::Retired;
    }
    if call.committed {
        return IngressDecision::Duplicate;
    }
    IngressDecision::Accept
}

fn commit_tool_ingress(world: &mut World) {
    let mut commands = world.resource_mut::<ValidatedToolIngressQueue>().0.drain();
    commands.sort_by_key(|command| command.completion.ordinal);
    for command in commands {
        let completion = command.completion;
        let committed = world
            .query::<&mut ToolCallNode>()
            .iter_mut(world)
            .find(|call| call.id == completion.call)
            .is_some_and(|mut call| {
                if call.committed {
                    false
                } else {
                    call.committed = true;
                    true
                }
            });
        let decision = if committed {
            record_progress(world, completion.identity.run);
            IngressDecision::Accept
        } else {
            IngressDecision::Duplicate
        };
        let _ = world
            .resource_mut::<ToolIngressOutcomeQueue>()
            .0
            .push(ToolIngressOutcome {
                call: completion.call,
                decision,
            });
    }
}

fn prepare_store_effects(world: &mut World) {
    let mut intents = world.resource_mut::<StoreDispatchQueue>().0.drain();
    intents.sort_by_key(|intent| intent.run);
    for intent in intents {
        let outcome = prepare_store_effect(world, intent);
        if let Err(outcome) = world.resource_mut::<PreparedStoreQueue>().0.push(outcome) {
            let run = match outcome {
                StoreDispatchOutcome::Prepared(prepared) => prepared.request.identity.run,
                StoreDispatchOutcome::NoBinding { run }
                | StoreDispatchOutcome::Rejected { run, .. } => run,
            };
            let tick = world.resource::<RuntimeTick>().0;
            commit_terminal(world, run, TerminalReason::Livelock, tick);
        }
    }
}

fn prepare_store_effect(
    world: &mut World,
    intent: crate::effects::StoreDispatchIntent,
) -> StoreDispatchOutcome {
    let run = world
        .query::<&RunNode>()
        .iter(world)
        .find(|run| run.id == intent.run)
        .cloned();
    let Some(run) = run else {
        return StoreDispatchOutcome::Rejected {
            run: intent.run,
            decision: IngressDecision::UnknownOperation,
        };
    };
    if run.agent != intent.agent {
        return StoreDispatchOutcome::Rejected {
            run: intent.run,
            decision: IngressDecision::WrongCorrelation,
        };
    }
    if matches!(run.phase, RunPhase::Terminal | RunPhase::CleanupEligible) {
        return StoreDispatchOutcome::Rejected {
            run: intent.run,
            decision: IngressDecision::Late,
        };
    }
    let binding = world
        .query::<(&AgentNode, Option<&StoreBinding>)>()
        .iter(world)
        .find(|(agent, _)| agent.id == intent.agent)
        .and_then(|(_, binding)| binding.cloned());
    let Some(binding) = binding else {
        return StoreDispatchOutcome::NoBinding { run: intent.run };
    };
    let store_operation = StoreOperationId::allocate();
    let identity = EffectIdentity {
        world: world.resource::<RuntimeWorld>().0,
        tenant: run.tenant,
        run: run.id,
        operation: OperationId::allocate(),
        generation: run.generation,
        correlation: store_operation.0,
    };
    let request = StoreEffectRequest {
        identity,
        store_operation,
        kind: intent.kind,
    };
    world.spawn((
        StoreOperation {
            id: store_operation,
            effect: identity,
            committed: false,
            retired: false,
        },
        request.clone(),
    ));
    record_progress(world, run.id);
    StoreDispatchOutcome::Prepared(PreparedStoreEffect {
        binding: binding.implementation,
        request,
    })
}

fn validate_store_ingress(world: &mut World) {
    let mut commands = world.resource_mut::<StoreIngressQueue>().0.drain();
    commands.sort_by_key(|command| command.store_operation);
    for command in commands {
        let decision = classify_store_ingress(world, &command);
        if decision == IngressDecision::Accept {
            if let Err(command) = world
                .resource_mut::<ValidatedStoreIngressQueue>()
                .0
                .push(command)
            {
                let tick = world.resource::<RuntimeTick>().0;
                commit_terminal(world, command.identity.run, TerminalReason::Livelock, tick);
            }
        } else {
            world
                .resource_mut::<IngressDiagnostics>()
                .0
                .push(RejectedIngress {
                    identity: command.identity,
                    decision,
                });
            let _ = world
                .resource_mut::<StoreIngressOutcomeQueue>()
                .0
                .push(StoreIngressOutcome {
                    store_operation: command.store_operation,
                    applied: StoreIngressApplied::Rejected(decision),
                });
        }
    }
}

fn classify_store_ingress(
    world: &mut World,
    command: &crate::effects::StoreIngressCommand,
) -> IngressDecision {
    if command.identity.world != world.resource::<RuntimeWorld>().0 {
        return IngressDecision::ForeignWorld;
    }
    let Some(run) = world
        .query::<&RunNode>()
        .iter(world)
        .find(|run| run.id == command.identity.run)
        .cloned()
    else {
        return IngressDecision::UnknownOperation;
    };
    if run.tenant != command.identity.tenant {
        return IngressDecision::WrongTenant;
    }
    if run.generation != command.identity.generation {
        return IngressDecision::StaleGeneration;
    }
    if matches!(run.phase, RunPhase::Terminal | RunPhase::CleanupEligible) {
        return IngressDecision::Late;
    }
    let Some(operation) = world
        .query::<&StoreOperation>()
        .iter(world)
        .find(|operation| operation.id == command.store_operation)
    else {
        return IngressDecision::UnknownOperation;
    };
    if operation.effect != command.identity {
        return IngressDecision::WrongCorrelation;
    }
    if operation.retired {
        return IngressDecision::Retired;
    }
    if operation.committed {
        return IngressDecision::Duplicate;
    }
    IngressDecision::Accept
}

fn commit_store_ingress(world: &mut World) {
    let mut commands = world.resource_mut::<ValidatedStoreIngressQueue>().0.drain();
    commands.sort_by_key(|command| command.store_operation);
    for command in commands {
        let applied = match command.completion {
            StoreEffectCompletion::Loaded(messages) => {
                let committed = world
                    .query::<(&mut RunNode, &mut CommittedTranscript)>()
                    .iter_mut(world)
                    .find(|(run, _)| run.id == command.identity.run)
                    .map(|(mut run, mut transcript)| {
                        transcript.0 = messages.clone();
                        run.phase = RunPhase::Ready;
                    })
                    .is_some();
                if committed {
                    StoreIngressApplied::Loaded(messages)
                } else {
                    StoreIngressApplied::Rejected(IngressDecision::UnknownOperation)
                }
            }
            StoreEffectCompletion::Appended => StoreIngressApplied::Appended,
            StoreEffectCompletion::Failed(error) => {
                let tick = world.resource::<RuntimeTick>().0;
                commit_terminal(
                    world,
                    command.identity.run,
                    TerminalReason::StoreFailure("store effect failed".into()),
                    tick,
                );
                StoreIngressApplied::Failed(error)
            }
        };
        let operation_changed = if let Some(mut operation) = world
            .query::<&mut StoreOperation>()
            .iter_mut(world)
            .find(|operation| operation.id == command.store_operation)
        {
            match applied {
                StoreIngressApplied::Failed(_) => operation.retired = true,
                _ => operation.committed = true,
            }
            true
        } else {
            false
        };
        if operation_changed {
            record_progress(world, command.identity.run);
        }
        let _ = world
            .resource_mut::<StoreIngressOutcomeQueue>()
            .0
            .push(StoreIngressOutcome {
                store_operation: command.store_operation,
                applied,
            });
    }
}

fn append_instruction(preamble: &mut Option<String>, instruction: String) {
    match preamble {
        Some(preamble) => {
            preamble.push_str("\n\n");
            preamble.push_str(&instruction);
        }
        None => *preamble = Some(instruction),
    }
}

fn increment_tick(mut tick: ResMut<'_, RuntimeTick>) {
    tick.0 = tick.0.saturating_add(1);
}

fn resolve_cancellation(
    mut commands: Commands<'_, '_>,
    tick: Res<'_, RuntimeTick>,
    mut progress: ResMut<'_, ScheduleProgress>,
    runs: Query<
        '_,
        '_,
        (
            Entity,
            &RunNode,
            &CancellationRequested,
            Option<&TerminalState>,
            &RetentionWindow,
        ),
    >,
) {
    let mut pending = runs
        .iter()
        .filter(|(_, _, _, terminal, _)| terminal.is_none())
        .map(|(entity, run, cancellation, _, retention)| {
            (run.id, entity, cancellation.0.clone(), retention.0)
        })
        .collect::<Vec<_>>();
    pending.sort_by_key(|(run, _, _, _)| *run);
    for (run, entity, reason, retention) in pending {
        commands.entity(entity).insert((
            TerminalState {
                reason: TerminalReason::Cancelled(reason),
                committed_tick: tick.0,
            },
            RetainUntil(tick.0.saturating_add(retention)),
        ));
        progress.record(run);
    }
}

fn publish_terminal_phase(
    mut runs: Query<'_, '_, &mut RunNode, With<TerminalState>>,
    mut progress: ResMut<'_, ScheduleProgress>,
) {
    for mut run in &mut runs {
        if run.phase != RunPhase::Terminal {
            run.phase = RunPhase::Terminal;
            progress.record(run.id);
        }
    }
}

fn track_progress(
    tick: Res<'_, RuntimeTick>,
    mut schedule_progress: ResMut<'_, ScheduleProgress>,
    mut runs: Query<
        '_,
        '_,
        (
            Entity,
            &mut ProgressState,
            &RunNode,
            Option<&TerminalState>,
            &RetentionWindow,
        ),
    >,
    mut commands: Commands<'_, '_>,
) {
    let mut ordered = runs.iter_mut().collect::<Vec<_>>();
    ordered.sort_by_key(|(_, _, run, _, _)| run.id);
    for (entity, mut progress, run, terminal, retention) in ordered {
        let changes = schedule_progress.take(run.id);
        if terminal.is_some() {
            continue;
        }
        if changes > 0 {
            progress.changes = progress.changes.saturating_add(changes);
            progress.idle_passes = 0;
        } else {
            progress.idle_passes = progress.idle_passes.saturating_add(1);
            if progress.idle_passes > progress.max_idle_passes {
                commands.entity(entity).insert((
                    TerminalState {
                        reason: TerminalReason::Livelock,
                        committed_tick: tick.0,
                    },
                    RetainUntil(tick.0.saturating_add(retention.0)),
                ));
            }
        }
    }
    schedule_progress.changes.clear();
}

fn mark_cleanup_eligible(
    tick: Res<'_, RuntimeTick>,
    mut runs: Query<'_, '_, (&mut RunNode, &RetainUntil, &TerminalState)>,
) {
    for (mut run, retain, _terminal) in &mut runs {
        if tick.0 >= retain.0 {
            run.phase = RunPhase::CleanupEligible;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        components::{CallBudget, CommittedTranscript, UsageLedger},
        topology::{AgentId, Generation, RunId, TenantId},
    };
    use rig_core::completion::Usage;

    #[test]
    fn cancellation_is_observable_before_cleanup() {
        let mut world = World::new();
        initialize_world(&mut world);
        let entity = world
            .spawn((
                RunNode {
                    id: RunId::allocate(),
                    agent: AgentId::allocate(),
                    tenant: TenantId(1),
                    generation: Generation(0),
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
                    max_idle_passes: 10,
                },
                CancellationRequested("caller".into()),
                RetainUntil(u64::MAX),
                RetentionWindow(10),
            ))
            .id();

        world.run_schedule(RigSchedule);
        let terminal = world
            .entity(entity)
            .get::<TerminalState>()
            .cloned()
            .expect("terminal state should be visible");
        assert_eq!(terminal.reason, TerminalReason::Cancelled("caller".into()));
        assert_eq!(
            world.entity(entity).get::<RunNode>().map(|run| run.phase),
            Some(RunPhase::Terminal)
        );
        let run = world.entity(entity).get::<RunNode>().expect("run node").id;
        assert!(
            !world
                .resource::<ScheduleProgress>()
                .changes
                .contains_key(&run)
        );
    }

    #[test]
    fn progress_from_one_run_does_not_mask_another_runs_livelock() {
        let mut world = World::new();
        initialize_world(&mut world);
        let active = RunId::allocate();
        let stuck = RunId::allocate();
        for run in [active, stuck] {
            world.spawn((
                RunNode {
                    id: run,
                    agent: AgentId::allocate(),
                    tenant: TenantId(1),
                    generation: Generation(0),
                    phase: RunPhase::Waiting,
                },
                ProgressState {
                    changes: 0,
                    idle_passes: 0,
                    max_idle_passes: 1,
                },
                RetainUntil(u64::MAX),
                RetentionWindow(10),
            ));
        }

        for _ in 0..2 {
            world.resource_mut::<ScheduleProgress>().record(active);
            world.run_schedule(RigSchedule);
        }

        let states = world
            .query::<(&RunNode, &ProgressState, Option<&TerminalState>)>()
            .iter(&world)
            .map(|(run, progress, terminal)| {
                (
                    run.id,
                    (
                        progress.idle_passes,
                        terminal.map(|state| state.reason.clone()),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(states.get(&active), Some(&(0, None)));
        assert_eq!(
            states.get(&stuck),
            Some(&(2, Some(TerminalReason::Livelock)))
        );
    }
}
