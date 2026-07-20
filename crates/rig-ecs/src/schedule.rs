//! Deterministic ordered ECS schedule and progression systems.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_ecs::{
    prelude::*,
    schedule::{IntoScheduleConfigs, ScheduleLabel, SystemSet},
};
use rig_core::{
    OneOrMany,
    completion::{AssistantContent, CompletionRequest, Message, ToolDefinition},
    message::{ToolCall, ToolChoice, ToolResult, UserContent},
    tool::ToolOutput,
};
use sha2::{Digest, Sha256};

use crate::{
    CapabilityId, CorrelationId, EffectHeader, EffectIngress, EffectRejection,
    EffectRejectionReason, InvalidToolPolicy, OperationId, OutputMode, ResponseRetryPolicy,
    RunEvent, RunId, RuntimeConfig, StreamingMode, TerminalReason,
    components::{
        AcceptedDeltas, ActiveOperation, ActiveOperationKind, ActiveOperations,
        AdvertisedCapability, AgentNode, CancellationRequest, CanonicalTranscript,
        CapabilitiesToDrop, CapabilityNode, CapabilityReferences, EffectQueueWait, GrantNode,
        InvalidToolRetryState, MemoryProgress, ModelCallRecord, PendingEffects, PendingIngress,
        PendingModelOutcome, PendingToolBatch, PlannedToolCall, RawFinalRecord, RecoveryFeedback,
        RejectionLog, ResponseRetryState, RunAccounting, RunEvents, RunNode, RunPhase, RunProgress,
        RuntimeIdentity, RuntimeTick, StructuredOutputState, TerminalCause, TerminalDiagnostic,
        TerminalObservation, TerminalState, TopologyIndex, TurnCapabilitySnapshot,
    },
    effects::{
        EffectCompletion, EffectIntent, MemoryEffectIntent, MemoryEffectOutput, ModelEffectIntent,
        ToolEffectIntent,
    },
};

/// Label for one complete Rig ECS progression pass.
#[derive(Clone, Debug, Hash, Eq, PartialEq, ScheduleLabel)]
pub struct RigSchedule;

/// Explicitly ordered lifecycle stages.
#[derive(Clone, Debug, Hash, Eq, PartialEq, SystemSet)]
pub enum RigSystemSet {
    /// Cancellation and terminal arbitration runs first.
    Cancellation,
    /// Correlated external messages enter authoritative state.
    Ingress,
    /// ECS policy interprets accepted outcomes.
    Policy,
    /// Owned effects leave the world.
    Dispatch,
    /// Completed batches and terminal outcomes become observable.
    Observation,
    /// Observed retained state and retired capabilities are safely removed.
    Cleanup,
}

pub(crate) fn build_schedule() -> Schedule {
    let mut schedule = Schedule::new(RigSchedule);
    schedule.configure_sets(
        (
            RigSystemSet::Cancellation,
            RigSystemSet::Ingress,
            RigSystemSet::Policy,
            RigSystemSet::Dispatch,
            RigSystemSet::Observation,
            RigSystemSet::Cleanup,
        )
            .chain(),
    );
    // Every system here is an exclusive `&mut World` system, so there are no
    // deferred command queues to apply between stages; the chained tuple is
    // the complete ordering story.
    schedule.add_systems(
        (
            advance_tick.in_set(RigSystemSet::Cancellation),
            process_cancellation.in_set(RigSystemSet::Cancellation),
            process_ingress.in_set(RigSystemSet::Ingress),
            evaluate_model_outcomes.in_set(RigSystemSet::Policy),
            dispatch_memory_effects.in_set(RigSystemSet::Dispatch),
            dispatch_model_effects.in_set(RigSystemSet::Dispatch),
            dispatch_tool_effects.in_set(RigSystemSet::Dispatch),
            commit_tool_batches.in_set(RigSystemSet::Observation),
            cleanup_terminal_runs.in_set(RigSystemSet::Cleanup),
            cleanup_retired_capabilities.in_set(RigSystemSet::Cleanup),
        )
            .chain(),
    );
    schedule
}

fn bump_progress(world: &mut World, entity: Entity) {
    if let Some(mut progress) = world.get_mut::<RunProgress>(entity) {
        progress.0 = progress.0.saturating_add(1);
    }
}

fn advance_tick(world: &mut World) {
    world.resource_mut::<RuntimeTick>().0 = world.resource::<RuntimeTick>().0.saturating_add(1);
}

fn publish(world: &mut World, entity: Entity, event: RunEvent) {
    if let Some(mut events) = world.get_mut::<RunEvents>(entity) {
        events.publish(event);
    }
}

fn set_terminal(
    world: &mut World,
    entity: Entity,
    reason: TerminalReason,
    diagnostic: Option<String>,
) {
    if world.entity(entity).contains::<TerminalState>() {
        return;
    }
    let tick = world.resource::<RuntimeTick>().0;
    let run_id = world.get::<RunNode>(entity).map(|run| run.id);
    if let Some(mut run) = world.get_mut::<RunNode>(entity) {
        run.phase = RunPhase::Terminal;
    }
    if !matches!(reason, TerminalReason::Completed)
        && let Some(mut operations) = world.get_mut::<ActiveOperations>(entity)
    {
        for operation in operations.0.values_mut() {
            operation.completed = true;
        }
    }
    let preserve_raw = matches!(reason, TerminalReason::Completed);
    let mut entity_mut = world.entity_mut(entity);
    entity_mut.insert(TerminalState {
        reason: reason.clone(),
        terminal_tick: tick,
    });
    entity_mut.remove::<PendingModelOutcome>();
    entity_mut.remove::<PendingToolBatch>();
    entity_mut.remove::<TurnCapabilitySnapshot>();
    entity_mut.remove::<EffectQueueWait>();
    if let Some(message) = diagnostic {
        entity_mut.insert(TerminalDiagnostic { message });
    }
    if !preserve_raw {
        entity_mut.remove::<RawFinalRecord>();
    }
    publish(world, entity, RunEvent::Terminal(reason));
    tracing::info!(
        target: "rig::ecs",
        run_id = ?run_id,
        terminal_reason = ?world.get::<TerminalState>(entity).map(|state| &state.reason),
        "run reached terminal state"
    );
    bump_progress(world, entity);
}

fn fail_run(world: &mut World, entity: Entity, code: &str, diagnostic: impl Into<String>) {
    set_terminal(
        world,
        entity,
        TerminalReason::Failed {
            code: code.to_string(),
        },
        Some(diagnostic.into()),
    );
}

fn process_cancellation(world: &mut World) {
    let entities = {
        let mut query = world.query_filtered::<Entity, With<CancellationRequest>>();
        query.iter(world).collect::<Vec<_>>()
    };
    for entity in entities {
        world.entity_mut(entity).remove::<CancellationRequest>();
        set_terminal(world, entity, TerminalReason::Cancelled, None);
    }
}

fn reject_ingress(
    world: &mut World,
    entity: Option<Entity>,
    header: EffectHeader,
    reason: EffectRejectionReason,
) {
    tracing::warn!(
        target: "rig::ecs",
        run_id = %header.run_id,
        operation_id = %header.operation_id,
        rejection_reason = ?reason,
        "effect ingress rejected"
    );
    let capacity = world
        .resource::<RuntimeConfigResource>()
        .0
        .rejection_capacity;
    let mut rejections = world.resource_mut::<RejectionLog>();
    if rejections.0.len() == capacity {
        rejections.0.remove(0);
    }
    rejections.0.push(EffectRejection { header, reason });
    if let Some(entity) = entity {
        if let Some(mut accounting) = world.get_mut::<RunAccounting>(entity) {
            accounting.rejected_effects = accounting.rejected_effects.saturating_add(1);
        }
        publish(world, entity, RunEvent::EffectRejected(reason));
    }
}

fn validate_ingress(
    world: &World,
    header: EffectHeader,
) -> Result<(Entity, ActiveOperation), (Option<Entity>, EffectRejectionReason)> {
    if header.runtime_id != world.resource::<RuntimeIdentity>().0 {
        return Err((None, EffectRejectionReason::ForeignRuntime));
    }
    let Some(entity) = world
        .resource::<TopologyIndex>()
        .runs
        .get(&header.run_id)
        .copied()
    else {
        return Err((None, EffectRejectionReason::UnknownRun));
    };
    let Some(run) = world.get::<RunNode>(entity) else {
        return Err((None, EffectRejectionReason::UnknownRun));
    };
    if run.tenant_id != header.tenant_id {
        return Err((Some(entity), EffectRejectionReason::WrongTenant));
    }
    if world.entity(entity).contains::<TerminalState>() {
        return Err((Some(entity), EffectRejectionReason::Late));
    }
    if run.generation != header.generation {
        return Err((Some(entity), EffectRejectionReason::WrongGeneration));
    }
    let Some(operation) = world
        .get::<ActiveOperations>(entity)
        .and_then(|operations| operations.0.get(&header.operation_id))
        .cloned()
    else {
        return Err((Some(entity), EffectRejectionReason::UnknownOperation));
    };
    if operation.completed {
        return Err((Some(entity), EffectRejectionReason::Duplicate));
    }
    if operation.header.correlation_id != header.correlation_id {
        return Err((Some(entity), EffectRejectionReason::WrongCorrelation));
    }
    if operation.header.capability_id != header.capability_id
        || operation.header.grant_id != header.grant_id
        || operation.header.capability_revision != header.capability_revision
    {
        return Err((Some(entity), EffectRejectionReason::WrongAuthorization));
    }
    Ok((entity, operation))
}

fn mark_operation_complete(world: &mut World, entity: Entity, operation_id: OperationId) {
    if let Some(mut operations) = world.get_mut::<ActiveOperations>(entity)
        && let Some(operation) = operations.0.get_mut(&operation_id)
    {
        operation.completed = true;
    }
    if let Some(mut accepted) = world.get_mut::<AcceptedDeltas>(entity) {
        accepted.0.remove(&operation_id);
    }
}

fn accept_delta_sequence(
    accepted: &mut AcceptedDeltas,
    operation_id: OperationId,
    sequence: u64,
) -> Result<(), EffectRejectionReason> {
    let expected = accepted.0.entry(operation_id).or_insert(0);
    if sequence < *expected {
        return Err(EffectRejectionReason::Duplicate);
    }
    if sequence > *expected {
        return Err(EffectRejectionReason::OutOfOrder);
    }
    *expected = expected.saturating_add(1);
    Ok(())
}

fn process_ingress(world: &mut World) {
    let ingress = std::mem::take(&mut world.resource_mut::<PendingIngress>().0);
    for message in ingress {
        let header = match &message {
            EffectIngress::Delta { header, .. } => *header,
            EffectIngress::Completion(completion) => *completion.header(),
        };
        let (entity, operation) = match validate_ingress(world, header) {
            Ok(validated) => validated,
            Err((entity, reason)) => {
                reject_ingress(world, entity, header, reason);
                continue;
            }
        };
        match message {
            EffectIngress::Delta {
                sequence, delta, ..
            } => {
                if operation.kind != ActiveOperationKind::Model {
                    reject_ingress(
                        world,
                        Some(entity),
                        header,
                        EffectRejectionReason::WrongAuthorization,
                    );
                    continue;
                }
                let sequence_result =
                    world
                        .get_mut::<AcceptedDeltas>(entity)
                        .map_or(Ok(()), |mut accepted| {
                            accept_delta_sequence(&mut accepted, header.operation_id, sequence)
                        });
                if let Err(reason) = sequence_result {
                    reject_ingress(world, Some(entity), header, reason);
                    continue;
                }
                publish(
                    world,
                    entity,
                    RunEvent::Provisional {
                        operation_id: header.operation_id,
                        delta: Box::new(delta),
                    },
                );
                bump_progress(world, entity);
            }
            EffectIngress::Completion(completion) => {
                if let Some(reason) = completion_rejection(&completion, &operation) {
                    reject_ingress(world, Some(entity), header, reason);
                    continue;
                }
                mark_operation_complete(world, entity, header.operation_id);
                accept_completion(world, entity, operation.kind, completion);
            }
        }
    }
}

fn completion_rejection(
    completion: &EffectCompletion,
    operation: &ActiveOperation,
) -> Option<EffectRejectionReason> {
    match completion {
        EffectCompletion::Model { .. } => (operation.kind != ActiveOperationKind::Model)
            .then_some(EffectRejectionReason::WrongAuthorization),
        EffectCompletion::Tool {
            tool_call_id,
            order,
            ..
        } => {
            if operation.kind != ActiveOperationKind::Tool {
                Some(EffectRejectionReason::WrongAuthorization)
            } else if operation.expected_tool_call_id.as_deref() != Some(tool_call_id.as_str())
                || operation.expected_tool_order != Some(*order)
            {
                Some(EffectRejectionReason::WrongPayload)
            } else {
                None
            }
        }
        EffectCompletion::Memory { result, .. } => match result {
            Ok(MemoryEffectOutput::Loaded(_)) => (operation.kind
                != ActiveOperationKind::MemoryLoad)
                .then_some(EffectRejectionReason::WrongPayload),
            Ok(MemoryEffectOutput::Appended) => (operation.kind
                != ActiveOperationKind::MemoryAppend)
                .then_some(EffectRejectionReason::WrongPayload),
            Err(_)
                if matches!(
                    operation.kind,
                    ActiveOperationKind::MemoryLoad | ActiveOperationKind::MemoryAppend
                ) =>
            {
                None
            }
            Err(_) => Some(EffectRejectionReason::WrongAuthorization),
        },
    }
}

fn accept_completion(
    world: &mut World,
    entity: Entity,
    operation_kind: ActiveOperationKind,
    completion: EffectCompletion,
) {
    match completion {
        EffectCompletion::Model { header, result } => match result {
            Ok(output) => {
                let Some(aggregate_usage) = world
                    .get::<RunAccounting>(entity)
                    .and_then(|accounting| checked_usage_sum(accounting.usage, output.usage))
                else {
                    fail_run(
                        world,
                        entity,
                        "usage_overflow",
                        "provider usage exceeded the supported accounting range",
                    );
                    return;
                };
                if let Some(mut accounting) = world.get_mut::<RunAccounting>(entity) {
                    accounting.usage = aggregate_usage;
                    accounting.model_calls.push(ModelCallRecord {
                        operation_id: header.operation_id,
                        usage: output.usage,
                        accepted: false,
                    });
                }
                let mut entity_mut = world.entity_mut(entity);
                if let Some(raw) = output.raw_final {
                    entity_mut.insert(RawFinalRecord {
                        operation_id: header.operation_id,
                        raw,
                    });
                } else {
                    entity_mut.remove::<RawFinalRecord>();
                }
                entity_mut.insert(PendingModelOutcome {
                    operation_id: header.operation_id,
                    choice: output.choice,
                    message_id: output.message_id,
                });
                if let Some(mut run) = entity_mut.get_mut::<RunNode>() {
                    run.phase = RunPhase::EvaluatingModel;
                }
                bump_progress(world, entity);
            }
            Err(error) => {
                let diagnostic = error.to_string();
                world
                    .entity_mut(entity)
                    .insert(TerminalCause::Model(Arc::new(error)));
                fail_run(world, entity, "model_effect", diagnostic);
            }
        },
        EffectCompletion::Tool {
            header,
            tool_call_id,
            order,
            result,
        } => {
            let output = match result {
                crate::effects::ToolEffectOutput::Success(output) => output,
                crate::effects::ToolEffectOutput::Failure(error) => error.model_output().clone(),
            };
            let mut matched = false;
            if let Some(mut batch) = world.get_mut::<PendingToolBatch>(entity)
                && let Some(call) = batch.calls.iter_mut().find(|call| {
                    call.order == order
                        && call.call.id == tool_call_id
                        && call.operation_id == Some(header.operation_id)
                })
            {
                call.result = Some(output);
                matched = true;
            }
            if !matched {
                reject_ingress(
                    world,
                    Some(entity),
                    header,
                    EffectRejectionReason::WrongCorrelation,
                );
            } else {
                bump_progress(world, entity);
            }
        }
        EffectCompletion::Memory { result, .. } => match result {
            Ok(MemoryEffectOutput::Loaded(mut messages)) => {
                // Loaded history is foreign content: validate it here, where
                // failure is cheap and correctly attributed to the memory,
                // before any model call sees it.
                let candidate_state = {
                    let existing = world
                        .get::<CanonicalTranscript>(entity)
                        .map(|transcript| transcript.messages.as_slice())
                        .unwrap_or_default();
                    crate::persistence::TranscriptValidator::over(&messages).and_then(
                        |mut validator| {
                            existing
                                .iter()
                                .try_for_each(|message| validator.observe(message))
                                .map(|()| validator)
                        },
                    )
                };
                match candidate_state {
                    Err(error) => {
                        fail_run(
                            world,
                            entity,
                            "memory_history",
                            format!(
                                "conversation memory returned a non-canonical history: {error}"
                            ),
                        );
                    }
                    Ok(validator) => {
                        // Transcript and validator must move together; a
                        // missing transcript fails terminally rather than
                        // silently desyncing the pair.
                        if let Some(mut transcript) = world.get_mut::<CanonicalTranscript>(entity) {
                            let prompt = std::mem::take(&mut transcript.messages);
                            transcript.new_messages_start = messages.len();
                            messages.extend(prompt);
                            transcript.messages = messages;
                            if let Some(mut validation) =
                                world.get_mut::<crate::persistence::TranscriptValidation>(entity)
                            {
                                validation.0 = validator;
                            }
                            if let Some(mut memory) = world.get_mut::<MemoryProgress>(entity) {
                                memory.loaded = true;
                            }
                            if let Some(mut run) = world.get_mut::<RunNode>(entity) {
                                run.phase = RunPhase::ReadyModel;
                            }
                            bump_progress(world, entity);
                        } else {
                            fail_run(
                                world,
                                entity,
                                "canonical_transcript",
                                "run has no canonical transcript to accept loaded history into",
                            );
                        }
                    }
                }
            }
            Ok(MemoryEffectOutput::Appended) => {
                if let Some(mut memory) = world.get_mut::<MemoryProgress>(entity) {
                    memory.appended = true;
                }
                set_terminal(world, entity, TerminalReason::Completed, None);
            }
            Err(error) if operation_kind == ActiveOperationKind::MemoryAppend => {
                let diagnostic =
                    format!("conversation memory append failed after canonical commit: {error}");
                world
                    .entity_mut(entity)
                    .insert(TerminalCause::Memory(Arc::new(error)));
                set_terminal(world, entity, TerminalReason::Completed, Some(diagnostic));
            }
            Err(error) => {
                let diagnostic = error.to_string();
                world
                    .entity_mut(entity)
                    .insert(TerminalCause::Memory(Arc::new(error)));
                fail_run(world, entity, "memory_effect", diagnostic);
            }
        },
    }
}

pub(crate) fn checked_usage_sum(
    left: rig_core::completion::Usage,
    right: rig_core::completion::Usage,
) -> Option<rig_core::completion::Usage> {
    Some(rig_core::completion::Usage {
        input_tokens: left.input_tokens.checked_add(right.input_tokens)?,
        output_tokens: left.output_tokens.checked_add(right.output_tokens)?,
        total_tokens: left.total_tokens.checked_add(right.total_tokens)?,
        cached_input_tokens: left
            .cached_input_tokens
            .checked_add(right.cached_input_tokens)?,
        cache_creation_input_tokens: left
            .cache_creation_input_tokens
            .checked_add(right.cache_creation_input_tokens)?,
        tool_use_prompt_tokens: left
            .tool_use_prompt_tokens
            .checked_add(right.tool_use_prompt_tokens)?,
        reasoning_tokens: left.reasoning_tokens.checked_add(right.reasoning_tokens)?,
    })
}

fn agent_for_run(world: &World, run: &RunNode) -> Option<AgentNode> {
    let entity = world
        .resource::<TopologyIndex>()
        .agents
        .get(&run.agent_id)
        .copied()?;
    world.get::<AgentNode>(entity).cloned()
}

fn mark_call_accepted(world: &mut World, entity: Entity, operation_id: OperationId) {
    if let Some(mut accounting) = world.get_mut::<RunAccounting>(entity)
        && let Some(record) = accounting
            .model_calls
            .iter_mut()
            .find(|record| record.operation_id == operation_id)
    {
        record.accepted = true;
    }
}

fn clear_recovery_feedback(world: &mut World, entity: Entity) {
    if let Some(mut feedback) = world.get_mut::<RecoveryFeedback>(entity) {
        feedback.0.clear();
    }
}

// The one rollback contract for every model-response retry: corrective
// feedback in, provisional raw final out, run back to ReadyModel.
fn reset_for_model_retry(world: &mut World, entity: Entity, feedback: String) {
    if let Some(mut messages) = world.get_mut::<RecoveryFeedback>(entity) {
        messages.0.push(feedback);
    }
    world.entity_mut(entity).remove::<RawFinalRecord>();
    if let Some(mut run) = world.get_mut::<RunNode>(entity) {
        run.phase = RunPhase::ReadyModel;
    }
    publish(world, entity, RunEvent::ResponseRetried);
    bump_progress(world, entity);
}

trait RetryCounter: Component<Mutability = bevy_ecs::component::Mutable> {
    fn count(&self) -> usize;
    fn bump(&mut self);
}

impl RetryCounter for ResponseRetryState {
    fn count(&self) -> usize {
        self.0
    }
    fn bump(&mut self) {
        self.0 = self.0.saturating_add(1);
    }
}

impl RetryCounter for InvalidToolRetryState {
    fn count(&self) -> usize {
        self.0
    }
    fn bump(&mut self) {
        self.0 = self.0.saturating_add(1);
    }
}

// The one retry ladder: consume budget and re-ask the model, or fail the run.
fn bump_retry_or_fail<C: RetryCounter>(
    world: &mut World,
    entity: Entity,
    max_retries: usize,
    feedback: &str,
    code: &str,
    exhausted: String,
) {
    let retries = world.get::<C>(entity).map_or(0, RetryCounter::count);
    if retries < max_retries {
        if let Some(mut state) = world.get_mut::<C>(entity) {
            state.bump();
        }
        reset_for_model_retry(world, entity, feedback.to_string());
    } else {
        fail_run(world, entity, code, exhausted);
    }
}

// Duplicate identities can never commit canonically (the same validation
// restore enforces), so every model turn — structured-output turns included —
// passes through this ladder before any handler runs. Returns whether the
// outcome was consumed; `Skip` falls through and the handlers drop repeats.
fn apply_duplicate_identity_policy(
    world: &mut World,
    entity: Entity,
    agent: &AgentNode,
    duplicate_id: &str,
) -> bool {
    match agent.spec.invalid_tool_policy {
        InvalidToolPolicy::Skip => false,
        InvalidToolPolicy::Fail | InvalidToolPolicy::Repair => {
            fail_run(
                world,
                entity,
                "duplicate_tool_call",
                format!("model emitted duplicate tool call identity `{duplicate_id}`"),
            );
            true
        }
        InvalidToolPolicy::Retry { max_retries } => {
            bump_retry_or_fail::<InvalidToolRetryState>(
                world,
                entity,
                max_retries,
                "Tool call identities must be unique. Re-issue each tool call with a distinct id.",
                "duplicate_tool_call",
                format!("duplicate tool call retry exhausted for `{duplicate_id}`"),
            );
            true
        }
        InvalidToolPolicy::Stop => {
            set_terminal(world, entity, TerminalReason::Stopped, None);
            true
        }
    }
}

fn finalize_accepted_turn(world: &mut World, entity: Entity, operation_id: OperationId) {
    mark_call_accepted(world, entity, operation_id);
    clear_recovery_feedback(world, entity);
    publish_provider_final(world, entity, operation_id);
    complete_success(world, entity);
}

fn find_output_call(choice: &[AssistantContent], output_name: &str) -> Option<ToolCall> {
    choice.iter().find_map(|content| match content {
        AssistantContent::ToolCall(call) if call.function.name == output_name => Some(call.clone()),
        _ => None,
    })
}

// Structured finals never carry tool calls; the output call survives as text.
fn structured_final_content(
    choice: Vec<AssistantContent>,
    output_call: Option<&ToolCall>,
) -> Vec<AssistantContent> {
    let mut content = choice
        .into_iter()
        .filter(|content| !matches!(content, AssistantContent::ToolCall(_)))
        .collect::<Vec<_>>();
    if let Some(call) = output_call {
        content.push(AssistantContent::text(call.function.arguments.to_string()));
    }
    content
}

// Commit-time gate: the transcript must satisfy the same canonical invariants
// `restore` enforces, so the runtime can never persist state it refuses to load.
fn commit_assistant_turn(
    world: &mut World,
    entity: Entity,
    message_id: Option<String>,
    content: OneOrMany<AssistantContent>,
    results: Option<OneOrMany<UserContent>>,
) -> bool {
    let mut staged = vec![Message::Assistant {
        id: message_id,
        content,
    }];
    if let Some(content) = results {
        staged.push(Message::User { content });
    }
    // Tail-only validation: loaded history and prior turns were validated at
    // their own boundaries, so only the appended turn is checked here, against
    // the run's live resumable validator state.
    let mut validator = world
        .get::<crate::persistence::TranscriptValidation>(entity)
        .map(|validation| validation.0.clone());
    let outcome = validator.as_mut().map_or(
        Err("run has no transcript validation state".to_string()),
        |validator| {
            staged
                .iter()
                .try_for_each(|message| validator.observe(message))
                .and_then(|()| validator.finish())
                .map_err(|error| format!("refusing to commit a non-canonical model turn: {error}"))
        },
    );
    if let Err(diagnostic) = outcome {
        fail_run(world, entity, "canonical_transcript", diagnostic);
        return false;
    }
    let Some(mut transcript) = world.get_mut::<CanonicalTranscript>(entity) else {
        fail_run(
            world,
            entity,
            "canonical_transcript",
            "run has no canonical transcript to commit into",
        );
        return false;
    };
    transcript.messages.extend(staged);
    if let (Some(mut validation), Some(validator)) = (
        world.get_mut::<crate::persistence::TranscriptValidation>(entity),
        validator,
    ) {
        validation.0 = validator;
    }
    true
}

fn publish_provider_final(world: &mut World, entity: Entity, operation_id: OperationId) {
    let provider_type = world
        .get::<RawFinalRecord>(entity)
        .filter(|record| record.operation_id == operation_id)
        .map(|record| record.raw.type_name());
    if let Some(provider_type) = provider_type {
        publish(
            world,
            entity,
            RunEvent::ProviderFinal {
                operation_id,
                provider_type,
            },
        );
    }
}

fn assistant_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<String>()
}

fn complete_success(world: &mut World, entity: Entity) {
    let pending_append = world
        .get::<MemoryProgress>(entity)
        .is_some_and(|memory| !memory.appended);
    if pending_append {
        if let Some(mut run) = world.get_mut::<RunNode>(entity) {
            run.phase = RunPhase::AppendingMemory;
        }
        bump_progress(world, entity);
    } else {
        set_terminal(world, entity, TerminalReason::Completed, None);
    }
}

fn validate_schema(value: &serde_json::Value, schema: &serde_json::Value) -> bool {
    jsonschema::validator_for(schema).is_ok_and(|validator| validator.is_valid(value))
}

fn retry_structured_output(
    world: &mut World,
    entity: Entity,
    choice: Vec<AssistantContent>,
    message_id: Option<String>,
    operation_id: OperationId,
) {
    let Some(state) = world.get::<StructuredOutputState>(entity) else {
        return;
    };
    let can_retry = state.retries < state.policy.max_retries;
    let best_effort = state.policy.best_effort;
    if can_retry {
        if let Some(mut state) = world.get_mut::<StructuredOutputState>(entity) {
            state.retries = state.retries.saturating_add(1);
        }
        reset_for_model_retry(
            world,
            entity,
            "The previous response did not satisfy the required JSON schema. Return a valid structured response."
                .to_string(),
        );
    } else if best_effort {
        // A best-effort turn is final: tool calls can never be answered after it,
        // so the schema-invalid output call is preserved as text, like the valid path.
        let output_call = world
            .get::<StructuredOutputState>(entity)
            .and_then(|state| state.output_tool_name.clone())
            .and_then(|name| find_output_call(&choice, &name));
        let final_content = structured_final_content(choice, output_call.as_ref());
        if let Some(content) = OneOrMany::from_iter_optional(final_content) {
            if commit_assistant_turn(world, entity, message_id, content, None) {
                finalize_accepted_turn(world, entity, operation_id);
            }
        } else {
            fail_run(
                world,
                entity,
                "structured_output",
                "best-effort structured output was empty",
            );
        }
    } else {
        fail_run(
            world,
            entity,
            "structured_output",
            "structured output recovery was exhausted",
        );
    }
}

fn evaluate_model_outcomes(world: &mut World) {
    let entities = {
        let mut query = world.query_filtered::<Entity, With<PendingModelOutcome>>();
        query.iter(world).collect::<Vec<_>>()
    };
    for entity in entities {
        if world.entity(entity).contains::<TerminalState>() {
            continue;
        }
        let Some(outcome) = world.entity_mut(entity).take::<PendingModelOutcome>() else {
            continue;
        };
        let Some(run) = world.get::<RunNode>(entity).cloned() else {
            continue;
        };
        let Some(agent) = agent_for_run(world, &run) else {
            fail_run(
                world,
                entity,
                "unknown_agent",
                "run agent topology disappeared",
            );
            continue;
        };

        let duplicate_id = {
            let mut seen = BTreeSet::new();
            outcome.choice.iter().find_map(|content| match content {
                AssistantContent::ToolCall(call) if !seen.insert(call.id.clone()) => {
                    Some(call.id.clone())
                }
                _ => None,
            })
        };
        if let Some(duplicate_id) = duplicate_id
            && apply_duplicate_identity_policy(world, entity, &agent, &duplicate_id)
        {
            continue;
        }

        let output_call = world
            .get::<StructuredOutputState>(entity)
            .and_then(|state| state.output_tool_name.clone())
            .and_then(|name| find_output_call(&outcome.choice, &name));
        if let Some(output_call) = output_call {
            handle_structured_output_call(world, entity, outcome, output_call);
            continue;
        }

        let tool_calls = outcome
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !tool_calls.is_empty() {
            evaluate_tool_calls(world, entity, &agent, outcome, tool_calls);
            continue;
        }

        if let Some(resolved_mode) = world
            .get::<StructuredOutputState>(entity)
            .map(|state| state.resolved_mode)
        {
            if matches!(resolved_mode, OutputMode::Tool) {
                retry_structured_output(
                    world,
                    entity,
                    outcome.choice,
                    outcome.message_id,
                    outcome.operation_id,
                );
                continue;
            }
            let text = assistant_text(&outcome.choice);
            let parsed = serde_json::from_str::<serde_json::Value>(text.trim()).ok();
            let valid = parsed.as_ref().is_some_and(|value| {
                world
                    .get::<StructuredOutputState>(entity)
                    .is_some_and(|state| validate_schema(value, state.schema.as_value()))
            });
            if valid {
                if let Some(mut state) = world.get_mut::<StructuredOutputState>(entity) {
                    state.value = parsed;
                }
            } else {
                retry_structured_output(
                    world,
                    entity,
                    outcome.choice,
                    outcome.message_id,
                    outcome.operation_id,
                );
                continue;
            }
        }

        let text = assistant_text(&outcome.choice);
        if text.trim().is_empty()
            && let ResponseRetryPolicy::RejectEmpty { max_retries } =
                agent.spec.response_retry_policy
        {
            handle_empty_response_retry(world, entity, max_retries);
            continue;
        }

        let Some(content) = OneOrMany::from_iter_optional(outcome.choice) else {
            fail_run(
                world,
                entity,
                "empty_choice",
                "provider returned an empty assistant choice",
            );
            continue;
        };
        if !commit_assistant_turn(world, entity, outcome.message_id, content, None) {
            continue;
        }
        finalize_accepted_turn(world, entity, outcome.operation_id);
    }
}

fn handle_structured_output_call(
    world: &mut World,
    entity: Entity,
    outcome: PendingModelOutcome,
    output_call: ToolCall,
) {
    // Every tool call other than the first occurrence of the output call is
    // suppressed — including a same-id repeat surviving under `Skip`.
    let mut saw_output_call = false;
    for peer in outcome.choice.iter().filter_map(|content| match content {
        AssistantContent::ToolCall(call) => Some(call),
        _ => None,
    }) {
        if !saw_output_call && peer.id == output_call.id {
            saw_output_call = true;
            continue;
        }
        publish(
            world,
            entity,
            RunEvent::ToolSuppressed {
                tool_call_id: peer.id.clone(),
            },
        );
    }
    let valid = world
        .get::<StructuredOutputState>(entity)
        .is_some_and(|state| {
            validate_schema(&output_call.function.arguments, state.schema.as_value())
        });
    if valid {
        if let Some(mut state) = world.get_mut::<StructuredOutputState>(entity) {
            state.value = Some(output_call.function.arguments.clone());
        }
        let final_content = structured_final_content(outcome.choice, Some(&output_call));
        let Some(final_content) = OneOrMany::from_iter_optional(final_content) else {
            fail_run(
                world,
                entity,
                "empty_structured_final",
                "structured finalization produced no canonical content",
            );
            return;
        };
        if commit_assistant_turn(world, entity, outcome.message_id, final_content, None) {
            finalize_accepted_turn(world, entity, outcome.operation_id);
        }
    } else {
        retry_structured_output(
            world,
            entity,
            outcome.choice,
            outcome.message_id,
            outcome.operation_id,
        );
    }
}

fn handle_empty_response_retry(world: &mut World, entity: Entity, max_retries: usize) {
    bump_retry_or_fail::<ResponseRetryState>(
        world,
        entity,
        max_retries,
        "The previous response was empty. Return a substantive answer.",
        "response_retry_exhausted",
        "empty response retry policy was exhausted".to_string(),
    );
}

fn evaluate_tool_calls(
    world: &mut World,
    entity: Entity,
    agent: &AgentNode,
    outcome: PendingModelOutcome,
    tool_calls: Vec<ToolCall>,
) {
    // Duplicate identities were already routed through the recovery ladder in
    // `evaluate_model_outcomes`; reaching here under `Skip` means each repeat
    // is dropped and only the first occurrence runs.
    let snapshot = world
        .get::<TurnCapabilitySnapshot>(entity)
        .cloned()
        .unwrap_or_default();
    let mut definitions = BTreeMap::new();
    for entry in snapshot.entries {
        definitions.insert(entry.definition.name.clone(), entry);
    }
    let mut planned = Vec::new();
    let mut invalid = None;
    let mut planned_ids = BTreeSet::new();
    for (order, mut call) in tool_calls.into_iter().enumerate() {
        if !planned_ids.insert(call.id.clone()) {
            // Only reachable under `Skip`: the repeat is dropped from the turn
            // entirely so the committed pairing stays canonical.
            publish(
                world,
                entity,
                RunEvent::ToolSuppressed {
                    tool_call_id: call.id,
                },
            );
            continue;
        }
        let mut entry = definitions.get(&call.function.name).cloned();
        if entry.is_none() && matches!(agent.spec.invalid_tool_policy, InvalidToolPolicy::Repair) {
            let matches = definitions
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case(&call.function.name))
                .map(|(_, entry)| entry.clone())
                .collect::<Vec<_>>();
            if let [repaired] = matches.as_slice() {
                call.function.name.clone_from(&repaired.definition.name);
                entry = Some(repaired.clone());
            }
        }
        if let Some(entry) = entry {
            planned.push(PlannedToolCall {
                call,
                capability_id: entry.capability_id,
                grant_id: entry.grant_id,
                revision: entry.revision,
                order,
                operation_id: None,
                result: None,
                suppressed: false,
            });
        } else if matches!(agent.spec.invalid_tool_policy, InvalidToolPolicy::Skip) {
            let call_id = call.id.clone();
            planned.push(PlannedToolCall {
                call,
                capability_id: CapabilityId::new(),
                grant_id: crate::GrantId::new(),
                revision: 0,
                order,
                operation_id: None,
                result: Some(ToolOutput::text(
                    "tool call skipped because it was not advertised",
                )),
                suppressed: true,
            });
            publish(
                world,
                entity,
                RunEvent::ToolSuppressed {
                    tool_call_id: call_id,
                },
            );
        } else {
            invalid = Some(call.function.name);
            break;
        }
    }
    if let Some(name) = invalid {
        match agent.spec.invalid_tool_policy {
            InvalidToolPolicy::Fail | InvalidToolPolicy::Repair => fail_run(
                world,
                entity,
                "invalid_tool",
                format!("model requested unadvertised tool `{name}`"),
            ),
            InvalidToolPolicy::Retry { max_retries } => {
                bump_retry_or_fail::<InvalidToolRetryState>(
                    world,
                    entity,
                    max_retries,
                    &format!(
                        "Tool `{name}` is unavailable. Use only an advertised tool or answer without it."
                    ),
                    "invalid_tool_retry_exhausted",
                    format!("invalid tool retry exhausted for `{name}`"),
                );
            }
            InvalidToolPolicy::Stop => set_terminal(world, entity, TerminalReason::Stopped, None),
            InvalidToolPolicy::Skip => {}
        }
        return;
    }

    let mut content_ids = BTreeSet::new();
    let repaired_content = outcome
        .choice
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(original) => {
                if !content_ids.insert(original.id.clone()) {
                    return None;
                }
                Some(
                    planned
                        .iter()
                        .find(|planned| planned.call.id == original.id)
                        .map_or(AssistantContent::ToolCall(original), |planned| {
                            AssistantContent::ToolCall(planned.call.clone())
                        }),
                )
            }
            content => Some(content),
        })
        .collect::<Vec<_>>();
    world.entity_mut(entity).insert(PendingToolBatch {
        assistant_content: repaired_content,
        message_id: outcome.message_id,
        calls: planned,
    });
    if let Some(mut run) = world.get_mut::<RunNode>(entity) {
        run.phase = RunPhase::WaitingTools;
    }
    mark_call_accepted(world, entity, outcome.operation_id);
    clear_recovery_feedback(world, entity);
    publish_provider_final(world, entity, outcome.operation_id);
    bump_progress(world, entity);
}

fn new_header(world: &World, run: &RunNode) -> EffectHeader {
    EffectHeader {
        runtime_id: world.resource::<RuntimeIdentity>().0,
        run_id: run.id,
        operation_id: OperationId::new(),
        generation: run.generation,
        correlation_id: CorrelationId::new(),
        tenant_id: run.tenant_id,
        capability_id: None,
        grant_id: None,
        capability_revision: None,
    }
}

fn has_active_kind(world: &World, entity: Entity, kind: ActiveOperationKind) -> bool {
    world
        .get::<ActiveOperations>(entity)
        .is_some_and(|operations| {
            operations
                .0
                .values()
                .any(|operation| operation.kind == kind && !operation.completed)
        })
}

fn register_operation(
    world: &mut World,
    entity: Entity,
    header: EffectHeader,
    kind: ActiveOperationKind,
) {
    if let Some(mut operations) = world.get_mut::<ActiveOperations>(entity) {
        operations.0.insert(
            header.operation_id,
            ActiveOperation {
                header,
                kind,
                expected_tool_call_id: None,
                expected_tool_order: None,
                completed: false,
            },
        );
    }
}

fn register_tool_operation(
    world: &mut World,
    entity: Entity,
    header: EffectHeader,
    tool_call_id: String,
    order: usize,
) {
    if let Some(mut operations) = world.get_mut::<ActiveOperations>(entity) {
        operations.0.insert(
            header.operation_id,
            ActiveOperation {
                header,
                kind: ActiveOperationKind::Tool,
                expected_tool_call_id: Some(tool_call_id),
                expected_tool_order: Some(order),
                completed: false,
            },
        );
    }
}

fn enqueue_effect(world: &mut World, entity: Entity, intent: EffectIntent) -> bool {
    let capacity = world
        .resource::<RuntimeConfigResource>()
        .0
        .effect_queue_capacity;
    if world.resource::<PendingEffects>().0.len() >= capacity {
        // Record a real waiting transition once. Repeated full-queue passes do
        // not manufacture progress and therefore cannot exhaust the target
        // run's livelock budget while another run owns the bounded capacity.
        if !world.entity(entity).contains::<EffectQueueWait>() {
            world.entity_mut(entity).insert(EffectQueueWait);
            bump_progress(world, entity);
        }
        false
    } else {
        world.entity_mut(entity).remove::<EffectQueueWait>();
        world.resource_mut::<PendingEffects>().0.push(intent);
        true
    }
}

fn dispatch_memory_effects(world: &mut World) {
    let mut entities = {
        let mut query = world.query::<(Entity, &RunNode, &MemoryProgress)>();
        query
            .iter(world)
            .filter_map(|(entity, run, memory)| {
                let kind = match run.phase {
                    RunPhase::LoadingMemory if !memory.loaded => {
                        Some(ActiveOperationKind::MemoryLoad)
                    }
                    RunPhase::AppendingMemory if !memory.appended => {
                        Some(ActiveOperationKind::MemoryAppend)
                    }
                    _ => None,
                }?;
                (!has_active_kind(world, entity, kind)).then_some((
                    entity,
                    run.clone(),
                    memory.memory_id,
                    memory.conversation_id.clone(),
                    kind,
                ))
            })
            .collect::<Vec<_>>()
    };
    entities.sort_by_key(|(_, run, ..)| run.id);
    for (entity, run, memory_id, conversation_id, kind) in entities {
        let header = new_header(world, &run);
        let intent = if kind == ActiveOperationKind::MemoryLoad {
            MemoryEffectIntent::Load {
                header,
                memory_id,
                conversation_id,
            }
        } else {
            let messages = world
                .get::<CanonicalTranscript>(entity)
                .and_then(|transcript| {
                    transcript
                        .messages
                        .get(transcript.new_messages_start..)
                        .map(<[_]>::to_vec)
                })
                .unwrap_or_default();
            MemoryEffectIntent::Append {
                header,
                memory_id,
                conversation_id,
                messages,
            }
        };
        if !enqueue_effect(world, entity, EffectIntent::Memory(intent)) {
            continue;
        }
        register_operation(world, entity, header, kind);
        bump_progress(world, entity);
    }
}

fn collect_turn_capabilities(world: &World, run: &RunNode) -> Vec<AdvertisedCapability> {
    let index = world.resource::<TopologyIndex>();
    let mut entries = index
        .grants
        .values()
        .filter_map(|entity| world.get::<GrantNode>(*entity))
        .filter(|grant| {
            !grant.revoked && grant.agent_id == run.agent_id && grant.tenant_id == run.tenant_id
        })
        .filter_map(|grant| {
            let entity = index.capabilities.get(&grant.capability_id)?;
            let capability = world.get::<CapabilityNode>(*entity)?;
            (!capability.retired && capability.tenant_id == run.tenant_id)
                .then_some((grant, capability))
        })
        .filter_map(|(grant, capability)| {
            let definition = capability.definition.clone()?;
            Some(AdvertisedCapability {
                capability_id: capability.id,
                grant_id: grant.id,
                revision: capability.revision,
                definition,
            })
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        left.definition
            .name
            .cmp(&right.definition.name)
            .then(left.capability_id.cmp(&right.capability_id))
            .then(left.grant_id.cmp(&right.grant_id))
    });
    entries
}

fn output_tool_callable(tool_choice: Option<&ToolChoice>, output_tool_name: &str) -> bool {
    match tool_choice {
        None | Some(ToolChoice::Auto | ToolChoice::Required) => true,
        Some(ToolChoice::Specific { function_names }) => {
            function_names.iter().any(|name| name == output_tool_name)
        }
        Some(ToolChoice::None) => false,
    }
}

fn resolve_output_mode(
    has_executable_tools: bool,
    output_tool_callable: bool,
    provider_composes_native: bool,
    requested: OutputMode,
) -> OutputMode {
    match requested {
        OutputMode::Native => OutputMode::Native,
        OutputMode::Prompted => OutputMode::Prompted,
        OutputMode::Tool if output_tool_callable => OutputMode::Tool,
        OutputMode::Tool => OutputMode::Native,
        OutputMode::Auto
            if has_executable_tools && output_tool_callable && !provider_composes_native =>
        {
            OutputMode::Tool
        }
        OutputMode::Auto => OutputMode::Native,
    }
}

#[derive(Debug, thiserror::Error)]
enum ToolChoicePolicyError {
    #[error("ToolChoice::Required forces a tool call, but no tools are advertised")]
    RequiredWithoutTools,
    #[error("ToolChoice::Specific requires at least one function name")]
    EmptySpecific,
    #[error("ToolChoice::Specific requested tools not advertised this turn: {missing:?}")]
    MissingSpecific { missing: Vec<String> },
}

fn allowed_executable_tool_names(
    executable_tool_names: &BTreeSet<String>,
    tool_choice: Option<&ToolChoice>,
    output_tool_name: Option<&str>,
) -> Result<BTreeSet<String>, ToolChoicePolicyError> {
    match tool_choice {
        None | Some(ToolChoice::Auto) => Ok(executable_tool_names.clone()),
        Some(ToolChoice::None) => Ok(BTreeSet::new()),
        Some(ToolChoice::Required) => {
            if executable_tool_names.is_empty() && output_tool_name.is_none() {
                Err(ToolChoicePolicyError::RequiredWithoutTools)
            } else {
                Ok(executable_tool_names.clone())
            }
        }
        Some(ToolChoice::Specific { function_names }) => {
            if function_names.is_empty() {
                return Err(ToolChoicePolicyError::EmptySpecific);
            }
            let requested = function_names.iter().cloned().collect::<BTreeSet<_>>();
            let missing = requested
                .iter()
                .filter(|name| {
                    !executable_tool_names.contains(*name)
                        && output_tool_name != Some(name.as_str())
                })
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(ToolChoicePolicyError::MissingSpecific { missing });
            }
            Ok(requested
                .intersection(executable_tool_names)
                .cloned()
                .collect())
        }
    }
}

fn output_tool_name(schema: &schemars::Schema, names: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(schema.as_value().to_string().as_bytes());
    let digest = hasher.finalize();
    let prefix = digest
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut candidate = format!("__rig_output_{prefix}");
    while names.contains(&candidate) {
        candidate.push('_');
    }
    candidate
}

fn dispatch_model_effects(world: &mut World) {
    let mut entities = {
        let mut query = world.query::<(Entity, &RunNode)>();
        query
            .iter(world)
            .filter(|(entity, run)| {
                run.phase == RunPhase::ReadyModel
                    && !has_active_kind(world, *entity, ActiveOperationKind::Model)
                    && !world.entity(*entity).contains::<TerminalState>()
            })
            .map(|(entity, run)| (entity, run.clone()))
            .collect::<Vec<_>>()
    };
    entities.sort_by_key(|(_, run)| run.id);
    for (entity, run) in entities {
        let Some(agent) = agent_for_run(world, &run) else {
            fail_run(
                world,
                entity,
                "unknown_agent",
                "run agent topology disappeared",
            );
            continue;
        };
        let dispatched = world
            .get::<RunAccounting>(entity)
            .map_or(0, |accounting| accounting.model_calls_dispatched);
        if dispatched >= agent.spec.max_model_calls {
            set_terminal(
                world,
                entity,
                TerminalReason::ModelCallBudgetExhausted,
                None,
            );
            continue;
        }

        let mut entries = collect_turn_capabilities(world, &run);
        let executable_tool_names = entries
            .iter()
            .map(|entry| entry.definition.name.clone())
            .collect::<BTreeSet<_>>();
        let mut preamble = agent.spec.preamble.clone();
        let mut output_schema = None;
        let mut selected_output_name = None;
        let mut output_tool_schema = None;
        if let Some(state) = world.get::<StructuredOutputState>(entity).map(|state| {
            (
                state.schema.clone(),
                state.policy.clone(),
                state.output_tool_name.clone(),
            )
        }) {
            let candidate = state
                .2
                .clone()
                .unwrap_or_else(|| output_tool_name(&state.0, &executable_tool_names));
            let callable = output_tool_callable(agent.spec.tool_choice.as_ref(), &candidate);
            let resolved = if state.2.is_some() {
                OutputMode::Tool
            } else {
                resolve_output_mode(
                    !entries.is_empty(),
                    callable,
                    agent.composes_native_output_with_tools,
                    state.1.mode,
                )
            };
            if matches!(resolved, OutputMode::Tool)
                && (!callable || executable_tool_names.contains(&candidate))
            {
                fail_run(
                    world,
                    entity,
                    "invalid_tool_choice",
                    if !callable {
                        format!(
                            "the active tool choice cannot call structured-output tool `{candidate}`"
                        )
                    } else {
                        format!(
                            "real tool `{candidate}` conflicts with the structured-output tool reserved for this run"
                        )
                    },
                );
                continue;
            }
            match resolved {
                OutputMode::Native => output_schema = Some(state.0.clone()),
                OutputMode::Prompted => {
                    let instruction = format!(
                        "Respond with only JSON satisfying this schema: {}",
                        state.0.as_value()
                    );
                    preamble = Some(match preamble {
                        Some(base) => format!("{base}\n\n{instruction}"),
                        None => instruction,
                    });
                }
                OutputMode::Tool => {
                    let instruction = format!(
                        "When the answer is ready, call `{candidate}` exactly once with the structured final result."
                    );
                    preamble = Some(match preamble {
                        Some(base) => format!("{base}\n\n{instruction}"),
                        None => instruction,
                    });
                    selected_output_name = Some(candidate);
                    output_tool_schema = Some(state.0.clone());
                }
                OutputMode::Auto => {}
            }
            if let Some(mut output) = world.get_mut::<StructuredOutputState>(entity) {
                output.resolved_mode = resolved;
                if selected_output_name.is_some() {
                    output.output_tool_name.clone_from(&selected_output_name);
                }
            }
        }
        let allowed_tool_names = match allowed_executable_tool_names(
            &executable_tool_names,
            agent.spec.tool_choice.as_ref(),
            selected_output_name.as_deref(),
        ) {
            Ok(allowed) => allowed,
            Err(error) => {
                fail_run(world, entity, "invalid_tool_choice", error.to_string());
                continue;
            }
        };
        entries.retain(|entry| allowed_tool_names.contains(&entry.definition.name));
        let mut definitions = entries
            .iter()
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        if let (Some(name), Some(schema)) = (&selected_output_name, output_tool_schema) {
            definitions.push(ToolDefinition {
                name: name.clone(),
                description: "Finalize the run with arguments satisfying the output schema."
                    .to_string(),
                parameters: schema.to_value(),
            });
        }

        let Some(transcript) = world.get::<CanonicalTranscript>(entity) else {
            continue;
        };
        let mut history = transcript.messages.clone();
        if let Some(feedback) = world.get::<RecoveryFeedback>(entity) {
            history.extend(feedback.0.iter().cloned().map(Message::user));
        }
        let Some(chat_history) = OneOrMany::from_iter_optional(history) else {
            fail_run(
                world,
                entity,
                "empty_history",
                "model request had no canonical messages",
            );
            continue;
        };
        let request = CompletionRequest {
            model: None,
            preamble,
            chat_history,
            documents: Vec::new(),
            tools: definitions,
            temperature: agent.spec.temperature,
            max_tokens: agent.spec.max_tokens,
            tool_choice: agent.spec.tool_choice.clone(),
            additional_params: agent.spec.additional_params.clone(),
            output_schema,
            record_telemetry_content: agent.spec.record_telemetry_content,
        };
        let header = new_header(world, &run);
        if !enqueue_effect(
            world,
            entity,
            EffectIntent::Model(Box::new(ModelEffectIntent {
                header,
                model_id: run.model_id,
                request,
                streaming: matches!(run.streaming, StreamingMode::Streaming),
            })),
        ) {
            continue;
        }
        register_operation(world, entity, header, ActiveOperationKind::Model);
        for entry in &entries {
            world
                .resource_mut::<CapabilityReferences>()
                .0
                .entry(entry.capability_id)
                .or_default()
                .insert(run.id);
        }
        world.entity_mut(entity).insert(TurnCapabilitySnapshot {
            entries,
            output_tool_name: selected_output_name,
        });
        if let Some(mut accounting) = world.get_mut::<RunAccounting>(entity) {
            accounting.model_calls_dispatched = accounting.model_calls_dispatched.saturating_add(1);
        }
        if let Some(mut run) = world.get_mut::<RunNode>(entity) {
            run.phase = RunPhase::WaitingModel;
        }
        publish(
            world,
            entity,
            RunEvent::ModelDispatched(header.operation_id),
        );
        bump_progress(world, entity);
    }
}

fn dispatch_tool_effects(world: &mut World) {
    let mut entities = {
        let mut query = world.query_filtered::<Entity, With<PendingToolBatch>>();
        query.iter(world).collect::<Vec<_>>()
    };
    entities.sort_by_key(|entity| world.get::<RunNode>(*entity).map(|run| run.id));
    for entity in entities {
        if world.entity(entity).contains::<TerminalState>() {
            continue;
        }
        let Some(run) = world.get::<RunNode>(entity).cloned() else {
            continue;
        };
        let mut calls = world
            .get::<PendingToolBatch>(entity)
            .map(|batch| {
                batch
                    .calls
                    .iter()
                    .filter(|call| {
                        !call.suppressed && call.operation_id.is_none() && call.result.is_none()
                    })
                    .map(|call| {
                        (
                            call.order,
                            call.call.clone(),
                            call.capability_id,
                            call.grant_id,
                            call.revision,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        calls.sort_by_key(|(order, ..)| *order);
        for (order, call, capability_id, grant_id, revision) in calls {
            let mut header = new_header(world, &run);
            header.capability_id = Some(capability_id);
            header.grant_id = Some(grant_id);
            header.capability_revision = Some(revision);
            let tool_call_id = call.id.clone();
            let effect = EffectIntent::Tool(ToolEffectIntent {
                header,
                tool_call_id: tool_call_id.clone(),
                provider_call_id: call.call_id.clone(),
                name: call.function.name,
                arguments: call.function.arguments,
                order,
            });
            if !enqueue_effect(world, entity, effect) {
                break;
            }
            register_tool_operation(world, entity, header, tool_call_id.clone(), order);
            if let Some(mut batch) = world.get_mut::<PendingToolBatch>(entity)
                && let Some(planned) = batch
                    .calls
                    .iter_mut()
                    .find(|planned| planned.order == order)
            {
                planned.operation_id = Some(header.operation_id);
            }
            publish(
                world,
                entity,
                RunEvent::ToolDispatched {
                    operation_id: header.operation_id,
                    tool_call_id,
                },
            );
            bump_progress(world, entity);
        }
    }
}

fn commit_tool_batches(world: &mut World) {
    let entities = {
        let mut query = world.query_filtered::<Entity, With<PendingToolBatch>>();
        query
            .iter(world)
            .filter(|entity| {
                world
                    .get::<PendingToolBatch>(*entity)
                    .is_some_and(|batch| batch.calls.iter().all(|call| call.result.is_some()))
            })
            .collect::<Vec<_>>()
    };
    for entity in entities {
        if world.entity(entity).contains::<TerminalState>() {
            continue;
        }
        let Some(mut batch) = world.entity_mut(entity).take::<PendingToolBatch>() else {
            continue;
        };
        let run_id = world.get::<RunNode>(entity).map(|run| run.id);
        let tool_count = batch.calls.len();
        batch.calls.sort_by_key(|call| call.order);
        let Some(assistant_content) = OneOrMany::from_iter_optional(batch.assistant_content) else {
            fail_run(
                world,
                entity,
                "empty_tool_turn",
                "tool batch had no assistant content",
            );
            continue;
        };
        let results = batch
            .calls
            .iter()
            .filter_map(|call| {
                let output = call.result.clone()?;
                Some(UserContent::ToolResult(ToolResult {
                    id: call.call.id.clone(),
                    call_id: call.call.call_id.clone(),
                    content: output.into_content(),
                }))
            })
            .collect::<Vec<_>>();
        let Some(results) = OneOrMany::from_iter_optional(results) else {
            fail_run(
                world,
                entity,
                "missing_tool_result",
                "tool batch completed without paired results",
            );
            continue;
        };
        if !commit_assistant_turn(
            world,
            entity,
            batch.message_id,
            assistant_content,
            Some(results),
        ) {
            continue;
        }
        for call in batch.calls {
            if let Some(operation_id) = call.operation_id {
                publish(
                    world,
                    entity,
                    RunEvent::ToolCommitted {
                        operation_id,
                        tool_call_id: call.call.id,
                    },
                );
            }
        }
        if let Some(mut run) = world.get_mut::<RunNode>(entity) {
            run.phase = RunPhase::ReadyModel;
        }
        tracing::info!(
            target: "rig::ecs",
            run_id = ?run_id,
            tool_count,
            "tool batch committed to canonical transcript"
        );
        bump_progress(world, entity);
    }
}

fn cleanup_terminal_runs(world: &mut World) {
    let (observed_retention, unobserved_retention) = world
        .get_resource::<RuntimeConfigResource>()
        .map_or((16, 1_024), |config| {
            (
                config.0.terminal_retention_ticks,
                config.0.unobserved_terminal_retention_ticks,
            )
        });
    let tick = world.resource::<RuntimeTick>().0;
    let entities = {
        let mut query = world.query::<(
            Entity,
            &RunNode,
            &TerminalState,
            Option<&TerminalObservation>,
        )>();
        query
            .iter(world)
            .filter(|(entity, _, terminal, observation)| {
                let (age_origin, retention) = match observation {
                    Some(observation) => (
                        observation.observed_tick.max(terminal.terminal_tick),
                        observed_retention,
                    ),
                    None => (terminal.terminal_tick, unobserved_retention),
                };
                tick.saturating_sub(age_origin) >= retention
                    && world
                        .get::<ActiveOperations>(*entity)
                        .is_none_or(|operations| operations.0.values().all(|op| op.completed))
            })
            .map(|(entity, run, _, _)| (entity, run.id))
            .collect::<Vec<_>>()
    };
    for (entity, run_id) in entities {
        world.resource_mut::<TopologyIndex>().runs.remove(&run_id);
        let mut capability_references = world.resource_mut::<CapabilityReferences>();
        for references in capability_references.0.values_mut() {
            references.remove(&run_id);
        }
        capability_references
            .0
            .retain(|_, references| !references.is_empty());
        let _ = world.despawn(entity);
        bump_progress(world, entity);
    }
}

fn cleanup_retired_capabilities(world: &mut World) {
    let candidates = {
        let mut query = world.query::<(Entity, &CapabilityNode)>();
        query
            .iter(world)
            .filter(|(_, capability)| capability.retired)
            .filter(|(_, capability)| {
                world
                    .resource::<CapabilityReferences>()
                    .0
                    .get(&capability.id)
                    .is_none_or(BTreeSet::is_empty)
            })
            .filter(|(_, capability)| {
                world
                    .resource::<TopologyIndex>()
                    .grants
                    .values()
                    .filter_map(|entity| world.get::<GrantNode>(*entity))
                    .all(|grant| grant.capability_id != capability.id || grant.revoked)
            })
            .map(|(entity, capability)| (entity, capability.id))
            .collect::<Vec<_>>()
    };
    for (entity, capability_id) in candidates {
        let grant_entities = world
            .resource::<TopologyIndex>()
            .grants
            .iter()
            .filter_map(|(grant_id, grant_entity)| {
                world
                    .get::<GrantNode>(*grant_entity)
                    .is_some_and(|grant| grant.capability_id == capability_id)
                    .then_some((*grant_id, *grant_entity))
            })
            .collect::<Vec<_>>();
        {
            let mut index = world.resource_mut::<TopologyIndex>();
            for (grant_id, _) in &grant_entities {
                index.grants.remove(grant_id);
            }
            index.capabilities.remove(&capability_id);
        }
        world
            .resource_mut::<CapabilityReferences>()
            .0
            .remove(&capability_id);
        for (_, grant_entity) in grant_entities {
            let _ = world.despawn(grant_entity);
        }
        world
            .resource_mut::<CapabilitiesToDrop>()
            .0
            .push(capability_id);
        let _ = world.despawn(entity);
        bump_progress(world, entity);
    }
}

/// Runtime config resource kept private so schedule cleanup uses the same bounds.
#[derive(Resource, Clone)]
pub(crate) struct RuntimeConfigResource(pub(crate) RuntimeConfig);

/// Force a terminal failure when an external wait path cannot continue.
pub(crate) fn fail_effect_wait(world: &mut World, run_id: RunId, error: String) {
    let entity = world.resource::<TopologyIndex>().runs.get(&run_id).copied();
    if let Some(entity) = entity {
        fail_run(world, entity, "effect_wait", error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentId, Generation, GrantId, ModelId, TenantId};

    #[test]
    fn stream_sequence_tracking_is_constant_space_and_strictly_monotonic() {
        let operation_id = OperationId::new();
        let mut accepted = AcceptedDeltas::default();

        for sequence in 0..100_000 {
            assert_eq!(
                accept_delta_sequence(&mut accepted, operation_id, sequence),
                Ok(())
            );
        }

        assert_eq!(accepted.0.len(), 1);
        assert_eq!(accepted.0.get(&operation_id), Some(&100_000));
        assert_eq!(
            accept_delta_sequence(&mut accepted, operation_id, 99_999),
            Err(EffectRejectionReason::Duplicate)
        );
        assert_eq!(
            accept_delta_sequence(&mut accepted, operation_id, 100_001),
            Err(EffectRejectionReason::OutOfOrder)
        );
        assert_eq!(accepted.0.len(), 1);
    }

    #[test]
    fn repeated_retirement_and_observation_cleanup_releases_all_bookkeeping() {
        let mut world = World::new();
        world.insert_resource(TopologyIndex::default());
        world.insert_resource(CapabilityReferences::default());
        world.insert_resource(CapabilitiesToDrop::default());
        world.insert_resource(RuntimeTick(100));
        world.insert_resource(RuntimeConfigResource(RuntimeConfig {
            terminal_retention_ticks: 0,
            ..RuntimeConfig::default()
        }));

        for _ in 0..64 {
            let tenant_id = TenantId::new();
            let capability_id = CapabilityId::new();
            let grant_id = GrantId::new();
            let capability_entity = world
                .spawn(CapabilityNode {
                    id: capability_id,
                    tenant_id,
                    kind: crate::CapabilityKind::Tool,
                    definition: None,
                    revision: 1,
                    retired: true,
                })
                .id();
            let grant_entity = world
                .spawn(GrantNode {
                    id: grant_id,
                    agent_id: AgentId::new(),
                    capability_id,
                    tenant_id,
                    revoked: true,
                })
                .id();
            {
                let mut index = world.resource_mut::<TopologyIndex>();
                index.capabilities.insert(capability_id, capability_entity);
                index.grants.insert(grant_id, grant_entity);
            }
            world
                .resource_mut::<CapabilityReferences>()
                .0
                .insert(capability_id, BTreeSet::new());

            cleanup_retired_capabilities(&mut world);

            assert!(world.get::<CapabilityNode>(capability_entity).is_none());
            assert!(world.get::<GrantNode>(grant_entity).is_none());
            assert!(world.resource::<TopologyIndex>().capabilities.is_empty());
            assert!(world.resource::<TopologyIndex>().grants.is_empty());
            assert!(world.resource::<CapabilityReferences>().0.is_empty());
        }
        let dropped = &world.resource::<CapabilitiesToDrop>().0;
        assert_eq!(dropped.len(), 64);
        assert_eq!(dropped.iter().copied().collect::<BTreeSet<_>>().len(), 64);

        for _ in 0..64 {
            let run_id = RunId::new();
            let capability_id = CapabilityId::new();
            let run_entity = world
                .spawn((
                    RunNode {
                        id: run_id,
                        agent_id: AgentId::new(),
                        model_id: ModelId::new(),
                        tenant_id: TenantId::new(),
                        generation: Generation(0),
                        phase: RunPhase::Terminal,
                        streaming: StreamingMode::Blocking,
                        created_tick: 0,
                    },
                    TerminalState {
                        reason: TerminalReason::Completed,
                        terminal_tick: 1,
                    },
                    TerminalObservation { observed_tick: 1 },
                    ActiveOperations::default(),
                ))
                .id();
            world
                .resource_mut::<TopologyIndex>()
                .runs
                .insert(run_id, run_entity);
            world
                .resource_mut::<CapabilityReferences>()
                .0
                .insert(capability_id, BTreeSet::from([run_id]));

            cleanup_terminal_runs(&mut world);

            assert!(world.get::<RunNode>(run_entity).is_none());
            assert!(world.resource::<TopologyIndex>().runs.is_empty());
            assert!(world.resource::<CapabilityReferences>().0.is_empty());
        }
    }
}
