use crate::{
    AgentId, CapabilityGrant, CapabilityId, ConcurrencyLimit, CorrelationId, GrantId,
    InvalidToolPolicy, InvalidToolResolution, MemoryAppendEffect, MemoryAppendIngress,
    MemoryImplementation, MemoryLoadEffect, MemoryLoadIngress, MemoryRebinding, OperationId,
    PersistedMemoryBinding, PersistedToolBinding, ProviderFinalEnvelope, RecoveryPolicy, RunId,
    RunSnapshot, RunStatus, RuntimeEvent, SnapshotError, StoreId, TenantId, TerminalReason,
    ToolCapability, ToolEffectRequest, ToolImplementation, ToolIngress, ToolRebinding,
    ToolSnapshot, ToolSnapshotEntry, VectorSearchEffect, VectorSearchIngress, WorldId,
    capability::BoundTool,
    events::EventLog,
    schedule,
    state::{
        Accounting, CallBudget, Generation, InvalidResolution, InvalidRetryCount, Lifecycle,
        PendingOperation, PendingRequest, PendingTool, PendingVector, Retention, RunIdentity,
        StalledPasses, StreamCursor, ToolBatch, ToolContinuation, Transcript,
    },
    store::MemoryState,
};
use bevy_ecs::{entity::Entity, prelude::Resource, world::World};
use rig_core::completion::{AssistantContent, CompletionRequest, Usage};
use rig_core::vector_store::{
    VectorStoreIndexDyn,
    request::{Filter, VectorSearchRequest},
};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct EffectRequest {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub request: CompletionRequest,
}

#[derive(Clone, Debug)]
pub struct CompletionIngress {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub choice: Vec<AssistantContent>,
    pub usage: Usage,
}

#[derive(Resource, Default)]
pub(crate) struct DispatchQueue(VecDeque<Entity>);

#[derive(Resource, Default)]
pub(crate) struct DispatchOutbox(HashMap<RunId, Result<EffectRequest, RuntimeError>>);

struct ToolDispatchCommand {
    entity: Entity,
    snapshot: ToolSnapshot,
    call_index: usize,
    name: String,
    arguments: String,
    implementation: Arc<dyn ToolImplementation>,
}

#[derive(Resource, Default)]
pub(crate) struct ToolDispatchQueue(VecDeque<ToolDispatchCommand>);

#[derive(Resource, Default)]
pub(crate) struct ToolDispatchOutbox(HashMap<RunId, Result<ToolEffectRequest, RuntimeError>>);

struct MemoryDispatchCommand {
    entity: Entity,
    conversation_id: String,
    implementation: Arc<dyn MemoryImplementation>,
}

#[derive(Resource, Default)]
pub(crate) struct MemoryDispatchQueue(VecDeque<MemoryDispatchCommand>);

#[derive(Resource, Default)]
pub(crate) struct MemoryDispatchOutbox(HashMap<RunId, Result<MemoryLoadEffect, RuntimeError>>);

struct VectorDispatchCommand {
    entity: Entity,
    implementation: Arc<dyn VectorStoreIndexDyn>,
    request: VectorSearchRequest<Filter<serde_json::Value>>,
}

#[derive(Resource, Default)]
pub(crate) struct VectorDispatchQueue(VecDeque<VectorDispatchCommand>);

#[derive(Resource, Default)]
pub(crate) struct VectorDispatchOutbox(HashMap<RunId, Result<VectorSearchEffect, RuntimeError>>);

#[derive(Resource, Default)]
pub(crate) struct IngressQueue(VecDeque<CompletionIngress>);

#[derive(Resource, Default)]
pub(crate) struct IngressOutbox(HashMap<RunId, Result<Entity, RuntimeError>>);

#[derive(Resource, Default)]
pub(crate) struct ToolIngressQueue(VecDeque<ToolIngress>);

#[derive(Resource, Default)]
pub(crate) struct ToolIngressOutbox(HashMap<RunId, Result<(), RuntimeError>>);

#[derive(Resource, Default)]
pub(crate) struct ToolTurnIngressQueue(VecDeque<CompletionIngress>);

#[derive(Resource, Default)]
pub(crate) struct ToolTurnIngressOutbox(HashMap<RunId, Result<CompletionRequest, RuntimeError>>);

#[derive(Resource, Default)]
pub(crate) struct MemoryAppendIngressQueue(VecDeque<MemoryAppendIngress>);

#[derive(Resource, Default)]
pub(crate) struct MemoryAppendIngressOutbox(HashMap<RunId, Result<(), RuntimeError>>);

#[derive(Resource, Default)]
pub(crate) struct MemoryLoadIngressQueue(VecDeque<MemoryLoadIngress>);

#[derive(Resource, Default)]
pub(crate) struct MemoryLoadIngressOutbox(HashMap<RunId, Result<(), RuntimeError>>);

type VectorDocuments = Vec<(f64, String, serde_json::Value)>;

#[derive(Resource, Default)]
pub(crate) struct VectorIngressQueue(VecDeque<VectorSearchIngress>);

#[derive(Resource, Default)]
pub(crate) struct VectorIngressOutbox(HashMap<RunId, Result<VectorDocuments, RuntimeError>>);

#[derive(Resource, Default)]
pub(crate) struct StreamingIngressQueue(VecDeque<StreamingIngress>);

#[derive(Resource, Default)]
pub(crate) struct StreamingIngressOutbox(HashMap<RunId, Result<(), RuntimeError>>);

#[derive(Resource, Default)]
pub(crate) struct HostedEffectQueue(VecDeque<(RunId, HostedEffect)>);

pub(crate) enum PolicyCommand {
    InvalidTool {
        entity: Entity,
        call_index: usize,
        tool_name: String,
        policy: InvalidToolPolicy,
    },
    Cancel {
        entity: Entity,
        reason: String,
    },
    RetryResponse {
        completion: CompletionIngress,
        feedback: String,
    },
    ExhaustResponse {
        completion: CompletionIngress,
        message: String,
    },
    ContinueAfterTools {
        entity: Entity,
        request: Box<CompletionRequest>,
        results: Vec<rig_core::completion::Message>,
    },
}

pub(crate) enum PolicyCommandResult {
    Unit,
    InvalidTool(InvalidToolResolution),
}

#[derive(Resource, Default)]
pub(crate) struct PolicyQueue(VecDeque<(RunId, PolicyCommand)>);

#[derive(Resource, Default)]
pub(crate) struct PolicyOutbox(HashMap<RunId, Result<PolicyCommandResult, RuntimeError>>);

pub(crate) fn dispatch_system(world: &mut World) {
    let queued = world
        .resource_mut::<DispatchQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for entity_id in queued {
        let result = (|| {
            let mut entity = world.entity_mut(entity_id);
            let identity = *entity
                .get::<RunIdentity>()
                .ok_or(RuntimeError::MissingRun)?;
            if entity
                .get::<MemoryState>()
                .is_some_and(|memory| !memory.loaded)
            {
                return Err(RuntimeError::MemoryNotLoaded);
            }
            if matches!(
                entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?.0,
                RunStatus::Terminal(_)
            ) {
                return Err(RuntimeError::Terminal);
            }
            let budget = entity.get::<CallBudget>().ok_or(RuntimeError::MissingRun)?;
            if budget.completed_calls >= budget.max_calls {
                entity
                    .get_mut::<Lifecycle>()
                    .ok_or(RuntimeError::MissingRun)?
                    .0 = RunStatus::Terminal(TerminalReason::BudgetExhausted);
                return Err(RuntimeError::Terminal);
            }
            let request = entity
                .get_mut::<PendingRequest>()
                .ok_or(RuntimeError::MissingRun)?
                .0
                .take()
                .ok_or(RuntimeError::NoPendingOperation)?;
            let operation_id = OperationId::fresh();
            let correlation_id = CorrelationId::fresh();
            let generation = entity
                .get::<Generation>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::AwaitingModel;
            let effect = EffectRequest {
                world_id: identity.world_id,
                tenant_id: identity.tenant_id,
                run_id: identity.run_id,
                operation_id,
                correlation_id,
                generation,
                request,
            };
            entity.insert(PendingOperation {
                operation_id,
                correlation_id,
                generation,
                request: effect.request.clone(),
            });
            Ok(effect)
        })();
        let run_id = world
            .get::<RunIdentity>(entity_id)
            .map(|id| id.run_id)
            .unwrap_or_default();
        world
            .resource_mut::<DispatchOutbox>()
            .0
            .insert(run_id, result);
    }
}

pub(crate) fn tool_dispatch_system(world: &mut World) {
    let queued = world
        .resource_mut::<ToolDispatchQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for command in queued {
        let run_id = world
            .get::<RunIdentity>(command.entity)
            .map(|id| id.run_id)
            .unwrap_or_default();
        let result = (|| {
            let mut entity = world.entity_mut(command.entity);
            let identity = *entity
                .get::<RunIdentity>()
                .ok_or(RuntimeError::MissingRun)?;
            if matches!(
                entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?.0,
                RunStatus::Terminal(_)
            ) {
                return Err(RuntimeError::Terminal);
            }
            let generation = entity
                .get::<Generation>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            if command.snapshot.world_id != identity.world_id
                || command.snapshot.tenant_id != identity.tenant_id
                || command.snapshot.run_id != identity.run_id
                || command.snapshot.generation != generation
            {
                return Err(RuntimeError::StaleToolSnapshot);
            }
            let entry = command
                .snapshot
                .entries
                .iter()
                .find(|entry| entry.definition.name == command.name)
                .ok_or(RuntimeError::ToolNotAdvertised)?;
            let limit = entity
                .get::<ConcurrencyLimit>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            let operation_id = OperationId::fresh();
            let correlation_id = CorrelationId::fresh();
            let mut batch = entity
                .get_mut::<ToolBatch>()
                .ok_or(RuntimeError::MissingRun)?;
            if batch.pending.contains_key(&command.call_index)
                || batch.completed.contains_key(&command.call_index)
            {
                return Err(RuntimeError::DuplicateToolCall);
            }
            if batch.pending.len() >= limit {
                return Err(RuntimeError::Backpressure(limit));
            }
            batch.pending.insert(
                command.call_index,
                PendingTool {
                    operation_id,
                    correlation_id,
                    generation,
                    capability_id: entry.capability_id,
                    revision: entry.revision,
                },
            );
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::AwaitingTools;
            Ok(ToolEffectRequest {
                world_id: identity.world_id,
                tenant_id: identity.tenant_id,
                run_id: identity.run_id,
                operation_id,
                correlation_id,
                generation,
                call_index: command.call_index,
                capability_id: entry.capability_id,
                revision: entry.revision,
                name: command.name,
                arguments: command.arguments,
                implementation: command.implementation,
            })
        })();
        world
            .resource_mut::<ToolDispatchOutbox>()
            .0
            .insert(run_id, result);
    }
}

pub(crate) fn memory_dispatch_system(world: &mut World) {
    let queued = world
        .resource_mut::<MemoryDispatchQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for command in queued {
        let run_id = world
            .get::<RunIdentity>(command.entity)
            .map(|id| id.run_id)
            .unwrap_or_default();
        let result = (|| {
            let mut entity = world.entity_mut(command.entity);
            let identity = *entity
                .get::<RunIdentity>()
                .ok_or(RuntimeError::MissingRun)?;
            if matches!(
                entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?.0,
                RunStatus::Terminal(_)
            ) {
                return Err(RuntimeError::Terminal);
            }
            let generation = entity
                .get::<Generation>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            let store_id = StoreId::new();
            let operation_id = OperationId::new();
            let correlation_id = CorrelationId::new();
            entity.insert(MemoryState {
                store_id,
                conversation_id: command.conversation_id.clone(),
                loaded: false,
                load_correlation: Some((operation_id, correlation_id, generation)),
                appended_generation: None,
                append_correlation: None,
                persist_from: 0,
            });
            Ok(MemoryLoadEffect {
                world_id: identity.world_id,
                tenant_id: identity.tenant_id,
                run_id: identity.run_id,
                operation_id,
                correlation_id,
                generation,
                store_id,
                conversation_id: command.conversation_id,
                implementation: command.implementation,
            })
        })();
        world
            .resource_mut::<MemoryDispatchOutbox>()
            .0
            .insert(run_id, result);
    }
}

pub(crate) fn vector_dispatch_system(world: &mut World) {
    let queued = world
        .resource_mut::<VectorDispatchQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for command in queued {
        let run_id = world
            .get::<RunIdentity>(command.entity)
            .map(|id| id.run_id)
            .unwrap_or_default();
        let result = (|| {
            let mut entity = world.entity_mut(command.entity);
            let identity = *entity
                .get::<RunIdentity>()
                .ok_or(RuntimeError::MissingRun)?;
            if matches!(
                entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?.0,
                RunStatus::Terminal(_)
            ) {
                return Err(RuntimeError::Terminal);
            }
            let generation = entity
                .get::<Generation>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            let store_id = StoreId::new();
            let operation_id = OperationId::new();
            let correlation_id = CorrelationId::new();
            entity.insert(PendingVector {
                operation_id,
                correlation_id,
                generation,
                store_id,
            });
            Ok(VectorSearchEffect {
                world_id: identity.world_id,
                tenant_id: identity.tenant_id,
                run_id: identity.run_id,
                operation_id,
                correlation_id,
                generation,
                store_id,
                request: command.request,
                implementation: command.implementation,
            })
        })();
        world
            .resource_mut::<VectorDispatchOutbox>()
            .0
            .insert(run_id, result);
    }
}

pub(crate) fn completion_ingress_system(world: &mut World) {
    let queued = world
        .resource_mut::<IngressQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for completion in queued {
        let run_id = completion.run_id;
        let result = (|| {
            let entity_id = world
                .query::<(Entity, &RunIdentity)>()
                .iter(world)
                .find_map(|(entity, identity)| {
                    (identity.run_id == completion.run_id).then_some(entity)
                })
                .ok_or(RuntimeError::MissingRun)?;
            let mut entity = world.entity_mut(entity_id);
            let identity = entity
                .get::<RunIdentity>()
                .ok_or(RuntimeError::MissingRun)?;
            if identity.world_id != completion.world_id {
                return Err(RuntimeError::ForeignWorld);
            }
            if identity.tenant_id != completion.tenant_id {
                return Err(RuntimeError::WrongTenant);
            }
            if matches!(
                entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?.0,
                RunStatus::Terminal(_)
            ) {
                return Err(RuntimeError::Terminal);
            }
            let pending = entity
                .get::<PendingOperation>()
                .ok_or(RuntimeError::NoPendingOperation)?;
            if pending.operation_id != completion.operation_id
                || pending.correlation_id != completion.correlation_id
                || pending.generation != completion.generation
            {
                return Err(RuntimeError::StaleCompletion);
            }
            entity.remove::<PendingOperation>();
            entity
                .get_mut::<CallBudget>()
                .ok_or(RuntimeError::MissingRun)?
                .completed_calls += 1;
            entity
                .get_mut::<Accounting>()
                .ok_or(RuntimeError::MissingRun)?
                .usage += completion.usage;
            let choice = completion.choice;
            let transcript = &mut entity
                .get_mut::<Transcript>()
                .ok_or(RuntimeError::MissingRun)?;
            transcript.final_output = choice.clone();
            if let Some(content) = rig_core::OneOrMany::from_iter_optional(choice) {
                transcript
                    .history
                    .push(rig_core::completion::Message::from(content));
            }
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::Terminal(TerminalReason::Completed);
            let content = entity
                .get::<Transcript>()
                .ok_or(RuntimeError::MissingRun)?
                .final_output
                .clone();
            let events = &mut entity
                .get_mut::<EventLog>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            events.push(RuntimeEvent::AcceptedFinal { run_id, content });
            events.push(RuntimeEvent::Terminal {
                run_id,
                reason: TerminalReason::Completed,
            });
            Ok(entity_id)
        })();
        world
            .resource_mut::<IngressOutbox>()
            .0
            .insert(run_id, result);
    }
}

pub(crate) fn tool_ingress_system(world: &mut World) {
    let queued = world
        .resource_mut::<ToolIngressQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for ingress in queued {
        let run_id = ingress.run_id;
        let result = commit_tool_ingress(world, ingress);
        world
            .resource_mut::<ToolIngressOutbox>()
            .0
            .insert(run_id, result);
    }
}

pub(crate) fn tool_turn_ingress_system(world: &mut World) {
    let queued = world
        .resource_mut::<ToolTurnIngressQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for completion in queued {
        let run_id = completion.run_id;
        let result = commit_tool_turn_ingress(world, completion);
        world
            .resource_mut::<ToolTurnIngressOutbox>()
            .0
            .insert(run_id, result);
    }
}

fn commit_tool_turn_ingress(
    world: &mut World,
    completion: CompletionIngress,
) -> Result<CompletionRequest, RuntimeError> {
    if !completion
        .choice
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)))
    {
        return Err(RuntimeError::NoPendingOperation);
    }
    let entity_id = validate_effect_world(world, &completion)?;
    let mut entity = world.entity_mut(entity_id);
    let pending = entity
        .take::<PendingOperation>()
        .ok_or(RuntimeError::NoPendingOperation)?;
    entity
        .get_mut::<CallBudget>()
        .ok_or(RuntimeError::MissingRun)?
        .completed_calls += 1;
    entity
        .get_mut::<Accounting>()
        .ok_or(RuntimeError::MissingRun)?
        .usage += completion.usage;
    let content = rig_core::OneOrMany::from_iter_optional(completion.choice)
        .ok_or(RuntimeError::NoPendingOperation)?;
    entity
        .get_mut::<Transcript>()
        .ok_or(RuntimeError::MissingRun)?
        .history
        .push(rig_core::completion::Message::from(content));
    entity
        .get_mut::<Lifecycle>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = RunStatus::AwaitingTools;
    entity
        .get_mut::<ToolContinuation>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = Some(pending.request.clone());
    Ok(pending.request)
}

fn commit_tool_ingress(world: &mut World, ingress: ToolIngress) -> Result<(), RuntimeError> {
    let entity_id = world
        .query::<(Entity, &RunIdentity)>()
        .iter(world)
        .find_map(|(entity, identity)| (identity.run_id == ingress.run_id).then_some(entity))
        .ok_or(RuntimeError::MissingRun)?;
    let identity = world
        .get::<RunIdentity>(entity_id)
        .ok_or(RuntimeError::MissingRun)?;
    if identity.world_id != ingress.world_id {
        return Err(RuntimeError::ForeignWorld);
    }
    if identity.tenant_id != ingress.tenant_id {
        return Err(RuntimeError::WrongTenant);
    }
    if matches!(
        world
            .get::<Lifecycle>(entity_id)
            .ok_or(RuntimeError::MissingRun)?
            .0,
        RunStatus::Terminal(_)
    ) {
        return Err(RuntimeError::StaleToolCompletion);
    }
    let generation = world
        .get::<Generation>(entity_id)
        .ok_or(RuntimeError::MissingRun)?
        .0;
    if generation != ingress.generation {
        return Err(RuntimeError::StaleToolCompletion);
    }
    let mut entity = world.entity_mut(entity_id);
    let (committed, quiescent) = {
        let mut batch = entity
            .get_mut::<ToolBatch>()
            .ok_or(RuntimeError::MissingRun)?;
        let pending = batch
            .pending
            .get(&ingress.call_index)
            .copied()
            .ok_or(RuntimeError::StaleToolCompletion)?;
        if pending.operation_id != ingress.operation_id
            || pending.correlation_id != ingress.correlation_id
            || pending.generation != ingress.generation
            || pending.capability_id != ingress.capability_id
            || pending.revision != ingress.revision
        {
            return Err(RuntimeError::StaleToolCompletion);
        }
        batch.pending.remove(&ingress.call_index);
        batch
            .completed
            .insert(ingress.call_index, (ingress.name, ingress.result));
        let mut committed = Vec::new();
        loop {
            let index = batch.next_commit;
            let Some((name, result)) = batch.completed.remove(&index) else {
                break;
            };
            batch.next_commit += 1;
            committed.push((index, name, result));
        }
        let quiescent = batch.pending.is_empty() && batch.completed.is_empty();
        (committed, quiescent)
    };
    let events = &mut entity
        .get_mut::<EventLog>()
        .ok_or(RuntimeError::MissingRun)?
        .0;
    for (call_index, name, result) in committed {
        events.push(RuntimeEvent::ToolCommitted {
            run_id: ingress.run_id,
            call_index,
            name,
            result,
        });
    }
    if quiescent {
        entity
            .get_mut::<Lifecycle>()
            .ok_or(RuntimeError::MissingRun)?
            .0 = RunStatus::Quiescent;
    }
    Ok(())
}

pub(crate) fn memory_append_ingress_system(world: &mut World) {
    let queued = world
        .resource_mut::<MemoryAppendIngressQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for ingress in queued {
        let run_id = ingress.run_id;
        let result = commit_memory_append_ingress(world, ingress);
        world
            .resource_mut::<MemoryAppendIngressOutbox>()
            .0
            .insert(run_id, result);
    }
}

fn commit_memory_append_ingress(
    world: &mut World,
    ingress: MemoryAppendIngress,
) -> Result<(), RuntimeError> {
    let entity = world
        .query::<(Entity, &RunIdentity)>()
        .iter(world)
        .find_map(|(entity, identity)| (identity.run_id == ingress.run_id).then_some(entity))
        .ok_or(RuntimeError::MissingRun)?;
    let identity = world
        .get::<RunIdentity>(entity)
        .ok_or(RuntimeError::MissingRun)?;
    if identity.world_id != ingress.world_id {
        return Err(RuntimeError::ForeignWorld);
    }
    if identity.tenant_id != ingress.tenant_id {
        return Err(RuntimeError::WrongTenant);
    }
    if world
        .get::<Generation>(entity)
        .ok_or(RuntimeError::MissingRun)?
        .0
        != ingress.generation
    {
        return Err(RuntimeError::StaleMemoryCompletion);
    }
    let mut entity = world.entity_mut(entity);
    let mut memory = entity
        .get_mut::<MemoryState>()
        .ok_or(RuntimeError::StaleMemoryCompletion)?;
    if memory.store_id != ingress.store_id
        || memory.append_correlation
            != Some((
                ingress.operation_id,
                ingress.correlation_id,
                ingress.generation,
            ))
    {
        return Err(RuntimeError::StaleMemoryCompletion);
    }
    memory.append_correlation = None;
    match ingress.result {
        Ok(()) => {
            memory.appended_generation = Some(ingress.generation);
            Ok(())
        }
        Err(error) => Err(RuntimeError::MemoryFailure(error.to_string())),
    }
}

pub(crate) fn memory_load_ingress_system(world: &mut World) {
    let queued = world
        .resource_mut::<MemoryLoadIngressQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for ingress in queued {
        let run_id = ingress.run_id;
        let result = commit_memory_load_ingress(world, ingress);
        world
            .resource_mut::<MemoryLoadIngressOutbox>()
            .0
            .insert(run_id, result);
    }
}

fn commit_memory_load_ingress(
    world: &mut World,
    ingress: MemoryLoadIngress,
) -> Result<(), RuntimeError> {
    let entity_id = world
        .query::<(Entity, &RunIdentity)>()
        .iter(world)
        .find_map(|(entity, identity)| (identity.run_id == ingress.run_id).then_some(entity))
        .ok_or(RuntimeError::MissingRun)?;
    let identity = world
        .get::<RunIdentity>(entity_id)
        .ok_or(RuntimeError::MissingRun)?;
    if identity.world_id != ingress.world_id {
        return Err(RuntimeError::ForeignWorld);
    }
    if identity.tenant_id != ingress.tenant_id {
        return Err(RuntimeError::WrongTenant);
    }
    if world
        .get::<Generation>(entity_id)
        .ok_or(RuntimeError::MissingRun)?
        .0
        != ingress.generation
    {
        return Err(RuntimeError::StaleMemoryCompletion);
    }
    let mut entity = world.entity_mut(entity_id);
    let memory = entity
        .get::<MemoryState>()
        .ok_or(RuntimeError::StaleMemoryCompletion)?;
    if memory.store_id != ingress.store_id
        || memory.load_correlation
            != Some((
                ingress.operation_id,
                ingress.correlation_id,
                ingress.generation,
            ))
    {
        return Err(RuntimeError::StaleMemoryCompletion);
    }
    match ingress.result {
        Ok(mut loaded) => {
            let loaded_len = loaded.len();
            let transcript = &mut entity
                .get_mut::<Transcript>()
                .ok_or(RuntimeError::MissingRun)?;
            loaded.append(&mut transcript.history);
            transcript.history = loaded;
            let mut memory = entity
                .get_mut::<MemoryState>()
                .ok_or(RuntimeError::MissingRun)?;
            memory.loaded = true;
            memory.load_correlation = None;
            memory.persist_from = loaded_len;
            Ok(())
        }
        Err(error) => {
            let message = error.to_string();
            let reason = TerminalReason::Failed {
                message: message.clone(),
            };
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::Terminal(reason.clone());
            entity
                .get_mut::<EventLog>()
                .ok_or(RuntimeError::MissingRun)?
                .0
                .push(RuntimeEvent::Terminal {
                    run_id: ingress.run_id,
                    reason,
                });
            Err(RuntimeError::MemoryFailure(message))
        }
    }
}

pub(crate) fn vector_ingress_system(world: &mut World) {
    let queued = world
        .resource_mut::<VectorIngressQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for ingress in queued {
        let run_id = ingress.run_id;
        let result = commit_vector_ingress(world, ingress);
        world
            .resource_mut::<VectorIngressOutbox>()
            .0
            .insert(run_id, result);
    }
}

pub(crate) fn streaming_ingress_system(world: &mut World) {
    let queued = world
        .resource_mut::<StreamingIngressQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for ingress in queued {
        let run_id = match &ingress {
            StreamingIngress::Delta { run_id, .. }
            | StreamingIngress::ProviderFailure { run_id, .. } => *run_id,
            StreamingIngress::ProviderFinal { completion, .. } => completion.run_id,
        };
        let result = commit_streaming_ingress(world, ingress);
        world
            .resource_mut::<StreamingIngressOutbox>()
            .0
            .insert(run_id, result);
    }
}

fn stream_effect_entity(
    world: &mut World,
    world_id: WorldId,
    tenant_id: TenantId,
    run_id: RunId,
    operation_id: OperationId,
    correlation_id: CorrelationId,
    generation: u64,
) -> Result<Entity, RuntimeError> {
    let entity = world
        .query::<(Entity, &RunIdentity)>()
        .iter(world)
        .find_map(|(entity, identity)| (identity.run_id == run_id).then_some(entity))
        .ok_or(RuntimeError::MissingRun)?;
    let identity = world
        .get::<RunIdentity>(entity)
        .ok_or(RuntimeError::MissingRun)?;
    if identity.world_id != world_id {
        return Err(RuntimeError::ForeignWorld);
    }
    if identity.tenant_id != tenant_id {
        return Err(RuntimeError::WrongTenant);
    }
    if matches!(
        world
            .get::<Lifecycle>(entity)
            .ok_or(RuntimeError::MissingRun)?
            .0,
        RunStatus::Terminal(_)
    ) {
        return Err(RuntimeError::Terminal);
    }
    let pending = world
        .get::<PendingOperation>(entity)
        .ok_or(RuntimeError::NoPendingOperation)?;
    if pending.operation_id != operation_id
        || pending.correlation_id != correlation_id
        || pending.generation != generation
    {
        return Err(RuntimeError::StaleCompletion);
    }
    Ok(entity)
}

fn commit_streaming_ingress(
    world: &mut World,
    ingress: StreamingIngress,
) -> Result<(), RuntimeError> {
    match ingress {
        StreamingIngress::Delta {
            world_id,
            tenant_id,
            run_id,
            operation_id,
            correlation_id,
            generation,
            sequence,
            content,
        } => {
            let entity = stream_effect_entity(
                world,
                world_id,
                tenant_id,
                run_id,
                operation_id,
                correlation_id,
                generation,
            )?;
            let mut entity = world.entity_mut(entity);
            let cursor = entity
                .get::<StreamCursor>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            if sequence != cursor {
                return Err(RuntimeError::StaleCompletion);
            }
            entity
                .get_mut::<StreamCursor>()
                .ok_or(RuntimeError::MissingRun)?
                .0 += 1;
            entity
                .get_mut::<EventLog>()
                .ok_or(RuntimeError::MissingRun)?
                .0
                .push(RuntimeEvent::ProvisionalDelta {
                    run_id,
                    operation_id,
                    correlation_id,
                    sequence,
                    content,
                });
            Ok(())
        }
        StreamingIngress::ProviderFailure {
            world_id,
            tenant_id,
            run_id,
            operation_id,
            correlation_id,
            generation,
            message,
        } => {
            let entity = stream_effect_entity(
                world,
                world_id,
                tenant_id,
                run_id,
                operation_id,
                correlation_id,
                generation,
            )?;
            let mut entity = world.entity_mut(entity);
            entity.remove::<PendingOperation>();
            entity
                .get_mut::<Generation>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = generation.saturating_add(1);
            let reason = TerminalReason::Failed {
                message: message.clone(),
            };
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::Terminal(reason.clone());
            let events = &mut entity
                .get_mut::<EventLog>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            events.push(RuntimeEvent::ProviderFailure { run_id, message });
            events.push(RuntimeEvent::Terminal { run_id, reason });
            Ok(())
        }
        StreamingIngress::ProviderFinal {
            completion,
            final_response,
        } => {
            let run_id = completion.run_id;
            world.resource_mut::<IngressQueue>().0.push_back(completion);
            completion_ingress_system(world);
            world
                .resource_mut::<IngressOutbox>()
                .0
                .remove(&run_id)
                .ok_or(RuntimeError::MissingRun)??;
            let entity = world
                .query::<(Entity, &RunIdentity)>()
                .iter(world)
                .find_map(|(entity, identity)| (identity.run_id == run_id).then_some(entity))
                .ok_or(RuntimeError::MissingRun)?;
            let events = &mut world
                .get_mut::<EventLog>(entity)
                .ok_or(RuntimeError::MissingRun)?
                .0;
            let final_index = events.len().saturating_sub(2);
            events.insert(
                final_index,
                RuntimeEvent::ProviderFinal {
                    run_id,
                    final_response,
                },
            );
            Ok(())
        }
    }
}

fn commit_vector_ingress(
    world: &mut World,
    ingress: VectorSearchIngress,
) -> Result<VectorDocuments, RuntimeError> {
    let entity_id = world
        .query::<(Entity, &RunIdentity)>()
        .iter(world)
        .find_map(|(entity, identity)| (identity.run_id == ingress.run_id).then_some(entity))
        .ok_or(RuntimeError::MissingRun)?;
    let identity = world
        .get::<RunIdentity>(entity_id)
        .ok_or(RuntimeError::MissingRun)?;
    if identity.world_id != ingress.world_id {
        return Err(RuntimeError::ForeignWorld);
    }
    if identity.tenant_id != ingress.tenant_id {
        return Err(RuntimeError::WrongTenant);
    }
    if matches!(
        world
            .get::<Lifecycle>(entity_id)
            .ok_or(RuntimeError::MissingRun)?
            .0,
        RunStatus::Terminal(_)
    ) || world
        .get::<Generation>(entity_id)
        .ok_or(RuntimeError::MissingRun)?
        .0
        != ingress.generation
    {
        return Err(RuntimeError::StaleVectorCompletion);
    }
    let pending = world
        .get::<PendingVector>(entity_id)
        .ok_or(RuntimeError::StaleVectorCompletion)?;
    if pending.operation_id != ingress.operation_id
        || pending.correlation_id != ingress.correlation_id
        || pending.generation != ingress.generation
        || pending.store_id != ingress.store_id
    {
        return Err(RuntimeError::StaleVectorCompletion);
    }
    world.entity_mut(entity_id).remove::<PendingVector>();
    ingress
        .result
        .map_err(|error| RuntimeError::VectorFailure(error.to_string()))
}

pub(crate) fn cleanup_system(world: &mut World) {
    let removable = world
        .query::<(Entity, &Lifecycle, &Retention)>()
        .iter(world)
        .filter_map(|(entity, lifecycle, retention)| {
            (matches!(lifecycle.0, RunStatus::Terminal(_))
                && retention.cleanup_requested
                && retention.observations_remaining == 0)
                .then_some(entity)
        })
        .collect::<Vec<_>>();
    for entity in removable {
        let _ = world.despawn(entity);
    }
}

pub(crate) fn policy_progress_system(world: &mut World) {
    let entities = world
        .query::<(Entity, &Lifecycle, &CallBudget)>()
        .iter(world)
        .filter_map(|(entity, lifecycle, budget)| {
            (matches!(lifecycle.0, RunStatus::Ready) && budget.completed_calls >= budget.max_calls)
                .then_some(entity)
        })
        .collect::<Vec<_>>();
    for entity in entities {
        if let Some(mut lifecycle) = world.entity_mut(entity).get_mut::<Lifecycle>() {
            lifecycle.0 = RunStatus::Terminal(TerminalReason::BudgetExhausted);
        }
    }
}

pub(crate) fn quiescence_progress_system(world: &mut World) {
    let entities = world
        .query::<(Entity, &Lifecycle, &ToolBatch)>()
        .iter(world)
        .filter_map(|(entity, lifecycle, batch)| {
            (matches!(lifecycle.0, RunStatus::AwaitingTools)
                && batch.pending.is_empty()
                && batch.completed.is_empty())
            .then_some(entity)
        })
        .collect::<Vec<_>>();
    for entity in entities {
        if let Some(mut lifecycle) = world.entity_mut(entity).get_mut::<Lifecycle>() {
            lifecycle.0 = RunStatus::Quiescent;
        }
    }
}

pub(crate) fn terminal_progress_system(world: &mut World) {
    const MAX_STALLED_PASSES: u16 = 32;
    let entities = world
        .query::<(Entity, &Lifecycle, &PendingRequest, &mut StalledPasses)>()
        .iter_mut(world)
        .filter_map(|(entity, lifecycle, request, mut stalled)| {
            let cannot_progress = matches!(lifecycle.0, RunStatus::Quiescent)
                || (matches!(lifecycle.0, RunStatus::Ready) && request.0.is_none());
            if cannot_progress {
                stalled.0 = stalled.0.saturating_add(1);
                (stalled.0 >= MAX_STALLED_PASSES).then_some(entity)
            } else {
                stalled.0 = 0;
                None
            }
        })
        .collect::<Vec<_>>();
    for entity in entities {
        let mut entity = world.entity_mut(entity);
        let Some(run_id) = entity.get::<RunIdentity>().map(|identity| identity.run_id) else {
            continue;
        };
        if let Some(mut lifecycle) = entity.get_mut::<Lifecycle>() {
            lifecycle.0 = RunStatus::Terminal(TerminalReason::Livelock);
        }
        if let Some(mut events) = entity.get_mut::<EventLog>() {
            events.0.push(RuntimeEvent::Terminal {
                run_id,
                reason: TerminalReason::Livelock,
            });
        }
    }
}

pub(crate) fn policy_command_system(world: &mut World) {
    let queued = world
        .resource_mut::<PolicyQueue>()
        .0
        .drain(..)
        .collect::<Vec<_>>();
    for (run_id, command) in queued {
        let result = match command {
            PolicyCommand::InvalidTool {
                entity,
                call_index,
                tool_name,
                policy,
            } => commit_invalid_tool_policy(world, entity, call_index, tool_name, policy)
                .map(PolicyCommandResult::InvalidTool),
            PolicyCommand::Cancel { entity, reason } => {
                commit_cancellation(world, entity, run_id, reason)
                    .map(|()| PolicyCommandResult::Unit)
            }
            PolicyCommand::RetryResponse {
                completion,
                feedback,
            } => commit_response_retry(world, completion, feedback)
                .map(|()| PolicyCommandResult::Unit),
            PolicyCommand::ExhaustResponse {
                completion,
                message,
            } => commit_response_exhaustion(world, completion, message)
                .map(|()| PolicyCommandResult::Unit),
            PolicyCommand::ContinueAfterTools {
                entity,
                request,
                results,
            } => commit_tool_continuation(world, entity, *request, results)
                .map(|()| PolicyCommandResult::Unit),
        };
        world
            .resource_mut::<PolicyOutbox>()
            .0
            .insert(run_id, result);
    }
}

fn validate_effect_world(
    world: &mut World,
    completion: &CompletionIngress,
) -> Result<Entity, RuntimeError> {
    let entity = world
        .query::<(Entity, &RunIdentity)>()
        .iter(world)
        .find_map(|(entity, identity)| (identity.run_id == completion.run_id).then_some(entity))
        .ok_or(RuntimeError::MissingRun)?;
    let identity = world
        .get::<RunIdentity>(entity)
        .ok_or(RuntimeError::MissingRun)?;
    if identity.world_id != completion.world_id {
        return Err(RuntimeError::ForeignWorld);
    }
    if identity.tenant_id != completion.tenant_id {
        return Err(RuntimeError::WrongTenant);
    }
    if matches!(
        world
            .get::<Lifecycle>(entity)
            .ok_or(RuntimeError::MissingRun)?
            .0,
        RunStatus::Terminal(_)
    ) {
        return Err(RuntimeError::Terminal);
    }
    let pending = world
        .get::<PendingOperation>(entity)
        .ok_or(RuntimeError::NoPendingOperation)?;
    if pending.operation_id != completion.operation_id
        || pending.correlation_id != completion.correlation_id
        || pending.generation != completion.generation
    {
        return Err(RuntimeError::StaleCompletion);
    }
    Ok(entity)
}

fn commit_response_retry(
    world: &mut World,
    completion: CompletionIngress,
    feedback: String,
) -> Result<(), RuntimeError> {
    let entity_id = validate_effect_world(world, &completion)?;
    let mut entity = world.entity_mut(entity_id);
    let pending = entity
        .take::<PendingOperation>()
        .ok_or(RuntimeError::NoPendingOperation)?;
    let mut request = pending.request;
    request
        .chat_history
        .push(rig_core::completion::Message::user(feedback));
    entity
        .get_mut::<CallBudget>()
        .ok_or(RuntimeError::MissingRun)?
        .completed_calls += 1;
    entity
        .get_mut::<Accounting>()
        .ok_or(RuntimeError::MissingRun)?
        .usage += completion.usage;
    entity
        .get_mut::<Accounting>()
        .ok_or(RuntimeError::MissingRun)?
        .rejected_effects += 1;
    entity
        .get_mut::<PendingRequest>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = Some(request);
    entity
        .get_mut::<Generation>()
        .ok_or(RuntimeError::MissingRun)?
        .0 += 1;
    entity
        .get_mut::<Lifecycle>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = RunStatus::Ready;
    entity
        .get_mut::<EventLog>()
        .ok_or(RuntimeError::MissingRun)?
        .0
        .push(RuntimeEvent::Rollback {
            run_id: completion.run_id,
            reason: "response rejected for retry".to_string(),
        });
    Ok(())
}

fn commit_response_exhaustion(
    world: &mut World,
    completion: CompletionIngress,
    message: String,
) -> Result<(), RuntimeError> {
    let entity_id = validate_effect_world(world, &completion)?;
    let mut entity = world.entity_mut(entity_id);
    entity
        .take::<PendingOperation>()
        .ok_or(RuntimeError::NoPendingOperation)?;
    entity
        .get_mut::<CallBudget>()
        .ok_or(RuntimeError::MissingRun)?
        .completed_calls += 1;
    entity
        .get_mut::<Accounting>()
        .ok_or(RuntimeError::MissingRun)?
        .usage += completion.usage;
    entity
        .get_mut::<Accounting>()
        .ok_or(RuntimeError::MissingRun)?
        .rejected_effects += 1;
    let reason = TerminalReason::Failed { message };
    entity
        .get_mut::<Lifecycle>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = RunStatus::Terminal(reason.clone());
    let events = &mut entity
        .get_mut::<EventLog>()
        .ok_or(RuntimeError::MissingRun)?
        .0;
    events.push(RuntimeEvent::Rollback {
        run_id: completion.run_id,
        reason: "structured response recovery exhausted".to_string(),
    });
    events.push(RuntimeEvent::Terminal {
        run_id: completion.run_id,
        reason,
    });
    Ok(())
}

fn commit_tool_continuation(
    world: &mut World,
    entity_id: Entity,
    mut request: CompletionRequest,
    results: Vec<rig_core::completion::Message>,
) -> Result<(), RuntimeError> {
    let mut entity = world.entity_mut(entity_id);
    let transcript = &mut entity
        .get_mut::<Transcript>()
        .ok_or(RuntimeError::MissingRun)?;
    transcript.history.extend(results);
    request.chat_history = rig_core::OneOrMany::from_iter_optional(transcript.history.clone())
        .ok_or(RuntimeError::NoPendingOperation)?;
    entity
        .get_mut::<PendingRequest>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = Some(request);
    entity
        .get_mut::<Generation>()
        .ok_or(RuntimeError::MissingRun)?
        .0 += 1;
    entity
        .get_mut::<Lifecycle>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = RunStatus::Ready;
    *entity
        .get_mut::<ToolBatch>()
        .ok_or(RuntimeError::MissingRun)? = ToolBatch::default();
    entity
        .get_mut::<ToolContinuation>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = None;
    Ok(())
}

fn commit_cancellation(
    world: &mut World,
    entity_id: Entity,
    run_id: RunId,
    reason: String,
) -> Result<(), RuntimeError> {
    let mut entity = world.entity_mut(entity_id);
    let lifecycle = entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?;
    if matches!(lifecycle.0, RunStatus::Terminal(_)) {
        return Err(RuntimeError::Terminal);
    }
    let generation = entity
        .get::<Generation>()
        .ok_or(RuntimeError::MissingRun)?
        .0;
    entity
        .get_mut::<Generation>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = generation.saturating_add(1);
    let reason = TerminalReason::Cancelled { reason };
    entity
        .get_mut::<Lifecycle>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = RunStatus::Terminal(reason.clone());
    entity
        .get_mut::<EventLog>()
        .ok_or(RuntimeError::MissingRun)?
        .0
        .push(RuntimeEvent::Terminal { run_id, reason });
    entity.remove::<PendingOperation>();
    entity.remove::<PendingVector>();
    if let Some(mut memory) = entity.get_mut::<MemoryState>() {
        memory.load_correlation = None;
        memory.append_correlation = None;
    }
    Ok(())
}

fn commit_invalid_tool_policy(
    world: &mut World,
    entity_id: Entity,
    call_index: usize,
    tool_name: String,
    policy: InvalidToolPolicy,
) -> Result<InvalidToolResolution, RuntimeError> {
    let identity = *world
        .get::<RunIdentity>(entity_id)
        .ok_or(RuntimeError::MissingRun)?;
    let mut entity = world.entity_mut(entity_id);
    let (reason, resolution) = match policy {
        InvalidToolPolicy::Fail => (
            format!("unknown or invalid tool `{tool_name}`"),
            InvalidToolResolution::Failed,
        ),
        InvalidToolPolicy::Retry => (
            format!("retry invalid tool `{tool_name}`"),
            InvalidToolResolution::Retry {
                feedback: format!(
                    "Tool `{tool_name}` is unavailable. Choose one of the advertised tools."
                ),
            },
        ),
        InvalidToolPolicy::Repair {
            replacement_name,
            replacement_arguments,
        } => (
            format!("repair `{tool_name}` as `{replacement_name}`"),
            InvalidToolResolution::Repair {
                name: replacement_name,
                arguments: replacement_arguments,
            },
        ),
        InvalidToolPolicy::Skip { reason } => {
            let result = rig_core::tool::ToolResult::skipped(reason.clone());
            (reason, InvalidToolResolution::Skip { result })
        }
        InvalidToolPolicy::Stop { reason } => (reason, InvalidToolResolution::Stopped),
    };
    entity
        .get_mut::<Accounting>()
        .ok_or(RuntimeError::MissingRun)?
        .rejected_effects += 1;
    entity
        .get_mut::<EventLog>()
        .ok_or(RuntimeError::MissingRun)?
        .0
        .push(RuntimeEvent::RejectedEffect {
            run_id: identity.run_id,
            reason: reason.clone(),
        });
    match &resolution {
        InvalidToolResolution::Skip { result } => {
            let (committed, quiescent) = {
                let mut batch = entity
                    .get_mut::<ToolBatch>()
                    .ok_or(RuntimeError::MissingRun)?;
                batch
                    .completed
                    .insert(call_index, (tool_name.clone(), result.clone()));
                let mut committed = Vec::new();
                loop {
                    let index = batch.next_commit;
                    let Some((name, result)) = batch.completed.remove(&index) else {
                        break;
                    };
                    batch.next_commit += 1;
                    committed.push((index, name, result));
                }
                let quiescent = batch.pending.is_empty() && batch.completed.is_empty();
                (committed, quiescent)
            };
            for (call_index, name, result) in committed {
                entity
                    .get_mut::<EventLog>()
                    .ok_or(RuntimeError::MissingRun)?
                    .0
                    .push(RuntimeEvent::ToolCommitted {
                        run_id: identity.run_id,
                        call_index,
                        name,
                        result,
                    });
            }
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = if quiescent {
                RunStatus::Quiescent
            } else {
                RunStatus::AwaitingTools
            };
        }
        InvalidToolResolution::Retry { feedback } => {
            let limit = entity
                .get::<RecoveryPolicy>()
                .ok_or(RuntimeError::MissingRun)?
                .max_invalid_tool_retries;
            let retries = entity
                .get::<InvalidRetryCount>()
                .ok_or(RuntimeError::MissingRun)?
                .0;
            if retries >= limit {
                return commit_invalid_tool_policy(
                    world,
                    entity_id,
                    call_index,
                    tool_name,
                    InvalidToolPolicy::Fail,
                );
            }
            entity
                .get_mut::<InvalidRetryCount>()
                .ok_or(RuntimeError::MissingRun)?
                .0 += 1;
            let mut request = entity
                .get_mut::<ToolContinuation>()
                .ok_or(RuntimeError::MissingRun)?
                .0
                .take()
                .ok_or(RuntimeError::NoPendingOperation)?;
            request
                .chat_history
                .push(rig_core::completion::Message::user(feedback.clone()));
            entity
                .get_mut::<PendingRequest>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = Some(request);
            entity
                .get_mut::<Generation>()
                .ok_or(RuntimeError::MissingRun)?
                .0 += 1;
            *entity
                .get_mut::<ToolBatch>()
                .ok_or(RuntimeError::MissingRun)? = ToolBatch::default();
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::Ready;
        }
        InvalidToolResolution::Repair { .. } => {
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::AwaitingTools;
        }
        InvalidToolResolution::Failed | InvalidToolResolution::Stopped => {
            entity
                .get_mut::<Generation>()
                .ok_or(RuntimeError::MissingRun)?
                .0 += 1;
            *entity
                .get_mut::<ToolBatch>()
                .ok_or(RuntimeError::MissingRun)? = ToolBatch::default();
            let terminal = match &resolution {
                InvalidToolResolution::Stopped => TerminalReason::Cancelled {
                    reason: reason.clone(),
                },
                _ => TerminalReason::Failed {
                    message: format!("invalid tool `{tool_name}`"),
                },
            };
            entity
                .get_mut::<Lifecycle>()
                .ok_or(RuntimeError::MissingRun)?
                .0 = RunStatus::Terminal(terminal.clone());
            entity
                .get_mut::<EventLog>()
                .ok_or(RuntimeError::MissingRun)?
                .0
                .push(RuntimeEvent::Terminal {
                    run_id: identity.run_id,
                    reason: terminal,
                });
        }
    }
    entity
        .get_mut::<InvalidResolution>()
        .ok_or(RuntimeError::MissingRun)?
        .0 = Some(resolution.clone());
    Ok(resolution)
}

/// Owned streaming ingress. Only `ProviderFinal` commits canonical output.
#[derive(Clone, Debug)]
pub enum StreamingIngress {
    Delta {
        world_id: WorldId,
        tenant_id: TenantId,
        run_id: RunId,
        operation_id: OperationId,
        correlation_id: CorrelationId,
        generation: u64,
        sequence: u64,
        content: AssistantContent,
    },
    ProviderFinal {
        completion: CompletionIngress,
        final_response: ProviderFinalEnvelope,
    },
    ProviderFailure {
        world_id: WorldId,
        tenant_id: TenantId,
        run_id: RunId,
        operation_id: OperationId,
        correlation_id: CorrelationId,
        generation: u64,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeHandle {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    entity: Entity,
}

/// Type-erased handle for an executor that repeatedly drives the same ECS
/// schedules as local blocking and streaming adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostedHandle(RuntimeHandle);

/// Type-erased owned effect emitted by the hosted driver.
#[derive(Clone)]
pub enum HostedEffect {
    Model(Box<EffectRequest>),
    Tool(Box<ToolEffectRequest>),
    MemoryLoad(Box<MemoryLoadEffect>),
    MemoryAppend(Box<MemoryAppendEffect>),
    Vector(Box<VectorSearchEffect>),
}

impl HostedEffect {
    fn generation(&self) -> u64 {
        match self {
            Self::Model(effect) => effect.generation,
            Self::Tool(effect) => effect.generation,
            Self::MemoryLoad(effect) => effect.generation,
            Self::MemoryAppend(effect) => effect.generation,
            Self::Vector(effect) => effect.generation,
        }
    }
}

impl std::fmt::Debug for HostedEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(effect) => f.debug_tuple("Model").field(effect).finish(),
            Self::Tool(effect) => f.debug_tuple("Tool").field(effect).finish(),
            Self::MemoryLoad(effect) => f.debug_tuple("MemoryLoad").field(effect).finish(),
            Self::MemoryAppend(effect) => f.debug_tuple("MemoryAppend").field(effect).finish(),
            Self::Vector(effect) => f.debug_tuple("Vector").field(effect).finish(),
        }
    }
}

/// One nonblocking hosted-driver observation.
#[derive(Clone, Debug)]
pub enum HostedPoll {
    Effect(HostedEffect),
    Pending,
    Terminal(TerminalReason),
}

/// Redacted explanation derived from authoritative ECS facts. Prompts, tool
/// arguments, memory keys, provider payloads, and credentials are omitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunExplanation {
    pub run_id: RunId,
    pub tenant_id: TenantId,
    pub generation: u64,
    pub status: RunStatus,
    pub completed_calls: usize,
    pub max_calls: usize,
    pub pending_model: bool,
    pub pending_tools: usize,
    pub event_count: usize,
    pub rejected_effects: usize,
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("run handle belongs to another runtime world")]
    ForeignWorld,
    #[error("run handle belongs to another tenant")]
    WrongTenant,
    #[error("run no longer exists")]
    MissingRun,
    #[error("run is already terminal")]
    Terminal,
    #[error("completion does not match the pending operation")]
    StaleCompletion,
    #[error("run has no pending model operation")]
    NoPendingOperation,
    #[error("capability belongs to another tenant")]
    ForeignCapability,
    #[error("agent has no grant for capability")]
    MissingGrant,
    #[error("capability is retired or missing")]
    MissingCapability,
    #[error("tool snapshot is stale or belongs to another run")]
    StaleToolSnapshot,
    #[error("tool is not present in the immutable turn snapshot")]
    ToolNotAdvertised,
    #[error("tool completion does not match an in-flight call")]
    StaleToolCompletion,
    #[error("tool call index is already in flight")]
    DuplicateToolCall,
    #[error("tool effect queue reached its configured concurrency limit {0}")]
    Backpressure(usize),
    #[error("memory must finish loading before model dispatch")]
    MemoryNotLoaded,
    #[error("memory completion is stale or mismatched")]
    StaleMemoryCompletion,
    #[error("memory backend failed: {0}")]
    MemoryFailure(String),
    #[error("vector completion is stale or mismatched")]
    StaleVectorCompletion,
    #[error("vector backend failed: {0}")]
    VectorFailure(String),
    #[error("owned effect input could not be encoded: {0}")]
    EffectEncoding(String),
    #[error("response retry is only valid for a tool-free model response")]
    RetryWithTools,
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
}

pub struct Runtime {
    world_id: WorldId,
    world: World,
    runs: HashMap<RunId, Entity>,
    capabilities: HashMap<CapabilityId, BoundTool>,
    implementations: HashMap<(CapabilityId, u64), Arc<dyn ToolImplementation>>,
    grants: HashMap<(AgentId, CapabilityId), Entity>,
    memories: HashMap<StoreId, Arc<dyn MemoryImplementation>>,
    memory_appends: VecDeque<MemoryAppendEffect>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    pub fn new() -> Self {
        let mut world = World::new();
        schedule::install(&mut world);
        Self {
            world_id: WorldId::fresh(),
            world,
            runs: HashMap::new(),
            capabilities: HashMap::new(),
            implementations: HashMap::new(),
            grants: HashMap::new(),
            memories: HashMap::new(),
            memory_appends: VecDeque::new(),
        }
    }

    /// Stable handles for currently retained runs, ordered by run ID.
    pub fn handles(&self) -> Vec<RuntimeHandle> {
        let mut handles = self
            .runs
            .iter()
            .filter_map(|(&run_id, &entity)| {
                let identity = self.world.get::<RunIdentity>(entity)?;
                Some(RuntimeHandle {
                    world_id: self.world_id,
                    tenant_id: identity.tenant_id,
                    run_id,
                    entity,
                })
            })
            .collect::<Vec<_>>();
        handles.sort_by_key(|handle| handle.run_id);
        handles
    }

    /// Drive one explicit runtime schedule. Blocking and streaming surfaces use
    /// this same entry point for schedule-owned policy/progression systems.
    pub fn drive(&mut self, schedule: crate::RuntimeSchedule) {
        self.world.run_schedule(schedule);
    }

    /// Convert a local handle into the type-erased hosted surface.
    pub fn hosted(&self, handle: RuntimeHandle) -> Result<HostedHandle, RuntimeError> {
        self.validate(handle)?;
        Ok(HostedHandle(handle))
    }

    /// Advance policy/quiescence/terminal systems once and emit an owned model
    /// effect when the run is ready. Executors submit results through the same
    /// correlated ingress APIs used by local adapters.
    pub fn poll_hosted(&mut self, handle: HostedHandle) -> Result<HostedPoll, RuntimeError> {
        self.validate(handle.0)?;
        self.world.run_schedule(crate::RuntimeSchedule::Progress);
        let status = self
            .world
            .get::<Lifecycle>(handle.0.entity)
            .ok_or(RuntimeError::MissingRun)?
            .0
            .clone();
        if let RunStatus::Terminal(reason) = status {
            self.purge_queued_effects(handle.0.run_id);
            return Ok(HostedPoll::Terminal(reason));
        }
        let generation = self
            .world
            .get::<Generation>(handle.0.entity)
            .ok_or(RuntimeError::MissingRun)?
            .0;
        self.world
            .resource_mut::<HostedEffectQueue>()
            .0
            .retain(|(run_id, effect)| {
                *run_id != handle.0.run_id || effect.generation() == generation
            });
        if let Some(position) = self
            .world
            .resource::<HostedEffectQueue>()
            .0
            .iter()
            .position(|(run_id, _)| *run_id == handle.0.run_id)
            && let Some((_, effect)) = self
                .world
                .resource_mut::<HostedEffectQueue>()
                .0
                .remove(position)
        {
            return Ok(HostedPoll::Effect(effect));
        }
        if let Some(position) = self
            .memory_appends
            .iter()
            .position(|effect| effect.run_id == handle.0.run_id)
            && let Some(effect) = self.memory_appends.remove(position)
        {
            return Ok(HostedPoll::Effect(HostedEffect::MemoryAppend(Box::new(
                effect,
            ))));
        }
        let status = self
            .world
            .get::<Lifecycle>(handle.0.entity)
            .ok_or(RuntimeError::MissingRun)?
            .0
            .clone();
        match status {
            RunStatus::Ready => self
                .dispatch(handle.0)
                .map(Box::new)
                .map(HostedEffect::Model)
                .map(HostedPoll::Effect),
            RunStatus::Terminal(reason) => Ok(HostedPoll::Terminal(reason)),
            RunStatus::AwaitingModel | RunStatus::AwaitingTools | RunStatus::Quiescent => {
                Ok(HostedPoll::Pending)
            }
        }
    }

    pub fn cancel_hosted(
        &mut self,
        handle: HostedHandle,
        reason: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        self.cancel(handle.0, reason)
    }

    /// Bind hosted memory and enqueue its owned load effect.
    pub fn hosted_bind_memory(
        &mut self,
        handle: HostedHandle,
        conversation_id: impl Into<String>,
        implementation: Arc<dyn MemoryImplementation>,
    ) -> Result<(), RuntimeError> {
        let effect = self.bind_memory(handle.0, conversation_id, implementation)?;
        self.world
            .resource_mut::<HostedEffectQueue>()
            .0
            .push_back((handle.0.run_id, HostedEffect::MemoryLoad(Box::new(effect))));
        Ok(())
    }

    /// Enqueue a hosted vector search effect.
    pub fn hosted_dispatch_vector(
        &mut self,
        handle: HostedHandle,
        implementation: Arc<dyn VectorStoreIndexDyn>,
        request: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> Result<(), RuntimeError> {
        let effect = self.dispatch_vector(handle.0, implementation, request)?;
        self.world
            .resource_mut::<HostedEffectQueue>()
            .0
            .push_back((handle.0.run_id, HostedEffect::Vector(Box::new(effect))));
        Ok(())
    }

    /// Commit a hosted model response and enqueue exact snapshotted tool
    /// effects when it contains calls.
    pub fn submit_hosted_completion(
        &mut self,
        handle: HostedHandle,
        completion: CompletionIngress,
    ) -> Result<(), RuntimeError> {
        self.submit_hosted_completion_with_policy(handle, completion, InvalidToolPolicy::Fail)
    }

    /// Commit a hosted response while applying the same invalid-tool policy
    /// semantics as the local executor.
    pub fn submit_hosted_completion_with_policy(
        &mut self,
        handle: HostedHandle,
        completion: CompletionIngress,
        invalid_policy: InvalidToolPolicy,
    ) -> Result<(), RuntimeError> {
        self.validate(handle.0)?;
        let calls = completion
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if calls.is_empty() {
            return self.submit(completion);
        }
        self.commit_tool_turn(completion)?;
        let identity = *self
            .world
            .get::<RunIdentity>(handle.0.entity)
            .ok_or(RuntimeError::MissingRun)?;
        let capabilities = self
            .grants
            .keys()
            .filter_map(|(agent_id, capability_id)| {
                (*agent_id == identity.agent_id).then_some(*capability_id)
            })
            .collect::<Vec<_>>();
        let snapshot = self.snapshot_tools(handle.0, capabilities)?;
        let mut staged = Vec::with_capacity(calls.len());
        for (call_index, call) in calls.into_iter().enumerate() {
            let arguments = serde_json::to_string(&call.function.arguments)
                .map_err(|error| RuntimeError::EffectEncoding(error.to_string()))?;
            let effect = match self.dispatch_tool(
                handle.0,
                &snapshot,
                call_index,
                &call.function.name,
                arguments,
            ) {
                Ok(effect) => effect,
                Err(RuntimeError::ToolNotAdvertised) => match self.apply_invalid_tool_policy(
                    handle.0,
                    call_index,
                    &call.function.name,
                    invalid_policy.clone(),
                )? {
                    InvalidToolResolution::Repair { name, arguments } => {
                        self.dispatch_tool(handle.0, &snapshot, call_index, &name, arguments)?
                    }
                    InvalidToolResolution::Skip { .. } => continue,
                    InvalidToolResolution::Retry { .. } => return Ok(()),
                    InvalidToolResolution::Failed | InvalidToolResolution::Stopped => return Ok(()),
                },
                Err(error) => return Err(error),
            };
            staged.push(effect);
        }
        for effect in staged {
            self.world
                .resource_mut::<HostedEffectQueue>()
                .0
                .push_back((handle.0.run_id, HostedEffect::Tool(Box::new(effect))));
        }
        if matches!(
            self.world
                .get::<Lifecycle>(handle.0.entity)
                .ok_or(RuntimeError::MissingRun)?
                .0,
            RunStatus::Quiescent
        ) {
            self.continue_hosted_after_tools(handle)?;
        }
        Ok(())
    }

    /// Commit a hosted tool result and automatically prepare the next model
    /// request once the batch becomes quiescent.
    pub fn submit_hosted_tool(
        &mut self,
        handle: HostedHandle,
        ingress: ToolIngress,
    ) -> Result<(), RuntimeError> {
        self.validate(handle.0)?;
        self.submit_tool(ingress)?;
        if !matches!(
            self.world
                .get::<Lifecycle>(handle.0.entity)
                .ok_or(RuntimeError::MissingRun)?
                .0,
            RunStatus::Quiescent
        ) {
            return Ok(());
        }
        self.continue_hosted_after_tools(handle)
    }

    fn continue_hosted_after_tools(&mut self, handle: HostedHandle) -> Result<(), RuntimeError> {
        let calls = self
            .world
            .get::<Transcript>(handle.0.entity)
            .ok_or(RuntimeError::MissingRun)?
            .history
            .last()
            .into_iter()
            .flat_map(|message| match message {
                rig_core::completion::Message::Assistant { content, .. } => content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::ToolCall(call) => Some(call.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect::<Vec<_>>();
        let events = self.events(handle.0)?;
        let results = calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| {
                let result = events
                    .iter()
                    .rev()
                    .find_map(|event| match event {
                        RuntimeEvent::ToolCommitted {
                            call_index, result, ..
                        } if *call_index == index => Some(result),
                        _ => None,
                    })
                    .ok_or(RuntimeError::StaleToolCompletion)?;
                Ok(rig_core::completion::Message::from(
                    rig_core::message::ToolResult {
                        id: call.id,
                        call_id: call.call_id,
                        content: result.output().as_content().clone(),
                    },
                ))
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        let request = self
            .world
            .get::<ToolContinuation>(handle.0.entity)
            .ok_or(RuntimeError::MissingRun)?
            .0
            .clone()
            .ok_or(RuntimeError::NoPendingOperation)?;
        self.continue_after_tools(handle.0, request, results)
    }

    pub fn spawn_run(
        &mut self,
        tenant_id: TenantId,
        request: CompletionRequest,
        max_calls: usize,
    ) -> RuntimeHandle {
        let run_id = RunId::fresh();
        let history = request.chat_history.clone().into_iter().collect();
        let entity = self
            .world
            .spawn((
                RunIdentity {
                    world_id: self.world_id,
                    tenant_id,
                    agent_id: AgentId::fresh(),
                    run_id,
                },
                Generation(0),
                CallBudget {
                    max_calls,
                    completed_calls: 0,
                },
                Lifecycle(RunStatus::Ready),
                PendingRequest(Some(request)),
                Transcript {
                    history,
                    final_output: Vec::new(),
                },
                Accounting::default(),
                StreamCursor::default(),
                StalledPasses::default(),
                EventLog::default(),
                ToolBatch::default(),
                RecoveryPolicy::default(),
                ConcurrencyLimit(8),
                Retention::default(),
            ))
            .id();
        self.world.entity_mut(entity).insert((
            ToolContinuation::default(),
            InvalidResolution::default(),
            InvalidRetryCount::default(),
        ));
        self.runs.insert(run_id, entity);
        RuntimeHandle {
            world_id: self.world_id,
            tenant_id,
            run_id,
            entity,
        }
    }

    pub fn dispatch(&mut self, handle: RuntimeHandle) -> Result<EffectRequest, RuntimeError> {
        self.validate(handle)?;
        self.world
            .resource_mut::<DispatchQueue>()
            .0
            .push_back(handle.entity);
        self.world.run_schedule(crate::RuntimeSchedule::Dispatch);
        self.world
            .resource_mut::<DispatchOutbox>()
            .0
            .remove(&handle.run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Reject a completed tool-free response, preserve its billed usage, roll
    /// back its content, and prepare a fresh request with corrective feedback.
    /// The next dispatch consumes the same total model-call budget as every
    /// other completion.
    pub fn retry_response(
        &mut self,
        completion: CompletionIngress,
        corrective_feedback: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        if completion
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)))
        {
            return Err(RuntimeError::RetryWithTools);
        }
        self.retry_response_inner(completion, corrective_feedback.into())
    }

    /// Retry a failed synthetic structured-output call without permitting
    /// ordinary executable tools to enter response recovery.
    pub fn retry_tool_output(
        &mut self,
        completion: CompletionIngress,
        output_tool_name: &str,
        corrective_feedback: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        if completion.choice.iter().any(|content| {
            matches!(content, AssistantContent::ToolCall(call) if call.function.name != output_tool_name)
        }) {
            return Err(RuntimeError::RetryWithTools);
        }
        self.retry_response_inner(completion, corrective_feedback.into())
    }

    fn retry_response_inner(
        &mut self,
        completion: CompletionIngress,
        corrective_feedback: String,
    ) -> Result<(), RuntimeError> {
        let run_id = completion.run_id;
        self.world.resource_mut::<PolicyQueue>().0.push_back((
            run_id,
            PolicyCommand::RetryResponse {
                completion,
                feedback: corrective_feedback,
            },
        ));
        self.world.run_schedule(crate::RuntimeSchedule::Progress);
        self.take_policy_unit(run_id)
    }

    /// Reject the final successful-but-invalid provider response, preserving
    /// its billed usage while terminalizing without accepting its content.
    pub fn exhaust_response(
        &mut self,
        completion: CompletionIngress,
        message: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        let run_id = completion.run_id;
        self.world.resource_mut::<PolicyQueue>().0.push_back((
            run_id,
            PolicyCommand::ExhaustResponse {
                completion,
                message: message.into(),
            },
        ));
        self.world.run_schedule(crate::RuntimeSchedule::Progress);
        self.take_policy_unit(run_id)
    }

    /// Commit a billed model turn that contains tool calls without finalizing
    /// the run, returning the owned request template for the continuation.
    pub fn commit_tool_turn(
        &mut self,
        completion: CompletionIngress,
    ) -> Result<CompletionRequest, RuntimeError> {
        let run_id = completion.run_id;
        self.world
            .resource_mut::<ToolTurnIngressQueue>()
            .0
            .push_back(completion);
        self.world.run_schedule(crate::RuntimeSchedule::Ingress);
        self.world
            .resource_mut::<ToolTurnIngressOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Commit canonical tool-result messages and prepare the next model effect.
    pub fn continue_after_tools(
        &mut self,
        handle: RuntimeHandle,
        request: CompletionRequest,
        results: Vec<rig_core::completion::Message>,
    ) -> Result<(), RuntimeError> {
        self.validate(handle)?;
        self.world.resource_mut::<PolicyQueue>().0.push_back((
            handle.run_id,
            PolicyCommand::ContinueAfterTools {
                entity: handle.entity,
                request: Box::new(request),
                results,
            },
        ));
        self.world.run_schedule(crate::RuntimeSchedule::Progress);
        self.take_policy_unit(handle.run_id)
    }

    pub fn submit(&mut self, completion: CompletionIngress) -> Result<(), RuntimeError> {
        if completion.world_id != self.world_id {
            return Err(RuntimeError::ForeignWorld);
        }
        let run_id = completion.run_id;
        self.world
            .resource_mut::<IngressQueue>()
            .0
            .push_back(completion);
        self.world.run_schedule(crate::RuntimeSchedule::Ingress);
        let entity = self
            .world
            .resource_mut::<IngressOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)??;
        self.queue_memory_append(run_id, entity)?;
        Ok(())
    }

    /// Bind tenant-local conversation memory and return an owned load effect.
    pub fn bind_memory(
        &mut self,
        handle: RuntimeHandle,
        conversation_id: impl Into<String>,
        implementation: Arc<dyn MemoryImplementation>,
    ) -> Result<MemoryLoadEffect, RuntimeError> {
        self.validate(handle)?;
        let conversation_id = conversation_id.into();
        self.world
            .resource_mut::<MemoryDispatchQueue>()
            .0
            .push_back(MemoryDispatchCommand {
                entity: handle.entity,
                conversation_id,
                implementation,
            });
        self.world.run_schedule(crate::RuntimeSchedule::Dispatch);
        let effect = self
            .world
            .resource_mut::<MemoryDispatchOutbox>()
            .0
            .remove(&handle.run_id)
            .ok_or(RuntimeError::MissingRun)??;
        self.memories
            .insert(effect.store_id, effect.implementation.clone());
        Ok(effect)
    }

    /// Commit loaded history before the first model request.
    pub fn submit_memory_load(&mut self, ingress: MemoryLoadIngress) -> Result<(), RuntimeError> {
        let run_id = ingress.run_id;
        self.world
            .resource_mut::<MemoryLoadIngressQueue>()
            .0
            .push_back(ingress);
        self.world.run_schedule(crate::RuntimeSchedule::Ingress);
        self.world
            .resource_mut::<MemoryLoadIngressOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Drain committed-only append effects for execution outside ECS borrows.
    pub fn take_memory_appends(&mut self) -> impl Iterator<Item = MemoryAppendEffect> + '_ {
        self.memory_appends.drain(..)
    }

    /// Commit an externally executed append only when its full ownership and
    /// generation correlation still matches the authoritative run.
    pub fn submit_memory_append(
        &mut self,
        ingress: MemoryAppendIngress,
    ) -> Result<(), RuntimeError> {
        let run_id = ingress.run_id;
        self.world
            .resource_mut::<MemoryAppendIngressQueue>()
            .0
            .push_back(ingress);
        self.world.run_schedule(crate::RuntimeSchedule::Ingress);
        self.world
            .resource_mut::<MemoryAppendIngressOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Recreate a failed or lost append effect for the current committed
    /// generation. Already-acknowledged generations remain idempotent.
    pub fn retry_memory_append(&mut self, handle: RuntimeHandle) -> Result<(), RuntimeError> {
        self.validate(handle)?;
        self.queue_memory_append(handle.run_id, handle.entity)
    }

    /// Snapshot an owned vector effect for an executor. The query and backend
    /// pointer are never stored in persisted run state.
    pub fn dispatch_vector(
        &mut self,
        handle: RuntimeHandle,
        implementation: Arc<dyn VectorStoreIndexDyn>,
        request: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> Result<VectorSearchEffect, RuntimeError> {
        self.validate(handle)?;
        self.world
            .resource_mut::<VectorDispatchQueue>()
            .0
            .push_back(VectorDispatchCommand {
                entity: handle.entity,
                implementation,
                request,
            });
        self.world.run_schedule(crate::RuntimeSchedule::Dispatch);
        self.world
            .resource_mut::<VectorDispatchOutbox>()
            .0
            .remove(&handle.run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Validate a correlated vector result and return its owned documents.
    pub fn submit_vector(
        &mut self,
        ingress: VectorSearchIngress,
    ) -> Result<Vec<(f64, String, serde_json::Value)>, RuntimeError> {
        let run_id = ingress.run_id;
        self.world
            .resource_mut::<VectorIngressQueue>()
            .0
            .push_back(ingress);
        self.world.run_schedule(crate::RuntimeSchedule::Ingress);
        self.world
            .resource_mut::<VectorIngressOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Register a tenant-owned portable tool capability.
    pub fn register_tool(
        &mut self,
        tenant_id: TenantId,
        implementation: Arc<dyn ToolImplementation>,
    ) -> CapabilityId {
        let capability_id = CapabilityId::new();
        let revision = 1;
        let definition = implementation.definition();
        let entity = self
            .world
            .spawn(ToolCapability {
                capability_id,
                tenant_id,
                revision,
                definition,
                retired: false,
            })
            .id();
        self.implementations
            .insert((capability_id, revision), implementation.clone());
        self.capabilities.insert(
            capability_id,
            BoundTool {
                entity,
                revision,
                implementation,
            },
        );
        capability_id
    }

    /// Replace future snapshots with a new revision while retaining old revisions
    /// for already-snapshotted in-flight calls.
    pub fn replace_tool(
        &mut self,
        capability_id: CapabilityId,
        implementation: Arc<dyn ToolImplementation>,
    ) -> Result<u64, RuntimeError> {
        let bound = self
            .capabilities
            .get_mut(&capability_id)
            .ok_or(RuntimeError::MissingCapability)?;
        let mut capability = self
            .world
            .get_mut::<ToolCapability>(bound.entity)
            .ok_or(RuntimeError::MissingCapability)?;
        let revision = capability.revision.saturating_add(1);
        capability.revision = revision;
        capability.definition = implementation.definition();
        capability.retired = false;
        bound.revision = revision;
        bound.implementation = implementation.clone();
        self.implementations
            .insert((capability_id, revision), implementation);
        Ok(revision)
    }

    /// Prevent a capability from appearing in future snapshots.
    pub fn retire_tool(&mut self, capability_id: CapabilityId) -> Result<(), RuntimeError> {
        let bound = self
            .capabilities
            .get(&capability_id)
            .ok_or(RuntimeError::MissingCapability)?;
        self.world
            .get_mut::<ToolCapability>(bound.entity)
            .ok_or(RuntimeError::MissingCapability)?
            .retired = true;
        Ok(())
    }

    /// Grant a run's agent access to a capability after tenant validation.
    pub fn grant_tool(
        &mut self,
        handle: RuntimeHandle,
        capability_id: CapabilityId,
    ) -> Result<GrantId, RuntimeError> {
        self.validate(handle)?;
        let identity = *self
            .world
            .get::<RunIdentity>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?;
        let bound = self
            .capabilities
            .get(&capability_id)
            .ok_or(RuntimeError::MissingCapability)?;
        let capability = self
            .world
            .get::<ToolCapability>(bound.entity)
            .ok_or(RuntimeError::MissingCapability)?;
        if capability.tenant_id != identity.tenant_id {
            return Err(RuntimeError::ForeignCapability);
        }
        let grant_id = GrantId::new();
        let entity = self
            .world
            .spawn(CapabilityGrant {
                grant_id,
                tenant_id: identity.tenant_id,
                agent_id: identity.agent_id,
                capability_id,
            })
            .id();
        self.grants
            .insert((identity.agent_id, capability_id), entity);
        Ok(grant_id)
    }

    /// Snapshot exact, granted revisions for one turn in stable capability order.
    pub fn snapshot_tools(
        &self,
        handle: RuntimeHandle,
        capability_ids: impl IntoIterator<Item = CapabilityId>,
    ) -> Result<ToolSnapshot, RuntimeError> {
        self.validate(handle)?;
        let identity = *self
            .world
            .get::<RunIdentity>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?;
        let generation = self
            .world
            .get::<Generation>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?
            .0;
        let mut entries = Vec::new();
        for capability_id in capability_ids {
            let bound = self
                .capabilities
                .get(&capability_id)
                .ok_or(RuntimeError::MissingCapability)?;
            let capability = self
                .world
                .get::<ToolCapability>(bound.entity)
                .ok_or(RuntimeError::MissingCapability)?;
            if capability.tenant_id != identity.tenant_id {
                return Err(RuntimeError::ForeignCapability);
            }
            if capability.retired {
                return Err(RuntimeError::MissingCapability);
            }
            if !self
                .grants
                .contains_key(&(identity.agent_id, capability_id))
            {
                return Err(RuntimeError::MissingGrant);
            }
            entries.push(ToolSnapshotEntry {
                capability_id,
                revision: capability.revision,
                definition: capability.definition.clone(),
            });
        }
        entries.sort_by_key(|entry| entry.capability_id);
        Ok(ToolSnapshot {
            world_id: self.world_id,
            tenant_id: identity.tenant_id,
            run_id: identity.run_id,
            generation,
            entries,
        })
    }

    /// Set the bounded number of concurrently executing tool effects.
    pub fn set_concurrency_limit(
        &mut self,
        handle: RuntimeHandle,
        limit: usize,
    ) -> Result<(), RuntimeError> {
        self.validate(handle)?;
        self.world
            .entity_mut(handle.entity)
            .insert(ConcurrencyLimit(limit.max(1)));
        Ok(())
    }

    /// Create an owned effect resolving the exact snapshotted implementation.
    pub fn dispatch_tool(
        &mut self,
        handle: RuntimeHandle,
        snapshot: &ToolSnapshot,
        call_index: usize,
        name: &str,
        arguments: String,
    ) -> Result<ToolEffectRequest, RuntimeError> {
        self.validate(handle)?;
        let identity = *self
            .world
            .get::<RunIdentity>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?;
        let entry = snapshot
            .entries
            .iter()
            .find(|entry| entry.definition.name == name)
            .ok_or(RuntimeError::ToolNotAdvertised)?;
        if !self
            .grants
            .contains_key(&(identity.agent_id, entry.capability_id))
        {
            return Err(RuntimeError::MissingGrant);
        }
        let implementation = self
            .implementations
            .get(&(entry.capability_id, entry.revision))
            .cloned()
            .ok_or(RuntimeError::MissingCapability)?;
        self.world
            .resource_mut::<ToolDispatchQueue>()
            .0
            .push_back(ToolDispatchCommand {
                entity: handle.entity,
                snapshot: snapshot.clone(),
                call_index,
                name: name.to_string(),
                arguments,
                implementation,
            });
        self.world.run_schedule(crate::RuntimeSchedule::Dispatch);
        self.world
            .resource_mut::<ToolDispatchOutbox>()
            .0
            .remove(&handle.run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Commit tool completions in model-call order regardless of arrival order.
    pub fn submit_tool(&mut self, ingress: ToolIngress) -> Result<(), RuntimeError> {
        let run_id = ingress.run_id;
        self.world
            .resource_mut::<ToolIngressQueue>()
            .0
            .push_back(ingress);
        self.world.run_schedule(crate::RuntimeSchedule::Ingress);
        self.world
            .resource_mut::<ToolIngressOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Apply an invalid-tool decision before any executable effect can exist.
    pub fn apply_invalid_tool_policy(
        &mut self,
        handle: RuntimeHandle,
        call_index: usize,
        tool_name: impl Into<String>,
        policy: InvalidToolPolicy,
    ) -> Result<InvalidToolResolution, RuntimeError> {
        self.validate(handle)?;
        self.world.resource_mut::<PolicyQueue>().0.push_back((
            handle.run_id,
            PolicyCommand::InvalidTool {
                entity: handle.entity,
                call_index,
                tool_name: tool_name.into(),
                policy,
            },
        ));
        self.world.run_schedule(crate::RuntimeSchedule::Progress);
        match self
            .world
            .resource_mut::<PolicyOutbox>()
            .0
            .remove(&handle.run_id)
            .ok_or(RuntimeError::MissingRun)??
        {
            PolicyCommandResult::InvalidTool(resolution) => Ok(resolution),
            PolicyCommandResult::Unit => Err(RuntimeError::MissingRun),
        }
    }

    /// Validate and ingest one streaming observation.
    pub fn submit_stream(&mut self, ingress: StreamingIngress) -> Result<(), RuntimeError> {
        let run_id = match &ingress {
            StreamingIngress::Delta { run_id, .. }
            | StreamingIngress::ProviderFailure { run_id, .. } => *run_id,
            StreamingIngress::ProviderFinal { completion, .. } => completion.run_id,
        };
        self.world
            .resource_mut::<StreamingIngressQueue>()
            .0
            .push_back(ingress);
        self.world.run_schedule(crate::RuntimeSchedule::Ingress);
        self.world
            .resource_mut::<StreamingIngressOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)?
    }

    /// Return retained observations in deterministic insertion order.
    pub fn events(&self, handle: RuntimeHandle) -> Result<&[RuntimeEvent], RuntimeError> {
        self.validate(handle)?;
        Ok(&self
            .world
            .get::<EventLog>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?
            .0)
    }

    /// Explain current progression without exposing sensitive effect inputs.
    pub fn explain(&self, handle: RuntimeHandle) -> Result<RunExplanation, RuntimeError> {
        self.validate(handle)?;
        let entity = self.world.entity(handle.entity);
        let identity = entity
            .get::<RunIdentity>()
            .ok_or(RuntimeError::MissingRun)?;
        let generation = entity.get::<Generation>().ok_or(RuntimeError::MissingRun)?;
        let lifecycle = entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?;
        let budget = entity.get::<CallBudget>().ok_or(RuntimeError::MissingRun)?;
        let tools = entity.get::<ToolBatch>().ok_or(RuntimeError::MissingRun)?;
        let events = entity.get::<EventLog>().ok_or(RuntimeError::MissingRun)?;
        let accounting = entity.get::<Accounting>().ok_or(RuntimeError::MissingRun)?;
        Ok(RunExplanation {
            run_id: identity.run_id,
            tenant_id: identity.tenant_id,
            generation: generation.0,
            status: lifecycle.0.clone(),
            completed_calls: budget.completed_calls,
            max_calls: budget.max_calls,
            pending_model: entity.contains::<PendingOperation>(),
            pending_tools: tools.pending.len(),
            event_count: events.0.len(),
            rejected_effects: accounting.rejected_effects,
        })
    }

    /// Acknowledge that a terminal observer has consumed retained state.
    pub fn acknowledge_terminal(&mut self, handle: RuntimeHandle) -> Result<(), RuntimeError> {
        self.validate(handle)?;
        if !matches!(
            self.world
                .get::<Lifecycle>(handle.entity)
                .ok_or(RuntimeError::MissingRun)?
                .0,
            RunStatus::Terminal(_)
        ) {
            return Err(RuntimeError::NoPendingOperation);
        }
        let mut retention = self
            .world
            .get_mut::<Retention>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?;
        retention.observations_remaining = retention.observations_remaining.saturating_sub(1);
        Ok(())
    }

    /// Request retention-aware cleanup. Supporting state is removed only after
    /// all configured terminal observations have been acknowledged.
    pub fn cleanup(&mut self, handle: RuntimeHandle) -> Result<bool, RuntimeError> {
        self.validate(handle)?;
        self.world
            .get_mut::<Retention>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?
            .cleanup_requested = true;
        self.world.run_schedule(crate::RuntimeSchedule::Cleanup);
        let removed = self.world.get_entity(handle.entity).is_err();
        if removed {
            self.runs.remove(&handle.run_id);
            self.purge_queued_effects(handle.run_id);
        }
        Ok(removed)
    }

    pub fn cancel(
        &mut self,
        handle: RuntimeHandle,
        reason: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        self.validate(handle)?;
        self.world.resource_mut::<PolicyQueue>().0.push_back((
            handle.run_id,
            PolicyCommand::Cancel {
                entity: handle.entity,
                reason: reason.into(),
            },
        ));
        self.world.run_schedule(crate::RuntimeSchedule::Progress);
        self.take_policy_unit(handle.run_id)?;
        self.purge_queued_effects(handle.run_id);
        Ok(())
    }

    fn purge_queued_effects(&mut self, run_id: RunId) {
        self.world
            .resource_mut::<HostedEffectQueue>()
            .0
            .retain(|(queued_run_id, _)| *queued_run_id != run_id);
        self.memory_appends.retain(|effect| effect.run_id != run_id);
    }

    pub fn snapshot(&self, handle: RuntimeHandle) -> Result<RunSnapshot, RuntimeError> {
        self.validate(handle)?;
        let entity = self.world.entity(handle.entity);
        if entity.contains::<PendingOperation>()
            || matches!(
                entity.get::<Lifecycle>().map(|lifecycle| &lifecycle.0),
                Some(RunStatus::AwaitingModel | RunStatus::AwaitingTools | RunStatus::Quiescent)
            )
        {
            return Err(SnapshotError::InFlightOperation.into());
        }
        let identity = entity
            .get::<RunIdentity>()
            .ok_or(RuntimeError::MissingRun)?;
        let generation = entity.get::<Generation>().ok_or(RuntimeError::MissingRun)?;
        let budget = entity.get::<CallBudget>().ok_or(RuntimeError::MissingRun)?;
        let lifecycle = entity.get::<Lifecycle>().ok_or(RuntimeError::MissingRun)?;
        let transcript = entity.get::<Transcript>().ok_or(RuntimeError::MissingRun)?;
        let accounting = entity.get::<Accounting>().ok_or(RuntimeError::MissingRun)?;
        let mut required_tools = self
            .grants
            .iter()
            .filter_map(|(&(agent_id, capability_id), _)| {
                if agent_id != identity.agent_id {
                    return None;
                }
                let bound = self.capabilities.get(&capability_id)?;
                let capability = self.world.get::<ToolCapability>(bound.entity)?;
                Some(PersistedToolBinding {
                    capability_id,
                    tenant_id: capability.tenant_id,
                    revision: capability.revision,
                    definition: capability.definition.clone(),
                    retired: capability.retired,
                })
            })
            .collect::<Vec<_>>();
        required_tools.sort_by_key(|binding| binding.capability_id);
        Ok(RunSnapshot {
            schema_version: 4,
            world_id: identity.world_id,
            tenant_id: identity.tenant_id,
            agent_id: identity.agent_id,
            run_id: identity.run_id,
            generation: generation.0,
            max_calls: budget.max_calls,
            completed_calls: budget.completed_calls,
            status: lifecycle.0.clone(),
            pending_request: entity
                .get::<PendingRequest>()
                .ok_or(RuntimeError::MissingRun)?
                .0
                .clone(),
            history: transcript.history.clone(),
            output: transcript.final_output.clone(),
            usage: accounting.usage,
            rejected_effects: accounting.rejected_effects,
            required_tools,
            required_memory: entity
                .get::<MemoryState>()
                .map(|memory| PersistedMemoryBinding {
                    store_id: memory.store_id,
                    conversation_id: memory.conversation_id.clone(),
                    loaded: memory.loaded,
                    persist_from: memory.persist_from,
                    appended_generation: memory.appended_generation,
                }),
        })
    }

    /// Restore explicit versioned domain state and rebind executable tools.
    /// No Bevy entity, task, channel, provider final, client, or credential is restored.
    pub fn restore(
        snapshot: RunSnapshot,
        rebindings: impl IntoIterator<Item = ToolRebinding>,
    ) -> Result<(Self, RuntimeHandle), RuntimeError> {
        Self::restore_with_memory(snapshot, rebindings, std::iter::empty())
    }

    /// Restore domain state with explicit tool and memory implementation
    /// rebindings. Missing backends fail before world reconstruction.
    pub fn restore_with_memory(
        snapshot: RunSnapshot,
        rebindings: impl IntoIterator<Item = ToolRebinding>,
        memory_rebindings: impl IntoIterator<Item = MemoryRebinding>,
    ) -> Result<(Self, RuntimeHandle), RuntimeError> {
        if snapshot.schema_version != 4 {
            return Err(SnapshotError::UnsupportedVersion(snapshot.schema_version).into());
        }
        if snapshot.completed_calls > snapshot.max_calls {
            return Err(SnapshotError::InvalidIntegrity(
                "completed model calls exceed configured budget".to_string(),
            )
            .into());
        }
        if matches!(snapshot.status, RunStatus::Ready) && snapshot.pending_request.is_none() {
            return Err(SnapshotError::InvalidIntegrity(
                "ready snapshot is missing its resumable request".to_string(),
            )
            .into());
        }
        if matches!(
            snapshot.status,
            RunStatus::AwaitingModel | RunStatus::AwaitingTools | RunStatus::Quiescent
        ) {
            return Err(SnapshotError::InvalidIntegrity(
                "snapshots must be taken at a stable ready or terminal boundary".to_string(),
            )
            .into());
        }
        let rebound = rebindings
            .into_iter()
            .map(|binding| {
                (
                    (binding.capability_id, binding.revision),
                    binding.implementation,
                )
            })
            .collect::<HashMap<_, _>>();
        for required in &snapshot.required_tools {
            let implementation = rebound
                .get(&(required.capability_id, required.revision))
                .ok_or(SnapshotError::MissingImplementation(
                    required.capability_id,
                    required.revision,
                ))?;
            if implementation.definition() != required.definition {
                return Err(SnapshotError::MismatchedImplementation(
                    required.capability_id,
                    required.revision,
                )
                .into());
            }
        }
        let rebound_memory = memory_rebindings
            .into_iter()
            .map(|binding| (binding.store_id, binding.implementation))
            .collect::<HashMap<_, _>>();
        if let Some(required) = &snapshot.required_memory
            && !rebound_memory.contains_key(&required.store_id)
        {
            return Err(SnapshotError::MissingMemoryImplementation(required.store_id).into());
        }

        let mut world = World::new();
        schedule::install(&mut world);
        let mut runtime = Self {
            world_id: snapshot.world_id,
            world,
            runs: HashMap::new(),
            capabilities: HashMap::new(),
            implementations: HashMap::new(),
            grants: HashMap::new(),
            memories: HashMap::new(),
            memory_appends: VecDeque::new(),
        };
        let run_entity = runtime
            .world
            .spawn((
                RunIdentity {
                    world_id: snapshot.world_id,
                    tenant_id: snapshot.tenant_id,
                    agent_id: snapshot.agent_id,
                    run_id: snapshot.run_id,
                },
                Generation(snapshot.generation),
                CallBudget {
                    max_calls: snapshot.max_calls,
                    completed_calls: snapshot.completed_calls,
                },
                Lifecycle(snapshot.status.clone()),
                PendingRequest(snapshot.pending_request.clone()),
                Transcript {
                    history: snapshot.history,
                    final_output: snapshot.output,
                },
                Accounting {
                    usage: snapshot.usage,
                    rejected_effects: snapshot.rejected_effects,
                },
                StreamCursor::default(),
                StalledPasses::default(),
                EventLog::default(),
                ToolBatch::default(),
                RecoveryPolicy::default(),
                ConcurrencyLimit(8),
                Retention::default(),
            ))
            .id();
        runtime.world.entity_mut(run_entity).insert((
            ToolContinuation::default(),
            InvalidResolution::default(),
            InvalidRetryCount::default(),
        ));
        runtime.runs.insert(snapshot.run_id, run_entity);

        if let Some(required) = snapshot.required_memory {
            let implementation = rebound_memory.get(&required.store_id).cloned().ok_or(
                SnapshotError::MissingMemoryImplementation(required.store_id),
            )?;
            runtime.world.entity_mut(run_entity).insert(MemoryState {
                store_id: required.store_id,
                conversation_id: required.conversation_id,
                loaded: required.loaded,
                load_correlation: None,
                appended_generation: required.appended_generation,
                append_correlation: None,
                persist_from: required.persist_from,
            });
            runtime.memories.insert(required.store_id, implementation);
        }

        for required in snapshot.required_tools {
            let implementation = rebound
                .get(&(required.capability_id, required.revision))
                .cloned()
                .ok_or(SnapshotError::MissingImplementation(
                    required.capability_id,
                    required.revision,
                ))?;
            let capability_entity = runtime
                .world
                .spawn(ToolCapability {
                    capability_id: required.capability_id,
                    tenant_id: required.tenant_id,
                    revision: required.revision,
                    definition: required.definition,
                    retired: required.retired,
                })
                .id();
            runtime.capabilities.insert(
                required.capability_id,
                BoundTool {
                    entity: capability_entity,
                    revision: required.revision,
                    implementation: implementation.clone(),
                },
            );
            runtime
                .implementations
                .insert((required.capability_id, required.revision), implementation);
            let grant_id = GrantId::new();
            let grant_entity = runtime
                .world
                .spawn(CapabilityGrant {
                    grant_id,
                    tenant_id: snapshot.tenant_id,
                    agent_id: snapshot.agent_id,
                    capability_id: required.capability_id,
                })
                .id();
            runtime
                .grants
                .insert((snapshot.agent_id, required.capability_id), grant_entity);
        }

        let handle = RuntimeHandle {
            world_id: snapshot.world_id,
            tenant_id: snapshot.tenant_id,
            run_id: snapshot.run_id,
            entity: run_entity,
        };
        Ok((runtime, handle))
    }

    fn validate(&self, handle: RuntimeHandle) -> Result<(), RuntimeError> {
        if handle.world_id != self.world_id {
            return Err(RuntimeError::ForeignWorld);
        }
        let run = self
            .world
            .get::<RunIdentity>(handle.entity)
            .ok_or(RuntimeError::MissingRun)?;
        if run.run_id != handle.run_id {
            return Err(RuntimeError::MissingRun);
        }
        if run.tenant_id != handle.tenant_id {
            return Err(RuntimeError::WrongTenant);
        }
        Ok(())
    }

    fn take_policy_unit(&mut self, run_id: RunId) -> Result<(), RuntimeError> {
        match self
            .world
            .resource_mut::<PolicyOutbox>()
            .0
            .remove(&run_id)
            .ok_or(RuntimeError::MissingRun)??
        {
            PolicyCommandResult::Unit => Ok(()),
            PolicyCommandResult::InvalidTool(_) => Err(RuntimeError::MissingRun),
        }
    }

    fn queue_memory_append(&mut self, run_id: RunId, entity: Entity) -> Result<(), RuntimeError> {
        let generation = self
            .world
            .get::<Generation>(entity)
            .ok_or(RuntimeError::MissingRun)?
            .0;
        let Some(memory) = self.world.get::<MemoryState>(entity) else {
            return Ok(());
        };
        if !memory.loaded
            || memory.appended_generation == Some(generation)
            || memory.append_correlation.is_some()
        {
            return Ok(());
        }
        let implementation =
            self.memories
                .get(&memory.store_id)
                .cloned()
                .ok_or(RuntimeError::MemoryFailure(
                    "memory implementation is not bound".to_string(),
                ))?;
        let transcript = self
            .world
            .get::<Transcript>(entity)
            .ok_or(RuntimeError::MissingRun)?;
        let messages = transcript
            .history
            .get(memory.persist_from..)
            .unwrap_or_default()
            .to_vec();
        let identity = *self
            .world
            .get::<RunIdentity>(entity)
            .ok_or(RuntimeError::MissingRun)?;
        let operation_id = OperationId::new();
        let correlation_id = CorrelationId::new();
        let effect = MemoryAppendEffect {
            world_id: self.world_id,
            tenant_id: identity.tenant_id,
            run_id,
            operation_id,
            correlation_id,
            generation,
            store_id: memory.store_id,
            conversation_id: memory.conversation_id.clone(),
            messages,
            implementation,
        };
        self.world
            .get_mut::<MemoryState>(entity)
            .ok_or(RuntimeError::MissingRun)?
            .append_correlation = Some((operation_id, correlation_id, generation));
        self.memory_appends.push_back(effect);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::{
        OneOrMany,
        completion::{Message, ToolDefinition},
        tool::{ToolOutput, ToolResult},
        vector_store::{TopNResults, VectorStoreError, VectorStoreIndexDyn},
        wasm_compat::WasmBoxedFuture,
    };
    use rig_runtime_conformance::{Adapter, Observation, Outcome, Scenario, assert_outcome};

    struct VersionedTool(&'static str);

    struct RecordingVector;

    impl VectorStoreIndexDyn for RecordingVector {
        fn top_n<'a>(
            &'a self,
            request: VectorSearchRequest<Filter<serde_json::Value>>,
        ) -> WasmBoxedFuture<'a, TopNResults> {
            Box::pin(async move {
                Ok(vec![(
                    0.9,
                    "doc-1".to_string(),
                    serde_json::json!({ "query": request.query() }),
                )])
            })
        }

        fn top_n_ids<'a>(
            &'a self,
            _request: VectorSearchRequest<Filter<serde_json::Value>>,
        ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
            Box::pin(async { Ok(vec![(0.9, "doc-1".to_string())]) })
        }
    }

    #[derive(Default)]
    struct RecordingMemory {
        loaded: std::sync::Mutex<Vec<Message>>,
        appended: std::sync::Mutex<Vec<Vec<Message>>>,
    }

    impl MemoryImplementation for RecordingMemory {
        fn load(
            &self,
            _conversation_id: String,
        ) -> WasmBoxedFuture<'_, Result<Vec<Message>, crate::MemoryEffectError>> {
            let messages = self.loaded.lock().expect("memory lock").clone();
            Box::pin(async move { Ok(messages) })
        }

        fn append(
            &self,
            _conversation_id: String,
            messages: Vec<Message>,
        ) -> WasmBoxedFuture<'_, Result<(), crate::MemoryEffectError>> {
            self.appended.lock().expect("memory lock").push(messages);
            Box::pin(async { Ok(()) })
        }
    }

    impl ToolImplementation for VersionedTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "lookup".to_string(),
                description: self.0.to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(&self, _arguments: String) -> WasmBoxedFuture<'_, ToolResult> {
            let version = self.0;
            Box::pin(async move { ToolResult::success(ToolOutput::text(version)) })
        }
    }

    fn request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::user("hello")),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            additional_params: None,
            tool_choice: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    fn completion(effect: &EffectRequest, text: &str, tokens: u64) -> CompletionIngress {
        CompletionIngress {
            world_id: effect.world_id,
            tenant_id: effect.tenant_id,
            run_id: effect.run_id,
            operation_id: effect.operation_id,
            correlation_id: effect.correlation_id,
            generation: effect.generation,
            choice: vec![AssistantContent::text(text)],
            usage: Usage {
                total_tokens: tokens,
                ..Usage::new()
            },
        }
    }

    #[test]
    fn duplicate_completion_cannot_double_commit_usage() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let effect = runtime.dispatch(handle).expect("dispatch");
        let ingress = CompletionIngress {
            world_id: effect.world_id,
            tenant_id: effect.tenant_id,
            run_id: effect.run_id,
            operation_id: effect.operation_id,
            correlation_id: effect.correlation_id,
            generation: effect.generation,
            choice: vec![AssistantContent::text("done")],
            usage: Usage {
                input_tokens: 2,
                output_tokens: 3,
                total_tokens: 5,
                ..Usage::new()
            },
        };
        runtime.submit(ingress.clone()).expect("first commit");
        assert!(matches!(
            runtime.submit(ingress),
            Err(RuntimeError::Terminal)
        ));
        let snapshot = runtime.snapshot(handle).expect("snapshot");
        assert_eq!(snapshot.completed_calls, 1);
        assert_eq!(snapshot.usage.total_tokens, 5);
    }

    #[test]
    fn response_retry_rolls_back_content_and_consumes_total_budget() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 2);
        let first = runtime.dispatch(handle).expect("first dispatch");
        runtime
            .retry_response(completion(&first, "rejected", 3), "return valid output")
            .expect("retry");
        let after_reject = runtime.snapshot(handle).expect("snapshot");
        assert_eq!(after_reject.completed_calls, 1);
        assert_eq!(after_reject.usage.total_tokens, 3);
        assert!(after_reject.output.is_empty());
        assert!(
            after_reject
                .history
                .iter()
                .all(|message| !format!("{message:?}").contains("rejected"))
        );

        let second = runtime.dispatch(handle).expect("second dispatch");
        assert_eq!(second.generation, 1);
        runtime
            .submit(completion(&second, "accepted", 4))
            .expect("accept");
        let final_snapshot = runtime.snapshot(handle).expect("final snapshot");
        assert_eq!(final_snapshot.completed_calls, 2);
        assert_eq!(final_snapshot.usage.total_tokens, 7);
        assert!(matches!(
            runtime.dispatch(handle),
            Err(RuntimeError::Terminal)
        ));
    }

    #[test]
    fn cleanup_waits_for_terminal_observer_acknowledgement() {
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(TenantId::new(), request(), 1);
        let effect = runtime.dispatch(handle).expect("dispatch");
        runtime
            .submit(completion(&effect, "done", 1))
            .expect("commit");
        assert!(!runtime.cleanup(handle).expect("retained"));
        assert!(runtime.snapshot(handle).is_ok());
        runtime.acknowledge_terminal(handle).expect("ack");
        assert!(runtime.cleanup(handle).expect("removed"));
        assert!(matches!(
            runtime.snapshot(handle),
            Err(RuntimeError::MissingRun)
        ));
    }

    #[test]
    fn every_invalid_tool_policy_is_explicit_and_suppresses_execution() {
        let policies = [
            InvalidToolPolicy::Fail,
            InvalidToolPolicy::Retry,
            InvalidToolPolicy::Repair {
                replacement_name: "safe".to_string(),
                replacement_arguments: "{}".to_string(),
            },
            InvalidToolPolicy::Skip {
                reason: "skip".to_string(),
            },
            InvalidToolPolicy::Stop {
                reason: "stop".to_string(),
            },
        ];
        for (index, policy) in policies.into_iter().enumerate() {
            let mut runtime = Runtime::new();
            let handle = runtime.spawn_run(TenantId::new(), request(), 2);
            runtime
                .world
                .get_mut::<ToolContinuation>(handle.entity)
                .expect("continuation")
                .0 = Some(request());
            let resolution = runtime
                .apply_invalid_tool_policy(handle, 0, "missing", policy)
                .expect("policy");
            assert!(matches!(
                (index, resolution),
                (0, InvalidToolResolution::Failed)
                    | (1, InvalidToolResolution::Retry { .. })
                    | (2, InvalidToolResolution::Repair { .. })
                    | (3, InvalidToolResolution::Skip { .. })
                    | (4, InvalidToolResolution::Stopped)
            ));
            assert_eq!(
                runtime.explain(handle).expect("explain").rejected_effects,
                1
            );
            assert!(
                runtime
                    .events(handle)
                    .expect("events")
                    .iter()
                    .any(|event| { matches!(event, RuntimeEvent::RejectedEffect { .. }) })
            );
        }
    }

    #[test]
    fn tool_queue_is_bounded_and_terminal_runs_suppress_dispatch() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 2);
        runtime.set_concurrency_limit(handle, 1).expect("limit");
        let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("v1")));
        runtime.grant_tool(handle, capability).expect("grant");
        let snapshot = runtime
            .snapshot_tools(handle, [capability])
            .expect("snapshot");
        runtime
            .dispatch_tool(handle, &snapshot, 0, "lookup", "{}".to_string())
            .expect("first");
        assert!(matches!(
            runtime.dispatch_tool(handle, &snapshot, 1, "lookup", "{}".to_string()),
            Err(RuntimeError::Backpressure(1))
        ));
        runtime.cancel(handle, "cancel").expect("cancel");
        assert!(matches!(
            runtime.dispatch_tool(handle, &snapshot, 1, "lookup", "{}".to_string()),
            Err(RuntimeError::Terminal)
        ));
    }

    #[test]
    fn cancellation_rejects_late_completion() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let effect = runtime.dispatch(handle).expect("dispatch");
        runtime.cancel(handle, "caller dropped").expect("cancel");
        let result = runtime.submit(CompletionIngress {
            world_id: effect.world_id,
            tenant_id: effect.tenant_id,
            run_id: effect.run_id,
            operation_id: effect.operation_id,
            correlation_id: effect.correlation_id,
            generation: effect.generation,
            choice: vec![AssistantContent::text("late")],
            usage: Usage::new(),
        });
        assert!(matches!(result, Err(RuntimeError::Terminal)));
    }

    #[test]
    fn foreign_world_effect_is_rejected() {
        let tenant = TenantId::new();
        let mut first = Runtime::new();
        let mut second = Runtime::new();
        let handle = first.spawn_run(tenant, request(), 1);
        let effect = first.dispatch(handle).expect("dispatch");
        let result = second.submit(CompletionIngress {
            world_id: effect.world_id,
            tenant_id: effect.tenant_id,
            run_id: effect.run_id,
            operation_id: effect.operation_id,
            correlation_id: effect.correlation_id,
            generation: effect.generation,
            choice: Vec::new(),
            usage: Usage::new(),
        });
        assert!(matches!(result, Err(RuntimeError::ForeignWorld)));
    }

    #[test]
    fn provisional_deltas_are_observable_but_never_committed() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let effect = runtime.dispatch(handle).expect("dispatch");
        runtime
            .submit_stream(StreamingIngress::Delta {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                sequence: 0,
                content: AssistantContent::text("looks final"),
            })
            .expect("delta");

        let snapshot = runtime.snapshot(handle);
        assert!(matches!(
            snapshot,
            Err(RuntimeError::Snapshot(SnapshotError::InFlightOperation))
        ));
        assert!(matches!(
            runtime.events(handle).expect("events"),
            [RuntimeEvent::ProvisionalDelta { .. }]
        ));
    }

    #[test]
    fn provider_failure_after_final_looking_delta_never_exposes_success() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let effect = runtime.dispatch(handle).expect("dispatch");
        runtime
            .submit_stream(StreamingIngress::Delta {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                sequence: 0,
                content: AssistantContent::text("done"),
            })
            .expect("delta");
        runtime
            .submit_stream(StreamingIngress::ProviderFailure {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                message: "stream failed".to_string(),
            })
            .expect("failure");

        let snapshot = runtime.snapshot(handle).expect("terminal snapshot");
        assert!(snapshot.output.is_empty());
        assert!(matches!(
            snapshot.status,
            RunStatus::Terminal(TerminalReason::Failed { .. })
        ));
        assert!(
            !runtime
                .events(handle)
                .expect("events")
                .iter()
                .any(|event| { matches!(event, RuntimeEvent::AcceptedFinal { .. }) })
        );
    }

    #[test]
    fn streaming_provider_final_has_canonical_event_order() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let effect = runtime.dispatch(handle).expect("dispatch");
        runtime
            .submit_stream(StreamingIngress::ProviderFinal {
                completion: CompletionIngress {
                    world_id: effect.world_id,
                    tenant_id: effect.tenant_id,
                    run_id: effect.run_id,
                    operation_id: effect.operation_id,
                    correlation_id: effect.correlation_id,
                    generation: effect.generation,
                    choice: vec![AssistantContent::text("done")],
                    usage: Usage::new(),
                },
                final_response: ProviderFinalEnvelope {
                    provider: "scripted".to_string(),
                    type_name: "ScriptedFinal".to_string(),
                    diagnostic_json: serde_json::json!({"id": "final-1"}),
                },
            })
            .expect("final");

        assert!(matches!(
            runtime.events(handle).expect("events"),
            [
                RuntimeEvent::ProviderFinal { .. },
                RuntimeEvent::AcceptedFinal { .. },
                RuntimeEvent::Terminal { .. },
            ]
        ));
    }

    #[tokio::test]
    async fn snapshotted_revision_survives_replacement_and_retirement() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 2);
        let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("v1")));
        runtime.grant_tool(handle, capability).expect("grant");
        let snapshot = runtime
            .snapshot_tools(handle, [capability])
            .expect("snapshot");
        assert_eq!(snapshot.entries[0].revision, 1);

        assert_eq!(
            runtime
                .replace_tool(capability, Arc::new(VersionedTool("v2")))
                .expect("replace"),
            2
        );
        runtime.retire_tool(capability).expect("retire");
        let effect = runtime
            .dispatch_tool(handle, &snapshot, 0, "lookup", "{}".to_string())
            .expect("old snapshot remains executable");
        assert_eq!(effect.revision, 1);
        let ingress = effect.execute().await;
        assert_eq!(ingress.result.output().as_text(), Some("v1"));
        runtime.submit_tool(ingress).expect("commit");
        assert!(matches!(
            runtime.snapshot_tools(handle, [capability]),
            Err(RuntimeError::MissingCapability)
        ));
    }

    #[test]
    fn tool_completions_commit_in_call_order_not_arrival_order() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 2);
        let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("v1")));
        runtime.grant_tool(handle, capability).expect("grant");
        let snapshot = runtime
            .snapshot_tools(handle, [capability])
            .expect("snapshot");
        let first = runtime
            .dispatch_tool(handle, &snapshot, 0, "lookup", "{}".to_string())
            .expect("first");
        let second = runtime
            .dispatch_tool(handle, &snapshot, 1, "lookup", "{}".to_string())
            .expect("second");

        let ingress = |effect: ToolEffectRequest| ToolIngress {
            world_id: effect.world_id,
            tenant_id: effect.tenant_id,
            run_id: effect.run_id,
            operation_id: effect.operation_id,
            correlation_id: effect.correlation_id,
            generation: effect.generation,
            call_index: effect.call_index,
            capability_id: effect.capability_id,
            revision: effect.revision,
            name: effect.name,
            result: ToolResult::success(ToolOutput::text(format!("call-{}", effect.call_index))),
        };
        runtime
            .submit_tool(ingress(second))
            .expect("second arrives");
        assert!(runtime.events(handle).expect("events").is_empty());
        runtime.submit_tool(ingress(first)).expect("first arrives");
        let indices = runtime
            .events(handle)
            .expect("events")
            .iter()
            .filter_map(|event| match event {
                RuntimeEvent::ToolCommitted { call_index, .. } => Some(*call_index),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn foreign_tenant_cannot_receive_or_snapshot_capability() {
        let mut runtime = Runtime::new();
        let first = runtime.spawn_run(TenantId::new(), request(), 1);
        let second = runtime.spawn_run(TenantId::new(), request(), 1);
        let capability = runtime.register_tool(first.tenant_id, Arc::new(VersionedTool("v1")));
        assert!(matches!(
            runtime.grant_tool(second, capability),
            Err(RuntimeError::ForeignCapability)
        ));
        runtime.grant_tool(first, capability).expect("owner grant");
        assert!(runtime.snapshot_tools(first, [capability]).is_ok());
    }

    #[test]
    fn restoration_requires_exact_explicit_rebinding() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("v1")));
        runtime.grant_tool(handle, capability).expect("grant");
        let snapshot = runtime.snapshot(handle).expect("snapshot");

        assert!(matches!(
            Runtime::restore(snapshot.clone(), []),
            Err(RuntimeError::Snapshot(SnapshotError::MissingImplementation(id, 1))) if id == capability
        ));
        assert!(matches!(
            Runtime::restore(snapshot.clone(), [ToolRebinding {
                capability_id: capability,
                revision: 1,
                implementation: Arc::new(VersionedTool("different-definition")),
            }]),
            Err(RuntimeError::Snapshot(SnapshotError::MismatchedImplementation(id, 1))) if id == capability
        ));

        let (restored, restored_handle) = Runtime::restore(
            snapshot,
            [ToolRebinding {
                capability_id: capability,
                revision: 1,
                implementation: Arc::new(VersionedTool("v1")),
            }],
        )
        .expect("restore");
        let restored_snapshot = restored
            .snapshot(restored_handle)
            .expect("restored snapshot");
        assert_eq!(restored_snapshot.run_id, handle.run_id);
        assert_eq!(restored_snapshot.tenant_id, tenant);
        assert_eq!(restored_snapshot.required_tools.len(), 1);
    }

    #[test]
    fn snapshot_schema_rejects_corrupt_accounting() {
        let tenant = TenantId::new();
        let runtime = Runtime::new();
        let mut snapshot = RunSnapshot {
            schema_version: 4,
            world_id: runtime.world_id,
            tenant_id: tenant,
            agent_id: AgentId::new(),
            run_id: RunId::new(),
            generation: 0,
            max_calls: 1,
            completed_calls: 2,
            status: RunStatus::Ready,
            pending_request: Some(request()),
            history: Vec::new(),
            output: Vec::new(),
            usage: Usage::new(),
            rejected_effects: 0,
            required_tools: Vec::new(),
            required_memory: None,
        };
        assert!(matches!(
            Runtime::restore(snapshot.clone(), []),
            Err(RuntimeError::Snapshot(SnapshotError::InvalidIntegrity(_)))
        ));
        snapshot.schema_version = 99;
        assert!(matches!(
            Runtime::restore(snapshot, []),
            Err(RuntimeError::Snapshot(SnapshotError::UnsupportedVersion(
                99
            )))
        ));
    }

    #[tokio::test]
    async fn vector_effects_are_owned_correlated_and_single_commit() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let vector_request = VectorSearchRequest::builder()
            .query("private query")
            .samples(1)
            .build();
        let effect = runtime
            .dispatch_vector(handle, Arc::new(RecordingVector), vector_request)
            .expect("vector dispatch");
        assert!(!format!("{effect:?}").contains("private query"));
        let ingress = effect.execute().await;
        let duplicate = VectorSearchIngress {
            world_id: ingress.world_id,
            tenant_id: ingress.tenant_id,
            run_id: ingress.run_id,
            operation_id: ingress.operation_id,
            correlation_id: ingress.correlation_id,
            generation: ingress.generation,
            store_id: ingress.store_id,
            result: Ok(Vec::new()),
        };
        let documents = runtime.submit_vector(ingress).expect("vector commit");
        assert_eq!(documents[0].1, "doc-1");
        assert!(matches!(
            runtime.submit_vector(duplicate),
            Err(RuntimeError::StaleVectorCompletion)
        ));
    }

    #[tokio::test]
    async fn memory_load_precedes_dispatch_and_append_contains_only_new_commits() {
        let tenant = TenantId::new();
        let memory = Arc::new(RecordingMemory::default());
        memory
            .loaded
            .lock()
            .expect("memory lock")
            .push(Message::user("old"));
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let load = runtime
            .bind_memory(handle, "thread-secret", memory.clone())
            .expect("bind");
        assert!(matches!(
            runtime.dispatch(handle),
            Err(RuntimeError::MemoryNotLoaded)
        ));
        runtime
            .submit_memory_load(load.execute().await)
            .expect("load");

        let effect = runtime.dispatch(handle).expect("dispatch after load");
        runtime
            .submit(CompletionIngress {
                world_id: effect.world_id,
                tenant_id: effect.tenant_id,
                run_id: effect.run_id,
                operation_id: effect.operation_id,
                correlation_id: effect.correlation_id,
                generation: effect.generation,
                choice: vec![AssistantContent::text("new answer")],
                usage: Usage::new(),
            })
            .expect("commit");
        let appends = runtime.take_memory_appends().collect::<Vec<_>>();
        assert_eq!(appends.len(), 1);
        assert_eq!(
            appends[0].messages.len(),
            2,
            "new prompt and accepted assistant only"
        );
        let append_ingress = appends.into_iter().next().expect("append").execute().await;
        runtime
            .submit_memory_append(append_ingress)
            .expect("append commit");
        assert_eq!(memory.appended.lock().expect("memory lock").len(), 1);
    }

    #[tokio::test]
    async fn restoration_requires_explicit_memory_rebinding() {
        let tenant = TenantId::new();
        let memory = Arc::new(RecordingMemory::default());
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let load = runtime
            .bind_memory(handle, "conversation", memory.clone())
            .expect("bind");
        runtime
            .submit_memory_load(load.execute().await)
            .expect("load");
        let effect = runtime.dispatch(handle).expect("dispatch");
        runtime
            .submit(completion(&effect, "done", 1))
            .expect("complete");
        let snapshot = runtime.snapshot(handle).expect("snapshot");
        let store_id = snapshot.required_memory.as_ref().expect("memory").store_id;

        assert!(matches!(
            Runtime::restore(snapshot.clone(), []),
            Err(RuntimeError::Snapshot(
                SnapshotError::MissingMemoryImplementation(id)
            )) if id == store_id
        ));
        let (restored, restored_handle) = Runtime::restore_with_memory(
            snapshot,
            [],
            [MemoryRebinding {
                store_id,
                implementation: memory,
            }],
        )
        .expect("restore");
        assert_eq!(
            restored
                .snapshot(restored_handle)
                .expect("restored snapshot")
                .required_memory
                .expect("restored memory")
                .store_id,
            store_id
        );
    }

    #[test]
    fn ready_snapshot_restores_a_dispatchable_request() {
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(TenantId::new(), request(), 1);
        let snapshot = runtime.snapshot(handle).expect("stable ready snapshot");
        let (mut restored, restored_handle) = Runtime::restore(snapshot, []).expect("restore");
        let effect = restored
            .dispatch(restored_handle)
            .expect("resumed dispatch");
        assert_eq!(effect.generation, 0);
    }

    #[test]
    fn hosted_progress_detects_livelock_and_retains_terminal_state() {
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(TenantId::new(), request(), 1);
        runtime
            .world
            .get_mut::<Lifecycle>(handle.entity)
            .expect("lifecycle")
            .0 = RunStatus::Quiescent;
        let hosted = runtime.hosted(handle).expect("hosted handle");
        for _ in 0..31 {
            assert!(matches!(
                runtime.poll_hosted(hosted).expect("poll"),
                HostedPoll::Pending
            ));
        }
        assert!(matches!(
            runtime.poll_hosted(hosted).expect("terminal poll"),
            HostedPoll::Terminal(TerminalReason::Livelock)
        ));
        assert!(runtime.events(handle).expect("events").iter().any(|event| {
            matches!(
                event,
                RuntimeEvent::Terminal {
                    reason: TerminalReason::Livelock,
                    ..
                }
            )
        }));
    }

    #[tokio::test]
    async fn hosted_driver_progresses_model_tool_and_continuation_effects() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 2);
        let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("hosted")));
        runtime.grant_tool(handle, capability).expect("grant");
        let hosted = runtime.hosted(handle).expect("hosted");
        let model = match runtime.poll_hosted(hosted).expect("model poll") {
            HostedPoll::Effect(HostedEffect::Model(effect)) => effect,
            other => panic!("expected model effect, got {other:?}"),
        };
        runtime
            .submit_hosted_completion(
                hosted,
                CompletionIngress {
                    world_id: model.world_id,
                    tenant_id: model.tenant_id,
                    run_id: model.run_id,
                    operation_id: model.operation_id,
                    correlation_id: model.correlation_id,
                    generation: model.generation,
                    choice: vec![AssistantContent::ToolCall(
                        rig_core::message::ToolCall::new(
                            "hosted-call".to_string(),
                            rig_core::message::ToolFunction::new(
                                "lookup".to_string(),
                                serde_json::json!({}),
                            ),
                        ),
                    )],
                    usage: Usage::new(),
                },
            )
            .expect("tool turn");
        let tool = match runtime.poll_hosted(hosted).expect("tool poll") {
            HostedPoll::Effect(HostedEffect::Tool(effect)) => effect,
            other => panic!("expected tool effect, got {other:?}"),
        };
        runtime
            .submit_hosted_tool(hosted, tool.execute().await)
            .expect("tool result");
        let model = match runtime.poll_hosted(hosted).expect("continuation poll") {
            HostedPoll::Effect(HostedEffect::Model(effect)) => effect,
            other => panic!("expected continuation effect, got {other:?}"),
        };
        runtime
            .submit_hosted_completion(hosted, completion(&model, "done", 1))
            .expect("final");
        assert!(matches!(
            runtime.poll_hosted(hosted).expect("terminal"),
            HostedPoll::Terminal(TerminalReason::Completed)
        ));
    }

    #[test]
    fn hosted_invalid_peer_never_exposes_staged_sibling_effect() {
        for policy in [
            InvalidToolPolicy::Fail,
            InvalidToolPolicy::Stop {
                reason: "stop".into(),
            },
            InvalidToolPolicy::Retry,
        ] {
            let tenant = TenantId::new();
            let mut runtime = Runtime::new();
            let handle = runtime.spawn_run(tenant, request(), 2);
            let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("valid")));
            runtime.grant_tool(handle, capability).expect("grant");
            let hosted = runtime.hosted(handle).expect("hosted");
            let model = match runtime.poll_hosted(hosted).expect("model poll") {
                HostedPoll::Effect(HostedEffect::Model(effect)) => effect,
                other => panic!("expected model effect, got {other:?}"),
            };
            let call = |id: &str, name: &str| {
                AssistantContent::ToolCall(rig_core::message::ToolCall::new(
                    id.to_string(),
                    rig_core::message::ToolFunction::new(name.to_string(), serde_json::json!({})),
                ))
            };
            runtime
                .submit_hosted_completion_with_policy(
                    hosted,
                    CompletionIngress {
                        world_id: model.world_id,
                        tenant_id: model.tenant_id,
                        run_id: model.run_id,
                        operation_id: model.operation_id,
                        correlation_id: model.correlation_id,
                        generation: model.generation,
                        choice: vec![call("valid", "lookup"), call("invalid", "missing")],
                        usage: Usage::new(),
                    },
                    policy,
                )
                .expect("policy applied");
            assert!(!matches!(
                runtime.poll_hosted(hosted).expect("post-policy poll"),
                HostedPoll::Effect(HostedEffect::Tool(_))
            ));
        }
    }

    #[test]
    fn terminal_run_rejects_direct_and_hosted_memory_binding() {
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(TenantId::new(), request(), 1);
        runtime.cancel(handle, "stop").expect("cancel");
        assert!(matches!(
            runtime.bind_memory(handle, "late", Arc::new(RecordingMemory::default())),
            Err(RuntimeError::Terminal)
        ));
        let hosted = runtime.hosted(handle).expect("hosted");
        assert!(matches!(
            runtime.hosted_bind_memory(hosted, "late", Arc::new(RecordingMemory::default())),
            Err(RuntimeError::Terminal)
        ));
    }

    #[tokio::test]
    async fn terminal_poll_and_cleanup_purge_owned_effect_queues() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 1);
        let hosted = runtime.hosted(handle).expect("hosted");
        let load = runtime
            .bind_memory(handle, "conversation", Arc::new(RecordingMemory::default()))
            .expect("memory effect");
        runtime
            .submit_memory_load(load.execute().await)
            .expect("memory loaded");
        runtime
            .hosted_dispatch_vector(
                hosted,
                Arc::new(RecordingVector),
                VectorSearchRequest::builder()
                    .query("queued")
                    .samples(1)
                    .build(),
            )
            .expect("queued vector");
        let model = runtime.dispatch(handle).expect("model");
        runtime
            .submit(completion(&model, "done", 1))
            .expect("complete");
        assert!(!runtime.memory_appends.is_empty());
        assert!(matches!(
            runtime.poll_hosted(hosted).expect("terminal poll"),
            HostedPoll::Terminal(TerminalReason::Completed)
        ));
        assert!(runtime.memory_appends.is_empty());
        assert!(runtime.world.resource::<HostedEffectQueue>().0.is_empty());

        runtime.acknowledge_terminal(handle).expect("acknowledge");
        assert!(runtime.cleanup(handle).expect("cleanup"));
        assert!(runtime.memory_appends.is_empty());
        assert!(runtime.world.resource::<HostedEffectQueue>().0.is_empty());
    }

    #[tokio::test]
    async fn cancellation_rejects_correlated_vector_completion() {
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(TenantId::new(), request(), 1);
        let effect = runtime
            .dispatch_vector(
                handle,
                Arc::new(RecordingVector),
                VectorSearchRequest::builder()
                    .query("secret")
                    .samples(1)
                    .build(),
            )
            .expect("vector effect");
        runtime.cancel(handle, "stop").expect("cancel");
        assert!(matches!(
            runtime.submit_vector(effect.execute().await),
            Err(RuntimeError::StaleVectorCompletion)
        ));
        assert!(matches!(
            runtime.dispatch_vector(
                handle,
                Arc::new(RecordingVector),
                VectorSearchRequest::builder()
                    .query("secret")
                    .samples(1)
                    .build(),
            ),
            Err(RuntimeError::Terminal)
        ));
    }

    #[test]
    fn suppressed_lower_tool_index_unblocks_ordered_commit() {
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(TenantId::new(), request(), 2);
        runtime
            .world
            .get_mut::<ToolBatch>(handle.entity)
            .expect("batch")
            .completed
            .insert(
                1,
                (
                    "second".to_string(),
                    ToolResult::success(ToolOutput::text("two")),
                ),
            );
        runtime
            .apply_invalid_tool_policy(
                handle,
                0,
                "missing",
                InvalidToolPolicy::Skip {
                    reason: "policy".to_string(),
                },
            )
            .expect("skip");
        let indexes = runtime
            .events(handle)
            .expect("events")
            .iter()
            .filter_map(|event| match event {
                RuntimeEvent::ToolCommitted { call_index, .. } => Some(*call_index),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(indexes, vec![0, 1]);
    }

    #[test]
    fn terminal_invalid_policy_rejects_already_dispatched_sibling() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 2);
        let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("v1")));
        runtime.grant_tool(handle, capability).expect("grant");
        let snapshot = runtime
            .snapshot_tools(handle, [capability])
            .expect("snapshot");
        let sibling = runtime
            .dispatch_tool(handle, &snapshot, 1, "lookup", "{}".to_string())
            .expect("sibling");
        assert!(matches!(
            runtime
                .apply_invalid_tool_policy(handle, 0, "missing", InvalidToolPolicy::Fail)
                .expect("policy"),
            InvalidToolResolution::Failed
        ));
        let ingress = ToolIngress {
            world_id: sibling.world_id,
            tenant_id: sibling.tenant_id,
            run_id: sibling.run_id,
            operation_id: sibling.operation_id,
            correlation_id: sibling.correlation_id,
            generation: sibling.generation,
            call_index: sibling.call_index,
            capability_id: sibling.capability_id,
            revision: sibling.revision,
            name: sibling.name,
            result: ToolResult::success(ToolOutput::text("late")),
        };
        assert!(matches!(
            runtime.submit_tool(ingress),
            Err(RuntimeError::StaleToolCompletion)
        ));
        assert!(matches!(
            runtime.events(handle).expect("events").last(),
            Some(RuntimeEvent::Terminal { .. })
        ));
    }

    #[test]
    fn every_stale_completion_class_preserves_authoritative_state() {
        let tenant = TenantId::new();
        let mut runtime = Runtime::new();
        let handle = runtime.spawn_run(tenant, request(), 2);
        let effect = runtime.dispatch(handle).expect("dispatch");
        let mut wrong_generation = completion(&effect, "wrong", 9);
        wrong_generation.generation += 1;
        assert!(matches!(
            runtime.submit(wrong_generation),
            Err(RuntimeError::StaleCompletion)
        ));
        let mut wrong_tenant = completion(&effect, "wrong", 9);
        wrong_tenant.tenant_id = TenantId::new();
        assert!(matches!(
            runtime.submit(wrong_tenant),
            Err(RuntimeError::WrongTenant)
        ));
        let rejected = completion(&effect, "rejected", 3);
        runtime
            .retry_response(rejected.clone(), "try again")
            .expect("retry");
        assert!(matches!(
            runtime.submit(rejected),
            Err(RuntimeError::StaleCompletion | RuntimeError::NoPendingOperation)
        ));
        let next = runtime.dispatch(handle).expect("fresh generation");
        runtime
            .submit(completion(&next, "accepted", 4))
            .expect("accept");
        let snapshot = runtime.snapshot(handle).expect("snapshot");
        assert_eq!(snapshot.completed_calls, 2);
        assert_eq!(snapshot.usage.total_tokens, 7);
        assert!(snapshot.output.iter().any(
            |content| matches!(content, AssistantContent::Text(text) if text.text == "accepted")
        ));
    }

    struct BevyAdapter;

    fn concrete_conformance_evidence(scenario: Scenario) -> Outcome {
        fn observed(runtime: &Runtime, handle: RuntimeHandle) -> Outcome {
            let snapshot = runtime.snapshot(handle).expect("observable stable state");
            let terminal_reason = match snapshot.status {
                RunStatus::Terminal(TerminalReason::Completed) => "completed",
                RunStatus::Terminal(TerminalReason::Cancelled { .. }) => "cancelled",
                RunStatus::Terminal(TerminalReason::Failed { .. }) => "failed",
                RunStatus::Terminal(TerminalReason::BudgetExhausted) => "budget_exhausted",
                RunStatus::Terminal(TerminalReason::Livelock) => "livelock",
                _ => "nonterminal",
            };
            let events = runtime.events(handle).expect("events");
            Outcome {
                history_messages: snapshot.history.len(),
                committed_history: snapshot.history,
                usage: snapshot.usage,
                model_calls: snapshot.completed_calls,
                tool_calls: events
                    .iter()
                    .filter_map(|event| match event {
                        RuntimeEvent::ToolCommitted { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
                tool_call_count: events
                    .iter()
                    .filter(|event| matches!(event, RuntimeEvent::ToolCommitted { .. }))
                    .count(),
                terminal_reason: terminal_reason.to_string(),
                provisional_events: events
                    .iter()
                    .filter(|event| matches!(event, RuntimeEvent::ProvisionalDelta { .. }))
                    .count(),
                rejected_effects: snapshot.rejected_effects,
                output_modes_observed: 0,
                observations: Default::default(),
            }
        }

        fn completed() -> (Runtime, RuntimeHandle) {
            let mut runtime = Runtime::new();
            let handle = runtime.spawn_run(TenantId::new(), request(), 2);
            let effect = runtime.dispatch(handle).expect("dispatch");
            runtime
                .submit(completion(&effect, "accepted", 2))
                .expect("completion");
            (runtime, handle)
        }

        let (runtime, handle) = completed();
        let mut outcome = observed(&runtime, handle);
        match scenario {
            Scenario::ModelCallBudgets => {
                let mut runtime = Runtime::new();
                let handle = runtime.spawn_run(TenantId::new(), request(), 2);
                let first = runtime.dispatch(handle).expect("first");
                runtime
                    .retry_response(completion(&first, "rejected", 1), "retry")
                    .expect("retry");
                let second = runtime.dispatch(handle).expect("second");
                runtime
                    .submit(completion(&second, "accepted", 1))
                    .expect("accepted");
                outcome = observed(&runtime, handle);
            }
            Scenario::CanonicalTranscript | Scenario::ProvisionalStreaming => {
                let mut runtime = Runtime::new();
                let handle = runtime.spawn_run(TenantId::new(), request(), 1);
                let effect = runtime.dispatch(handle).expect("dispatch");
                runtime
                    .submit_stream(StreamingIngress::Delta {
                        world_id: effect.world_id,
                        tenant_id: effect.tenant_id,
                        run_id: effect.run_id,
                        operation_id: effect.operation_id,
                        correlation_id: effect.correlation_id,
                        generation: effect.generation,
                        sequence: 0,
                        content: AssistantContent::text("provisional"),
                    })
                    .expect("delta");
                if scenario == Scenario::CanonicalTranscript {
                    runtime
                        .submit(completion(&effect, "accepted", 2))
                        .expect("accepted");
                } else {
                    runtime
                        .submit_stream(StreamingIngress::ProviderFailure {
                            world_id: effect.world_id,
                            tenant_id: effect.tenant_id,
                            run_id: effect.run_id,
                            operation_id: effect.operation_id,
                            correlation_id: effect.correlation_id,
                            generation: effect.generation,
                            message: "failed".to_string(),
                        })
                        .expect("failure");
                }
                outcome = observed(&runtime, handle);
            }
            Scenario::ToolPairing | Scenario::Concurrency => {
                let tenant = TenantId::new();
                let mut runtime = Runtime::new();
                let handle = runtime.spawn_run(tenant, request(), 2);
                let capability = runtime.register_tool(tenant, Arc::new(VersionedTool("v1")));
                runtime.grant_tool(handle, capability).expect("grant");
                let snapshot = runtime
                    .snapshot_tools(handle, [capability])
                    .expect("snapshot");
                let first = runtime
                    .dispatch_tool(handle, &snapshot, 0, "lookup", "{}".to_string())
                    .expect("first");
                let second = runtime
                    .dispatch_tool(handle, &snapshot, 1, "lookup", "{}".to_string())
                    .expect("second");
                for effect in [second, first] {
                    runtime
                        .submit_tool(ToolIngress {
                            world_id: effect.world_id,
                            tenant_id: effect.tenant_id,
                            run_id: effect.run_id,
                            operation_id: effect.operation_id,
                            correlation_id: effect.correlation_id,
                            generation: effect.generation,
                            call_index: effect.call_index,
                            capability_id: effect.capability_id,
                            revision: effect.revision,
                            name: effect.name,
                            result: ToolResult::success(ToolOutput::text("done")),
                        })
                        .expect("tool ingress");
                }
                outcome.tool_calls = runtime
                    .events(handle)
                    .expect("events")
                    .iter()
                    .filter_map(|event| match event {
                        RuntimeEvent::ToolCommitted { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect();
                outcome.tool_call_count = outcome.tool_calls.len();
            }
            Scenario::UsageAccounting => {}
            Scenario::InvalidToolRecovery | Scenario::ToolSuppression => {
                let policies = [
                    InvalidToolPolicy::Fail,
                    InvalidToolPolicy::Retry,
                    InvalidToolPolicy::Repair {
                        replacement_name: "safe".to_string(),
                        replacement_arguments: "{}".to_string(),
                    },
                    InvalidToolPolicy::Skip {
                        reason: "skip".to_string(),
                    },
                    InvalidToolPolicy::Stop {
                        reason: "stop".to_string(),
                    },
                ];
                outcome.rejected_effects = policies
                    .into_iter()
                    .map(|policy| {
                        let mut runtime = Runtime::new();
                        let handle = runtime.spawn_run(TenantId::new(), request(), 2);
                        runtime
                            .world
                            .get_mut::<ToolContinuation>(handle.entity)
                            .expect("continuation")
                            .0 = Some(request());
                        runtime
                            .apply_invalid_tool_policy(handle, 0, "missing", policy)
                            .expect("policy");
                        runtime.explain(handle).expect("explain").rejected_effects
                    })
                    .sum();
                outcome.tool_calls.clear();
                outcome.tool_call_count = 0;
            }
            Scenario::ResponseRetryRollback => {
                let mut runtime = Runtime::new();
                let handle = runtime.spawn_run(TenantId::new(), request(), 2);
                let first = runtime.dispatch(handle).expect("first");
                runtime
                    .retry_response(completion(&first, "rejected", 1), "retry")
                    .expect("retry");
                let second = runtime.dispatch(handle).expect("second");
                runtime
                    .submit(completion(&second, "accepted", 1))
                    .expect("accepted");
                outcome = observed(&runtime, handle);
            }
            Scenario::StopAndCancellation => {
                let mut runtime = Runtime::new();
                let handle = runtime.spawn_run(TenantId::new(), request(), 1);
                runtime.cancel(handle, "cancelled").expect("cancel");
                outcome = observed(&runtime, handle);
            }
            Scenario::StructuredOutput => {
                let mut runtime = Runtime::new();
                let handle = runtime.spawn_run(TenantId::new(), request(), 2);
                let first = runtime.dispatch(handle).expect("first");
                runtime
                    .retry_response(completion(&first, "invalid", 1), "retry")
                    .expect("retry");
                let second = runtime.dispatch(handle).expect("second");
                runtime
                    .submit(completion(&second, "accepted", 1))
                    .expect("accepted");
                outcome = observed(&runtime, handle);
            }
            Scenario::Memory
            | Scenario::BlockingStreamingParity
            | Scenario::ProviderFinalExposure => {}
            Scenario::StaleResultHandling => {
                let mut runtime = Runtime::new();
                let handle = runtime.spawn_run(TenantId::new(), request(), 1);
                let effect = runtime.dispatch(handle).expect("dispatch");
                let mut rejected = 0;
                let mut wrong_generation = completion(&effect, "wrong", 1);
                wrong_generation.generation += 1;
                rejected += usize::from(runtime.submit(wrong_generation).is_err());
                let mut wrong_tenant = completion(&effect, "wrong", 1);
                wrong_tenant.tenant_id = TenantId::new();
                rejected += usize::from(runtime.submit(wrong_tenant).is_err());
                runtime
                    .submit(completion(&effect, "accepted", 1))
                    .expect("accepted");
                rejected += usize::from(runtime.submit(completion(&effect, "late", 1)).is_err());
                let foreign = Runtime::new();
                let mut foreign_completion = completion(&effect, "foreign", 1);
                foreign_completion.world_id = foreign.world_id;
                rejected += usize::from(runtime.submit(foreign_completion).is_err());
                outcome = observed(&runtime, handle);
                outcome.rejected_effects = rejected;
            }
        }
        outcome
    }

    impl Adapter for BevyAdapter {
        type Error = std::convert::Infallible;

        fn run(&mut self, scenario: Scenario) -> Result<Outcome, Self::Error> {
            let mut executed_output_modes = 0;
            let observations = match scenario {
                Scenario::ModelCallBudgets => {
                    let mut runtime = Runtime::new();
                    let zero = runtime.spawn_run(TenantId::new(), request(), 0);
                    assert!(matches!(
                        runtime.dispatch(zero),
                        Err(RuntimeError::Terminal)
                    ));
                    response_retry_rolls_back_content_and_consumes_total_budget();
                    [
                        Observation::ZeroBudgetRejected,
                        Observation::ExactBudgetObserved,
                    ]
                    .into()
                }
                Scenario::CanonicalTranscript => {
                    provisional_deltas_are_observable_but_never_committed();
                    [
                        Observation::TranscriptValid,
                        Observation::ProvisionalExcluded,
                    ]
                    .into()
                }
                Scenario::ToolPairing => {
                    tool_completions_commit_in_call_order_not_arrival_order();
                    [Observation::CallsPaired, Observation::CommitOrderStable].into()
                }
                Scenario::UsageAccounting => {
                    duplicate_completion_cannot_double_commit_usage();
                    [Observation::AccountingIdempotent].into()
                }
                Scenario::InvalidToolRecovery => {
                    every_invalid_tool_policy_is_explicit_and_suppresses_execution();
                    [
                        Observation::AllInvalidPoliciesObserved,
                        Observation::SuppressedToolNotExecuted,
                    ]
                    .into()
                }
                Scenario::ResponseRetryRollback => {
                    response_retry_rolls_back_content_and_consumes_total_budget();
                    [
                        Observation::RetryRolledBack,
                        Observation::RetryUsedTotalBudget,
                    ]
                    .into()
                }
                Scenario::StopAndCancellation => {
                    cancellation_rejects_late_completion();
                    [
                        Observation::LateCommitRejected,
                        Observation::CancellationRetained,
                    ]
                    .into()
                }
                Scenario::StructuredOutput => {
                    executed_output_modes =
                        crate::local::tests::all_structured_output_modes_execute();
                    assert_eq!(
                        crate::synthetic_output_tool_name(["__rig_structured_output"]),
                        "__rig_structured_output_2"
                    );
                    assert!(RecoveryPolicy::default().max_response_retries > 0);
                    crate::local::tests::local_tool_output_is_collision_safe_and_suppresses_execution();
                    crate::local::tests::structured_output_recovers_within_total_budget();
                    crate::local::tests::structured_output_exhaustion_preserves_all_billed_usage();
                    [
                        Observation::AllOutputModesObserved,
                        Observation::StructuredRecoveryBounded,
                    ]
                    .into()
                }
                Scenario::Memory => {
                    let tenant = TenantId::new();
                    let mut runtime = Runtime::new();
                    let handle = runtime.spawn_run(tenant, request(), 1);
                    let memory = Arc::new(RecordingMemory::default());
                    let load = runtime.bind_memory(handle, "shared", memory).expect("bind");
                    assert!(matches!(
                        runtime.dispatch(handle),
                        Err(RuntimeError::MemoryNotLoaded)
                    ));
                    runtime
                        .submit_memory_load(MemoryLoadIngress {
                            world_id: load.world_id,
                            tenant_id: load.tenant_id,
                            run_id: load.run_id,
                            operation_id: load.operation_id,
                            correlation_id: load.correlation_id,
                            generation: load.generation,
                            store_id: load.store_id,
                            result: Ok(Vec::new()),
                        })
                        .expect("load");
                    let effect = runtime.dispatch(handle).expect("dispatch");
                    runtime
                        .submit(completion(&effect, "done", 1))
                        .expect("commit");
                    assert_eq!(runtime.take_memory_appends().count(), 1);
                    [
                        Observation::MemoryLoadedBeforeModel,
                        Observation::MemoryCommittedOnly,
                    ]
                    .into()
                }
                Scenario::BlockingStreamingParity => {
                    let tenant = TenantId::new();
                    let mut blocking = Runtime::new();
                    let blocking_handle = blocking.spawn_run(tenant, request(), 1);
                    let blocking_effect = blocking.dispatch(blocking_handle).expect("blocking");
                    blocking
                        .submit(completion(&blocking_effect, "same", 2))
                        .expect("blocking commit");
                    let mut streaming = Runtime::new();
                    let streaming_handle = streaming.spawn_run(tenant, request(), 1);
                    let streaming_effect = streaming.dispatch(streaming_handle).expect("streaming");
                    streaming
                        .submit_stream(StreamingIngress::ProviderFinal {
                            completion: completion(&streaming_effect, "same", 2),
                            final_response: ProviderFinalEnvelope {
                                provider: "scripted".into(),
                                type_name: "Final".into(),
                                diagnostic_json: serde_json::json!({}),
                            },
                        })
                        .expect("streaming commit");
                    let left = blocking.snapshot(blocking_handle).expect("left");
                    let right = streaming.snapshot(streaming_handle).expect("right");
                    assert_eq!(left.history, right.history);
                    assert_eq!(left.usage, right.usage);
                    assert_eq!(left.status, right.status);
                    [Observation::BlockingStreamingEqual].into()
                }
                Scenario::ProviderFinalExposure => {
                    fn typed_surface<R>(_: Option<crate::LocalStreamingResult<R>>) {}
                    typed_surface::<rig_core::test_utils::MockResponse>(None);
                    provider_failure_after_final_looking_delta_never_exposes_success();
                    [
                        Observation::TypedProviderFinalObserved,
                        Observation::ProviderFailureNotSuccess,
                    ]
                    .into()
                }
                Scenario::ProvisionalStreaming => {
                    provisional_deltas_are_observable_but_never_committed();
                    provider_failure_after_final_looking_delta_never_exposes_success();
                    [
                        Observation::ProvisionalExcluded,
                        Observation::ProvisionalRollbackObserved,
                    ]
                    .into()
                }
                Scenario::ToolSuppression => {
                    every_invalid_tool_policy_is_explicit_and_suppresses_execution();
                    tool_queue_is_bounded_and_terminal_runs_suppress_dispatch();
                    [
                        Observation::AllSuppressionCausesObserved,
                        Observation::SuppressedToolNotExecuted,
                    ]
                    .into()
                }
                Scenario::Concurrency => {
                    tool_queue_is_bounded_and_terminal_runs_suppress_dispatch();
                    tool_completions_commit_in_call_order_not_arrival_order();
                    [
                        Observation::ConcurrencyBounded,
                        Observation::CommitOrderStable,
                        Observation::SiblingPolicyDeterministic,
                    ]
                    .into()
                }
                Scenario::StaleResultHandling => {
                    duplicate_completion_cannot_double_commit_usage();
                    cancellation_rejects_late_completion();
                    foreign_world_effect_is_rejected();
                    foreign_tenant_cannot_receive_or_snapshot_capability();
                    every_stale_completion_class_preserves_authoritative_state();
                    [
                        Observation::AllStaleClassesRejected,
                        Observation::AccountingIdempotent,
                    ]
                    .into()
                }
            };
            let mut outcome = concrete_conformance_evidence(scenario);
            outcome.observations = observations;
            outcome.output_modes_observed = executed_output_modes;
            Ok(outcome)
        }
    }

    #[test]
    fn shared_conformance_ledger_passes_for_bevy_adapter() {
        let mut adapter = BevyAdapter;
        for scenario in Scenario::ALL {
            let outcome = adapter.run(scenario).expect("infallible adapter");
            assert_outcome(scenario, &outcome).expect("shared contract");
        }
    }
}
