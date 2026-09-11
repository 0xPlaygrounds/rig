//! Checked operations on the authoritative run graph.

use bevy_ecs::{
    entity::{EntityCloner, EntityHashMap},
    prelude::*,
};

use crate::{
    agent::{
        Assembling, AwaitingModel, Batch, Cancelled, Cursor, Failed, InvalidCall, InvalidRetries,
        LoadingMemory, MemoryAppendScheduled, OutputRetries, OutputToolName, Remembering,
        ResolvingTools, Retrieving, Run, RunCounter, RunOf, RunResult, RunSeq, Settled, Turn,
        Usage,
    },
    bus::{EffectOutcome, InFlight, PendingEffect, Scope, Serving, Streaming},
    commands::OperationError,
    systems::{Fresh, Materialised},
};

/// The result of requesting run cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    /// The cancellation component was applied.
    Cancelled,
    /// The run already had a terminal result; its ending was preserved.
    AlreadyFinished,
}

fn run(world: &World, entity: Entity) -> Result<EntityRef<'_>, OperationError> {
    let target = world
        .get_entity(entity)
        .map_err(|_| OperationError::MissingEntity(entity))?;
    if !target.contains::<Run>() || !target.contains::<RunOf>() {
        return Err(OperationError::NotRun(entity));
    }
    Ok(target)
}

/// Cancel a live run. Never-issued effects are removed; issued work remains
/// owned by its handler. A finished run keeps its existing result or failure.
pub fn cancel(
    world: &mut World,
    entity: Entity,
    reason: impl Into<String>,
) -> Result<CancelOutcome, OperationError> {
    crate::commands::installed(world)?;
    let target = run(world, entity)?;
    if target.contains::<Settled>() || target.contains::<Failed>() {
        return Ok(CancelOutcome::AlreadyFinished);
    }
    world.entity_mut(entity).insert(Cancelled(reason.into()));
    Ok(CancelOutcome::Cancelled)
}

/// Fork a run before its first turn or between fully materialized turns.
///
/// History and ordinary graph components are cloned; effect entities and their
/// descendants are omitted, so completed tools are not dispatched again. Each
/// branch gets a fresh sequence and scope. Remembering runs are refused because
/// cloning their conversation would append both branches to the same store.
///
/// Application components follow Bevy's cloning behavior. This does not isolate
/// arbitrary external handles, application resources, or custom relationships.
/// Validation happens before allocation; application clone hooks and observers
/// are not transactional and may still change or reject the resulting graph.
pub fn fork(world: &mut World, source: Entity) -> Result<Entity, OperationError> {
    crate::commands::installed(world)?;
    let target = run(world, source)?;
    let valid = target.contains::<Assembling>()
        && target.contains::<RunSeq>()
        && target.contains::<Cursor>()
        && target.contains::<OutputRetries>()
        && target.contains::<InvalidRetries>()
        && target.contains::<OutputToolName>()
        && target.contains::<Usage>()
        && !target.contains::<Cancelled>()
        && !target.contains::<Failed>()
        && !target.contains::<Settled>()
        && !target.contains::<RunResult>()
        && !target.contains::<AwaitingModel>()
        && !target.contains::<ResolvingTools>()
        && !target.contains::<LoadingMemory>()
        && !target.contains::<Remembering>()
        && !target.contains::<MemoryAppendScheduled>();
    if !valid {
        return Err(OperationError::UnsafeFork(source));
    }
    let agent = target
        .get::<RunOf>()
        .ok_or(OperationError::NotRun(source))?
        .0;
    let agent_ref = world
        .get_entity(agent)
        .map_err(|_| OperationError::MissingEntity(agent))?;
    if agent_ref.contains::<Run>()
        || agent_ref.contains::<Turn>()
        || !agent_ref.contains::<crate::agent::UsesModel>()
    {
        return Err(OperationError::NotAgent(agent));
    }
    let owner = agent_ref
        .get::<crate::agent::Owner>()
        .ok_or(OperationError::NotAgent(agent))?
        .0
        .clone();

    // Traverse this run only. An effect subtree is checked but never cloned.
    let mut stack = vec![(source, false)];
    let mut graph = Vec::new();
    while let Some((entity, in_effect)) = stack.pop() {
        let node = world
            .get_entity(entity)
            .map_err(|_| OperationError::MissingEntity(entity))?;
        let effect = in_effect || node.contains::<PendingEffect>();
        if (entity != source && node.contains::<Run>())
            || node.contains::<Fresh>()
            || node.contains::<Retrieving>()
            || node.contains::<Batch>()
            || node.contains::<InvalidCall>()
            || node.contains::<InFlight>()
            || node.contains::<Serving>()
            || node.contains::<Streaming>()
            || (node.contains::<Turn>() && !node.contains::<Materialised>())
            || (node.contains::<PendingEffect>() && !node.contains::<EffectOutcome>())
        {
            return Err(OperationError::UnsafeFork(source));
        }
        if !effect {
            graph.push(entity);
        }
        if let Some(children) = node.get::<Children>() {
            stack.extend(children.iter().rev().map(|child| (child, effect)));
        }
    }
    let sequence = world.resource::<RunCounter>().0;
    let next = sequence
        .checked_add(1)
        .ok_or(OperationError::SequenceExhausted)?;
    world.resource_mut::<RunCounter>().0 = next;
    let mut mapped = EntityHashMap::default();
    for entity in &graph {
        mapped.insert(*entity, world.spawn_empty().id());
    }
    let Some(&clone) = mapped.get(&source) else {
        remove_clones(world, &mapped);
        return Err(OperationError::UnsafeFork(source));
    };
    world
        .entity_mut(clone)
        .insert((RunSeq(sequence), Scope(format!("{owner}/run#{sequence}"))));
    let mut builder = EntityCloner::build_opt_out(world);
    builder.deny::<(Children, RunSeq, Scope)>();
    let mut cloner = builder.finish();
    for entity in graph {
        // Binding/clone observers may remove a destination or a later source.
        // Do not attach further children to a removed root.
        let Some(&destination) = mapped.get(&entity) else {
            remove_clones(world, &mapped);
            return Err(OperationError::UnsafeFork(source));
        };
        for required in [clone, entity, destination] {
            if world.get_entity(required).is_err() {
                remove_clones(world, &mapped);
                return Err(OperationError::MissingEntity(required));
            }
        }
        cloner.clone_entity_mapped(world, entity, &mut mapped);
    }
    if let Err(error) = run(world, clone) {
        remove_clones(world, &mapped);
        return Err(error);
    }
    Ok(clone)
}

fn remove_clones(world: &mut World, mapped: &EntityHashMap<Entity>) {
    for entity in mapped.values() {
        if world.get_entity(*entity).is_ok() {
            world.despawn(*entity);
        }
    }
}

fn agent(world: &World, entity: Entity) -> Result<(), OperationError> {
    let target = world
        .get_entity(entity)
        .map_err(|_| OperationError::MissingEntity(entity))?;
    if target.contains::<Run>()
        || target.contains::<Turn>()
        || !target.contains::<crate::agent::Owner>()
        || !target.contains::<crate::agent::UsesModel>()
    {
        return Err(OperationError::NotAgent(entity));
    }
    Ok(())
}

/// Add an ordered static tool grant to an agent. Repeating the same static grant
/// returns the existing link without allocating another or changing order.
/// Retrieval-only links remain separate. Pending handler
/// registration is supported. Existing turn snapshots and issued work are unchanged.
pub fn grant_tool(
    world: &mut World,
    owner: Entity,
    tool: Entity,
) -> Result<Entity, OperationError> {
    crate::commands::installed(world)?;
    agent(world, owner)?;
    crate::commands::handler(world, tool, rig_core::effect::EffectFamily::Tool)?;
    crate::bus::handlers::materialize_registration(world, tool)
        .map_err(|_| OperationError::MissingEntity(tool))?;
    // Registration observers may remove either target.
    agent(world, owner)?;
    crate::commands::handler(world, tool, rig_core::effect::EffectFamily::Tool)?;
    if let Some(children) = world.get::<Children>(owner)
        && let Some(link) = children.iter().find(|child| {
            world.get::<crate::agent::Retrievable>(*child).is_none()
                && world
                    .get::<crate::agent::Grant>(*child)
                    .is_some_and(|grant| grant.0 == tool)
        })
    {
        return Ok(link);
    }
    let order = crate::commands::order(world)?;
    crate::commands::child(world, owner, (crate::agent::Grant(tool), order))
}

/// Remove all grants for this tool from the agent and return the number removed.
/// Repeated removal returns zero. A removed handler is allowed, so stale grants
/// can be cleaned up. Existing turn snapshots and effects are not revoked.
pub fn revoke_tool(
    world: &mut World,
    owner: Entity,
    tool: Entity,
) -> Result<usize, OperationError> {
    crate::commands::installed(world)?;
    agent(world, owner)?;
    let links: Vec<_> = world
        .get::<Children>(owner)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter(|child| {
            world
                .get::<crate::agent::Grant>(*child)
                .is_some_and(|grant| grant.0 == tool)
        })
        .collect();
    let mut removed = 0;
    for link in links {
        if world.get_entity(link).is_ok() {
            world.despawn(link);
            removed += 1;
        }
    }
    Ok(removed)
}

fn active_turn(
    world: &World,
    entity: Entity,
    operation: &'static str,
) -> Result<Entity, OperationError> {
    let target = world
        .get_entity(entity)
        .map_err(|_| OperationError::MissingEntity(entity))?;
    if !target.contains::<Turn>() {
        return Err(OperationError::NotTurn(entity));
    }
    let owner = target
        .get::<ChildOf>()
        .ok_or(OperationError::NotTurn(entity))?
        .parent();
    let run = run(world, owner)?;
    if run.contains::<Cancelled>() || run.contains::<Failed>() || run.contains::<Settled>() {
        return Err(OperationError::InvalidPhase { entity, operation });
    }
    Ok(owner)
}

/// Request retry of complete, tool-free output before Materialise reads it.
/// Provider errors still take precedence. Identical pending retries are no-ops;
/// conflicting feedback is an error. This neither revives a finished run nor
/// bypasses its turn budget. Use `Retry::default().feedback("...")` for feedback.
pub fn retry_turn(
    world: &mut World,
    turn: Entity,
    retry: crate::agent::Retry,
) -> Result<(), OperationError> {
    crate::commands::installed(world)?;
    let owner = active_turn(world, turn, "retry")?;
    let eligible = world.get::<AwaitingModel>(owner).is_some()
        && world.get::<Materialised>(turn).is_none()
        && world.get::<crate::systems::Folded>(turn).is_some()
        && world
            .get::<crate::agent::Outputs>(turn)
            .is_some_and(|outputs| {
                outputs.done
                    && !outputs.content.iter().any(|part| {
                        matches!(part, rig_core::message::AssistantContent::ToolCall(_))
                    })
            });
    if !eligible {
        return Err(OperationError::InvalidPhase {
            entity: turn,
            operation: "retry",
        });
    }
    if let Some(previous) = world.get::<crate::agent::Retry>(turn) {
        return if previous == &retry {
            Ok(())
        } else {
            Err(OperationError::ConflictingRetry(turn))
        };
    }
    world.entity_mut(turn).insert(retry);
    Ok(())
}

/// Merge a patch on a Fresh turn before Assemble. Calls compose in application
/// order using RequestPatch::merge (including tool-set intersection and appended
/// context). Repeating an additive patch appends its context again; this operation
/// does not promise idempotency or alter an already-folded request.
pub fn patch_turn(
    world: &mut World,
    turn: Entity,
    patch: crate::agent::RequestPatch,
) -> Result<(), OperationError> {
    crate::commands::installed(world)?;
    active_turn(world, turn, "patch")?;
    if world.get::<Fresh>(turn).is_none() || world.get::<Materialised>(turn).is_some() {
        return Err(OperationError::InvalidPhase {
            entity: turn,
            operation: "patch",
        });
    }
    let patch = world
        .get::<crate::agent::RequestPatch>(turn)
        .cloned()
        .unwrap_or_default()
        .merge(patch);
    world.entity_mut(turn).insert(patch);
    Ok(())
}
