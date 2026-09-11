//! Read-only views derived from the run graph. Use [`RunView`] in ordinary
//! systems, and [`inspect`] when the host holds a World. Neither stores a second
//! mutable lifecycle state. Detailed effect inspection walks only the selected
//! run's descendants, and only when requested.

use bevy_ecs::prelude::*;

use crate::{
    agent::{
        Assembling, AwaitingModel, Failed, Failure, LoadingMemory, ResolvingTools, Run, RunOf,
        RunResult, Settled, Usage,
    },
    bus::{EffectOutcome, Held, HoldOwners, InFlight, PendingEffect},
    commands::OperationError,
};

/// The state currently represented by the run's components, not a stall diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Ready to assemble the next turn.
    Ready,
    /// Loading the configured conversation.
    LoadingMemory,
    /// Waiting for a model effect; it may be held or in flight.
    AwaitingModel,
    /// Resolving a tool batch; individual effects may be held or in flight.
    ResolvingTools,
    /// A successful terminal result.
    Settled,
    /// A terminal failure.
    Failed,
    /// No recognized active phase, or conflicting active markers.
    Unknown,
}

fn status(
    failed: bool,
    settled: bool,
    assembling: bool,
    memory: bool,
    model: bool,
    tools: bool,
) -> RunStatus {
    if failed {
        return RunStatus::Failed;
    }
    if settled {
        return RunStatus::Settled;
    }
    if [assembling, memory, model, tools]
        .into_iter()
        .filter(|active| *active)
        .count()
        != 1
    {
        return RunStatus::Unknown;
    }
    if memory {
        RunStatus::LoadingMemory
    } else if model {
        RunStatus::AwaitingModel
    } else if tools {
        RunStatus::ResolvingTools
    } else {
        RunStatus::Ready
    }
}

/// A named query for normal systems: `Query<RunView>` needs no lifetime annotations.
#[derive(bevy_ecs::query::QueryData)]
pub struct RunView {
    /// Run identity.
    pub entity: Entity,
    /// The owning agent.
    pub agent: &'static RunOf,
    /// The answer, if published.
    pub result: Option<&'static RunResult>,
    /// The structured failure, if published.
    pub failure: Option<&'static Failed>,
    /// Accumulated usage, when present.
    pub usage: Option<&'static Usage>,
    run: &'static Run,
    settled: Has<Settled>,
    assembling: Has<Assembling>,
    memory: Has<LoadingMemory>,
    model: Has<AwaitingModel>,
    tools: Has<ResolvingTools>,
}

impl<'w, 's> RunViewItem<'w, 's> {
    /// The phase represented by the current components.
    pub fn status(&self) -> RunStatus {
        status(
            self.failure.is_some(),
            self.settled,
            self.assembling,
            self.memory,
            self.model,
            self.tools,
        )
    }

    /// Whether the run has a terminal ending, successful or failed.
    pub fn is_finished(&self) -> bool {
        self.settled || self.failure.is_some()
    }

    /// Borrow the current answer without copying it.
    pub fn answer(&self) -> Option<&'w str> {
        self.result.map(|result| result.0.as_str())
    }
}

/// An on-demand borrowed view for a host holding the World.
pub struct RunInfo<'w> {
    world: &'w World,
    entity: Entity,
}

/// Inspect a live run without scanning the world or copying its messages.
pub fn inspect(world: &World, entity: Entity) -> Result<RunInfo<'_>, OperationError> {
    let target = world
        .get_entity(entity)
        .map_err(|_| OperationError::MissingEntity(entity))?;
    if !target.contains::<Run>() || !target.contains::<RunOf>() {
        return Err(OperationError::NotRun(entity));
    }
    Ok(RunInfo { world, entity })
}

impl<'w> RunInfo<'w> {
    /// The run identity.
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// The phase represented by the current components.
    pub fn status(&self) -> RunStatus {
        status(
            self.world.get::<Failed>(self.entity).is_some(),
            self.world.get::<Settled>(self.entity).is_some(),
            self.world.get::<Assembling>(self.entity).is_some(),
            self.world.get::<LoadingMemory>(self.entity).is_some(),
            self.world.get::<AwaitingModel>(self.entity).is_some(),
            self.world.get::<ResolvingTools>(self.entity).is_some(),
        )
    }

    /// Whether the run has a terminal ending.
    pub fn is_finished(&self) -> bool {
        matches!(self.status(), RunStatus::Settled | RunStatus::Failed)
    }

    /// Borrow the published answer.
    pub fn answer(&self) -> Option<&'w str> {
        self.world
            .get::<RunResult>(self.entity)
            .map(|result| result.0.as_str())
    }

    /// Borrow the structured failure, retaining its underlying provider evidence.
    pub fn failure(&self) -> Option<&'w Failure> {
        self.world
            .get::<Failed>(self.entity)
            .map(|failure| &failure.0)
    }

    /// Borrow accumulated usage.
    pub fn usage(&self) -> Option<&'w rig_core::completion::Usage> {
        self.world.get::<Usage>(self.entity).map(|usage| &usage.0)
    }

    /// Inspect effects in this run's subtree. Held/in-flight/outcome are facts;
    /// a hold is not assumed to be a human approval or a particular stall reason.
    pub fn effects(&self) -> impl Iterator<Item = EffectInfo<'w>> + 'w {
        let world = self.world;
        let mut stack = vec![self.entity];
        let mut effects = Vec::new();
        while let Some(entity) = stack.pop() {
            if let Some(children) = world.get::<Children>(entity) {
                stack.extend(children.iter().rev());
            }
            if let Some(effect) = world.get::<PendingEffect>(entity) {
                effects.push(EffectInfo {
                    entity,
                    effect,
                    outcome: world.get::<EffectOutcome>(entity),
                    held: world.get::<Held>(entity).is_some(),
                    in_flight: world.get::<InFlight>(entity).is_some(),
                    hold_owners: world.get::<HoldOwners>(entity),
                    approval: world.get::<crate::approval::ApprovalRequest>(entity),
                });
            }
        }
        effects.into_iter()
    }
}

/// Borrowed facts about an effect. Detailed payload access is deliberate;
/// this type does not print arguments or provider responses automatically.
pub struct EffectInfo<'w> {
    /// The approval snapshot, if this effect uses the built-in workflow.
    /// Check `is_pending()`; a resolved snapshot may still be retained.
    pub approval: Option<&'w crate::approval::ApprovalRequest>,
    /// Effect identity in this World.
    pub entity: Entity,
    /// Handler key and proposed operation.
    pub effect: &'w PendingEffect,
    /// Current consumer-visible outcome, possibly rewritten after recording.
    pub outcome: Option<&'w EffectOutcome>,
    /// Dispatch is held.
    pub held: bool,
    /// Dispatch is in flight.
    pub in_flight: bool,
    /// Named hold owners, when present.
    pub hold_owners: Option<&'w HoldOwners>,
}
