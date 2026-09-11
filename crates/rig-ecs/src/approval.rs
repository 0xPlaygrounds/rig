//! Human decisions on tool effects, using the existing owned hold barrier.
//!
//! Insert [`ApprovalRequired`] on a tool effect in `BusSet::Gate`. After Gate
//! and before Dispatch, the runtime captures its proposal and acquires a hold.
//! Query [`ApprovalRequest`] on later passes and feed its ticket to [`decide`]
//! from CLI, UI, or test code. No input source or executor is installed here.
//!
//! A changed key, tool name, arguments, owner, or run invalidates the previous
//! ticket and requires a new decision. Gate policies must finish their edits
//! before the approval guard; do not mutate proposals between it and Dispatch.
//! Approval is not a transaction over external source state: handlers still
//! validate their own preconditions. Requests have no automatic expiration.

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectKind, HandlerKey},
    error::{ErrorKind, ErrorReport},
    observe::Emitter,
};

use crate::{
    agent::{Cancelled, Failed, Run, Settled},
    bus::{EffectOutcome, HoldOwners, Issued, PendingEffect, acquire_hold, release_hold},
};

/// Ask for human approval before this tool effect dispatches. Use a distinct
/// owner name for this workflow; other owners' holds are never released by it.
#[derive(Component, Clone)]
pub struct ApprovalRequired(pub Emitter);

/// Identity of one proposal revision in this world. Keep the ticket that was
/// displayed to the reviewer; never replace it with the latest ticket on input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApprovalTicket {
    effect: Entity,
    world: bevy_ecs::world::WorldId,
    revision: u64,
}

impl ApprovalTicket {
    /// The exact effect entity, including its generation.
    pub fn effect(self) -> Entity {
        self.effect
    }
    /// A monotonically allocated proposal revision in this world.
    pub fn revision(self) -> u64 {
        self.revision
    }
}

/// A reviewable snapshot on the effect entity. Resolved requests remain until
/// the effect is removed or its proposal changes, making repeat input explicit.
/// This transient UI state is not a persistence or replay authorization token.
#[derive(Component, Clone)]
pub struct ApprovalRequest {
    ticket: ApprovalTicket,
    proposal: Proposal,
    decision: Option<ApprovalChoice>,
}

impl ApprovalRequest {
    /// The identity to return with the decision.
    pub fn ticket(&self) -> ApprovalTicket {
        self.ticket
    }
    /// The owning run at capture time.
    pub fn run(&self) -> Entity {
        self.proposal.run
    }
    /// The selected handler key.
    pub fn key(&self) -> &HandlerKey {
        &self.proposal.key
    }
    /// The proposed tool name.
    pub fn name(&self) -> &str {
        &self.proposal.name
    }
    /// The exact wire arguments shown for this revision.
    pub fn args(&self) -> &str {
        &self.proposal.args
    }
    /// Whether this revision is waiting for input.
    pub fn is_pending(&self) -> bool {
        self.decision.is_none()
    }
}

/// A human's decision about the displayed proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalChoice {
    /// Release this workflow's hold; other policies may still prevent dispatch.
    Approve,
    /// Return a denied tool result without dispatching it.
    Deny(String),
    /// Cancel the owning run under its usual issued-work cancellation contract.
    Cancel(String),
}

/// Outcome of applying a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionOutcome {
    /// The decision was applied.
    Applied,
    /// This exact decision was previously applied to this revision.
    AlreadyApplied,
}

/// An approval operation that could not be applied. Failed preparation is also
/// retained as a component and denies dispatch rather than silently proceeding.
#[derive(Component, Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApprovalError {
    /// The effect no longer exists.
    #[error("approval effect {0:?} no longer exists")]
    Missing(Entity),
    /// Approval requires a tool effect below a live run.
    #[error("entity {0:?} is not a tool effect below a live run")]
    InvalidTarget(Entity),
    /// The proposal changed or the ticket belongs to another revision.
    #[error("approval proposal changed; display the new request before deciding")]
    Stale,
    /// The effect dispatched, answered, or its run ended before the decision.
    #[error("the operation can no longer accept an approval decision")]
    TooLate,
    /// A conflicting decision was already applied.
    #[error("this approval revision already has a different decision")]
    AlreadyDecided,
    /// The workflow's hold was removed outside this API.
    #[error("the approval workflow no longer owns its hold")]
    HoldLost,
    /// No further unique proposal revisions can be allocated.
    #[error("approval revision counter exhausted")]
    Exhausted,
}

#[derive(Clone, PartialEq, Eq)]
struct Proposal {
    run: Entity,
    key: HandlerKey,
    name: String,
    args: String,
    owner: Emitter,
}

#[derive(Resource, Default)]
struct Revisions(u64);

fn proposal(world: &World, effect: Entity, active: bool) -> Result<Proposal, ApprovalError> {
    let target = world
        .get_entity(effect)
        .map_err(|_| ApprovalError::Missing(effect))?;
    if active && (target.contains::<Issued>() || target.contains::<EffectOutcome>()) {
        return Err(ApprovalError::TooLate);
    }
    let required = target
        .get::<ApprovalRequired>()
        .ok_or(ApprovalError::Stale)?;
    let pending = target
        .get::<PendingEffect>()
        .ok_or(ApprovalError::InvalidTarget(effect))?;
    let EffectKind::ToolCall { name, args } = &pending.kind else {
        return Err(ApprovalError::InvalidTarget(effect));
    };
    let mut ancestor = effect;
    let mut visited = std::collections::HashSet::new();
    let run = loop {
        if !visited.insert(ancestor) {
            return Err(ApprovalError::InvalidTarget(effect));
        }
        let node = world
            .get_entity(ancestor)
            .map_err(|_| ApprovalError::InvalidTarget(effect))?;
        if node.contains::<Run>() {
            if !node.contains::<crate::agent::RunOf>() {
                return Err(ApprovalError::InvalidTarget(effect));
            }
            if active
                && (node.contains::<Cancelled>()
                    || node.contains::<Failed>()
                    || node.contains::<Settled>())
            {
                return Err(ApprovalError::TooLate);
            }
            break ancestor;
        }
        ancestor = node
            .get::<ChildOf>()
            .ok_or(ApprovalError::InvalidTarget(effect))?
            .parent();
    };
    Ok(Proposal {
        run,
        key: pending.key.clone(),
        name: name.clone(),
        args: args.clone(),
        owner: required.0.clone(),
    })
}

fn owns_hold(world: &World, effect: Entity, owner: &str) -> bool {
    world.get::<crate::bus::Held>(effect).is_some()
        && world
            .get::<HoldOwners>(effect)
            .is_some_and(|holds| holds.owners().any(|item| item.name == owner))
}

fn prepare(world: &mut World, effect: Entity) -> Result<(), ApprovalError> {
    let next = proposal(world, effect, true)?;
    let previous = world.get::<ApprovalRequest>(effect).cloned();
    if let Some(previous) = &previous
        && previous.proposal == next
        && previous.ticket.world == world.id()
        && previous.ticket.effect == effect
    {
        if previous.is_pending() && !owns_hold(world, effect, &next.owner.name) {
            return Err(ApprovalError::HoldLost);
        }
        return Ok(());
    }
    let mut revisions = world.get_resource_or_init::<Revisions>();
    revisions.0 = revisions.0.checked_add(1).ok_or(ApprovalError::Exhausted)?;
    let revision = revisions.0;
    let ticket = ApprovalTicket {
        effect,
        world: world.id(),
        revision,
    };
    acquire_hold(world, effect, next.owner.clone());
    // Hold observers are application code and can change/remove the target.
    if proposal(world, effect, true)? != next {
        return Err(ApprovalError::Stale);
    }
    if !owns_hold(world, effect, &next.owner.name) {
        return Err(ApprovalError::HoldLost);
    }
    world.entity_mut(effect).insert(ApprovalRequest {
        ticket,
        proposal: next.clone(),
        decision: None,
    });
    if let Some(previous) = previous
        && previous.proposal.owner.name != next.owner.name
    {
        release_hold(world, effect, &previous.proposal.owner.name);
    }
    if proposal(world, effect, true)? != next {
        return Err(ApprovalError::Stale);
    }
    let current = world
        .get::<ApprovalRequest>(effect)
        .ok_or(ApprovalError::Stale)?;
    if current.ticket != ticket || current.proposal != next {
        return Err(ApprovalError::Stale);
    }
    if current.is_pending() && !owns_hold(world, effect, &next.owner.name) {
        return Err(ApprovalError::HoldLost);
    }
    Ok(())
}

/// Apply input to the proposal that was actually displayed. This immediate
/// operation can be queued from ordinary Commands. Errors must be reported by
/// the application. A stale response never releases a hold or changes a run.
pub fn decide(
    world: &mut World,
    ticket: ApprovalTicket,
    choice: ApprovalChoice,
) -> Result<DecisionOutcome, ApprovalError> {
    if ticket.world != world.id() {
        return Err(ApprovalError::Stale);
    }
    let entity = world
        .get_entity(ticket.effect)
        .map_err(|_| ApprovalError::Missing(ticket.effect))?;
    let request = entity
        .get::<ApprovalRequest>()
        .ok_or(ApprovalError::Stale)?
        .clone();
    if request.ticket != ticket {
        return Err(ApprovalError::Stale);
    }
    if proposal(world, ticket.effect, false)? != request.proposal {
        return Err(ApprovalError::Stale);
    }
    if let Some(previous) = &request.decision {
        return if previous == &choice {
            Ok(DecisionOutcome::AlreadyApplied)
        } else {
            Err(ApprovalError::AlreadyDecided)
        };
    }
    if proposal(world, ticket.effect, true)? != request.proposal {
        return Err(ApprovalError::Stale);
    }
    if !owns_hold(world, ticket.effect, &request.proposal.owner.name) {
        return Err(ApprovalError::HoldLost);
    }
    if matches!(choice, ApprovalChoice::Cancel(_)) {
        crate::commands::installed(world).map_err(|_| ApprovalError::TooLate)?;
    }
    // Mutating the existing field avoids Insert observers between validation
    // and action. Hold/outcome observers run as part of the action itself.
    world
        .get_mut::<ApprovalRequest>(ticket.effect)
        .ok_or(ApprovalError::Stale)?
        .decision = Some(choice.clone());
    match choice {
        ApprovalChoice::Approve => {
            release_hold(world, ticket.effect, &request.proposal.owner.name);
        }
        ApprovalChoice::Deny(reason) => {
            world
                .entity_mut(ticket.effect)
                .insert(EffectOutcome(Err(ErrorReport::new(
                    ErrorKind::Denied,
                    reason,
                ))));
        }
        ApprovalChoice::Cancel(reason) => {
            crate::lifecycle::cancel(world, request.proposal.run, reason)
                .map_err(|_| ApprovalError::TooLate)?;
        }
    }
    Ok(DecisionOutcome::Applied)
}

type AwaitingApproval = (
    With<ApprovalRequired>,
    Without<Issued>,
    Without<EffectOutcome>,
);

pub(crate) fn guard(effects: Query<Entity, AwaitingApproval>, mut commands: Commands) {
    for effect in &effects {
        commands.queue(move |world: &mut World| {
            if let Err(error) = prepare(world, effect)
                && let Ok(mut target) = world.get_entity_mut(effect)
                && !target.contains::<Issued>()
                && !target.contains::<EffectOutcome>()
            {
                target.insert((
                    EffectOutcome(Err(ErrorReport::new(ErrorKind::Denied, error.to_string()))),
                    error,
                ));
            }
        });
    }
}
