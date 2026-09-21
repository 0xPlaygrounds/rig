//! Durable tool-turn commits and host-owned holds at their checkpoint boundary.
//!
//! A hold is armed before the desired tool turn. It only prevents advancement
//! after a real batch commit, never dispatch of the batch being awaited. Owners
//! release their own named hold; all remaining owners must release before the
//! run continues. A checkpoint preserves holds, but does not persist host tools
//! or external side effects.
//!
//! ```
//! use rig_ecs::agent::checkpoint::ToolTurnHolds;
//! let holds = ToolTurnHolds::default();
//! assert!(!holds.blocks(1));
//! ```

use bevy_reflect::Reflect;
use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Failed, Run, Settled};

/// A completed tool batch on its existing turn entity. The number is the run's
/// model-turn cursor, including earlier non-tool turns, not a batch count.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolTurnCommit {
    /// Model turn whose complete results became history.
    pub turn: usize,
}

/// The assistant utterance committed for this tool-bearing turn.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = AssistantForTurns)]
#[reflect(Component)]
pub struct TurnAssistant(pub Entity);

/// Turns whose assistant utterance is this entity.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = TurnAssistant)]
#[reflect(Component)]
pub struct AssistantForTurns(Vec<Entity>);

/// The single ordered user utterance containing the committed tool results.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = ResultsForTurns)]
#[reflect(Component)]
pub struct TurnResults(pub Entity);

/// Turns whose tool-result utterance is this entity.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = TurnResults)]
#[reflect(Component)]
pub struct ResultsForTurns(Vec<Entity>);

/// Armed checkpoint holds on a run, indexed by host owner name. Each value is
/// the earliest committed model-turn number at which that owner blocks advance.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolTurnHolds(BTreeMap<String, usize>);

impl ToolTurnHolds {
    /// Owners and their minimum committed model-turn numbers.
    pub fn owners(&self) -> impl Iterator<Item = (&str, usize)> {
        self.0.iter().map(|(owner, turn)| (owner.as_str(), *turn))
    }

    /// Whether any owner is holding this committed turn.
    pub fn blocks(&self, committed_turn: usize) -> bool {
        self.0.values().any(|turn| *turn <= committed_turn)
    }
}

/// A live batch commit notification, emitted after deferred graph writes and
/// phase changes are visible. A checkpoint load does not emit this event. Inspect
/// ToolTurnCommit for durable state; do not treat component insertion on load
/// as a notification that tools ran again. A terminal observer that removes
/// the owning graph before delivery suppresses this live event.
#[derive(Event, Debug, Clone, Copy)]
pub struct ToolTurnCommitted {
    /// Run owning the committed turn.
    pub run: Entity,
    /// Existing turn entity carrying ToolTurnCommit and utterance links.
    pub turn: Entity,
}

/// An invalid host request to arm or release a checkpoint hold.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CheckpointError {
    /// The entity does not name a live run.
    #[error("checkpoint hold needs a live run")]
    NotLiveRun,
    /// An owner must have a nonempty stable name.
    #[error("checkpoint hold owner must not be empty")]
    EmptyOwner,
    /// Model turns are numbered from one.
    #[error("checkpoint turn must be at least one")]
    ZeroTurn,
}

/// Arm a hold for this owner. Re-arming is idempotent and cannot move an
/// existing owner's boundary later; release that hold explicitly to replace it.
/// Returns true when a hold was added or moved earlier. Hosts must arm before
/// advancement; this cannot retract a request that has already dispatched.
/// Returns an error for a non-live run, empty owner, or zero turn.
pub fn hold_after_tool_turn(
    world: &mut World,
    run: Entity,
    owner: impl Into<String>,
    turn: usize,
) -> Result<bool, CheckpointError> {
    if world.get::<Run>(run).is_none()
        || world.get::<Failed>(run).is_some()
        || world.get::<Settled>(run).is_some()
    {
        return Err(CheckpointError::NotLiveRun);
    }
    let owner = owner.into();
    if owner.is_empty() {
        return Err(CheckpointError::EmptyOwner);
    }
    if turn == 0 {
        return Err(CheckpointError::ZeroTurn);
    }
    let mut holds = world.get::<ToolTurnHolds>(run).cloned().unwrap_or_default();
    let previous = holds.0.get(&owner).copied();
    if previous.is_some_and(|previous| previous <= turn) {
        return Ok(false);
    }
    holds.0.insert(owner, turn);
    world.entity_mut(run).insert(holds);
    Ok(true)
}

/// Release only the named owner's hold. Also permitted on a terminal run for
/// host cleanup. Unknown owners are an idempotent no-op. The next schedule
/// update continues a nonterminal run once no hold blocks its committed turn.
/// Returns whether an entry was removed, or an error for an empty owner or
/// entity without [`Run`].
pub fn release_tool_turn_hold(
    world: &mut World,
    run: Entity,
    owner: &str,
) -> Result<bool, CheckpointError> {
    if world.get::<Run>(run).is_none() {
        return Err(CheckpointError::NotLiveRun);
    }
    if owner.is_empty() {
        return Err(CheckpointError::EmptyOwner);
    }
    let Some(mut holds) = world.get_mut::<ToolTurnHolds>(run) else {
        return Ok(false);
    };
    let removed = holds.0.remove(owner).is_some();
    if holds.0.is_empty() {
        world.entity_mut(run).remove::<ToolTurnHolds>();
    }
    Ok(removed)
}

/// The invariants a loaded tool-turn checkpoint holds on `entity`: a hold
/// on a run with named owners and non-zero turns; a commit or its links on
/// a turn of a run, naming utterances of that run in their roles; a commit
/// with both links, no batch out, and a turn the run's cursor has reached.
pub(crate) fn validate(world: &World, entity: Entity) -> Result<(), rig_core::error::ErrorReport> {
    use super::{Batch, Cursor, Role, Turn, Utterance};
    let refused = |message: &str| {
        rig_core::error::ErrorReport::new(rig_core::error::ErrorKind::Request, message)
    };
    if let Some(holds) = world.get::<ToolTurnHolds>(entity)
        && (world.get::<Run>(entity).is_none()
            || holds
                .owners()
                .any(|(owner, turn)| owner.is_empty() || turn == 0))
    {
        return Err(refused("invalid tool-turn hold"));
    }
    let assistant = world.get::<TurnAssistant>(entity);
    let results = world.get::<TurnResults>(entity);
    let commit = world.get::<ToolTurnCommit>(entity);
    if assistant.is_none() && results.is_none() && commit.is_none() {
        return Ok(());
    }
    let run = world
        .get::<ChildOf>(entity)
        .map(ChildOf::parent)
        .filter(|run| world.get::<Run>(*run).is_some())
        .ok_or_else(|| refused("tool-turn link has no run"))?;
    if world.get::<Turn>(entity).is_none() {
        return Err(refused("tool-turn link is not on a turn"));
    }
    for (target, role) in [
        (assistant.map(|link| link.0), Role::Assistant),
        (results.map(|link| link.0), Role::User),
    ] {
        if let Some(target) = target
            && (world.get::<ChildOf>(target).map(ChildOf::parent) != Some(run)
                || world.get::<Utterance>(target).is_none()
                || world.get::<Role>(target) != Some(&role))
        {
            return Err(refused("tool-turn utterance has invalid ownership or role"));
        }
    }
    if let Some(commit) = commit {
        if commit.turn == 0
            || assistant.is_none()
            || results.is_none()
            || world.get::<Batch>(entity).is_some()
            || world
                .get::<Cursor>(run)
                .is_none_or(|cursor| commit.turn > cursor.turn)
        {
            return Err(refused("invalid committed tool turn"));
        }
    } else if results.is_some() {
        return Err(refused("tool results have no commit"));
    }
    Ok(())
}
