//! What each agent's model calls used and cost, and how full its context is.
//! Every finished model call of a turn, a plugin's
//! [`ModelRequest`](super::turn::ModelRequest) too,
//! adds the usage its provider reported to its agent's [`Spending`], which
//! its log saves, and to its turn's [`TurnSpending`], also when the
//! turn-failure rule rejects the reply. rig-core prices a reply its provider did not price at
//! the model's catalog [`Pricing`](rig_core::catalog::Pricing), so the cost
//! here is the provider's figure or the catalog's list price.
//!
//! The context in use is the agent's `LastUsage`: the next request sends
//! all of it again. rig-core's [`UsageTotals`] and `ContextUse` label the
//! sums and the context against the connected model's window.

use bevy_ecs::prelude::*;
use bevy_log::info;
use bevy_reflect::prelude::*;
use rig_core::completion::UsageTotals;
use serde::{Deserialize, Serialize};

use super::agent::{AgentId, CallOf, TurnOf};
use super::calls::Done;
use super::journal::ReflectSaved;
use super::turn::ModelReply;

/// An agent's model calls' usage summed, as rig-core's [`UsageTotals`]
/// sums it, saved with the session. A turn's is a [`TurnSpending`].
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Spending(pub UsageTotals);

/// What the running turn's model calls used so far, on the turn entity.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Debug, Serialize, Deserialize)]
pub struct TurnSpending(pub UsageTotals);

/// Adds a finished model call's usage to its agent's and its turn's.
pub(crate) fn record_spending(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Done<ModelReply>)>,
    mut turns: Query<(&TurnOf, &mut TurnSpending)>,
    mut agents: Query<&mut Spending>,
) {
    let Ok((&CallOf(turn), Done(Ok(response)))) = calls.get(done.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), mut turn_spent)) = turns.get_mut(turn) else {
        return;
    };
    turn_spent.0.record(&response.usage);
    if let Ok(mut spent) = agents.get_mut(agent) {
        spent.0.record(&response.usage);
    }
}

/// Logs what a turn used when its entity goes away, however it ended.
pub(crate) fn log_turn_spending(
    end: On<Remove<TurnSpending>>,
    turns: Query<(&TurnOf, &TurnSpending)>,
    agents: Query<&AgentId>,
) {
    let Ok((&TurnOf(agent), TurnSpending(spent))) = turns.get(end.entity) else {
        return;
    };
    if spent.calls == 0 {
        return;
    }
    let agent = agents.get(agent).map_or("-", |id| id.0.as_str());
    info!(agent, "turn used {spent}");
}
