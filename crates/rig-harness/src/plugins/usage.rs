//! What each agent's model calls used and cost, and `/usage`. Every
//! finished model call of a turn, a plugin's [`ModelRequest`] too, adds
//! the usage its provider reported to its agent's [`Spending`], which its
//! log saves, and to its turn's [`TurnSpending`], also when the
//! turn-failure rule rejects the reply. rig-core prices a reply its
//! provider did not price at the model's catalog
//! [`Pricing`](rig_core::catalog::Pricing), so the cost here is the
//! provider's figure or the catalog's list price. The context in use is the
//! agent's [`LastUsage`], which the kernel keeps: the next request sends
//! all of it again.

use rig_core::completion::{ContextUse, UsageTotals};
use rig_ecs::prelude::*;
use serde::{Deserialize, Serialize};

/// Sums every agent's and turn's model calls, and adds `/usage`.
#[derive(Default)]
pub struct UsagePlugin;

impl Plugin for UsagePlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, Spending>()
            .register_required_components::<TurnOf, TurnSpending>()
            .add_command(
                "usage",
                "Show the tokens, cost and context this agent, its subagents and the session used",
                usage,
            )
            .add_observer(record_spending);
    }
}

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
fn record_spending(
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

/// `/usage`: the shown agent's tokens, cost and context, what each agent
/// it spawned used, and the session's total over every agent.
fn usage(
    In(args): In<CommandArgs>,
    agents: Query<(
        &Spending,
        &LastUsage,
        Option<&Connection>,
        Option<&ActiveTurn>,
    )>,
    turns: Query<&TurnSpending>,
    everyone: Query<(&AgentId, Option<&Name>, &Spending), With<Agent>>,
    families: Query<&Spawned>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Spending(spent), last, connection, turn)) = agents.get(args.agent) else {
        return;
    };
    let mut total = UsageTotals::default();
    let mut spenders = 0;
    for (.., Spending(other)) in &everyone {
        if other.calls > 0 {
            total.add(other);
            spenders += 1;
        }
    }
    if total.calls == 0 {
        notices.write(Notice::info(args.agent, "No model call yet."));
        return;
    }
    let mut lines = vec![if spent.calls == 0 {
        "This agent: no model call yet.".to_owned()
    } else {
        format!("This agent: {spent}.")
    }];
    if let Some(TurnSpending(turn)) = turn.and_then(|turn| turns.get(turn.turn()).ok())
        && turn.calls > 0
    {
        lines.push(format!("This turn: {turn}."));
    }
    let window = connection.and_then(|connection| connection.spec.context_window);
    match last.context().map(|tokens| ContextUse { tokens, window }) {
        Some(context) => lines.push(format!(
            "Context: {context} tokens{}.",
            if context.window.is_none() {
                ", the model's window is not in the catalog"
            } else {
                ""
            }
        )),
        None if spent.calls > 0 => {
            lines.push("Context: not reported by the provider.".to_owned());
        }
        None => {}
    }
    let subagents: Vec<String> = families
        .iter_descendants_depth_first::<Spawned>(args.agent)
        .filter_map(|child| everyone.get(child).ok())
        .filter(|(.., Spending(spent))| spent.calls > 0)
        .map(|(id, name, Spending(spent))| {
            let title =
                name.map_or_else(|| format!("agent {}", id.short()), |name| name.to_string());
            let calls = match spent.calls {
                1 => "1 call".to_owned(),
                calls => format!("{calls} calls"),
            };
            format!("  {title}: {}, {calls}", spent.cost_or_tokens())
        })
        .collect();
    if !subagents.is_empty() {
        lines.push("Its subagents:".to_owned());
        lines.extend(subagents);
    }
    if spenders > 1 {
        lines.push(format!("Session, {spenders} agents: {total}."));
    }
    if total.unpriced > 0 {
        lines.push(format!(
            "{} of {} calls had no price: their provider did not report one and the catalog \
             lists none, or the model is local or billed by subscription.",
            total.unpriced, total.calls
        ));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}
