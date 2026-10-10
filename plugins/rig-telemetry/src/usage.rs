//! What each agent's model calls used and cost, and `/usage`. Every
//! finished model call of a turn, a plugin's [`ModelRequest`] too, adds
//! the usage its provider reported to its agent's [`Spending`], which its
//! log saves, and to its turn's [`TurnSpending`], also when the
//! turn-failure rule rejects the reply. rig-core prices a reply its
//! provider did not price at the model's catalog
//! [`Pricing`](rig_core::catalog::Pricing), so the cost here is the
//! provider's figure or the catalog's list price. The context in use is the
//! agent's [`LastUsage`], which the kernel keeps: the next request sends
//! all of it again. Each agent's status line shows its meter (tokens in
//! and out, cached input, cost, context) and what its turn spent so far.

use rig_core::completion::{ContextUse, UsageTotals, tokens_label};
use rig_harness::prelude::*;
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
            .add_systems(PostUpdate, show_meter.in_set(StatusSystems))
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

/// Where what the running turn spent is in the status line.
const TURN: StatusItem = StatusItem::at(Side::Left, 80, 11);
/// Where the meter's items are: the cached input goes first when the
/// line is too narrow, then the cost, the tokens and the context.
const TOKENS: StatusItem = StatusItem::at(Side::Right, 10, 4);
const CACHE: StatusItem = StatusItem::at(Side::Right, 20, 1);
const COST: StatusItem = StatusItem::at(Side::Right, 30, 3);
const CONTEXT: StatusItem = StatusItem::at(Side::Right, 40, 5);

/// Each agent's meter, once a model call of it finished: its uncached
/// input and output tokens, cache reads, cost, then the context against
/// the model's window, yellow past 70% and red past 90%. `/usage` details
/// the cache writes and reasoning.
fn show_meter(
    mut agents: Query<
        (&Spending, &LastUsage, Option<&Connection>, &mut StatusItems),
        Or<(Changed<Spending>, Changed<LastUsage>, Changed<Connection>)>,
    >,
) {
    for (Spending(spent), last, connection, mut items) in &mut agents {
        let shown = spent.calls > 0;
        let window = connection.and_then(|connection| connection.spec.context_window);
        let context = last.context().map(|tokens| ContextUse { tokens, window });
        let tokens = format!(
            "↑{} ↓{}",
            tokens_label(spent.uncached_input()),
            tokens_label(spent.tokens.output_tokens.unwrap_or(0))
        );
        let cache = (spent.tokens.cached_input_tokens)
            .filter(|read| *read > 0)
            .map(|read| format!("cache {}", tokens_label(read)));
        let tone = match context.and_then(|context| context.percent()) {
            Some(90..) => Tone::Red,
            Some(70..) => Tone::Yellow,
            _ => Tone::Dim,
        };
        let context = context.map(|context| format!("ctx {context}"));
        let text = |text: Option<String>| text.filter(|_| shown).unwrap_or_default();
        items.show(TOKENS.says(text(Some(tokens)), Tone::Dim));
        items.show(CACHE.says(text(cache), Tone::Dim));
        items.show(COST.says(text(spent.cost_label()), Tone::Dim));
        items.show(CONTEXT.says(text(context), tone));
    }
}

/// Adds a finished model call's usage to its agent's and its turn's, and
/// shows the turn's in the agent's status line.
fn record_spending(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Done<ModelReply>)>,
    mut turns: Query<(&TurnOf, &mut TurnSpending, Option<&mut StatusItems>)>,
    mut agents: Query<&mut Spending>,
) {
    let Ok((&CallOf(turn), Done(Ok(response)))) = calls.get(done.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), mut turn_spent, items)) = turns.get_mut(turn) else {
        return;
    };
    turn_spent.0.record(&response.usage);
    if let Some(mut items) = items {
        let used = turn_spent.0.cost_or_tokens();
        items.show(TURN.says(format!("this turn {used}"), Tone::Dim));
    }
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
    everyone: Query<(&Name, &Spending), With<Agent>>,
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
        .map(|(name, Spending(spent))| {
            let calls = match spent.calls {
                1 => "1 call".to_owned(),
                calls => format!("{calls} calls"),
            };
            format!("  {name}: {}, {calls}", spent.cost_or_tokens())
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
