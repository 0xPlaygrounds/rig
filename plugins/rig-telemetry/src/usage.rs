//! What model calls used and cost, and `/usage`. Every model call
//! ([`ModelRequest`]) that settles with a reply, a turn's or a plugin's,
//! adds the usage its provider reported to its agent's [`Spending`], to the
//! session's [`SessionSpending`], both saved with the session, and to its
//! turn's [`TurnSpending`], also when the turn-failure rule rejects the
//! reply. Each [`Spending`] keeps the total, and the same split per model
//! that answered and per purpose the call was made for (`turn`, `compaction`,
//! a plugin's own), as plain data: a budget plugin reads
//! [`SessionSpending`] in a Gate system, and a plugin that pays for work
//! that is no model request [`record`](Spending::record)s it in both.
//! rig-core prices a reply its provider did not price at the model's
//! catalog [`Pricing`](rig_core::catalog::Pricing), so the cost here is
//! the provider's figure or the catalog's list price. The context in use
//! is the agent's [`LastUsage`], which the kernel keeps: the next request
//! sends all of it again. Other plugins add lines to `/usage` with
//! [`UsageRow`]s. With the default `tui` feature, each agent's status
//! line in the terminal view shows its meter (tokens in and out, cached
//! input, cost, context) and what its turn spent so far.

use std::collections::BTreeMap;

use rig_core::completion::{ContextUse, Usage, UsageTotals};
use rig_harness::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tui")]
mod status;

/// Sums every agent's, turn's and the session's model calls, and adds
/// `/usage`.
#[derive(Default)]
pub struct UsagePlugin;

impl Plugin for UsagePlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, Spending>()
            .register_required_components::<TurnOf, TurnSpending>()
            .init_resource::<SessionSpending>()
            .add_command(
                "usage",
                "Show the tokens, cost and context this agent, its subagents and the session used",
                usage,
            )
            .add_observer(record_spending);
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// Model calls' usage summed, as rig-core's [`UsageTotals`] sums it: in
/// all, and split per model and per purpose. On every agent (its calls),
/// saved with the session; the session's is [`SessionSpending`] and a
/// turn's a [`TurnSpending`].
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Spending {
    /// Every call.
    pub total: UsageTotals,
    /// Per model that answered, by catalog reference (`vendor/model`).
    pub models: BTreeMap<String, UsageTotals>,
    /// Per purpose the call was made for, such as `turn` or `compaction`;
    /// a call that named none is in the total only.
    pub purposes: BTreeMap<String, UsageTotals>,
}

impl Spending {
    /// Adds a finished call's `usage`, answered by `model` and made for
    /// `purpose`.
    pub fn record(&mut self, model: Option<&str>, purpose: Option<&str>, usage: &Usage) {
        self.total.record(usage);
        if let Some(model) = model {
            self.models
                .entry(model.to_owned())
                .or_default()
                .record(usage);
        }
        if let Some(purpose) = purpose {
            let purpose = self.purposes.entry(purpose.to_owned()).or_default();
            purpose.record(usage);
        }
    }
}

/// What every model call of the session used, the calls of agents since
/// despawned included. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
pub struct SessionSpending(pub Spending);

/// What the running turn's model calls used so far, on the turn entity.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Debug, Serialize, Deserialize)]
pub struct TurnSpending(pub UsageTotals);

/// A line another plugin adds to `/usage`, such as what is left of a
/// budget: spawn one and keep its `text` current. Lines come after the
/// usage, by `tag`; an empty `text`, or Bevy's `Disabled` on the entity,
/// leaves it out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct UsageRow {
    /// What sorts it, such as `budget`.
    pub tag: String,
    /// The line.
    pub text: String,
}

impl UsageRow {
    /// The line `text`, sorted by `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// Adds a settled model call's usage to its agent's, its turn's and the
/// session's, under the model it went to (the call's own connection, else
/// its agent's, else the one its reply names) and its purpose.
fn record_spending(
    settled: On<Add<Settled>>,
    calls: Query<(
        &Outcome,
        &ModelRequest,
        &Effect,
        Option<&ForAgent>,
        Option<&CallOf>,
    )>,
    mut turns: Query<&mut TurnSpending>,
    mut agents: Query<(Option<&mut Spending>, Option<&Connection>)>,
    mut session: ResMut<SessionSpending>,
) {
    let Ok((outcome, request, effect, agent, of)) = calls.get(settled.entity) else {
        return;
    };
    let Some(response) = outcome.completion() else {
        return;
    };
    let usage = &response.usage;
    if let Some(mut turn_spent) = of.and_then(|of| turns.get_mut(of.0).ok()) {
        turn_spent.0.record(usage);
    }
    let purpose = effect.purpose.as_deref();
    let (spent, own) = agent
        .and_then(|agent| agents.get_mut(agent.0).ok())
        .unwrap_or((None, None));
    let model = match request.connection.as_ref().or(own) {
        Some(connection) => connection.spec.reference(),
        None => format!("{}/{}", response.origin.provider, response.origin.model),
    };
    if let Some(mut spent) = spent {
        spent.record(Some(&model), purpose, usage);
    }
    session.0.record(Some(&model), purpose, usage);
}

/// `/usage`: the shown agent's tokens, cost and context, what each agent
/// it spawned used, the session's total with its split per model and per
/// purpose, and the other plugins' [`UsageRow`]s.
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
    session: Res<SessionSpending>,
    rows: Query<&UsageRow>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Spending { total: spent, .. }, last, connection, turn)) = agents.get(args.agent) else {
        return;
    };
    let SessionSpending(session) = &*session;
    let total = &session.total;
    let mut lines = Vec::new();
    if total.calls == 0 {
        lines.push("No model call yet.".to_owned());
    } else if spent.calls == 0 {
        lines.push("This agent: no model call yet.".to_owned());
    } else {
        lines.push(format!("This agent: {spent}."));
    }
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
        .filter(|(_, spent)| spent.total.calls > 0)
        .map(|(name, spent)| format!("  {name}: {}", calls(&spent.total)))
        .collect();
    if !subagents.is_empty() {
        lines.push("Its subagents:".to_owned());
        lines.extend(subagents);
    }
    if total.calls > spent.calls {
        lines.push(format!("Session: {total}."));
    }
    if session.models.len() > 1 {
        lines.push("Per model:".to_owned());
        lines.extend(split(&session.models));
    }
    if session.purposes.keys().any(|purpose| purpose != "turn") {
        lines.push("Per purpose:".to_owned());
        lines.extend(split(&session.purposes));
    }
    if total.unpriced > 0 {
        lines.push(format!(
            "{} of {} calls had no price: their provider did not report one and the catalog \
             lists none, or the model is local or billed by subscription.",
            total.unpriced, total.calls
        ));
    }
    let mut rows: Vec<&UsageRow> = rows.iter().filter(|row| !row.text.is_empty()).collect();
    rows.sort_by(|a, b| a.tag.cmp(&b.tag));
    lines.extend(rows.into_iter().map(|row| row.text.clone()));
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// `spent`'s cost, or tokens, and its calls.
fn calls(spent: &UsageTotals) -> String {
    match spent.calls {
        1 => format!("{}, 1 call", spent.cost_or_tokens()),
        calls => format!("{}, {calls} calls", spent.cost_or_tokens()),
    }
}

/// One indented line per key of `split`.
fn split(split: &BTreeMap<String, UsageTotals>) -> impl Iterator<Item = String> + '_ {
    split
        .iter()
        .map(|(key, spent)| format!("  {key}: {}", calls(spent)))
}

#[cfg(test)]
mod tests;
