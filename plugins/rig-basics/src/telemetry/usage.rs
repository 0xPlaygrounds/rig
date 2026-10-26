//! What model calls used and cost, and `/usage`.
//!
//! - **What counts.** Every settled model call ([`ModelRequest`]), a turn's
//!   or a plugin's, adds its usage to its agent's [`AgentUsage`], the
//!   session's [`SessionUsage`] (both saved) and its turn's [`TurnUsage`].
//!   A reply the turn-failure rule rejects still counts. A call settled
//!   without a reply (cancelled, superseded, failed mid-stream) counts its
//!   [`CallUsage`], if one was put on it.
//! - **How it is kept.** Each [`AgentUsage`] keeps a total, and splits per
//!   model ([`SentTo`]) and per purpose (`turn`, `compaction`, a plugin's
//!   own). Cost is the provider's figure, else the catalog's
//!   [`Pricing`](rig_core::catalog::Pricing).
//! - **Context in use.** The agent's [`LastUsage`], against the input
//!   limit of the model its last turn reply came from.
//! - **Extending.** A budget plugin reads [`SessionUsage`] in a Gate
//!   system, and a plugin paying for other work
//!   [`record`](AgentUsage::record)s it. [`UsageRow`]s add lines to
//!   `/usage`. With the `tui` feature, each agent's status line shows its
//!   meter and what its turn used so far.

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
        app.register_required_components::<Agent, AgentUsage>()
            .register_required_components::<TurnOf, TurnUsage>()
            .init_resource::<SessionUsage>()
            .add_command(
                "usage",
                "Show the tokens, cost and context this agent, its subagents and the session used",
                usage,
            )
            .add_observer(record_usage)
            .add_observer(forget_input_limit);
        super::resume::add(app);
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// Model calls' usage summed, as rig-core's [`UsageTotals`] sums it: in
/// all, and split per model and per purpose. On every agent (its calls),
/// saved with the session; the session's is [`SessionUsage`] and a
/// turn's a [`TurnUsage`].
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentUsage {
    /// Every call.
    pub total: UsageTotals,
    /// Per model that answered, by catalog reference (`vendor/model`).
    pub models: BTreeMap<String, UsageTotals>,
    /// Per purpose the call was made for, such as `turn` or `compaction`;
    /// a call that named none is in the total only.
    pub purposes: BTreeMap<String, UsageTotals>,
}

impl AgentUsage {
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
pub struct SessionUsage(pub AgentUsage);

/// What the running turn's model calls used so far, on the turn entity.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Debug, Serialize, Deserialize)]
pub struct TurnUsage(pub UsageTotals);

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

/// What a model call used that settled without a reply (cancelled,
/// superseded, or failed mid-stream), as whoever knows says: a plugin's
/// estimate, or the provider's partial count. Put it on the call before it
/// settles, such as on a race's loser before superseding it; it counts as
/// a reply's usage does, under the model the call went to and its purpose.
/// A reply's own usage wins over it.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct CallUsage(pub Usage);

/// The [input limit](rig_core::catalog::ModelSpec::input_limit) of the
/// model the agent's last turn reply that reported its tokens came from, as
/// its call's [`SentTo`] says; until one did since the agent's model last
/// changed, the limit of the agent's [`Connection`] stands.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InputLimit(pub(crate) Option<u32>);

impl InputLimit {
    /// The input limit `last` says, else that of the agent's `connection`.
    pub(crate) fn of(last: Option<&Self>, connection: Option<&Connection>) -> Option<u32> {
        match last {
            Some(InputLimit(limit)) => *limit,
            None => connection.and_then(|connection| connection.spec.input_limit()),
        }
    }
}

/// Forgets the input limit of the last reply's model when the agent's
/// model changes, such as on `/model`: the next request goes to the new one.
fn forget_input_limit(changed: On<Insert<Connection>>, mut commands: Commands) {
    commands.entity(changed.entity).try_remove::<InputLimit>();
}

/// A model call as [`record_usage`] reads it.
type UsageCall<'a> = (
    (&'a ModelRequest, &'a Effect, Option<&'a SentTo>),
    (Option<&'a Outcome>, Option<&'a CallUsage>),
    (Option<&'a ForAgent>, Option<&'a CallOf>, Has<TurnCall>),
);

/// Adds a settled model call's usage, its reply's, else its [`CallUsage`],
/// to its agent's, its turn's and the session's, under the model it went
/// to (where it was sent, else the call's own connection, else its
/// agent's, else the one its reply names) and its purpose. A turn's reply
/// that reported its tokens sets its agent's [`InputLimit`].
fn record_usage(
    settled: On<Add<Settled>>,
    calls: Query<UsageCall>,
    mut turns: Query<&mut TurnUsage>,
    mut agents: Query<(Option<&mut AgentUsage>, Option<&Connection>)>,
    mut session: ResMut<SessionUsage>,
    mut commands: Commands,
) {
    let Ok(((request, effect, sent), (outcome, estimate), (agent, of, turn_call))) =
        calls.get(settled.entity)
    else {
        return;
    };
    let response = outcome.and_then(Outcome::completion);
    let Some(usage) =
        (response.map(|response| &response.usage)).or(estimate.map(|CallUsage(usage)| usage))
    else {
        return;
    };
    if let Some(mut turn_spent) = of.and_then(|of| turns.get_mut(of.0).ok()) {
        turn_spent.0.record(usage);
    }
    let purpose = effect.purpose.as_deref();
    let (spent, own) = agent
        .and_then(|agent| agents.get_mut(agent.0).ok())
        .unwrap_or((None, None));
    let went_to = (sent.map(|SentTo(connection)| connection))
        .or(request.connection.as_ref())
        .or(own);
    if let (Some(agent), true, Some(_), Some(_)) =
        (agent, turn_call, response, usage.context_tokens())
    {
        let limit = went_to.and_then(|connection| connection.spec.input_limit());
        commands.entity(agent.0).try_insert(InputLimit(limit));
    }
    let model = match (went_to, response) {
        (Some(connection), _) => Some(connection.spec.reference()),
        (None, Some(response)) => Some(format!(
            "{}/{}",
            response.origin.provider, response.origin.model
        )),
        (None, None) => None,
    };
    if let Some(mut spent) = spent {
        spent.record(model.as_deref(), purpose, usage);
    }
    session.0.record(model.as_deref(), purpose, usage);
}

/// `/usage`: the shown agent's tokens, cost and context, what each agent
/// it spawned used, the session's total with its split per model and per
/// purpose, and the other plugins' [`UsageRow`]s.
fn usage(
    In(args): In<CommandArgs>,
    agents: Query<(
        &AgentUsage,
        &LastUsage,
        Has<ContextStale>,
        (Option<&Connection>, Option<&InputLimit>),
        Option<&ActiveTurn>,
    )>,
    turns: Query<&TurnUsage>,
    everyone: Query<(&Name, &AgentUsage), With<Agent>>,
    families: Query<&Spawned>,
    session: Res<SessionUsage>,
    rows: Query<&UsageRow>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((AgentUsage { total: spent, .. }, last, stale, (connection, limit), turn)) =
        agents.get(args.agent)
    else {
        return;
    };
    let SessionUsage(session) = &*session;
    let total = &session.total;
    let mut lines = Vec::new();
    if total.calls == 0 {
        lines.push("No model call yet.".to_owned());
    } else if spent.calls == 0 {
        lines.push("This agent: no model call yet.".to_owned());
    } else {
        lines.push(format!("This agent: {spent}."));
    }
    if let Some(TurnUsage(turn)) = turn.and_then(|turn| turns.get(turn.turn()).ok())
        && turn.calls > 0
    {
        lines.push(format!("This turn: {turn}."));
    }
    let limit = InputLimit::of(limit, connection);
    match (!stale)
        .then(|| last.context())
        .flatten()
        .map(|tokens| ContextUse { tokens, limit })
    {
        None if stale => {
            lines.push("Context: unknown until the next reply.".to_owned());
        }
        Some(context) => lines.push(format!(
            "Context: {context} tokens{}.",
            if context.limit.is_none() {
                ", the model's input limit is not in the catalog"
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
    if session
        .purposes
        .keys()
        .any(|purpose| purpose != rig_ecs::turn::PURPOSE)
    {
        lines.push("Per purpose:".to_owned());
        lines.extend(split(&session.purposes));
    }
    if total.calls > total.unpriced || total.cost > 0.0 {
        lines.push(
            "Costs use provider-reported amounts when available, otherwise API list-price \
             estimates, not subscription charges or verified invoices."
                .to_owned(),
        );
    }
    if total.unpriced > 0 {
        lines.push(format!(
            "{} of {} calls had no complete cost: rates or usage details were missing, \
             or the model is local or billed by subscription.",
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
