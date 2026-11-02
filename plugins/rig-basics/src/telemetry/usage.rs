//! What model calls used and cost, and `/usage`.
//!
//! - **What counts.** Every settled model call ([`ModelRequest`]), a turn's
//!   or a plugin's, is one [`Spent`], and so is each call a plugin paid for
//!   elsewhere, such as a `helper` child's. Each adds its usage to its
//!   agent's [`AgentUsage`] total, to the [`below`](AgentUsage::below) of
//!   every agent above it, to the session's [`SessionUsage`] (all saved) and
//!   to its turn's [`TurnUsage`]. A reply the turn-failure rule rejects
//!   still counts. A call settled without a reply (cancelled, superseded,
//!   failed mid-stream) counts its [`CallUsage`], if one was put on it.
//! - **How it is kept.** Each [`AgentUsage`] keeps a total, and splits per
//!   model ([`SentTo`]) and per purpose (`turn`, `compaction`, a plugin's
//!   own). Cost is the provider's figure, else the catalog's
//!   [`Pricing`](rig_core::catalog::Pricing).
//! - **Context in use.** The agent's [`LastUsage`], against the input
//!   limit of the model its last turn reply came from.
//! - **Extending.** A budget plugin reads [`SessionUsage`] in a Gate
//!   system, and a plugin paying for other work triggers a [`Spent`] for
//!   each call. [`UsageRow`]s add lines to `/usage`. Each agent's line in
//!   `/agents` says what it and the agents below it spent. With the `tui`
//!   feature, each agent's status line shows its meter and what its turn
//!   used so far.

use std::collections::BTreeMap;

use rig_core::completion::{ContextUse, Usage, UsageTotals};
use rig_ecs::session_log::Reset;
use rig_harness::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sessions::agents::{AgentColumn, ColumnOf};

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
            .init_resource::<UnknownSpend>()
            .add_command(
                "usage",
                "Show the tokens, cost and context this agent, its subagents and the session used",
                usage,
            )
            .add_observer(record_usage)
            .add_observer(account)
            .add_observer(add_column)
            .add_observer(forget_input_limit)
            .add_systems(Update, show_column);
        super::resume::add(app);
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// Model calls' usage summed, as rig-core's [`UsageTotals`] sums it: in
/// all, and split per model and per purpose. On every agent (its calls,
/// and those of the agents below it, despawned ones included), saved with
/// the session, and a fork starts from zero; the session's is
/// [`SessionUsage`] and a turn's a [`TurnUsage`].
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(
    opaque,
    Component,
    Saved(Reset),
    Default,
    Clone,
    Serialize,
    Deserialize
)]
#[serde(default)]
pub struct AgentUsage {
    /// Every call of its own.
    pub total: UsageTotals,
    /// Every call of the agents below it ([`Spawned`]), including those
    /// since despawned; `total` and `below` added are its tree's spend.
    pub below: UsageTotals,
    /// Per model that answered, by catalog reference (`vendor/model`).
    pub models: BTreeMap<String, UsageTotals>,
    /// Per purpose the call was made for, such as `turn` or `compaction`;
    /// a call that named none is in the total only.
    pub purposes: BTreeMap<String, UsageTotals>,
}

impl AgentUsage {
    /// Adds a finished call's `usage`, answered by `model` and made for
    /// `purpose`.
    pub(crate) fn record(&mut self, model: Option<&str>, purpose: Option<&str>, usage: &Usage) {
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

/// One model call's usage, to record on its agent, the agents above it,
/// its turn and the session: [`UsagePlugin`] triggers one for each settled
/// call, and a plugin paying for a call made elsewhere, such as a `helper`
/// child's, triggers its own. `rig -p --json` writes each as a `usage`
/// line.
#[derive(Event, Clone, Debug, PartialEq)]
pub struct Spent {
    /// The agent the call was for; `None` counts it for the session only.
    pub agent: Option<Entity>,
    /// The turn it counts in, if any.
    pub turn: Option<Entity>,
    /// The model that answered, by catalog reference (`vendor/model`).
    pub model: Option<String>,
    /// What it was for, such as `turn`, `compaction` or `helper`.
    pub purpose: Option<String>,
    /// What it used.
    pub usage: Usage,
}

/// How many reports of spend made elsewhere left some of it unknown, such
/// as a `helper` child's that died before it reported all of it, or whose
/// own helpers' was not all known: [`SessionUsage`] is then a lower bound.
/// Saved with the session.
#[derive(Resource, Reflect, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
pub struct UnknownSpend(pub u64);

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

/// Records a settled model call's usage, its reply's, else its
/// [`CallUsage`], as a [`Spent`] of its agent and turn, under the model it
/// went to (where it was sent, else the call's own connection, else its
/// agent's, else the one its reply names) and its purpose. A turn's reply
/// that reported its tokens sets its agent's [`InputLimit`].
fn record_usage(
    settled: On<Add<Settled>>,
    calls: Query<UsageCall>,
    agents: Query<Option<&Connection>>,
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
    let own = agent.and_then(|agent| agents.get(agent.0).ok().flatten());
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
    commands.trigger(Spent {
        agent: agent.map(|agent| agent.0),
        turn: of.map(|of| of.0),
        model,
        purpose: effect.purpose.as_deref().map(str::to_owned),
        usage: *usage,
    });
}

/// Adds a [`Spent`] to its turn's, its agent's, the `below` of every
/// agent above it, and the session's.
fn account(
    spent: On<Spent>,
    mut turns: Query<&mut TurnUsage>,
    mut agents: Query<&mut AgentUsage>,
    lineage: Query<&SpawnedBy>,
    mut session: ResMut<SessionUsage>,
) {
    let Spent {
        agent,
        turn,
        model,
        purpose,
        usage,
    } = spent.event();
    if let Some(mut turn_spent) = turn.and_then(|turn| turns.get_mut(turn).ok()) {
        turn_spent.0.record(usage);
    }
    if let Some(agent) = *agent {
        if let Ok(mut own) = agents.get_mut(agent) {
            own.record(model.as_deref(), purpose.as_deref(), usage);
        }
        for above in lineage.iter_ancestors(agent) {
            if let Ok(mut above) = agents.get_mut(above) {
                above.below.record(usage);
            }
        }
    }
    session
        .0
        .record(model.as_deref(), purpose.as_deref(), usage);
}

/// On an agent: its `/agents` column, what it and the agents below it
/// spent.
#[derive(Component)]
struct SpentColumn(Entity);

/// Gives each agent, spawned or restored, its `/agents` column.
fn add_column(added: On<Add<Agent>>, mut commands: Commands) {
    let agent = added.entity;
    let column = commands
        .spawn((AgentColumn::new("spent", ""), ColumnOf(agent)))
        .id();
    commands.entity(agent).insert(SpentColumn(column));
}

/// Keeps each agent's column current: `spent $0.12`, and `$0.40 below`
/// once an agent below it made a call.
fn show_column(
    agents: Query<(&SpentColumn, &AgentUsage), Or<(Added<SpentColumn>, Changed<AgentUsage>)>>,
    mut columns: Query<&mut AgentColumn>,
) {
    for (&SpentColumn(column), spent) in &agents {
        let text = match (spent.total.calls, spent.below.calls) {
            (0, 0) => String::new(),
            (_, 0) => format!("spent {}", spent.total.cost_or_tokens()),
            (_, _) => format!(
                "spent {}, {} below",
                spent.total.cost_or_tokens(),
                spent.below.cost_or_tokens()
            ),
        };
        if let Ok(mut column) = columns.get_mut(column) {
            column.set_if_neq(AgentColumn::new("spent", text));
        }
    }
}

/// `/usage`: the shown agent's tokens, cost and context, what each agent
/// it spawned used, what it and every agent below it used, despawned ones
/// included, the session's total with its split per model and per
/// purpose, and the other plugins' [`UsageRow`]s.
fn usage(
    In(args): In<NoArgs>,
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
    (session, unknown): (Res<SessionUsage>, Res<UnknownSpend>),
    rows: Query<&UsageRow>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((
        AgentUsage {
            total: spent,
            below,
            ..
        },
        last,
        stale,
        (connection, limit),
        turn,
    )) = agents.get(args.agent)
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
    match last
        .context()
        .filter(|_| !stale)
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
    if below.calls > 0 {
        let mut tree = *spent;
        tree.add(below);
        lines.push(format!("This agent and below: {tree}."));
    }
    if total.calls > spent.calls + below.calls {
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
    match unknown.0 {
        0 => {}
        1 => lines.push(
            "One helper's spend in this session is not all known: the totals are at least \
             what was spent."
                .to_owned(),
        ),
        n => lines.push(format!(
            "{n} helpers' spend in this session is not all known: the totals are at least \
             what was spent."
        )),
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
