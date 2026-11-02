//! A budget for what the session spends, and `/budget`. Off until set:
//! `[budget] usd` and `tokens` in `RIG_HOME/settings.toml` are the budget
//! a new session starts with, `/budget 2.00`, `/budget 500000 tokens` and
//! `/budget off` change the session's, which is saved with it, and
//! `/budget` shows what was spent against it.
//!
//! What counts is [`SessionUsage`]: every model call of every agent, those
//! of agents since despawned and those a `helper` child reported included,
//! priced as rig-core prices them, so `usd` is a lower bound of the bill,
//! and `tokens` counts the input and output tokens providers reported.
//! The status line shows it, red past 80%. Once the budget is reached, every model call about to start is refused, and every
//! running turn is stopped, which stops its `helper` children and theirs,
//! each with the reason. A call already sent finishes, so a tree can pass
//! the budget by the calls it had in flight.
//!
//! A `helper` child gets what is left in [`USD_VARIABLE`] and
//! [`TOKENS_VARIABLE`], and a run that has them never spends more, whatever
//! its own budget: so a child whose parent is gone stops too.

use bevy_log::warn;
use rig_core::completion::{UsageTotals, tokens_label};
use rig_harness::prelude::*;
use serde::Deserialize;

use crate::telemetry::usage::{SessionUsage, UsagePlugin};

#[cfg(test)]
mod tests;

/// The name the budget decides and stops as ([`Decider::plugin`]).
pub const ORIGIN: &str = "budget";

/// The variable a `helper` child gets the USD left of its parent's budget
/// in.
pub const USD_VARIABLE: &str = "RIG_BUDGET_USD";

/// The variable a `helper` child gets the tokens left of its parent's
/// budget in.
pub const TOKENS_VARIABLE: &str = "RIG_BUDGET_TOKENS";

/// The share of the budget past which the status line is red.
#[cfg(feature = "tui")]
const WARN_AT: f64 = 0.8;

/// Keeps the session within its budget, and adds `/budget`; adds
/// [`UsagePlugin`] when it is missing.
#[derive(Default)]
pub struct BudgetPlugin;

impl Plugin for BudgetPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<UsagePlugin>() {
            app.add_plugins(UsagePlugin);
        }
        app.add_setting::<Limit>("budget");
        let limit = app.world().get_resource::<Limit>().copied();
        let handed = Limit {
            usd: variable(USD_VARIABLE),
            tokens: variable(TOKENS_VARIABLE),
        };
        app.insert_resource(Budget(limit.unwrap_or_default()))
            .insert_resource(HandedDown(handed))
            .add_command(
                "budget",
                "Show what the session spent against its budget, set it (/budget 2.00, \
             /budget 500000 tokens) or remove it (/budget off)",
                budget,
            )
            .add_systems(AgentPass, refuse_when_reached.in_set(Stage::Gate))
            .add_systems(
                PostUpdate,
                stop_when_reached
                    .run_if(resource_changed::<SessionUsage>.or_eager(resource_changed::<Budget>)),
            );
        #[cfg(feature = "tui")]
        app.init_resource::<rig_tui::AppStatus>().add_systems(
            PostUpdate,
            show.in_set(rig_tui::StatusSystems::Show)
                .run_if(resource_changed::<SessionUsage>.or_eager(resource_changed::<Budget>)),
        );
    }
}

/// What a budget allows: at most `usd` and at most `tokens` (input and
/// output), each unlimited when `None`. As the `[budget]` settings table,
/// what a new session starts with.
#[derive(Resource, Reflect, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Limit {
    /// The most it may cost, in USD.
    pub usd: Option<f64>,
    /// The most input and output tokens it may use.
    pub tokens: Option<u64>,
}

impl Limit {
    /// What `spent` used of it, as `$0.04 of $0.05` and `120k of 200k
    /// tokens`, and whether that reaches it; nothing when unlimited.
    fn against(&self, spent: &UsageTotals) -> Option<(String, bool)> {
        let usd = self.usd.map(|usd| {
            let label = format!("${:.2} of ${usd:.2}", spent.cost);
            (label, spent.cost >= usd)
        });
        let tokens = self.tokens.map(|tokens| {
            let used = spent.total_tokens();
            let label = format!("{} of {} tokens", tokens_label(used), tokens_label(tokens));
            (label, used >= tokens)
        });
        let parts: Vec<(String, bool)> = usd.into_iter().chain(tokens).collect();
        if parts.is_empty() {
            return None;
        }
        let reached = parts.iter().any(|(_, reached)| *reached);
        let labels: Vec<String> = parts.into_iter().map(|(label, _)| label).collect();
        Some((labels.join(" and "), reached))
    }

    /// Whether 80% of it is spent.
    #[cfg(feature = "tui")]
    fn near(&self, spent: &UsageTotals) -> bool {
        let usd = self.usd.is_some_and(|usd| spent.cost >= usd * WARN_AT);
        let tokens = (self.tokens)
            .is_some_and(|tokens| spent.total_tokens() as f64 >= tokens as f64 * WARN_AT);
        usd || tokens
    }

    /// What is left of it after `spent`, never below zero.
    fn left(&self, spent: &UsageTotals) -> Self {
        Self {
            usd: self.usd.map(|usd| (usd - spent.cost).max(0.0)),
            tokens: (self.tokens).map(|tokens| tokens.saturating_sub(spent.total_tokens())),
        }
    }

    /// The lower of each of its and `other`'s.
    fn lower(self, other: Self) -> Self {
        let usd = match (self.usd, other.usd) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let tokens = match (self.tokens, other.tokens) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Self { usd, tokens }
    }
}

/// The session's budget, saved with it.
#[derive(Resource, Reflect, Clone, Copy, Debug, Default, PartialEq)]
#[reflect(Resource, Saved, Default)]
pub struct Budget(pub Limit);

/// What the run that started this one as a `helper` child left of its
/// budget ([`USD_VARIABLE`], [`TOKENS_VARIABLE`]): this run never spends
/// more. Not saved.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct HandedDown(pub Limit);

/// The variables a `helper` child is started with: what is left of the
/// session's budget and of the one handed down, after what the session
/// spent; none without a budget.
pub(crate) fn hand_down(
    budget: Option<&Budget>,
    handed: Option<&HandedDown>,
    spent: &SessionUsage,
) -> Vec<(&'static str, String)> {
    let spent = &spent.0.total;
    let own = budget.map(|budget| budget.0.left(spent));
    let handed = handed.map(|HandedDown(limit)| limit.left(spent));
    let left = own.unwrap_or_default().lower(handed.unwrap_or_default());
    let usd = left.usd.map(|usd| (USD_VARIABLE, usd.to_string()));
    let tokens = left
        .tokens
        .map(|tokens| (TOKENS_VARIABLE, tokens.to_string()));
    usd.into_iter().chain(tokens).collect()
}

/// The amount in the variable `name`; none when it is unset. One that does
/// not read as an amount is logged and allows nothing, so a garbled
/// handed-down budget never lifts it.
fn variable<T: std::str::FromStr + Default>(name: &str) -> Option<T> {
    let value = std::env::var(name).ok()?;
    match value.trim().parse() {
        Ok(amount) => Some(amount),
        Err(_) => {
            warn!("{name}={value} is not an amount: this run may spend nothing");
            Some(T::default())
        }
    }
}

/// Why the session may spend no more: its budget's or the handed-down
/// one's, as reached, naming how to raise it; `None` while neither is.
fn reached(budget: &Budget, handed: &HandedDown, spent: &UsageTotals) -> Option<String> {
    if let Some((used, true)) = budget.0.against(spent) {
        return Some(format!(
            "Budget reached: {used} spent. Raise it with /budget, or [budget] in settings.toml \
             for new sessions."
        ));
    }
    match handed.0.against(spent) {
        Some((used, true)) => Some(format!(
            "Budget reached: {used} spent of what the rig that started this one left it \
             ({USD_VARIABLE}, {TOKENS_VARIABLE})."
        )),
        _ => None,
    }
}

/// Once the budget is reached, refuses every model call about to start.
fn refuse_when_reached(
    mut calls: Query<&mut Decision, (ReadyToStart, ModelCalls)>,
    (budget, handed): (Res<Budget>, Res<HandedDown>),
    spending: Res<SessionUsage>,
) {
    if calls.is_empty() {
        return;
    }
    let Some(why) = reached(&budget, &handed, &spending.0.total) else {
        return;
    };
    for mut decision in &mut calls {
        decision.decide(Decider::plugin(ORIGIN), Outcome::refused(why.clone()));
    }
}

/// Once the budget is reached, as spending grows or the budget is lowered,
/// stops every running turn with the reason, as the user's stop does: from
/// each agent nothing spawned whose tree has one, so a subagent's report
/// says it was stopped and starts no turn. A turn's running `helper` call
/// stops its children, and they theirs.
fn stop_when_reached(
    roots: Query<Entity, (With<Agent>, Without<SpawnedBy>)>,
    (busy, below): (Query<(), With<ActiveTurn>>, Query<&Spawned>),
    (budget, handed): (Res<Budget>, Res<HandedDown>),
    spending: Res<SessionUsage>,
    mut commands: Commands,
) {
    let Some(why) = reached(&budget, &handed, &spending.0.total) else {
        return;
    };
    for root in &roots {
        let mut tree = std::iter::once(root).chain(below.iter_descendants(root));
        if tree.any(|agent| busy.contains(agent)) {
            commands.trigger(StopTurn::plugin(root, ORIGIN, &why));
        }
    }
}

/// `/budget`, `/budget 2.00`, `/budget 500000 tokens`, `/budget off`.
fn budget(
    In(args): In<CommandArgs>,
    (mut budget, handed): (ResMut<Budget>, Res<HandedDown>),
    spending: Res<SessionUsage>,
    mut notices: MessageWriter<Notice>,
) {
    let spent = &spending.0.total;
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let positive = |usd: &f64| usd.is_finite() && *usd > 0.0;
    let set = match words.as_slice() {
        [] => Some(budget.0),
        ["off"] => Some(Limit::default()),
        [amount] => {
            (amount.trim_start_matches('$').parse().ok().filter(positive)).map(|usd| Limit {
                usd: Some(usd),
                ..budget.0
            })
        }
        [amount, "tokens"] => {
            (amount.parse().ok().filter(|tokens| *tokens > 0)).map(|tokens| Limit {
                tokens: Some(tokens),
                ..budget.0
            })
        }
        _ => None,
    };
    let Some(limit) = set else {
        let why = format!(
            "Not a budget: {}; use /budget 2.00, /budget 500000 tokens or /budget off.",
            args.args
        );
        notices.write(Notice::error(args.agent, why));
        return;
    };
    budget.set_if_neq(Budget(limit));
    let mut text = match limit.against(spent) {
        Some((used, _)) => format!("Budget: {used} spent."),
        None => format!("No budget: {} spent.", spent.cost_or_tokens()),
    };
    if let Some((used, _)) = handed.0.against(spent) {
        text.push_str(&format!(
            " The rig that started this one left it at most {used}."
        ));
    }
    notices.write(Notice::info(args.agent, text));
}

/// Where the budget is in the status line: after the meter's cost.
#[cfg(feature = "tui")]
const ITEM: rig_tui::StatusItem = rig_tui::StatusItem::new("budget", rig_tui::Side::Right, 35, 6);

/// The session's spending against its budget in the status line,
/// `$0.04/$0.05` or `120k/200k tokens`, red past 80%; nothing without one.
#[cfg(feature = "tui")]
fn show(
    (budget, handed): (Res<Budget>, Res<HandedDown>),
    spending: Res<SessionUsage>,
    mut status: ResMut<rig_tui::AppStatus>,
) {
    let spent = &spending.0.total;
    let limit = budget.0.lower(handed.0);
    let usd = limit.usd.map(|usd| format!("${:.2}/${usd:.2}", spent.cost));
    let tokens = limit.tokens.map(|tokens| {
        let used = tokens_label(spent.total_tokens());
        format!("{used}/{} tokens", tokens_label(tokens))
    });
    let text: Vec<String> = usd.into_iter().chain(tokens).collect();
    let tone = match limit.near(spent) {
        true => rig_tui::Tone::Red,
        false => rig_tui::Tone::Dim,
    };
    status.0.show(ITEM.says(text.join(" "), tone));
}
