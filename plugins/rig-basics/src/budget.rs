//! A budget for what the session spends, and `/budget`. Off until set:
//! `[budget] usd` and `tokens` in `RIG_HOME/settings.toml` are the budget
//! a new session starts with, `/budget 2.00`, `/budget 500000 tokens` and
//! `/budget off` change the session's, which is saved with it, and
//! `/budget` shows what was spent against it.
//!
//! What counts is [`SessionUsage`]: every model call of every agent, those
//! of agents since despawned and those a `helper` child reported included,
//! priced as rig-core prices them, so `usd` is a lower bound of the bill
//! and leaves out calls with no price, such as a subscription model's,
//! which `/budget` and the status line (`$0.04+`) point out; `tokens`
//! counts the input and output tokens providers reported, priced or not.
//! The status line shows it, red past 80%. Once the budget is reached,
//! every model call about to start is refused, and every running turn is
//! stopped, with an error notice of the reason, which fails a `-p` run;
//! on unix a stopped turn's `helper` call stops its children, and they
//! theirs. A model call in flight then is cancelled, and what its
//! provider bills for it is not counted, so the bill can pass the budget.
//!
//! A `helper` child gets what was left when its call started in
//! [`USD_VARIABLE`] and [`TOKENS_VARIABLE`], and a run with this plugin
//! that has them holds its session to them as to a budget, whatever its
//! own: so a child whose parent is gone stops too. They count the whole
//! session's spend, its earlier runs' included. What a child starts
//! inherits the variables as the child got them. An amount that is not a
//! number allows nothing.

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
        let limit = limit.unwrap_or_default().checked("[budget] usd");
        let handed = handed_down(
            variable(std::env::var(USD_VARIABLE)),
            variable(std::env::var(TOKENS_VARIABLE)),
        );
        app.insert_resource(Budget(limit))
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

    /// The variables that hand it down, of those it sets, as `the limit in
    /// RIG_BUDGET_TOKENS` or `the limits in RIG_BUDGET_USD and
    /// RIG_BUDGET_TOKENS`.
    fn variables(&self) -> String {
        let usd = self.usd.map(|_| USD_VARIABLE);
        let tokens = self.tokens.map(|_| TOKENS_VARIABLE);
        let names: Vec<&str> = usd.into_iter().chain(tokens).collect();
        let limits = if names.len() > 1 { "limits" } else { "limit" };
        format!("the {limits} in {}", names.join(" and "))
    }

    /// Whether 80% of it is spent.
    #[cfg(feature = "tui")]
    fn near(&self, spent: &UsageTotals) -> bool {
        let usd = self.usd.is_some_and(|usd| spent.cost >= usd * WARN_AT);
        let tokens = (self.tokens)
            .is_some_and(|tokens| spent.total_tokens() as f64 >= tokens as f64 * WARN_AT);
        usd || tokens
    }

    /// It with a `usd` from `source` that is not a number, or is below
    /// zero, made zero, which allows nothing, and logged: a comparison
    /// with one that is not a number would never reach it.
    fn checked(self, source: &str) -> Self {
        match self.usd {
            Some(usd) if !(usd.is_finite() && usd >= 0.0) => {
                warn!("{source} = {usd} is not an amount: the session may spend nothing");
                Self {
                    usd: Some(0.0),
                    ..self
                }
            }
            _ => self,
        }
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

/// What [`USD_VARIABLE`] and [`TOKENS_VARIABLE`] allow, such as what the
/// run that started this one as a `helper` child left of its budget: held
/// as a budget is, against the session's spend. Not saved.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct HandedDown(pub(crate) Limit);

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

/// A variable's value: none when it is unset, and one that is not UTF-8
/// as a value that is no amount, so it allows nothing.
fn variable(value: Result<String, std::env::VarError>) -> Option<String> {
    match value {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => Some(String::new()),
    }
}

/// What [`USD_VARIABLE`] and [`TOKENS_VARIABLE`] allow, from their
/// values: none of an unset one, and
/// nothing of one that is not an amount, which is logged, so a garbled
/// handed-down budget never lifts it.
fn handed_down(usd: Option<String>, tokens: Option<String>) -> Limit {
    fn amount<T: std::str::FromStr + Default>(name: &str, value: Option<String>) -> Option<T> {
        let value = value?;
        let parsed = value.trim().parse().ok();
        if parsed.is_none() {
            warn!("{name}={value} is not an amount: this run may spend nothing");
        }
        Some(parsed.unwrap_or_default())
    }
    let handed = Limit {
        usd: amount(USD_VARIABLE, usd),
        tokens: amount(TOKENS_VARIABLE, tokens),
    };
    handed.checked(USD_VARIABLE)
}

/// Why the session may spend no more: its budget's or the handed-down
/// one's, as reached, naming how to raise it; `None` while neither is.
fn reached(budget: &Budget, handed: &HandedDown, spent: &UsageTotals) -> Option<String> {
    if let Some((used, true)) = budget.0.against(spent) {
        return Some(format!(
            "Budget reached: {used} spent. Raise it with /budget and send again, or \
             [budget] in settings.toml for new sessions."
        ));
    }
    match handed.0.against(spent) {
        Some((used, true)) => Some(format!(
            "Budget reached: {used} spent, {}, set by a helper's parent rig or by hand: \
             start this rig again with more.",
            handed.0.variables()
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
/// says it was stopped and starts no turn. An error notice tells the root,
/// and each agent stopped working, why. A turn's running `helper` call
/// stops its children, and they theirs.
fn stop_when_reached(
    roots: Query<Entity, (With<Agent>, Without<SpawnedBy>)>,
    (busy, below): (Query<(), With<ActiveTurn>>, Query<&Spawned>),
    (budget, handed): (Res<Budget>, Res<HandedDown>),
    spending: Res<SessionUsage>,
    (mut commands, mut notices): (Commands, MessageWriter<Notice>),
) {
    let Some(why) = reached(&budget, &handed, &spending.0.total) else {
        return;
    };
    for root in &roots {
        let working: Vec<Entity> = (below.iter_descendants(root))
            .filter(|agent| busy.contains(*agent))
            .collect();
        if working.is_empty() && !busy.contains(root) {
            continue;
        }
        commands.trigger(StopTurn::plugin(root, ORIGIN, &why));
        // Each stopped agent's view says why, as the root's does.
        for agent in std::iter::once(root).chain(working) {
            notices.write(Notice::error(agent, why.clone()));
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
        [amount, "tokens"] => (tokens(amount).filter(|tokens| *tokens > 0)).map(|tokens| Limit {
            tokens: Some(tokens),
            ..budget.0
        }),
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
        let variables = handed.0.variables();
        text.push_str(&format!(" Against {variables}: {used} spent."));
    }
    if limit.lower(handed.0).usd.is_some() && spent.unpriced > 0 {
        text.push_str(&format!(
            " {} of {} calls had no price, such as a subscription model's, and are not in \
             the dollars: /budget 500000 tokens counts them.",
            spent.unpriced, spent.calls
        ));
    }
    notices.write(Notice::info(args.agent, text));
}

/// A token count as `/budget` takes it: `500000`, `500k` or `2M`.
fn tokens(amount: &str) -> Option<u64> {
    let (digits, times) = match amount.strip_suffix(['k', 'K']) {
        Some(digits) => (digits, 1_000),
        None => match amount.strip_suffix('M') {
            Some(digits) => (digits, 1_000_000),
            None => (amount, 1),
        },
    };
    digits.parse::<u64>().ok()?.checked_mul(times)
}

/// Where the budget is in the status line: after the meter's cost.
#[cfg(feature = "tui")]
const ITEM: rig_tui::StatusItem = rig_tui::StatusItem::new("budget", rig_tui::Side::Right, 35, 6);

/// The session's spending against its budget in the status line,
/// `$0.04/$0.05` (`$0.04+/$0.05` when some calls had no price) or
/// `120k/200k tokens`, red past 80%; nothing without one.
#[cfg(feature = "tui")]
fn show(
    (budget, handed): (Res<Budget>, Res<HandedDown>),
    spending: Res<SessionUsage>,
    mut status: ResMut<rig_tui::AppStatus>,
) {
    let spent = &spending.0.total;
    let limit = budget.0.lower(handed.0);
    let more = if spent.unpriced > 0 { "+" } else { "" };
    let usd = (limit.usd).map(|usd| format!("${:.2}{more}/${usd:.2}", spent.cost));
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
