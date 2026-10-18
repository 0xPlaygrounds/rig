//! A dollar budget for the session. `/budget 2.00` sets it, `/budget`
//! shows what was spent against it and `/budget off` removes it; a new
//! session starts with the `default` of the `[budget]` settings table, if
//! any. What was spent is `rig_basics::telemetry`'s [`SessionUsage`]: every model
//! call of every agent, subagents' and plugins' calls included, priced as
//! rig-core prices them. At 80% a warning is shown once. Once the budget
//! is spent, a Gate system refuses every model request about to be sent:
//! a turn's refused request ends the turn and keeps the user's message for
//! `/retry` once the budget is raised. A call already running is never cut
//! off. The budget is saved with the session, and the status line shows
//! `$1.42/$2.00`, red past 80%.

use rig_basics::telemetry::usage::{SessionUsage, UsagePlugin};
use rig_harness::prelude::*;
use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The share of the budget past which it is nearly spent.
const WARN_AT: f64 = 0.8;
/// Where the budget is in the status line: after the meter's cost.
const ITEM: StatusItem = StatusItem::new("cost-budget", Side::Right, 35, 6);

/// Keeps the session within a dollar budget, and adds `/budget`.
#[derive(Default)]
pub struct CostBudgetPlugin;

impl Plugin for CostBudgetPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<UsagePlugin>() {
            app.add_plugins(UsagePlugin);
        }
        app.add_setting::<BudgetSettings>("budget");
        let settings = app.world().get_resource::<BudgetSettings>();
        let limit = settings.and_then(|settings| settings.default);
        app.insert_resource(Budget {
            limit: limit.filter(|limit| valid(*limit)),
            warned: false,
        })
        .init_resource::<AppStatus>()
        .add_command(
            "budget",
            "Show the session's spending and budget, set it (/budget 2.00) or remove it (/budget off)",
            budget,
        )
        .configure_sets(AgentPass, CostBudgetSystems.in_set(Stage::Gate))
        .add_systems(AgentPass, refuse_when_spent.in_set(CostBudgetSystems))
        .add_systems(
            PostUpdate,
            (
                warn.run_if(resource_changed::<SessionUsage>),
                show.in_set(StatusSystems::Show).run_if(
                    resource_changed::<SessionUsage>.or_eager(resource_changed::<Budget>),
                ),
            ),
        );
    }
}

/// The Gate system that refuses every model call once the budget is
/// spent, to order against.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CostBudgetSystems;

/// The `[budget]` settings table: `default = 2.0`, the budget of a new
/// session in USD.
#[derive(Resource, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetSettings {
    pub default: Option<f64>,
}

/// The session's budget, saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Resource, Saved, Default)]
pub struct Budget {
    /// The most the session may spend, in USD, if limited.
    pub limit: Option<f64>,
    /// Whether the warning at 80% of `limit` was shown.
    pub warned: bool,
}

impl Budget {
    /// Whether 80% of the limit is spent.
    fn near(&self, spent: f64) -> bool {
        self.limit.is_some_and(|limit| spent >= limit * WARN_AT)
    }
}

/// What the session spent, in USD.
fn spent(spending: &SessionUsage) -> f64 {
    spending.0.total.cost
}

/// `$1.42/$2.00`, or `$1.42 spent` without a limit.
fn label(limit: Option<f64>, spent: f64) -> String {
    match limit {
        Some(limit) => format!("${spent:.2}/${limit:.2}"),
        None => format!("${spent:.2} spent"),
    }
}

fn valid(limit: f64) -> bool {
    limit.is_finite() && limit > 0.0
}

/// Once the budget is spent, refuses every model request about to be sent.
fn refuse_when_spent(
    mut calls: Query<&mut Decision, (ReadyToStart, ModelCalls)>,
    budget: Res<Budget>,
    spending: Res<SessionUsage>,
) {
    let spent = spent(&spending);
    let Some(limit) = budget.limit.filter(|limit| spent >= *limit) else {
        return;
    };
    for mut decision in &mut calls {
        let why = format!(
            "Stopped: budget of ${limit:.2} reached (${spent:.2} spent). Raise it with /budget, \
             then /retry."
        );
        decision.decide(Decider::plugin("cost-budget"), Outcome::refused(why));
    }
}

/// Warns once when 80% of the budget is spent.
fn warn(
    spending: Res<SessionUsage>,
    mut budget: ResMut<Budget>,
    mut notices: MessageWriter<Notice>,
) {
    let spent = spent(&spending);
    if budget.near(spent) && !budget.warned {
        budget.warned = true;
        let warning = format!("Budget nearly spent: {}.", label(budget.limit, spent));
        notices.write(Notice::info(None, warning));
    }
}

/// `/budget`, `/budget 2.00`, `/budget off`.
fn budget(
    In(args): In<CommandArgs>,
    mut budget: ResMut<Budget>,
    spending: Res<SessionUsage>,
    mut notices: MessageWriter<Notice>,
) {
    let spent = spent(&spending);
    let text = match args.args.as_str() {
        "" if budget.limit.is_none() => format!("No budget: {}.", label(None, spent)),
        "" => format!("Budget: {}.", label(budget.limit, spent)),
        "off" => {
            budget.limit = None;
            budget.warned = false;
            format!("Budget off: {}.", label(None, spent))
        }
        amount => {
            let limit = amount.trim_start_matches('$').parse::<f64>().ok();
            let Some(limit) = limit.filter(|limit| valid(*limit)) else {
                notices.write(Notice::error(
                    args.agent,
                    format!("Not a budget: {amount}; use /budget 2.00 or /budget off."),
                ));
                return;
            };
            budget.limit = Some(limit);
            // A budget nearly spent already says so here.
            budget.warned = budget.near(spent);
            format!("Budget: {}.", label(budget.limit, spent))
        }
    };
    notices.write(Notice::info(args.agent, text));
}

/// The session's spending against its budget in the status line, red past
/// 80%; nothing without a budget.
fn show(budget: Res<Budget>, spending: Res<SessionUsage>, mut status: ResMut<AppStatus>) {
    let spent = spent(&spending);
    let tone = if budget.near(spent) {
        Tone::Red
    } else {
        Tone::Dim
    };
    let text = budget
        .limit
        .map(|limit| label(Some(limit), spent))
        .unwrap_or_default();
    status.0.show(ITEM.says(text, tone));
}
