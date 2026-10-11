//! A dollar budget for the session. `/budget 2.00` sets it, `/budget`
//! shows what was spent against it and `/budget off` removes it; a new
//! session starts with the `default` of `RIG_HOME/budget.json`
//! (`{"default": 2.0}`), if any. Every finished model call of every agent,
//! a plugin's [`ModelRequest`] too, adds what its reply cost (the
//! provider's figure, else rig-core's catalog price) to the session's
//! [`Budget`], which is saved with the session. At 80% a warning is shown
//! once; once the budget is spent, the next model request of any turn is
//! not sent and the turn ends, the conversation kept for `/retry` after the
//! budget is raised. A call already running is never cut off. The status
//! line shows `$1.42/$2.00`, red past 80%.

use rig_ecs::agent::Halt;
use rig_harness::prelude::*;
use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The share of the budget past which it is nearly spent.
const WARN_AT: f64 = 0.8;
/// Where the budget is in the status line: beside the cost of the meter.
const ITEM: StatusItem = StatusItem::at(Side::Right, 35, 6);
/// The file in `RIG_HOME` with the budget a new session starts with.
const CONFIG: &str = "budget.json";

/// Keeps the session within a dollar budget, and adds `/budget`.
#[derive(Default)]
pub struct CostBudgetPlugin;

impl Plugin for CostBudgetPlugin {
    fn build(&self, app: &mut App) {
        // Only an agent the launcher started reads `RIG_HOME`, not a test.
        let limit = app
            .world()
            .get_resource::<Invoked>()
            .and_then(|_| configured());
        app.insert_resource(Budget {
            limit,
            ..Budget::default()
        })
        .init_resource::<AppStatus>()
        .add_command(
            "budget",
            "Show the session's spending and budget, set it (/budget 2.00) or remove it (/budget off)",
            budget,
        )
        .add_observer(record)
        .add_observer(enforce)
        .add_systems(
            PostUpdate,
            show.in_set(StatusSystems)
                .run_if(resource_changed::<Budget>),
        );
    }
}

/// What the session's model calls cost, in USD, and its budget, saved with
/// the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Resource, Saved, Default)]
pub struct Budget {
    /// The most the session may spend, if limited.
    pub limit: Option<f64>,
    /// What its finished model calls cost.
    pub spent: f64,
    /// The finished calls with no price, which `spent` leaves out.
    pub unpriced: u64,
    /// Whether the warning at 80% of `limit` was shown.
    pub warned: bool,
}

impl Budget {
    /// The limit, once it is spent.
    fn reached(&self) -> Option<f64> {
        self.limit.filter(|limit| self.spent >= *limit)
    }

    /// Whether 80% of the limit is spent.
    fn near(&self) -> bool {
        self.limit
            .is_some_and(|limit| self.spent >= limit * WARN_AT)
    }

    /// `$1.42/$2.00`, or `$1.42 spent` without a limit.
    fn label(&self) -> String {
        match self.limit {
            Some(limit) => format!("${:.2}/${limit:.2}", self.spent),
            None => format!("${:.2} spent", self.spent),
        }
    }
}

/// The default budget of `RIG_HOME/budget.json`, if any.
fn configured() -> Option<f64> {
    #[derive(Deserialize)]
    struct Config {
        default: f64,
    }
    let text = std::fs::read_to_string(Home::from_env().root().join(CONFIG)).ok()?;
    let config: Config = serde_json::from_str(&text).ok()?;
    Some(config.default).filter(|limit| limit.is_finite() && *limit > 0.0)
}

/// Adds a finished model call's cost, of any agent and any turn, and warns
/// once at 80% of the budget.
fn record(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<&Done<ModelReply>>,
    mut budget: ResMut<Budget>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(Done(Ok(response))) = calls.get(done.entity) else {
        return;
    };
    match response.usage.cost {
        Some(cost) => budget.spent += cost.total,
        None => budget.unpriced += 1,
    }
    if budget.near() && !budget.warned {
        budget.warned = true;
        let warning = format!("Budget nearly spent: {}.", budget.label());
        notices.write(Notice::info(None, warning));
    }
}

/// Keeps a spent budget's turns from sending their next request: the turn
/// ends, and the conversation is kept for `/retry`.
fn enforce(
    prepare: On<PrepareRequest>,
    budget: Res<Budget>,
    mut agents: Query<&mut Conversation>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(limit) = budget.reached() else {
        return;
    };
    let (turn, agent) = (prepare.entity, prepare.agent);
    if let Ok(mut conversation) = agents.get_mut(agent) {
        commit.halt(agent, &mut conversation, Halt::Kept);
    }
    // Despawned before the observers' calls are waited for, the turn sends
    // nothing.
    commands.entity(turn).despawn();
    notices.write(Notice::error(
        agent,
        format!(
            "Stopped: budget of ${limit:.2} reached (${:.2} spent). Raise it with /budget, then \
             /retry.",
            budget.spent
        ),
    ));
}

/// `/budget`, `/budget 2.00`, `/budget off`.
fn budget(
    In(args): In<CommandArgs>,
    mut budget: ResMut<Budget>,
    mut notices: MessageWriter<Notice>,
) {
    let text = match args.args.as_str() {
        "" => {
            let unpriced = match budget.unpriced {
                0 => String::new(),
                calls => format!(" ({calls} calls had no price)"),
            };
            match budget.limit {
                Some(_) => format!("Budget: {}{unpriced}.", budget.label()),
                None => format!("No budget: {}{unpriced}.", budget.label()),
            }
        }
        "off" => {
            budget.limit = None;
            budget.warned = false;
            format!("Budget off: {}.", budget.label())
        }
        amount => {
            let limit = amount.trim_start_matches('$').parse::<f64>().ok();
            let Some(limit) = limit.filter(|limit| limit.is_finite() && *limit > 0.0) else {
                notices.write(Notice::error(
                    args.agent,
                    format!("Not a budget: {amount}; use /budget 2.00 or /budget off."),
                ));
                return;
            };
            budget.limit = Some(limit);
            // A budget nearly spent already says so here.
            budget.warned = budget.near();
            format!("Budget: {}.", budget.label())
        }
    };
    notices.write(Notice::info(args.agent, text));
}

/// The session's spending against its budget in the status line, red past
/// 80%; nothing without a budget.
fn show(budget: Res<Budget>, mut status: ResMut<AppStatus>) {
    let tone = if budget.near() { Tone::Red } else { Tone::Dim };
    let text = budget.limit.map(|_| budget.label()).unwrap_or_default();
    status.0.show(ITEM.says(text, tone));
}
