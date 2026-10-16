//! Spend by origin (spec `H27-spend-by-origin.md`): who spent the tokens.
//!
//! Every model call ([`ModelRequest`]) is given an origin as it is
//! admitted ([`SpendOrigin`]), and its reported usage is added under that
//! origin once it settles, in [`SpendByOrigin`], saved with the session.
//! The origin is, in this order:
//!
//! - a turn's call ([`TurnCall`]): its agent by name, a subagent as
//!   `parent › child`;
//! - a call a plugin marked as its own ([`ProvidedBy`], as the plugin guide
//!   asks of what a plugin spawns while the app runs): the plugin, with the
//!   call's purpose as its reason;
//! - a call made under a tool call (its effect's `parent`): the tool's
//!   plugin and the tool;
//! - else `unknown`. The default plugins mark their own calls, so none of
//!   theirs ends up there.
//!
//! `/spend` shows the session's table, `/spend turn` the last turn's (the
//! calls since an agent nothing spawned last started a turn). It only
//! reports.

use std::collections::BTreeMap;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Usage, UsageTotals};
use serde::{Deserialize, Serialize};

/// The origin of a call nothing names.
pub const UNKNOWN: &str = "unknown";

/// How deep a subagent's lineage is followed.
const MAX_DEPTH: usize = 16;

/// Sums every model call's usage per origin, and adds `/spend`.
#[derive(Default)]
pub struct SpendByOriginPlugin;

impl Plugin for SpendByOriginPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SpendByOrigin>()
            .add_command(
                "spend",
                "Who spent the tokens: each agent, subagent and plugin; /spend turn for the last turn",
                spend,
            )
            .add_observer(attribute)
            .add_observer(tally)
            .add_observer(new_turn);
    }
}

/// The origin a model call's usage goes to, given as it is admitted.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct SpendOrigin(pub String);

/// What the model calls spent per origin: in the whole session, and in the
/// last turn. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpendByOrigin {
    /// Every call of the session.
    pub session: BTreeMap<String, UsageTotals>,
    /// The calls since an agent nothing spawned last started a turn.
    pub turn: BTreeMap<String, UsageTotals>,
}

impl SpendByOrigin {
    /// Adds a settled call's `usage` under `origin`.
    pub fn record(&mut self, origin: &str, usage: &Usage) {
        for table in [&mut self.session, &mut self.turn] {
            table.entry(origin.to_owned()).or_default().record(usage);
        }
    }
}

/// Gives each model call its origin as it is admitted, while the tool call
/// it may have been made under is still there.
fn attribute(
    admitted: On<Add<Admitted>>,
    calls: Query<
        (
            &Effect,
            Option<&ForAgent>,
            Option<&ProvidedBy>,
            Has<TurnCall>,
        ),
        Or<(With<ModelRequest>, With<TurnCall>)>,
    >,
    agents: Query<(&Name, Option<&SpawnedBy>), With<Agent>>,
    tool_calls: Query<(&Effect, &ToolRequest), ToolCalls>,
    tools: Query<(&ToolDef, Option<&ProvidedBy>)>,
    plugins: Query<&Name, With<PluginSource>>,
    mut commands: Commands,
) {
    let Ok((effect, agent, provided, turn_call)) = calls.get(admitted.entity) else {
        return;
    };
    let reason = effect.purpose.as_deref();
    let plugin = |entity: Entity| plugins.get(entity).ok().map(short_name);
    let origin = match (turn_call, agent, provided) {
        (true, Some(&ForAgent(agent)), _) => lineage(agent, &agents),
        (_, _, Some(&ProvidedBy(by))) => with_reason(plugin(by), reason),
        _ => {
            let tool = effect.parent.and_then(|parent| {
                let (_, request) = tool_calls.iter().find(|(call, _)| call.id() == parent)?;
                let name = request.call.function.name.as_str();
                let by = tools.iter().find(|(def, _)| def.0.name.as_str() == name);
                let by = by.and_then(|(_, by)| by).and_then(|by| plugin(by.0));
                Some(match by {
                    Some(by) => format!("{by} › tool {name}"),
                    None => format!("tool {name}"),
                })
            });
            with_reason(tool, reason)
        }
    };
    commands
        .entity(admitted.entity)
        .try_insert(SpendOrigin(origin));
}

/// `name`, and `reason` in brackets when there is one; `unknown` without a
/// name.
fn with_reason(name: Option<String>, reason: Option<&str>) -> String {
    match (name, reason) {
        (Some(name), Some(reason)) => format!("{name} ({reason})"),
        (Some(name), None) => name,
        (None, _) => UNKNOWN.to_owned(),
    }
}

/// A plugin's type name without its path.
fn short_name(name: &Name) -> String {
    let name = name.as_str();
    name.rsplit("::").next().unwrap_or(name).to_owned()
}

/// `agent`'s name after the names of the agents that spawned it:
/// `main › scout`.
fn lineage(agent: Entity, agents: &Query<(&Name, Option<&SpawnedBy>), With<Agent>>) -> String {
    let mut names = Vec::new();
    let mut next = Some(agent);
    while let Some(agent) = next.filter(|_| names.len() < MAX_DEPTH) {
        let Ok((name, by)) = agents.get(agent) else {
            break;
        };
        names.push(name.as_str().to_owned());
        next = by.map(|by| by.0);
    }
    if names.is_empty() {
        return UNKNOWN.to_owned();
    }
    names.reverse();
    names.join(" › ")
}

/// Adds each settled model call's reported usage under its origin.
fn tally(
    settled: On<Add<Settled>>,
    calls: Query<(&Outcome, Option<&SpendOrigin>), With<ModelRequest>>,
    mut spend: ResMut<SpendByOrigin>,
) {
    let Ok((outcome, origin)) = calls.get(settled.entity) else {
        return;
    };
    if let Some(response) = outcome.completion() {
        let origin = origin.map_or(UNKNOWN, |origin| origin.0.as_str());
        spend.record(origin, &response.usage);
    }
}

/// Starts the last turn's table when an agent nothing spawned starts a
/// turn, not when a restart carries one on.
fn new_turn(
    added: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, Has<Restored>)>,
    roots: Query<(), (With<Agent>, Without<SpawnedBy>)>,
    mut spend: ResMut<SpendByOrigin>,
) {
    if let Ok((&TurnOf(agent), false)) = turns.get(added.entity)
        && roots.contains(agent)
    {
        spend.turn.clear();
    }
}

/// `/spend`, or `/spend turn`: a table of what each origin spent.
fn spend(In(args): In<CommandArgs>, spend: Res<SpendByOrigin>, mut notices: MessageWriter<Notice>) {
    let (table, what) = match args.args.trim() {
        "" => (&spend.session, "this session"),
        "turn" => (&spend.turn, "the last turn"),
        other => {
            let why = format!("/spend takes nothing or `turn`, not `{other}`");
            notices.write(Notice::error(args.agent, why));
            return;
        }
    };
    let text = if table.is_empty() {
        format!("No model call in {what} yet.")
    } else {
        format!("Spend by origin, {what}:\n{}", render(table))
    };
    notices.write(Notice::info(args.agent, text));
}

/// The table: one row per origin, the most expensive first, then the total.
fn render(table: &BTreeMap<String, UsageTotals>) -> String {
    let mut total = UsageTotals::default();
    for spent in table.values() {
        total.add(spent);
    }
    // Shares of the cost when any call was priced, else of the tokens.
    let by_cost = total.cost > 0.0;
    let weight = |spent: &UsageTotals| {
        if by_cost {
            spent.cost
        } else {
            spent.total_tokens() as f64
        }
    };
    let mut rows: Vec<(&String, &UsageTotals)> = table.iter().collect();
    rows.sort_by(|a, b| weight(b.1).total_cmp(&weight(a.1)).then(a.0.cmp(b.0)));
    let all = weight(&total);
    let share = |spent: &UsageTotals| {
        if all > 0.0 {
            format!("{:.0}%", 100.0 * weight(spent) / all)
        } else {
            "-".to_owned()
        }
    };
    let row = |origin: &str, spent: &UsageTotals| {
        [
            origin.to_owned(),
            spent.calls.to_string(),
            spent.tokens.input_tokens.unwrap_or(0).to_string(),
            spent.tokens.cached_input_tokens.unwrap_or(0).to_string(),
            spent.tokens.output_tokens.unwrap_or(0).to_string(),
            spent.cost_label().unwrap_or_else(|| "-".to_owned()),
            share(spent),
        ]
    };
    let header = [
        "origin", "calls", "input", "cached", "output", "cost", "share",
    ]
    .map(str::to_owned);
    let mut lines = vec![header];
    lines.extend(rows.into_iter().map(|(origin, spent)| row(origin, spent)));
    lines.push(row("total", &total));
    let mut widths = [0; 7];
    for line in &lines {
        for (width, cell) in widths.iter_mut().zip(line) {
            *width = (*width).max(cell.chars().count());
        }
    }
    let lines: Vec<String> = lines
        .iter()
        .map(|line| {
            let cells = line
                .iter()
                .zip(widths)
                .enumerate()
                .map(|(column, (cell, width))| {
                    let pad = width.saturating_sub(cell.chars().count());
                    if column == 0 {
                        format!("{cell}{}", " ".repeat(pad))
                    } else {
                        format!("{}{cell}", " ".repeat(pad))
                    }
                });
            cells.collect::<Vec<_>>().join("  ").trim_end().to_owned()
        })
        .collect();
    lines.join("\n")
}

#[cfg(test)]
mod tests;
