//! Plan mode (corpus spec `09-plan-mode.md`). `/plan` lets the shown agent
//! read but not change anything: a Prepare system offers its model calls
//! only the tools that change nothing and tells the model to plan, and a
//! Gate system refuses, with its own reason, a call that may change
//! something of the agent or of a subagent it started while planning. The
//! answer that ends a planning turn is the plan, kept with the mode and
//! written to `.rig/plan.md`; a picker then offers to execute it, keep
//! planning or discard it. The agent's tools are never changed, so leaving
//! gives it back exactly what it had.

use std::path::PathBuf;

use rig_harness::prelude::*;
use rig_tui::{PickItem, PickRequest, Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The status item: after the reasoning setting, before the status.
const PLAN_ITEM: StatusItem = StatusItem::at(Side::Left, 45, 17);

/// What every request of a planning agent adds to its system prompt.
pub const PLAN_ONLY: &str = "\n\nPlan mode: you may read and search, but change nothing. \
    Explore what you need, then end your turn with a step-by-step plan as your answer. \
    The plan is carried out only once the user approves it.";

/// Why a call that may change something is refused in plan mode.
pub const REFUSED: &str = "plan mode: changes are not allowed";

/// What the plan is sent with once the user executes it.
pub const EXECUTE: &str = "Execute this plan: ";

/// On an agent in plan mode: the plan of its last planning turn. Saved
/// with the session, and its removal too.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct PlanMode {
    /// The last planning turn's answer.
    pub plan: Option<String>,
}

/// Where a captured plan is written, relative to the working directory
/// unless absolute.
#[derive(Resource, Clone, Debug)]
pub struct PlanFile(pub PathBuf);

impl Default for PlanFile {
    fn default() -> Self {
        Self(PathBuf::from(".rig").join("plan.md"))
    }
}

/// `/plan`, the plan-only requests, the plan's capture and its picker.
#[derive(Default)]
pub struct PlanModePlugin;

impl Plugin for PlanModePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<PlanMode>()
            .init_resource::<PlanFile>()
            .add_message::<PickRequest>()
            .add_command(
                "plan",
                "Plan before changing anything: /plan, /plan off",
                plan,
            )
            .add_systems(
                AgentPass,
                (
                    plan_only.in_set(Stage::Prepare),
                    no_changes.in_set(Stage::Gate),
                ),
            )
            .add_observer(capture)
            .add_systems(PostUpdate, show.in_set(StatusSystems));
    }
}

/// `/plan` enters plan mode; `/plan off` and `/plan discard` leave it,
/// `/plan keep` stays, and `/plan execute` leaves it and sends the plan.
fn plan(
    In(args): In<CommandArgs>,
    agents: Query<(Option<&PlanMode>, Has<ActiveTurn>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((planning, busy)) = agents.get(agent) else {
        return;
    };
    // The tools change for the whole of a turn, never in its middle.
    if busy && args.args != "keep" {
        notices.write(Notice::error(agent, "A turn is running; stop it first."));
        return;
    }
    match (args.args.as_str(), planning) {
        ("", None) => {
            commands.entity(agent).insert(PlanMode { plan: None });
            notices.write(Notice::info(agent, "Plan mode: nothing can be changed."));
        }
        ("", Some(_)) => {
            notices.write(Notice::info(agent, "Already in plan mode."));
        }
        ("keep", Some(_)) => {
            notices.write(Notice::info(agent, "Still planning."));
        }
        ("off" | "discard", Some(_)) => {
            commands.entity(agent).remove::<PlanMode>();
            notices.write(Notice::info(agent, "Left plan mode."));
        }
        ("execute", Some(mode)) => {
            let Some(plan) = mode.plan.clone() else {
                notices.write(Notice::error(agent, "There is no plan to execute yet."));
                return;
            };
            commands.entity(agent).remove::<PlanMode>();
            // The user chose to execute it: their request, word for word.
            let text = format!("{EXECUTE}{plan}");
            commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue).literal());
        }
        (_, None) => {
            notices.write(Notice::error(agent, "Not in plan mode; /plan enters it."));
        }
        (other, Some(_)) => {
            notices.write(Notice::error(
                agent,
                format!("Unknown /plan argument `{other}`: use off, execute, keep or discard."),
            ));
        }
    }
}

/// Whether `agent` plans: it is in plan mode, or an agent that started it
/// is.
fn plans(agent: Entity, planning: &Query<(), With<PlanMode>>, parents: &Query<&SpawnedBy>) -> bool {
    std::iter::once(agent)
        .chain(parents.iter_ancestors(agent))
        .any(|agent| planning.contains(agent))
}

/// A planning agent's turn calls offer only the tools that change nothing,
/// and tell the model to plan only.
fn plan_only(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    tools: Query<(&ToolDef, &Footprint)>,
    planning: Query<(), With<PlanMode>>,
    parents: Query<&SpawnedBy>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        if !plans(agent, &planning, &parents) {
            continue;
        }
        let changes = |name: &str| {
            let tool = tools.iter().find(|(def, _)| def.0.name.as_str() == name);
            tool.is_none_or(|(_, footprint)| *footprint == Footprint::Exclusive)
        };
        call.request
            .tools
            .retain(|tool| !changes(tool.name.as_str()));
        if let Some(prompt) = call.system_prompt_mut() {
            prompt.push_str(PLAN_ONLY);
        }
    }
}

/// Refuses each call that may change something, of a planning agent, when
/// it is about to run.
fn no_changes(
    calls: Query<(Entity, &Footprint, &ForAgent), (Undecided, With<ToolRequest>)>,
    planning: Query<(), With<PlanMode>>,
    parents: Query<&SpawnedBy>,
    mut commands: Commands,
) {
    for (call, footprint, &ForAgent(agent)) in &calls {
        if *footprint == Footprint::Exclusive && plans(agent, &planning, &parents) {
            commands
                .entity(call)
                .insert_if_new(Outcome::refused(REFUSED));
        }
    }
}

/// Keeps the answer that ended a planning turn as the plan, writes it to
/// the [`PlanFile`], and asks the user what to do with it.
fn capture(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut planning: Query<&mut PlanMode>,
    file: Res<PlanFile>,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((TurnOutcome::Answered(answer), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let (Ok(mut mode), Some(plan)) = (planning.get_mut(agent), final_answer(answer)) else {
        return;
    };
    if let Err(error) = write_plan(&file.0, &plan) {
        notices.write(Notice::error(
            agent,
            format!("The plan was not saved to {}: {error}", file.0.display()),
        ));
    }
    mode.plan = Some(plan);
    let item = |label: &str, command: &str| PickItem {
        label: label.to_owned(),
        command: format!("plan {command}"),
    };
    picks.write(PickRequest {
        agent,
        title: "The plan".to_owned(),
        items: vec![
            item("Execute", "execute"),
            item("Keep planning", "keep"),
            item("Discard", "discard"),
        ],
        selected: 0,
    });
}

/// Writes `plan` to `path`, with its directory.
fn write_plan(path: &std::path::Path, plan: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, plan)
}

/// "PLAN" on the status line of an agent in plan mode.
fn show(
    entered: Query<Entity, Added<PlanMode>>,
    mut left: RemovedComponents<PlanMode>,
    mut agents: Query<(&mut StatusItems, Has<PlanMode>)>,
) {
    for agent in entered.iter().chain(left.read()) {
        if let Ok((mut items, planning)) = agents.get_mut(agent) {
            let text = if planning { "PLAN" } else { "" };
            items.show(PLAN_ITEM.says(text, Tone::Yellow));
        }
    }
}
