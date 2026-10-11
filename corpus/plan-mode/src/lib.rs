//! Plan mode (corpus spec `09-plan-mode.md`). `/plan` lets the shown agent
//! read but not change anything: its [`ToolAccess`] narrows to the tools
//! that change nothing, the tools it had are kept in its [`PlanMode`], and
//! each request tells the model to plan only. The answer that ends a
//! planning turn is the plan, kept with the mode and written to
//! `.rig/plan.md`; a picker then offers to execute it, keep planning or
//! discard it. Leaving gives the agent back exactly the tools it had.

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

/// What the plan is sent with once the user executes it.
pub const EXECUTE: &str = "Execute this plan: ";

/// On an agent in plan mode: the tools it had before, given back when it
/// leaves, and the plan of its last planning turn. Saved with the session,
/// and its removal too.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct PlanMode {
    /// The agent's tool access before plan mode.
    pub before: ToolAccess,
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
            .add_observer(plan_only)
            .add_observer(capture)
            .add_systems(PostUpdate, show.in_set(StatusSystems));
    }
}

/// `/plan` enters plan mode; `/plan off` and `/plan discard` leave it,
/// `/plan keep` stays, and `/plan execute` leaves it and sends the plan.
fn plan(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut ToolAccess, Option<&PlanMode>, Has<ActiveTurn>)>,
    tools: Query<(&ToolDef, &Footprint)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((mut access, planning, busy)) = agents.get_mut(agent) else {
        return;
    };
    // The tools change for the whole of a turn, never in its middle.
    if busy && args.args != "keep" {
        notices.write(Notice::error(agent, "A turn is running; stop it first."));
        return;
    }
    match (args.args.as_str(), planning) {
        ("", None) => {
            let before = access.clone();
            // Tools that change nothing; a subagent started now gets at
            // most these.
            let allowed = tools
                .iter()
                .filter(|(def, footprint)| {
                    **footprint != Footprint::Exclusive && before.allows(def.0.name.as_str())
                })
                .map(|(def, _)| def.0.name.as_str().to_owned())
                .collect();
            *access = ToolAccess::Only(allowed);
            commands
                .entity(agent)
                .insert(PlanMode { before, plan: None });
            notices.write(Notice::info(agent, "Plan mode: nothing can be changed."));
        }
        ("", Some(_)) => {
            notices.write(Notice::info(agent, "Already in plan mode."));
        }
        ("keep", Some(_)) => {
            notices.write(Notice::info(agent, "Still planning."));
        }
        ("off" | "discard", Some(mode)) => {
            *access = mode.before.clone();
            commands.entity(agent).remove::<PlanMode>();
            notices.write(Notice::info(agent, "Left plan mode."));
        }
        ("execute", Some(mode)) => {
            let Some(plan) = mode.plan.clone() else {
                notices.write(Notice::error(agent, "There is no plan to execute yet."));
                return;
            };
            *access = mode.before.clone();
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

/// A planning agent's requests tell the model to plan only. Its tools are
/// already narrowed by its [`ToolAccess`], which also refuses a call of
/// any other.
fn plan_only(mut prepare: On<PrepareRequest>, planning: Query<(), With<PlanMode>>) {
    if planning.contains(prepare.agent) {
        prepare.preamble.push_str(PLAN_ONLY);
    }
}

/// Keeps the answer that ended a planning turn as the plan, writes it to
/// the [`PlanFile`], and asks the user what to do with it.
fn capture(
    ended: On<TurnEnded>,
    mut planning: Query<&mut PlanMode>,
    file: Res<PlanFile>,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = ended.entity;
    let TurnOutcome::Answered(answer) = &ended.outcome else {
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
