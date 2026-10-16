//! Plan mode (corpus spec `09-plan-mode.md`). `/plan` lets the shown agent
//! read but not change anything: a Prepare system offers its model calls
//! only the read-only tools ([`Touches::Reads`], which a tool's
//! read-only annotation gives it) and tells the model to plan, and a Gate
//! system refuses, with its own reason, any other call, of the agent or
//! of an agent it spawned. A tool that changes nothing its reply's other
//! calls see ([`Touches::Independent`]: starting a subagent, scheduling
//! a job, a reload) still acts, so it waits until the plan is approved. The answer that ends
//! a planning turn is the plan: it is written to `.rig/plan.md` and the
//! user is asked whether to execute it, keep planning or discard it. The
//! agent's tools are never changed, so leaving gives back exactly what it
//! had.

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Section;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

#[cfg(test)]
mod tests;

/// The status item: after the reasoning setting, before the status.
const PLAN_ITEM: StatusItem = StatusItem::new("plan-mode", Side::Left, 45, 17);

/// What every request of a planning agent adds to its system prompt.
pub const PLAN_ONLY: &str = "Plan mode: you may read and search, but change nothing. \
    Explore what you need, then end your turn with a step-by-step plan as your answer. \
    The plan is carried out only once the user approves it.";

/// Why a call that may change something is refused in plan mode.
pub const REFUSED: &str = "plan mode: changes are not allowed";

/// What the plan is sent with once the user executes it.
pub const EXECUTE: &str = "Execute this plan: ";

/// Where a captured plan is written, under the project.
pub const PLAN_FILE: &str = ".rig/plan.md";

/// On an agent in plan mode; saved with the session, and its removal too.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved, Default)]
pub struct PlanMode;

/// On the question about a captured plan.
#[derive(Component)]
struct PlanQuestion;

/// The plan mode plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanModeSystems {
    /// In [`Stage::Prepare`], after the default prompt went in and before
    /// compaction measures the request: a planning agent's requests offer
    /// only the read-only tools.
    Prepare,
    /// In [`Stage::Gate`]: refuses a planning agent's calls of any other
    /// tool.
    Gate,
}

/// `/plan`, the plan-only requests and calls, the plan's capture and the
/// question about it.
#[derive(Default)]
pub struct PlanModePlugin;

impl Plugin for PlanModePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<PlanMode>()
            .init_resource::<ProjectRoot>()
            .add_command(
                "plan",
                "Plan before changing anything: /plan, /plan off",
                plan,
            )
            .configure_sets(
                AgentPass,
                (
                    PlanModeSystems::Prepare.in_set(PrepareSystems::Edit),
                    PlanModeSystems::Gate.in_set(Stage::Gate),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    plan_only.in_set(PlanModeSystems::Prepare),
                    no_changes.in_set(PlanModeSystems::Gate),
                ),
            )
            .add_observer(capture)
            .add_observer(decide)
            .add_systems(PostUpdate, show.in_set(StatusSystems::Show));
    }
}

/// `/plan` enters plan mode; `/plan off` leaves it.
fn plan(In(args): In<CommandArgs>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    let agent = args.agent;
    match args.args.as_str() {
        "" => {
            commands.entity(agent).insert(PlanMode);
            notices.write(Notice::info(agent, "Plan mode: nothing can be changed."));
        }
        "off" => {
            commands.entity(agent).remove::<PlanMode>();
            notices.write(Notice::info(agent, "Left plan mode."));
        }
        other => {
            notices.write(Notice::error(
                agent,
                format!("Unknown /plan argument `{other}`: use /plan or /plan off."),
            ));
        }
    }
}

/// Whether `agent` plans: it is in plan mode, or an agent that spawned it
/// is.
fn plans(agent: Entity, planning: &Query<(), With<PlanMode>>, parents: &Query<&SpawnedBy>) -> bool {
    std::iter::once(agent)
        .chain(parents.iter_ancestors(agent))
        .any(|agent| planning.contains(agent))
}

/// A planning agent's turn calls offer only the read-only tools, and tell
/// the model to plan only.
fn plan_only(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    tools: Query<(&ToolDef, &Footprint)>,
    planning: Query<(), With<PlanMode>>,
    parents: Query<&SpawnedBy>,
) {
    let reads: Vec<&str> = tools
        .iter()
        .filter(|(_, footprint)| footprint.0.reads_only())
        .map(|(def, _)| def.0.name.as_str())
        .collect();
    for (mut call, &ForAgent(agent)) in &mut calls {
        if plans(agent, &planning, &parents) {
            call.request_mut()
                .tools
                .retain(|tool| reads.contains(&tool.name.as_str()));
            if let Some(prompt) = &mut call.prompt {
                prompt.sections.push(Section::new("plan_mode", PLAN_ONLY));
            }
        }
    }
}

/// Refuses each call of a tool that is not read-only, of a planning agent,
/// when it is about to run.
fn no_changes(
    mut calls: Query<(&Footprint, &ForAgent, &mut Decision), (ReadyToStart, ToolCalls)>,
    planning: Query<(), With<PlanMode>>,
    parents: Query<&SpawnedBy>,
) {
    for (footprint, &ForAgent(agent), mut decision) in &mut calls {
        if !footprint.0.reads_only() && plans(agent, &planning, &parents) {
            decision.decide(Decider::plugin("plan-mode"), Outcome::refused(REFUSED));
        }
    }
}

/// Keeps the answer that ended a planning turn as the plan, in the
/// project's [`PLAN_FILE`], and asks the user what to do with it.
fn capture(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    planning: Query<(), With<PlanMode>>,
    root: Res<ProjectRoot>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((TurnOutcome::Answered(answer), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Some(plan) = answer.final_answer().filter(|_| planning.contains(agent)) else {
        return;
    };
    let path = root.0.join(PLAN_FILE);
    let written = path.parent().map_or(Ok(()), std::fs::create_dir_all);
    if let Err(error) = written.and_then(|()| std::fs::write(&path, &plan)) {
        notices.write(Notice::error(
            agent,
            format!("The plan was not saved to {}: {error}", path.display()),
        ));
    }
    if !answerers.can_answer(Some(agent)) {
        notices.write(Notice::info(
            agent,
            "Still in plan mode; /plan off leaves it.",
        ));
        return;
    }
    let question = Ask::new("The plan")
        .body(plan)
        .choice("execute", "Execute")
        .choice("keep", "Keep planning")
        .choice("discard", "Discard");
    commands.spawn((question, PlanQuestion, ForAgent(agent)));
}

/// Carries out the user's answer about a plan: Execute leaves plan mode and
/// sends the plan, Discard leaves it, and Keep planning, or no answer, stays.
fn decide(
    settled: On<Add<Settled>>,
    questions: Query<(&Ask, &Outcome, &ForAgent), With<PlanQuestion>>,
    mut commands: Commands,
) {
    let Ok((question, outcome, &ForAgent(agent))) = questions.get(settled.entity) else {
        return;
    };
    let Some(answer) = Answer::of(outcome) else {
        return;
    };
    if answer.is("execute") || answer.is("discard") {
        commands.entity(agent).remove::<PlanMode>();
    }
    if answer.is("execute") {
        // The user chose to execute it: their request, word for word.
        let text = format!("{EXECUTE}{}", question.body);
        commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue).literal());
    }
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
