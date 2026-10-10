//! `--print`: one prompt to the primary agent, then exit once no agent
//! works, so the subagents it started have answered. The answer's text
//! goes to stdout and failures to stderr; the exit code
//! is 0 when the turn ended with an answer, 2 for a refused slash command
//! (unknown, or arguments it does not take), 1 otherwise. Text piped in on
//! stdin follows the prompt, so `git diff | rig -p "review this"` works.
//!
//! The agent uses `--model`, else the model its restored session chose,
//! else the first model the environment has a key for.
//!
//! It is the front of a `--print` run, and of any run no other front took,
//! such as an agent built without the terminal view: that run answers what
//! is piped in.

use std::io::{IsTerminal, Read as _};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;

use rig_core::transcript::final_answer;
use rig_ecs::agent::{ActiveTurn, Agent, Conversation, Notice, NoticeLevel, PrimaryQuery, primary};
use rig_ecs::inbox::{DeliveryMode, Recalled};
use rig_ecs::model::{Connection, ModelChoice, Models, SetModel};
use rig_harness::front::{Busy, Front, RunMode, send_input};

/// Sends the prompt and exits when the turn ends.
#[derive(Default)]
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn build(&self, _: &mut App) {}

    /// Takes the run once every plugin is built, unless another front did.
    fn cleanup(&self, app: &mut App) {
        if app.world().contains_resource::<Front>() {
            return;
        }
        let prompt = app
            .world()
            .get_resource::<RunMode>()
            .and_then(|mode| mode.0.print.clone())
            .unwrap_or_default();
        app.insert_resource(Front("print".to_owned()))
            .insert_resource(PrintRun {
                prompt,
                step: Step::Start,
                failed: false,
                refused: false,
            })
            .add_systems(Update, (print_notices, drive).chain());
    }
}

/// Where the print run is.
#[derive(Resource)]
struct PrintRun {
    prompt: String,
    step: Step,
    /// Whether an error notice about the agent came while it ran.
    failed: bool,
    /// Whether the prompt was a refused slash command.
    refused: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Nothing done yet.
    Start,
    /// A model was chosen for `agent`, which connects it at once.
    Connecting { agent: Entity },
    /// The prompt went to `agent`, whose conversation had `before`
    /// messages.
    Sent { agent: Entity, before: usize },
    /// Exited.
    Done,
}

/// The prompt with what was piped in on stdin after it.
fn full_prompt(prompt: &str) -> String {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return prompt.to_owned();
    }
    let mut piped = String::new();
    if stdin.lock().read_to_string(&mut piped).is_err() || piped.trim().is_empty() {
        return prompt.to_owned();
    }
    if prompt.trim().is_empty() {
        piped
    } else {
        format!("{prompt}\n\n{piped}")
    }
}

/// The agents as the print run reads them.
#[derive(SystemParam)]
struct Agents<'w, 's> {
    primary: PrimaryQuery<'w, 's>,
    choices: Query<'w, 's, (Option<&'static ModelChoice>, Has<Connection>)>,
    conversations: Query<'w, 's, &'static Conversation>,
    /// Agents in a turn.
    working: Query<'w, 's, (), (With<Agent>, With<ActiveTurn>)>,
    /// Other work a front waits for, such as a sign-in.
    busy: Query<'w, 's, (), With<Busy>>,
}

/// Chooses the model, sends the prompt, and exits after the turn.
fn drive(
    mut run: ResMut<PrintRun>,
    mode: Res<RunMode>,
    agents: Agents,
    models: Res<Models>,
    mut commands: Commands,
    mut exits: MessageWriter<AppExit>,
) {
    // A command such as `/login` runs without a model.
    let command = run.prompt.trim_start().starts_with('/');
    match run.step {
        Step::Start => {
            let Some(agent) = primary(&agents.primary) else {
                return;
            };
            let Ok((chosen, _)) = agents.choices.get(agent) else {
                return;
            };
            let model = match (&mode.0.model, chosen) {
                // The front chose it, or the restored session did, and it
                // was connected, or refused with a notice, already.
                (Some(_), _) | (None, Some(_)) => None,
                (None, None) => match models.0.reachable().first() {
                    Some(spec) => Some(spec.reference()),
                    None if command => None,
                    None => {
                        eprintln!(
                            "rig: no model can be reached: set a provider's API key, or name one \
                             with --model"
                        );
                        run.step = Step::Done;
                        exits.write(AppExit::from_code(1));
                        return;
                    }
                },
            };
            if let Some(model) = model {
                commands.trigger(SetModel {
                    entity: agent,
                    model,
                });
            }
            run.step = Step::Connecting { agent };
        }
        Step::Connecting { agent } => {
            let connected = command
                || agents
                    .choices
                    .get(agent)
                    .is_ok_and(|(_, connected)| connected);
            if !connected {
                // The notice of the failed choice says why.
                run.step = Step::Done;
                exits.write(AppExit::from_code(1));
                return;
            }
            let text = full_prompt(&run.prompt);
            if text.trim().is_empty() {
                eprintln!("rig: no prompt: give one after --print, or pipe it in");
                run.step = Step::Done;
                exits.write(AppExit::from_code(2));
                return;
            }
            let before = agents
                .conversations
                .get(agent)
                .map_or(0, |conversation| conversation.messages().len());
            send_input(&mut commands, agent, text, DeliveryMode::Steer);
            run.step = Step::Sent { agent, before };
        }
        Step::Sent { agent, before } => {
            // The turn starts with the request; a command may start none,
            // or other work, such as a sign-in. A subagent's answer starts
            // another turn of the agent that started it.
            if !agents.working.is_empty() || !agents.busy.is_empty() {
                return;
            }
            let answer = agents
                .conversations
                .get(agent)
                .ok()
                .filter(|conversation| conversation.messages().len() > before)
                .and_then(|conversation| conversation.messages().last())
                .and_then(final_answer);
            let ok = !run.failed || answer.is_some();
            if let Some(answer) = &answer {
                println!("{}", answer.trim_end());
            }
            run.step = Step::Done;
            exits.write(match (run.refused, ok) {
                (true, _) => AppExit::from_code(2),
                (false, true) => AppExit::Success,
                (false, false) => AppExit::from_code(1),
            });
        }
        Step::Done => {}
    }
}

/// Notes failures and writes them to stderr, with every notice of a
/// `/command` prompt and why it was refused.
fn print_notices(
    mut run: ResMut<PrintRun>,
    mut notices: MessageReader<Notice>,
    mut recalled: MessageReader<Recalled>,
) {
    for why in recalled.read().filter_map(|recalled| recalled.why.as_ref()) {
        eprintln!("rig: {why}");
        run.refused = true;
    }
    // A command's answers are its notices.
    let command = run.prompt.trim_start().starts_with('/');
    for notice in notices.read() {
        if notice.level != NoticeLevel::Error {
            if command {
                eprintln!("{}", notice.text);
            }
            continue;
        }
        if let Step::Sent { agent, .. } | Step::Connecting { agent, .. } = run.step
            && notice.agent.is_none_or(|about| about == agent)
        {
            run.failed = true;
        }
        eprintln!("rig: {}", notice.text);
    }
}

#[cfg(test)]
mod tests;
