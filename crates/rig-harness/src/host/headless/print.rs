//! `--print`: one prompt to the primary agent, then exit once no agent
//! works, so the subagents it started have answered. The answer's text
//! goes to stdout and failures to stderr; the exit code
//! is 0 when the turn ended with an answer, 1 otherwise. Text piped in on
//! stdin follows the prompt, so `git diff | rig -p "review this"` works.
//!
//! The agent uses `--model`, else the model its restored session chose,
//! else the first model the environment has a key for.

use std::io::{IsTerminal, Read as _};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::completion::{AssistantContent, Message};

use super::{PrimaryQuery, RunMode, primary};
use crate::core::agent::{
    ActiveTurn, Agent, Connection, Conversation, ModelChoice, Notice, NoticeLevel, SetModel,
};
use crate::core::commands::send_input;
use crate::core::inbox::DeliveryMode;
use crate::core::login::PendingLogin;
use crate::core::models;

/// Frames to wait for a model to connect before giving up.
const CONNECT_FRAMES: u32 = 3;

/// Sends the prompt and exits when the turn ends.
pub(super) struct PrintPlugin {
    pub(super) prompt: String,
}

impl Plugin for PrintPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PrintRun {
            prompt: self.prompt.clone(),
            step: Step::Start,
            failed: false,
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Nothing done yet.
    Start,
    /// A model was chosen for `agent`; waiting `frames` more for it to
    /// connect.
    Connecting { agent: Entity, frames: u32 },
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

/// Chooses the model, sends the prompt, and exits after the turn.
#[allow(clippy::too_many_arguments)]
fn drive(
    mut run: ResMut<PrintRun>,
    mode: Res<RunMode>,
    agents: PrimaryQuery,
    models_of: Query<(Option<&ModelChoice>, Has<Connection>)>,
    working: Query<(), (With<Agent>, With<ActiveTurn>)>,
    conversations: Query<&Conversation>,
    logins: Query<(), With<PendingLogin>>,
    mut commands: Commands,
    mut exits: MessageWriter<AppExit>,
) {
    // A command such as `/login` runs without a model.
    let command = run.prompt.trim_start().starts_with('/');
    match run.step {
        Step::Start => {
            let Some(agent) = primary(&agents) else {
                return;
            };
            let Ok((chosen, connected)) = models_of.get(agent) else {
                return;
            };
            let model = match (&mode.0.model, chosen) {
                (Some(model), _) => Some(model.clone()),
                (None, Some(_)) if connected => None,
                (None, Some(chosen)) => Some(chosen.0.clone()),
                (None, None) => match models::available_models().first() {
                    Some(spec) => Some(models::reference(spec)),
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
            run.step = Step::Connecting {
                agent,
                frames: CONNECT_FRAMES,
            };
        }
        Step::Connecting { agent, frames } => {
            let connected = command || models_of.get(agent).is_ok_and(|(_, connected)| connected);
            if !connected {
                if frames == 0 {
                    run.step = Step::Done;
                    exits.write(AppExit::from_code(1));
                } else {
                    run.step = Step::Connecting {
                        agent,
                        frames: frames - 1,
                    };
                }
                return;
            }
            let text = full_prompt(&run.prompt);
            if text.trim().is_empty() {
                eprintln!("rig: no prompt: give one after --print, or pipe it in");
                run.step = Step::Done;
                exits.write(AppExit::from_code(2));
                return;
            }
            let before = conversations
                .get(agent)
                .map_or(0, |conversation| conversation.messages().len());
            send_input(&mut commands, agent, text, DeliveryMode::Steer);
            run.step = Step::Sent { agent, before };
        }
        Step::Sent { agent, before } => {
            // The turn starts with the request; a command may start none,
            // or a sign-in. A subagent's answer starts another turn of the
            // agent that started it.
            if !working.is_empty() || !logins.is_empty() {
                return;
            }
            let answer = conversations
                .get(agent)
                .ok()
                .filter(|conversation| conversation.messages().len() > before)
                .and_then(|conversation| conversation.messages().last())
                .and_then(answer_text);
            let ok = !run.failed || answer.is_some();
            if let Some(answer) = &answer {
                println!("{}", answer.trim_end());
            }
            run.step = Step::Done;
            exits.write(if ok {
                AppExit::Success
            } else {
                AppExit::from_code(1)
            });
        }
        Step::Done => {}
    }
}

/// The text of a final answer: the model's last message, with no tool
/// calls left to run.
fn answer_text(message: &Message) -> Option<String> {
    let Message::Assistant(reply) = message else {
        return None;
    };
    let mut text = String::new();
    for part in &reply.content {
        match part {
            AssistantContent::Text(part) => text.push_str(&part.text),
            AssistantContent::ToolCall(_) => return None,
            _ => {}
        }
    }
    Some(text)
}

/// Notes failures and writes them to stderr, with every notice of a
/// `/command` prompt.
fn print_notices(mut run: ResMut<PrintRun>, mut notices: MessageReader<Notice>) {
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
