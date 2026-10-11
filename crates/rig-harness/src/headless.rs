//! Driving a run nobody sits at, for a front such as `--print` or one
//! that streams JSON: [`AppHeadlessExt::run_headless`] sends one prompt to
//! the primary agent, then exits once no agent works, so the subagents it
//! started have answered. Just before the exit, [`HeadlessEnded`] says how
//! it went, for the front to write out.
//!
//! The agent uses `--model`, else the model its restored session chose,
//! else the first model the environment has a key for. A prompt that is a
//! slash command, such as `/login`, runs without a model. The exit code is
//! 0 when the turn ended with an answer, 2 for an empty prompt or a refused
//! slash command (unknown, or arguments it does not take), 1 otherwise,
//! such as an error notice about the agent and no answer. A warning
//! notice never fails it.
//!
//! ```
//! use rig_harness::headless::{AppHeadlessExt, HeadlessEnded};
//! use rig_harness::prelude::*;
//!
//! #[derive(Default)]
//! struct AnswerFront;
//!
//! impl Plugin for AnswerFront {
//!     fn build(&self, app: &mut App) {
//!         app.add_front(Front::new("answer").on_flag("answer-only", "Print only the answer"));
//!     }
//!
//!     fn finish(&self, app: &mut App) {
//!         if app.runs_front("answer") {
//!             app.run_headless("Say hello").add_observer(|ended: On<HeadlessEnded>| {
//!                 println!("{}", ended.answer.as_deref().unwrap_or_default());
//!             });
//!         }
//!     }
//! }
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use rig_core::completion::Message;
use rig_core::transcript::final_answer;
use rig_ecs::prelude::*;

use crate::host::exit::{Exit, ExitReason};
use crate::host::launcher::Invoked;
use crate::primary::{PrimaryQuery, primary};

/// Running a headless run.
pub trait AppHeadlessExt {
    /// Sends `prompt` to the primary agent once it has a model, and exits
    /// when no agent works any more. Call it once, from the front's
    /// `finish`.
    fn run_headless(&mut self, prompt: impl Into<String>) -> &mut Self;
}

impl AppHeadlessExt for App {
    fn run_headless(&mut self, prompt: impl Into<String>) -> &mut Self {
        let model = self
            .world()
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.args.model.is_some());
        self.insert_resource(HeadlessRun {
            prompt: prompt.into(),
            model,
            step: Step::Start,
            answer: None,
            failed: false,
            refused: false,
        })
        .add_systems(Update, drive)
        .add_observer(keep_answer)
    }
}

/// How a headless run ended, triggered as it asks to exit ([`Exit`]).
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub struct HeadlessEnded {
    /// The agent the prompt went to, if it got that far.
    pub agent: Option<Entity>,
    /// The final answer to the prompt, if it had one.
    pub answer: Option<String>,
    /// Why the run could not send the prompt, such as no model.
    pub why: Option<String>,
    /// Why it exits: [`ExitReason::Quit`] when it answered, else
    /// [`ExitReason::Failed`] with its exit code.
    pub exit: ExitReason,
}

/// Where the headless run is.
#[derive(Resource)]
struct HeadlessRun {
    prompt: String,
    /// Whether `--model` named the model, which a models plugin chooses.
    model: bool,
    step: Step,
    /// The message the last turn of the agent the prompt went to answered
    /// with, if one did.
    answer: Option<Message>,
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
    /// The prompt went to `agent`.
    Sent { agent: Entity },
    /// Exited.
    Done,
}

/// The agents as the run reads them.
#[derive(SystemParam)]
struct Agents<'w, 's> {
    primary: PrimaryQuery<'w, 's>,
    choices: Query<'w, 's, (Option<&'static ModelChoice>, Has<Connection>)>,
    /// Running turns.
    turns: Query<'w, 's, (), With<TurnOf>>,
    /// Work in progress, such as a sign-in; not a long-lived stream, such
    /// as a file watch.
    running: Query<'w, 's, &'static Running>,
    /// Messages on their way to an agent, such as the prompt itself.
    messages: Query<'w, 's, (), (With<Incoming>, Without<Settled>)>,
}

impl Agents<'_, '_> {
    /// Whether no agent works: no turn, task or message on its way.
    fn idle(&self) -> bool {
        self.turns.is_empty()
            && self.messages.is_empty()
            && self.running.iter().all(Running::is_stream)
    }
}

/// Chooses the model, sends the prompt, and exits after the turn.
fn drive(
    mut run: ResMut<HeadlessRun>,
    agents: Agents,
    models: Res<Models>,
    (mut notices, mut recalled): (MessageReader<Notice>, MessageReader<Recalled>),
    mut commands: Commands,
) {
    if recalled.read().any(|recalled| recalled.why.is_some()) {
        run.refused = true;
    }
    for notice in notices.read() {
        if let Step::Sent { agent, .. } | Step::Connecting { agent } = run.step
            && notice.level == NoticeLevel::Error
            && notice.agent.is_none_or(|about| about == agent)
        {
            run.failed = true;
        }
    }
    // A command such as `/login` runs without a model.
    let command = run.prompt.trim_start().starts_with('/');
    let failed = |agent: Option<Entity>, why: Option<&str>, code: u8| HeadlessEnded {
        agent,
        answer: None,
        why: why.map(str::to_owned),
        exit: ExitReason::Failed(code),
    };
    let ended = match run.step {
        Step::Start => {
            let Some(agent) = primary(&agents.primary) else {
                return;
            };
            let Ok((chosen, _)) = agents.choices.get(agent) else {
                return;
            };
            let model = match (run.model, chosen) {
                // `--model` chose it, or the restored session did, and it
                // was connected, or refused with a notice, already.
                (true, _) | (false, Some(_)) => None,
                (false, None) => match models.0.reachable().first() {
                    Some(spec) => Some(spec.reference()),
                    None if command => None,
                    None => {
                        let why = "no model can be reached: set a provider's API key, or name \
                                   one with --model";
                        return end(&mut run, failed(Some(agent), Some(why), 1), commands);
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
            return;
        }
        Step::Connecting { agent } => {
            let connected = command
                || agents
                    .choices
                    .get(agent)
                    .is_ok_and(|(_, connected)| connected);
            if !connected {
                // The notice of the failed choice says why.
                return end(&mut run, failed(Some(agent), None, 1), commands);
            }
            if run.prompt.trim().is_empty() {
                let why = "no prompt: give one after --print, or pipe it in";
                return end(&mut run, failed(Some(agent), Some(why), 2), commands);
            }
            let text = run.prompt.clone();
            commands.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            run.step = Step::Sent { agent };
            return;
        }
        Step::Sent { agent } => {
            // The turn starts with the request once the agent accepted
            // it; a command may start none, or other work, such as a
            // sign-in. A subagent's answer starts another turn of the agent
            // that started it.
            if !agents.idle() {
                return;
            }
            let answer = run.answer.as_ref().and_then(final_answer);
            let exit = match (run.refused, run.failed && answer.is_none()) {
                (true, _) => ExitReason::Failed(2),
                (false, false) => ExitReason::Quit,
                (false, true) => ExitReason::Failed(1),
            };
            HeadlessEnded {
                agent: Some(agent),
                answer,
                why: None,
                exit,
            }
        }
        Step::Done => return,
    };
    end(&mut run, ended, commands);
}

/// Keeps the answer of each turn of the agent the prompt went to that ended
/// answered, whatever its conversation looks like now, such as after a
/// compaction.
fn keep_answer(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut run: ResMut<HeadlessRun>,
) {
    if let Ok((TurnOutcome::Answered(message), &ForAgent(agent))) = turns.get(settled.entity)
        && run.step == (Step::Sent { agent })
    {
        run.answer = Some(message.clone());
    }
}

/// Ends the run as `ended` says.
fn end(run: &mut HeadlessRun, ended: HeadlessEnded, mut commands: Commands) {
    run.step = Step::Done;
    commands.trigger(Exit(ended.exit.clone()));
    commands.trigger(ended);
}

#[cfg(test)]
mod tests;
