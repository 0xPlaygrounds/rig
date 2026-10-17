//! Driving a run nobody sits at, for a front such as `--print` or one
//! that streams JSON: [`AppHeadlessExt::run_headless`] sends one prompt to
//! the primary agent, then exits once no agent works, so the subagents it
//! started have answered. Each time the agents' work settles after a turn
//! of the primary agent ended, [`HeadlessTurnEnded`] carries that turn's
//! answer; just before the exit, [`HeadlessEnded`] says how it went, for
//! the front to write out.
//!
//! A plugin that keeps sending the agent work, such as one that watches
//! files, keeps the run going with a [`KeepRunning`] entity: while one
//! exists, the run waits when its agents are idle, and it ends once none
//! does and no agent works. Such a run may start without a prompt, and so
//! may one whose restored session left a turn unfinished, such as
//! `rig -r <id> -p` after a crash: that turn finishes, and its answer is
//! the run's. The launcher restarts a run whose agent reloaded without its
//! prompt: what the reload left, such as its report to the model, carries
//! it on, and with nothing left it ends with code 0.
//!
//! The agent uses `--model`, else the model its restored session chose,
//! else the first model the environment has a key for. A prompt that is a
//! slash command, such as `/login`, runs without a model. The exit code is
//! 0 when the turn ended with an answer, 2 for no prompt (and no restored
//! turn to finish) or a refused slash command (unknown, or arguments it
//! does not take), 1 otherwise, such as an error notice about the agent
//! and no answer. A warning
//! notice never fails it. A run of several turns exits with the worst.
//!
//! ```
//! use rig_harness::headless::{AppHeadlessExt, HeadlessTurnEnded};
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
//!             let hello = Some("Say hello".to_owned());
//!             app.run_headless(hello).add_observer(|ended: On<HeadlessTurnEnded>| {
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
use rig_ecs::prelude::*;

use crate::host::exit::{Exit, ExitReason, ExitRequest};
use crate::host::launcher::Invoked;
use crate::host::session::AppStart;
use crate::main_agent::{MainAgentQuery, main_agent};
use rig_harness_protocol::StartReason;

/// Starts a run nobody sits at on an [`App`].
pub trait AppHeadlessExt {
    /// Sends `prompt` to the primary agent once it has a model, and exits
    /// when no agent works any more and no [`KeepRunning`] exists. Without
    /// a prompt, the run finishes the turns its restored session left
    /// unfinished, and waits for the work a `KeepRunning` plugin sends.
    /// Call it once, from the front's `finish`.
    fn run_headless(&mut self, prompt: Option<String>) -> &mut Self;
}

impl AppHeadlessExt for App {
    fn run_headless(&mut self, prompt: Option<String>) -> &mut Self {
        let model = self
            .world()
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.args.model.is_some());
        self.insert_resource(HeadlessRun {
            prompt: prompt.filter(|prompt| !prompt.trim().is_empty()),
            model,
            step: Step::Start,
            answer: None,
            turned: false,
            last: None,
            failed: false,
            refused: false,
            worst: 0,
        })
        .add_systems(Update, drive.after(AgentLoop))
        .add_observer(keep_answer)
    }
}

/// While any entity has it, a headless run does not end when its agents
/// are idle: a plugin that sends the agent work over time, such as one
/// that watches files, spawns one, and despawns it to let the run end.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct KeepRunning;

/// Triggered each time a headless run's work settles (no agent works)
/// after a turn of the agent its prompts go to ended, before the run waits
/// or ends.
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub struct HeadlessTurnEnded {
    /// The agent the prompts go to.
    pub agent: Entity,
    /// The answer of its last turn, if it had one.
    pub answer: Option<String>,
    /// Whether it failed: an error notice about the agent and no answer,
    /// or a refused slash command.
    pub failed: bool,
}

/// How a headless run ended, triggered as it asks to exit ([`Exit`]).
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub struct HeadlessEnded {
    /// The agent the prompt went to, if it got that far.
    pub agent: Option<Entity>,
    /// The answer of the last [`HeadlessTurnEnded`], if it had one.
    pub answer: Option<String>,
    /// Why the run could not send the prompt, such as no model.
    pub why: Option<String>,
    /// Why it exits: [`ExitReason::Quit`] when it answered, else
    /// [`ExitReason::Failed`] with the worst exit code of its turns.
    pub exit: ExitReason,
}

/// Where the headless run is.
#[derive(Resource)]
struct HeadlessRun {
    /// The prompt, if one was given.
    prompt: Option<String>,
    /// Whether `--model` named the model, which a models plugin chooses.
    model: bool,
    step: Step,
    /// The message the last turn of the agent the prompts go to answered
    /// with since the work last settled, if one did.
    answer: Option<Message>,
    /// Whether a turn of the agent ended since the work last settled.
    turned: bool,
    /// The answer of the last [`HeadlessTurnEnded`].
    last: Option<String>,
    /// Whether an error notice about the agent came since the work last
    /// settled.
    failed: bool,
    /// Whether a slash command was refused since the work last settled.
    refused: bool,
    /// The worst exit code so far; 0 for none.
    worst: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Nothing done yet.
    Start,
    /// A model was chosen for `agent`, which connects it at once.
    Connecting { agent: Entity },
    /// The prompt went to `agent`, or other work came, and the agents
    /// work.
    Sent { agent: Entity },
    /// The agents are idle, and a [`KeepRunning`] keeps the run going.
    Waiting { agent: Entity },
    /// Exited.
    Done,
}

/// The agents as the run reads them.
#[derive(SystemParam)]
struct Agents<'w, 's> {
    primary: MainAgentQuery<'w, 's>,
    choices: Query<'w, 's, (Option<&'static ModelChoice>, Has<Connection>)>,
    /// Running turns.
    turns: Query<'w, 's, (), With<TurnOf>>,
    /// Work in progress, such as a sign-in; not a long-lived stream, such
    /// as a file watch.
    running: Query<'w, 's, &'static Running>,
    /// Messages on their way to an agent, such as the prompt itself.
    messages: Query<'w, 's, (), (With<Incoming>, Without<Settled>)>,
    /// Effects not started yet, such as one held at Gate, or answered and
    /// not settled, such as one held for review; not an exit.
    held: Query<
        'w,
        's,
        (),
        (
            With<Effect>,
            Without<Settled>,
            Without<ExitRequest>,
            Or<(Without<Started>, With<Outcome>)>,
        ),
    >,
    /// What keeps the run going while its agents are idle.
    keep: Query<'w, 's, (), With<KeepRunning>>,
}

impl Agents<'_, '_> {
    /// Whether no agent works: no turn, task, held effect or message on its
    /// way. An open effect that started and waits on the world outside,
    /// such as a question nobody answers, is no work.
    fn idle(&self) -> bool {
        self.turns.is_empty()
            && self.messages.is_empty()
            && self.held.is_empty()
            && self.running.iter().all(Running::is_stream)
    }
}

/// Chooses the model, sends the prompt, and exits after the turn.
fn drive(
    mut run: ResMut<HeadlessRun>,
    agents: Agents,
    models: Res<Models>,
    (start, wake): (Option<Res<AppStart>>, Res<Wake>),
    (mut notices, mut recalled): (MessageReader<Notice>, MessageReader<Recalled>),
    mut commands: Commands,
) {
    if recalled.read().any(|recalled| recalled.why.is_some()) {
        run.refused = true;
    }
    for notice in notices.read() {
        if let Step::Sent { agent } | Step::Waiting { agent } | Step::Connecting { agent } =
            run.step
            && notice.level == NoticeLevel::Error
            && notice.agent.is_none_or(|about| about == agent)
        {
            run.failed = true;
        }
    }
    // The launcher restarts a run whose agent reloaded without its prompt.
    let restarted = start
        .is_some_and(|start| matches!(start.0, StartReason::Reloaded | StartReason::Restarted));
    // A command such as `/login` runs without a model.
    let command = run
        .prompt
        .as_ref()
        .is_some_and(|prompt| prompt.trim_start().starts_with('/'));
    let ended = match run.step {
        Step::Start => {
            let Some(agent) = main_agent(&agents.primary) else {
                return;
            };
            let Ok((chosen, connected)) = agents.choices.get(agent) else {
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
            // Connected already, as `--model` or the restored session
            // leaves it: the prompt goes out in the first frame, so a kill
            // soon after leaves it in the session to resume.
            if model.is_none() && (connected || command) {
                send(&mut run, agent, &agents, command, restarted, &mut commands)
            } else {
                if let Some(model) = model {
                    commands.trigger(SetModel {
                        entity: agent,
                        model,
                    });
                }
                run.step = Step::Connecting { agent };
                // The model connects as this frame ends, or a refusal says
                // why not; nothing else may ask for the frame that sees it.
                wake.wake();
                return;
            }
        }
        Step::Connecting { agent } => {
            send(&mut run, agent, &agents, command, restarted, &mut commands)
        }
        Step::Sent { agent } | Step::Waiting { agent } => {
            // The turn starts with the request once the agent accepted
            // it; a command may start none, or other work, such as a
            // sign-in. A subagent's answer starts another turn of the agent
            // that started it.
            if !agents.idle() {
                run.step = Step::Sent { agent };
                return;
            }
            let keep = !agents.keep.is_empty();
            // A turn may start and end between two runs of this system, as
            // a fast model's can within one run of the agent loop: the
            // turn's end, not having seen it work, says it happened.
            if run.step == (Step::Sent { agent }) || run.turned || !keep {
                settle(&mut run, agent, &mut commands);
            }
            if keep {
                run.step = Step::Waiting { agent };
                return;
            }
            let exit = match run.worst {
                0 => ExitReason::Quit,
                code => ExitReason::Failed(code),
            };
            Some(HeadlessEnded {
                agent: Some(agent),
                answer: run.last.clone(),
                why: None,
                exit,
            })
        }
        Step::Done => None,
    };
    if let Some(ended) = ended {
        end(&mut run, ended, commands);
    }
}

/// Sends the prompt to `agent` once it is connected, or else moves on to
/// the work there is; how the run ends, when it ends here.
fn send(
    run: &mut HeadlessRun,
    agent: Entity,
    agents: &Agents,
    command: bool,
    restarted: bool,
    commands: &mut Commands,
) -> Option<HeadlessEnded> {
    let connected = command
        || agents
            .choices
            .get(agent)
            .is_ok_and(|(_, connected)| connected);
    if !connected {
        // The notice of the failed choice says why.
        return Some(failed(Some(agent), None, 1));
    }
    match run.prompt.clone() {
        Some(text) => {
            commands.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            run.step = Step::Sent { agent };
        }
        // Something keeps the run going, or already did: a turn it started
        // may have ended while the model was connecting.
        None if !agents.keep.is_empty() || run.turned => run.step = Step::Waiting { agent },
        // Work the restored session left runs, such as an unfinished turn:
        // it finishes, and its answer is the run's.
        None if !agents.idle() => run.step = Step::Sent { agent },
        // The run restarted on a reload with nothing left to do.
        None if restarted => {
            return Some(HeadlessEnded {
                agent: Some(agent),
                answer: None,
                why: None,
                exit: ExitReason::Quit,
            });
        }
        None => {
            let why = "no prompt: give one after --print, or pipe it in";
            return Some(failed(Some(agent), Some(why), 2));
        }
    }
    None
}

/// A run that failed with `code`, for `why` when no notice says it.
fn failed(agent: Option<Entity>, why: Option<&str>, code: u8) -> HeadlessEnded {
    HeadlessEnded {
        agent,
        answer: None,
        why: why.map(str::to_owned),
        exit: ExitReason::Failed(code),
    }
}

/// Notes each turn of the agent the prompts go to that ended, with its
/// answer when it answered, whatever its conversation looks like now, such
/// as after a compaction.
fn keep_answer(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut run: ResMut<HeadlessRun>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    // While connecting too: a turn a plugin started may end before the
    // model the run chose is connected.
    if let Step::Connecting { agent: to } | Step::Sent { agent: to } | Step::Waiting { agent: to } =
        run.step
        && to == agent
    {
        run.turned = true;
        if let TurnOutcome::Answered(message) = outcome {
            run.answer = Some(message.clone());
        }
    }
}

/// Takes what came since the work last settled: the exit code it earns,
/// and a [`HeadlessTurnEnded`] when a turn of `agent` ended.
fn settle(run: &mut HeadlessRun, agent: Entity, commands: &mut Commands) {
    let answer = run.answer.take().and_then(|message| message.final_answer());
    let code = match (run.refused, run.failed && answer.is_none()) {
        (true, _) => 2,
        (false, true) => 1,
        (false, false) => 0,
    };
    run.worst = run.worst.max(code);
    if run.turned {
        commands.trigger(HeadlessTurnEnded {
            agent,
            answer: answer.clone(),
            failed: code != 0,
        });
        run.last = answer;
    }
    (run.turned, run.failed, run.refused) = (false, false, false);
}

/// Ends the run as `ended` says.
fn end(run: &mut HeadlessRun, ended: HeadlessEnded, mut commands: Commands) {
    run.step = Step::Done;
    commands.trigger(Exit(ended.exit.clone()));
    commands.trigger(ended);
}

#[cfg(test)]
mod tests;
