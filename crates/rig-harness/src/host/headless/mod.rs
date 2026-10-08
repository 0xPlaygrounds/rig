//! The modes without a terminal view, chosen by the process's arguments
//! ([`rig::harness_protocol::Invocation`], which the `rig` launcher passes
//! on): `--print` answers one prompt, `--print --json` streams every event
//! as a line of JSON, and `eval` runs the tasks of a spec across models ([`super::eval`]). Each is
//! a view like the terminal one: it reads agent components and messages and
//! sends the agents the same requests, and it never owns the loop.
//!
//! In these modes stdout carries only the mode's output; notices go to
//! stderr in print mode and into the stream otherwise, and the log stays in
//! the session's `agent.log`.

pub mod events;
mod print;

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use rig::harness_protocol::{Invocation, Mode};
use serde_json::Value;

use crate::core::agent::{Agent, AgentId, Notice};
use crate::core::approval::{ApprovalAnswer, Approve, AwaitingApproval};
use crate::core::rewind::Forked;
use crate::core::subagents::SubagentOf;

use super::eval::TrialAgent;

/// How this process runs: its [`Invocation`], read from its arguments
/// unless the app inserted one before adding [`ModePlugin`]. Views check
/// it: the terminal view stays out of every headless mode.
#[derive(Resource, Clone, Debug, Default)]
pub struct RunMode(pub Invocation);

impl RunMode {
    /// The mode.
    pub fn mode(&self) -> &Mode {
        &self.0.mode
    }

    /// Whether nobody sits at a terminal view.
    pub fn is_headless(&self) -> bool {
        self.0.mode.is_headless()
    }
}

/// Reads the [`RunMode`] and adds what its mode needs: the print loop, the
/// JSON event stream, or an eval run.
pub struct ModePlugin;

impl Plugin for ModePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<RunMode>() {
            let invocation = Invocation::from_env().unwrap_or_else(|failure| {
                // The launcher checked them; the binary run alone says why
                // it runs as usual.
                eprintln!("rig: {failure}; starting the terminal view");
                Invocation::default()
            });
            app.insert_resource(RunMode(invocation));
        }
        let mode = app
            .world()
            .get_resource::<RunMode>()
            .map(|mode| mode.0.mode.clone())
            .unwrap_or_default();
        if mode.is_headless() {
            app.add_systems(Last, exit_when_stdout_closes);
        }
        if mode.is_one_shot() {
            // Nobody can answer: a call the policy leaves to the user is
            // refused, and the model told why.
            app.add_observer(refuse_approvals);
        }
        match mode {
            Mode::Interactive => {}
            Mode::Print { prompt, json } => {
                if json {
                    app.add_plugins(events::EventStreamPlugin);
                }
                app.add_plugins(print::PrintPlugin { prompt, json });
            }
            Mode::Eval { spec, json } => {
                app.add_plugins(super::eval::EvalPlugin { spec, json });
            }
        }
    }
}

/// The agent a headless run talks to when none is named: the first agent
/// the user started, not a subagent, a fork or an eval trial.
pub fn primary_agent<'a>(
    agents: impl IntoIterator<Item = (Entity, &'a AgentId, bool)>,
) -> Option<Entity> {
    agents
        .into_iter()
        .min_by(|a, b| (a.2, &a.1.0).cmp(&(b.2, &b.1.0)))
        .map(|(entity, ..)| entity)
}

/// The agents for [`primary_agent`]: each with whether it is someone
/// else's (a subagent, a fork or a trial).
pub type PrimaryQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AgentId,
        Has<SubagentOf>,
        Has<Forked>,
        Has<TrialAgent>,
    ),
    With<Agent>,
>;

/// [`primary_agent`] of a [`PrimaryQuery`].
pub fn primary(agents: &PrimaryQuery) -> Option<Entity> {
    primary_agent(
        agents
            .iter()
            .map(|(entity, id, sub, fork, trial)| (entity, id, sub || fork || trial)),
    )
}

/// Refuses each call waiting for approval in a run nobody watches.
fn refuse_approvals(
    waiting: On<Add<AwaitingApproval>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    commands.trigger(Approve {
        entity: waiting.entity,
        answer: ApprovalAnswer::Deny {
            reason: "this run has nobody to approve it (a print or eval run); do without it, \
                     or say what you would need"
                .to_owned(),
        },
    });
    notices.write(Notice::info(
        None,
        "A tool call needed approval and was refused: nobody can answer in this mode.",
    ));
}

/// Set once stdout is closed: nobody reads the output any more.
static STDOUT_CLOSED: AtomicBool = AtomicBool::new(false);

/// Writes `value` to stdout as one line of JSON, at once. A closed stdout
/// ends the app at the end of the frame.
pub(crate) fn emit(value: &Value) {
    let mut out = std::io::stdout().lock();
    let written = serde_json::to_writer(&mut out, value)
        .map_err(std::io::Error::from)
        .and_then(|()| out.write_all(b"\n"))
        .and_then(|()| out.flush());
    if let Err(failure) = written
        && !STDOUT_CLOSED.swap(true, Ordering::Relaxed)
    {
        error!("could not write to stdout, so stopping: {failure}");
    }
}

/// Exits once [`emit`] found stdout closed.
fn exit_when_stdout_closes(mut exits: MessageWriter<AppExit>) {
    if STDOUT_CLOSED.load(Ordering::Relaxed) {
        exits.write(AppExit::from_code(1));
    }
}
