//! `/reload`: rebuild the agent through the `rig` launcher and restart on
//! the new build, in the same session. The build runs as a child process,
//! and its latest line of output is the progress. Quitting during the build
//! kills it with cargo and rustc.
//!
//! `/reload` typed while a turn runs is queued: the build starts once no
//! turn runs, and `/reload cancel` drops it. The model queues one with the
//! `reload` tool, and a plugin with [`ReloadStatus::ask`]. A turn started
//! while the build runs delays the restart until it ends. A build that
//! fails leaves this build running, and the launcher rolls back a new
//! build that fails to start. Views show where it is from the
//! [`ReloadStatus`] resource; with the `tui` feature, the terminal view's
//! status line does. A [`StopTurn`] of an idle agent (Esc in the
//! terminal view) cancels a running build.
//!
//! A failed build's first errors go into the conversation of the agent that
//! asked, as a message for its model ([`launcher::build_failure_message`]);
//! `rig build` keeps the whole output in `RIG_HOME/build.log`. A reload the
//! model asked for reports back ([`ReloadAsked`]): after the restart, a
//! failed build or a rollback, that agent gets one message that starts a
//! turn, saying the new build started, carrying the build's errors, or
//! saying the new build was rolled back; a user's `/reload` that fails only
//! leaves the errors as a note. A run nobody sits at (`--print`) waits for
//! a reload and restarts on the new build too.
//!
//! When the launcher started the agent, [`ReloadPlugin`] also spawns the
//! system prompt's section on how the agent extends itself
//! ([`self_extension`]): fixed text that names `plugins.toml`, the
//! launcher's plugin commands and the plugin guide, never a list of what
//! they hold.
//!
//! [`PluginsPlugin`](plugins::PluginsPlugin) adds `/plugins`, which turns the plugins of
//! `plugins.toml` on and off and reloads.

use rig_harness::harness_protocol;
use rig_harness::prelude::*;

pub mod plugins;
mod rebuild;
mod report;
mod status;
mod tool;

pub use tool::{SELF_EXTENSION_TAG, self_extension};

/// The name the reload tool decides effects and the reload reports write
/// as ([`Decider::plugin`], [`Author::plugin`]), in effect logs and
/// conversations: the module's.
pub const ORIGIN: &str = "reload";

/// `/reload`, the `reload` tool, the rebuild in flight and the restart,
/// and the system prompt's section on how the agent extends itself.
#[derive(Default)]
pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "reload",
            "Rebuild with the plugins in plugins.toml and restart once no turn runs; \
             `/reload cancel` cancels",
            reload,
        )
        .init_resource::<ReloadStatus>()
        .add_observer(on_cancel_reload)
        .add_observer(cancel_on_interrupt);
        rebuild::add(app);
        report::add(app);
        tool::add(app);
        status::add(app);
    }
}

/// Where `/reload` is, for views.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Clone, Debug, Default, PartialEq)]
pub enum ReloadStatus {
    /// No reload was asked for.
    #[default]
    Idle,
    /// Asked for by `/reload` or the `reload` tool, for `agent`, whose
    /// model is told when the build fails: the build starts once no turn
    /// runs. `/reload cancel` or [`CancelReload`] drops it.
    Queued {
        /// The agent that asked.
        agent: Entity,
    },
    /// The build runs.
    Building {
        /// The build's latest line: the launcher's `Resolving
        /// dependencies…`, or cargo's own, such as `Compiling serde
        /// v1.0.228`.
        latest: Option<String>,
    },
    /// The build succeeded; the restart waits for every agent to be idle.
    Ready,
    /// The last build failed; this build keeps running.
    Failed,
}

impl ReloadStatus {
    /// Queues a reload for `agent`: its build starts once no turn runs.
    /// Refused without the launcher, or with a reload queued or running.
    pub fn ask(&mut self, agent: Entity) -> Result<(), &'static str> {
        match self {
            ReloadStatus::Idle | ReloadStatus::Failed if harness_protocol::launcher().is_some() => {
                *self = ReloadStatus::Queued { agent };
                Ok(())
            }
            ReloadStatus::Idle | ReloadStatus::Failed => {
                Err("/reload needs the rig launcher: start the agent with `rig`.")
            }
            ReloadStatus::Queued { .. } => {
                Err("A reload is already queued for when no turn runs; /reload cancel cancels it.")
            }
            ReloadStatus::Building { .. } | ReloadStatus::Ready => {
                Err("A rebuild is already running; Esc cancels it.")
            }
        }
    }
}

/// The most reloads the model may ask for per message of the user.
pub const MAX_RELOADS: u8 = 3;

/// On the agent whose model asked for a reload with the `reload` tool: what
/// it said it would do once the new build runs, how many reloads it asked
/// for since the user's last message (at most [`MAX_RELOADS`]), and whether
/// it is owed the last one's result. Saved, so the result reaches it after
/// the restart: one message that starts a turn, that the new build started,
/// the failed build's errors, or that the new build was rolled back. The
/// user's next message starts the count again; the user stopping the
/// agent, or cancelling the reload, clears what it is owed.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug, PartialEq)]
pub struct ReloadAsked {
    /// What the model said it would do once the new build runs.
    pub then: String,
    /// The reloads asked for since the user's last message.
    pub attempts: u8,
    /// Whether the last one's result is still owed.
    pub owed: bool,
}

/// Stops the running rebuild, or drops the queued one, if any.
#[derive(Event, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Event, Clone, Debug, Default)]
pub struct CancelReload;

fn reload(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    mut status: ResMut<ReloadStatus>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if args.args == "cancel" {
        commands.trigger(CancelReload);
        return;
    }
    let asked = status.ask(args.agent);
    if asked.is_ok() {
        report::keep_run(&mut commands);
    }
    let notice = match asked {
        // The build starts this frame, with its own notice.
        Ok(()) if turns.is_empty() => return,
        Ok(()) => {
            "A turn is running: the agent rebuilds and restarts once no turn runs. \
             /reload cancel cancels it."
        }
        Err(why) => why,
    };
    notices.write(Notice::info(None, notice));
}

/// Cancels a running build when an idle agent is interrupted, as Esc does
/// in the terminal view; Esc on a running turn stops the turn only.
fn cancel_on_interrupt(
    interrupt: On<StopTurn>,
    busy: Query<Has<ActiveTurn>>,
    status: Res<ReloadStatus>,
    mut commands: Commands,
) {
    let idle = busy.get(interrupt.entity).is_ok_and(|busy| !busy);
    if idle && matches!(*status, ReloadStatus::Building { .. }) {
        commands.trigger(CancelReload);
    }
}

fn on_cancel_reload(
    _: On<CancelReload>,
    mut status: ResMut<ReloadStatus>,
    mut asked: Query<&mut ReloadAsked>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let notice = match *status {
        ReloadStatus::Queued { .. } => "Queued reload cancelled.",
        ReloadStatus::Building { .. } => {
            rebuild::stop(&mut commands);
            "Rebuild cancelled."
        }
        ReloadStatus::Idle | ReloadStatus::Ready | ReloadStatus::Failed => {
            notices.write(Notice::info(None, "No reload to cancel."));
            return;
        }
    };
    *status = ReloadStatus::Idle;
    report::clear(&mut asked);
    notices.write(Notice::info(None, notice));
}

#[cfg(test)]
mod tests;
