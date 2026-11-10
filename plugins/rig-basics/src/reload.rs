//! `/reload`: rebuild the agent through the `rig` launcher and restart on
//! the new build, in the same session, with the plugins `settings.toml`
//! turns on then. The build runs as a child process whose latest line of
//! output is the progress; quitting during the build kills it.
//!
//! `/reload` typed while a turn runs is queued: the build starts once no
//! turn runs, and `/reload cancel` drops it. A plugin queues one with
//! [`ReloadStatus::ask`], and views read progress from that resource. A turn
//! started while the build runs delays the restart until it ends. A stop that
//! reaches an idle agent no agent spawned (Esc in the terminal view) cancels
//! a running build. A build that fails leaves this build running, and its
//! first errors go to the agent that asked as a note for its model
//! ([`launcher::build_failure_message`]); `rig build` keeps the whole output
//! in `RIG_HOME/build.log`. The launcher rolls back a new build that fails to
//! start. A run nobody sits at (`--print`) waits for a reload too.
//!
//! `/update` reinstalls rig through the launcher's `rig update`, which says
//! the command it runs and builds the agent with the new launcher, then
//! restarts the same way; the launcher, whose own file was replaced, starts
//! the new one on the same session.
//!
//! [`PluginsPlugin`](plugins::PluginsPlugin) adds `/plugins`, which turns plugins on and
//! off in `settings.toml` and reloads.

use rig_harness::harness_protocol;
use rig_harness::prelude::*;

mod hold;
pub mod plugins;
mod rebuild;
#[cfg(feature = "tui")]
mod status;

/// `/reload` and `/update`, the rebuild in flight and the restart.
#[derive(Default)]
pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "reload",
            "Rebuild what changed and restart with the plugins settings.toml turns on once no \
             turn runs; `/reload cancel` cancels",
            reload,
        )
        .add_command(
            "update",
            "Reinstall rig with cargo (from crates.io, or the checkout it was installed from), \
             build the agent with it and restart once no turn runs; `/update cancel` cancels",
            update,
        )
        .init_resource::<ReloadStatus>()
        .add_observer(on_cancel_reload)
        .add_observer(cancel_on_interrupt);
        hold::add(app);
        rebuild::add(app);
        #[cfg(feature = "tui")]
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
    /// Asked for by `/reload`, `/update` or a plugin, for `agent`, whose
    /// model is told when the build fails: the build starts once no turn
    /// runs. `/reload cancel` or [`CancelReload`] drops it.
    Queued {
        /// The agent that asked.
        agent: Entity,
        /// What runs before the restart.
        rebuild: Rebuild,
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

/// What a reload runs through the launcher before it restarts.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Clone, Debug, Default, PartialEq)]
pub enum Rebuild {
    /// `rig build`: the agent with the plugins `settings.toml` turns on.
    #[default]
    Build,
    /// `rig update`: rig reinstalled, then the agent built with it.
    Update,
}

impl Rebuild {
    /// The launcher's command for it.
    fn command(self) -> &'static str {
        match self {
            Rebuild::Build => "build",
            Rebuild::Update => "update",
        }
    }
}

impl ReloadStatus {
    /// Queues a reload for `agent` that runs `rebuild` once no turn runs.
    /// Refused without the launcher, or with a reload queued or running.
    pub fn ask(&mut self, agent: Entity, rebuild: Rebuild) -> Result<(), &'static str> {
        match self {
            ReloadStatus::Idle | ReloadStatus::Failed if harness_protocol::launcher().is_some() => {
                *self = ReloadStatus::Queued { agent, rebuild };
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

/// Stops the running rebuild, or drops the queued one, if any.
#[derive(Event, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Event, Clone, Debug, Default)]
pub struct CancelReload;

fn reload(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    status: ResMut<ReloadStatus>,
    commands: Commands,
    notices: MessageWriter<Notice>,
) {
    queue(args, Rebuild::Build, turns, status, commands, notices);
}

fn update(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    status: ResMut<ReloadStatus>,
    commands: Commands,
    notices: MessageWriter<Notice>,
) {
    queue(args, Rebuild::Update, turns, status, commands, notices);
}

/// Queues `rebuild` for `/reload` or `/update`, or cancels on `cancel`.
fn queue(
    args: CommandArgs,
    rebuild: Rebuild,
    turns: Query<(), With<TurnOf>>,
    mut status: ResMut<ReloadStatus>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if args.args == "cancel" {
        commands.trigger(CancelReload);
        return;
    }
    let asked = status.ask(args.agent, rebuild);
    if asked.is_ok() {
        hold::keep_run(&mut commands);
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

/// Cancels a running build when a stop reaches an idle agent that no agent
/// spawned, as Esc does in the terminal view; Esc on a running turn stops
/// the turn only. The agents below, which the stop reaches too, are not
/// the one it was for.
fn cancel_on_interrupt(
    interrupt: On<StopReached>,
    spawned: Query<Has<SpawnedBy>>,
    status: Res<ReloadStatus>,
    mut commands: Commands,
) {
    let root = spawned
        .get(interrupt.event_target())
        .is_ok_and(|spawned| !spawned);
    if interrupt.idle() && root && matches!(*status, ReloadStatus::Building { .. }) {
        commands.trigger(CancelReload);
    }
}

fn on_cancel_reload(
    _: On<CancelReload>,
    mut status: ResMut<ReloadStatus>,
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
    notices.write(Notice::info(None, notice));
}

#[cfg(test)]
mod tests;
