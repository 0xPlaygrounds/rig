//! `/reload`: rebuild the agent through the `rig` launcher and restart on
//! the new build. The build runs as a child process whose stderr a std
//! thread forwards line by line; its latest line is the progress. Quitting
//! during the build kills it with cargo and rustc.
//!
//! `/reload` typed while a turn runs is queued: the build starts once no
//! turn runs, and `/reload cancel` drops it. A plugin queues one with
//! [`ReloadStatus::ask`], as the `reload` tool of the `rig-coding-tools`
//! plugin crate does for the model. A turn started while the build runs
//! delays the restart until it ends. A build that fails leaves this build
//! running, and the launcher rolls back a new build that fails to start.
//! Views show where it is from the [`ReloadStatus`] resource.
//!
//! A failed build's first errors go into the conversation of the agent that
//! asked, as a note for its model ([`super::launcher::build_failure_note`]);
//! `rig build` keeps the whole output in `RIG_HOME/build.log`.

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStderr, Command, Stdio};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use crossbeam_channel::{Receiver, Sender, TryRecvError};

use rig::harness_protocol::{Home, RELOAD_EXIT_CODE, first_errors};
use rig_tools::process::{detach, kill_group};

use super::launcher;
use rig_ecs::agent::{Notice, TurnOf};
use rig_ecs::calls::PollCalls;
use rig_ecs::calls::Wake;
use rig_ecs::commands::{AppCommandsExt, CommandArgs};

/// Lines of a failed build in the note to the model, from its first error.
const NOTE_LINES: usize = 40;

/// `/reload`, the rebuild in flight and the restart.
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
        .add_systems(
            Update,
            (start_queued_reload, drain_reload, finish_reload)
                .chain()
                .after(PollCalls),
        );
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

/// The rebuild's process. Dropping it while the build runs kills the
/// build, as the app's runner does when it drops the app on exit.
#[derive(Resource)]
struct ReloadBuild {
    child: Child,
    /// The agent that asked, whose model is told when the build fails.
    agent: Entity,
    lines: Receiver<String>,
    /// This build's output so far. Not read back from the build log, which
    /// a build queued behind this one, or an older one when this one fails
    /// before it starts its log, may have written.
    output: Vec<String>,
    exited: bool,
}

impl ReloadBuild {
    fn start(launcher: &std::ffi::OsStr, agent: Entity, wake: Wake) -> std::io::Result<Self> {
        let mut command = Command::new(launcher);
        command
            .arg("build")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        // Its own session and process group, so cancelling stops cargo and
        // rustc too, and a git prompt for a plugin cannot reach the screen.
        detach(&mut command);
        let mut child = command.spawn()?;
        let (sender, lines) = crossbeam_channel::unbounded();
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || forward(stderr, sender, wake));
        }
        Ok(Self {
            child,
            agent,
            lines,
            output: Vec::new(),
            exited: false,
        })
    }

    /// At most `count` lines of the output from the first error on, or its
    /// end when no line starts with `error`.
    fn errors(&self, count: usize) -> String {
        let lines = self.output.iter().map(String::as_str);
        let mut shown = first_errors(lines.clone(), count);
        if shown.is_empty() {
            shown = lines
                .skip(self.output.len().saturating_sub(count))
                .collect();
        }
        shown.join("\n")
    }
}

impl Drop for ReloadBuild {
    fn drop(&mut self) {
        if !self.exited {
            kill_group(&mut self.child);
            self.child.wait().ok();
        }
    }
}

/// Stops the running rebuild, or drops the queued one, if any.
#[derive(Event, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Event, Clone, Debug, Default)]
pub struct CancelReload;

/// Sends each line of the build's stderr, and wakes the loop for each and
/// at the end.
fn forward(stderr: ChildStderr, lines: Sender<String>, wake: Wake) {
    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
        if lines.send(line).is_err() {
            return;
        }
        wake.wake();
    }
    wake.wake();
}

impl ReloadStatus {
    /// Queues a reload for `agent`: its build starts once no turn runs.
    /// Refused without the launcher, or with a reload queued or running.
    pub fn ask(&mut self, agent: Entity) -> Result<(), String> {
        match self {
            ReloadStatus::Idle | ReloadStatus::Failed if launcher::executable().is_some() => {
                *self = ReloadStatus::Queued { agent };
                Ok(())
            }
            ReloadStatus::Idle | ReloadStatus::Failed => {
                Err("/reload needs the rig launcher: start the agent with `rig`.".to_owned())
            }
            ReloadStatus::Queued { .. } => Err(
                "A reload is already queued for when no turn runs; /reload cancel cancels it."
                    .to_owned(),
            ),
            ReloadStatus::Building { .. } | ReloadStatus::Ready => {
                Err("A rebuild is already running; Esc cancels it.".to_owned())
            }
        }
    }
}

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
    let notice = match status.ask(args.agent) {
        // The build starts this frame, with its own notice.
        Ok(()) if turns.is_empty() => return,
        Ok(()) => "A turn is running: the agent rebuilds and restarts once no turn runs. \
                   /reload cancel cancels it."
            .to_owned(),
        Err(why) => why,
    };
    notices.write(Notice::info(None, notice));
}

/// Starts the queued reload's build once no turn runs.
fn start_queued_reload(
    mut status: ResMut<ReloadStatus>,
    turns: Query<(), With<TurnOf>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let ReloadStatus::Queued { agent } = *status else {
        return;
    };
    if !turns.is_empty() {
        return;
    }
    *status = ReloadStatus::Idle;
    let Some(launcher) = launcher::executable() else {
        return;
    };
    let notice = match ReloadBuild::start(&launcher, agent, wake.clone()) {
        Ok(build) => {
            commands.insert_resource(build);
            *status = ReloadStatus::Building { latest: None };
            "Rebuilding the agent…".to_owned()
        }
        Err(failure) => format!("Could not start the rebuild: {failure}"),
    };
    notices.write(Notice::info(None, notice));
}

/// Reads the build's output; on its exit, reports a failure or marks the
/// build ready.
fn drain_reload(
    build: Option<ResMut<ReloadBuild>>,
    mut status: ResMut<ReloadStatus>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(mut build) = build else {
        return;
    };
    let ReloadStatus::Building { mut latest } = status.clone() else {
        return;
    };
    let ended = loop {
        match build.lines.try_recv() {
            Ok(line) if line.trim().is_empty() => {}
            Ok(line) => {
                // Without the source path cargo appends, so it fits the
                // status line: `Compiling serde v1.0.228`.
                let shown = line.split(" (").next().unwrap_or_default();
                latest = Some(shown.trim().to_owned());
                build.output.push(line);
            }
            Err(TryRecvError::Empty) => break false,
            Err(TryRecvError::Disconnected) => break true,
        }
    };
    status.set_if_neq(ReloadStatus::Building { latest });
    if !ended {
        return;
    }
    let exit = match build.child.try_wait() {
        Ok(Some(exit)) => exit,
        Ok(None) => return,
        Err(failure) => {
            notices.write(Notice::error(
                None,
                format!("The rebuild failed: {failure}"),
            ));
            commands.remove_resource::<ReloadBuild>();
            *status = ReloadStatus::Idle;
            return;
        }
    };
    build.exited = true;
    commands.remove_resource::<ReloadBuild>();
    if exit.success() {
        *status = ReloadStatus::Ready;
        notices.write(Notice::info(None, "Build ready; restarting."));
        return;
    }
    *status = ReloadStatus::Failed;
    // The notice is the one log line: the whole output is in the build log
    // it names.
    let first = first_errors(build.output.iter().map(String::as_str), 1)
        .first()
        .map(|line| format!(": {}", line.trim()))
        .unwrap_or_default();
    notices.write(Notice::error(
        None,
        format!(
            "The rebuild failed ({exit}){first}; this build keeps running. Whole output: {}",
            Home::from_env().build_log().display()
        ),
    ));
    launcher::note_build_failure(
        &mut commands,
        build.agent,
        launcher::build_failure_note(
            "/reload",
            &format!(
                "`rig build` exited with {exit}. Its output from the first error:\n{}",
                build.errors(NOTE_LINES)
            ),
        ),
    );
}

/// Exits with [`RELOAD_EXIT_CODE`] once the build is ready and no turn
/// runs, at most once.
fn finish_reload(
    status: Res<ReloadStatus>,
    turns: Query<(), With<TurnOf>>,
    mut exiting: Local<bool>,
    mut exit: MessageWriter<AppExit>,
) {
    if !*exiting && *status == ReloadStatus::Ready && turns.is_empty() {
        *exiting = true;
        exit.write(AppExit::from_code(RELOAD_EXIT_CODE));
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
            commands.remove_resource::<ReloadBuild>();
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
