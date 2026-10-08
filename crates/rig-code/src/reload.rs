//! `/reload`: rebuild the agent with the launcher and restart on the new
//! build.
//!
//! The command spawns a [`ReloadBuild`] entity whose task runs
//! `$RIG_LAUNCHER build` and streams its output back. Views read the
//! entity's progress, taken from cargo's own `Building [..] N/M` count. A
//! failed build is reported with [`BuildFailed`] and this binary keeps
//! running. A successful one waits until every agent is idle, saves the
//! session and exits with [`RELOAD_EXIT_CODE`], which tells the launcher to
//! start the new binary.

use std::{
    ffi::OsString,
    process::{Command, ExitStatus},
    sync::{
        Mutex, PoisonError,
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};

use bevy_app::{App, AppExit, Plugin, Update};
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, Task, futures::check_ready};

use crate::{
    RigCodeAppExt as _,
    agent::{Agent, AgentCalls, AgentStatus, NeedsReply, Notice, RigSet, turn_running},
    commands::CommandArgs,
    process::{POLL, Piped, sleep},
    session,
};

/// The exit code that asks the launcher for a restart on the new build.
pub const RELOAD_EXIT_CODE: u8 = 75;
/// Output lines a build keeps for its error report.
const KEPT_LINES: usize = 200;
/// Lines of a failed build shown when no line starts with `error`.
const TAIL_LINES: usize = 30;
/// How long output is still read after the build exits.
const DRAIN_GRACE: Duration = Duration::from_secs(1);

/// Registers `/reload` and the systems that follow its build.
pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<BuildFailed>()
            .add_command(
                "reload",
                "Rebuild rig-code with the plugin list and restart on it",
                reload_command,
            )
            .add_systems(
                Update,
                (poll_build, finish_reload).chain().after(RigSet::Finish),
            );
    }
}

/// A `/reload` build in progress.
#[derive(Component)]
pub struct ReloadBuild {
    /// The agent that asked for it, which gets its notices.
    pub agent: Entity,
    /// Units cargo finished.
    pub done: usize,
    /// Units cargo will build.
    pub total: usize,
    /// The crate cargo compiled last.
    pub current: String,
    lines: Vec<String>,
    output: Mutex<Receiver<String>>,
}

impl ReloadBuild {
    /// One line for views, e.g. `Compiling 37/41 crates (rig-code)`.
    pub fn progress(&self) -> String {
        let mut text = if self.total == 0 {
            "Preparing the build".to_owned()
        } else {
            format!("Compiling {}/{} crates", self.done, self.total)
        };
        if !self.current.is_empty() {
            text.push_str(&format!(" ({})", self.current));
        }
        text
    }

    /// Take one output segment: a progress bar updates the count, anything
    /// else is kept for the error report.
    fn take(&mut self, segment: &str) {
        let raw = segment.replace("\x1b[K", "");
        let raw = raw.trim_end();
        let text = raw.trim_start();
        if text.is_empty() {
            return;
        }
        if let Some((_, after)) = text.split_once("Building [")
            && let Some((_, counts)) = after.split_once(']')
        {
            let counts = counts.split(':').next().unwrap_or_default().trim();
            if let Some((done, total)) = counts.split_once('/')
                && let (Ok(done), Ok(total)) = (done.parse(), total.parse())
            {
                self.done = done;
                self.total = total;
            }
            return;
        }
        if let Some(name) = text
            .strip_prefix("Compiling ")
            .and_then(|rest| rest.split_whitespace().next())
        {
            self.current = name.to_owned();
        }
        // rustc's `-->` and `|` gutters keep their indentation.
        self.lines.push(raw.to_owned());
        let excess = self.lines.len().saturating_sub(KEPT_LINES);
        self.lines.drain(..excess);
    }

    /// The lines worth showing for a failed build: from the first error on,
    /// or the last few.
    fn errors(&self) -> Vec<String> {
        let start = self
            .lines
            .iter()
            .position(|line| line.starts_with("error"))
            .unwrap_or_else(|| self.lines.len().saturating_sub(TAIL_LINES));
        self.lines.iter().skip(start).cloned().collect()
    }
}

/// The task running the launcher's build.
#[derive(Component)]
struct BuildTask(Task<std::io::Result<ExitStatus>>);

/// The build succeeded; the app restarts once every agent is idle.
#[derive(Component)]
struct BuildReady;

/// A `/reload` build failed. Views show the lines.
#[derive(Message, Debug, Clone)]
pub struct BuildFailed {
    /// The compiler's and the launcher's report.
    pub lines: Vec<String>,
}

/// Whether any agent has a turn running.
fn busy(agents: &Query<(&AgentStatus, Has<NeedsReply>, Has<AgentCalls>), With<Agent>>) -> bool {
    agents
        .iter()
        .any(|(status, needs_reply, calls)| turn_running(*status, needs_reply, calls))
}

fn reload_command(
    In(CommandArgs { agent, .. }): In<CommandArgs>,
    agents: Query<(&AgentStatus, Has<NeedsReply>, Has<AgentCalls>), With<Agent>>,
    builds: Query<(), With<ReloadBuild>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut notice = |text: &str| {
        notices.write(Notice {
            agent,
            text: text.to_owned(),
        });
    };
    let Some(launcher) = std::env::var_os("RIG_LAUNCHER").filter(|path| !path.is_empty()) else {
        notice("Start rig-code with the rig launcher to use /reload.");
        return;
    };
    if busy(&agents) {
        notice("A turn is running. Press Esc to stop it, then /reload.");
        return;
    }
    if !builds.is_empty() {
        notice("A build is already running.");
        return;
    }
    let (sender, output) = mpsc::channel();
    let task =
        AsyncComputeTaskPool::get_or_init(Default::default).spawn(run_build(launcher, sender));
    commands.spawn((
        ReloadBuild {
            agent,
            done: 0,
            total: 0,
            current: String::new(),
            lines: Vec::new(),
            output: Mutex::new(output),
        },
        BuildTask(task),
    ));
    notice("Rebuilding rig-code. Keep working; it restarts when the build is done and idle.");
}

/// Run `launcher build` with stdout and stderr joined, sending each line
/// or progress-bar update as it arrives. Dropping the future kills the
/// build.
async fn run_build(launcher: OsString, output: Sender<String>) -> std::io::Result<ExitStatus> {
    let mut command = Command::new(launcher);
    command.arg("build").env("CARGO_TERM_COLOR", "never");
    let mut process = Piped::spawn(command)?;
    let mut pending = Vec::new();
    let mut exited = None;
    loop {
        let (bytes, open) = process.output();
        for byte in bytes {
            if byte == b'\n' || byte == b'\r' {
                // The app is gone when nobody listens; the build goes on.
                let _ = output.send(String::from_utf8_lossy(&pending).into_owned());
                pending.clear();
            } else {
                pending.push(byte);
            }
        }
        if exited.is_none() {
            exited = process.try_wait()?.map(|status| (status, Instant::now()));
        }
        // Wait for the pipe to close, unless something the build started
        // keeps it open after the build itself exited.
        if let Some((status, at)) = exited
            && (!open || at.elapsed() >= DRAIN_GRACE)
        {
            let _ = output.send(String::from_utf8_lossy(&pending).into_owned());
            return Ok(status);
        }
        sleep(POLL).await;
    }
}

/// Follow running builds: take their output, and on completion mark them
/// ready or report the failure.
fn poll_build(
    mut builds: Query<(Entity, &mut ReloadBuild, &mut BuildTask)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut failures: MessageWriter<BuildFailed>,
) {
    for (entity, mut build, mut task) in &mut builds {
        let build = &mut *build;
        let segments: Vec<String> = build
            .output
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .try_iter()
            .collect();
        for segment in segments {
            build.take(&segment);
        }
        let Some(finished) = check_ready(&mut task.0) else {
            continue;
        };
        let agent = build.agent;
        match finished {
            Ok(status) if status.success() => {
                commands
                    .entity(entity)
                    .remove::<BuildTask>()
                    .insert(BuildReady);
                notices.write(Notice {
                    agent,
                    text: "The build is done. Restarting once every agent is idle.".to_owned(),
                });
            }
            finished => {
                let mut lines = build.errors();
                lines.push(match finished {
                    Ok(status) => format!("rig build {status}"),
                    Err(error) => format!("cannot run rig build: {error}"),
                });
                failures.write(BuildFailed { lines });
                notices.write(Notice {
                    agent,
                    text: "The build failed; this build keeps running.".to_owned(),
                });
                commands.entity(entity).despawn();
            }
        }
    }
}

/// Once a build is ready and no agent has a turn running: save the session
/// and exit with the reload code.
fn finish_reload(
    ready: Query<Entity, With<BuildReady>>,
    agents: Query<(&AgentStatus, Has<NeedsReply>, Has<AgentCalls>), With<Agent>>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if ready.is_empty() || busy(&agents) {
        return;
    }
    for build in &ready {
        commands.entity(build).despawn();
    }
    commands.queue(session::save);
    exit.write(AppExit::from_code(RELOAD_EXIT_CODE));
}
