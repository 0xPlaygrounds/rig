//! `/reload`: rebuild the agent through the `rig` launcher and restart on
//! the new build. The build runs as a child process whose stderr a std
//! thread forwards line by line. Until cargo's own `done/total` counter
//! appears, the build is resolving dependencies and its latest line is the
//! progress. Quitting during the build kills it with cargo and rustc.

use std::collections::VecDeque;
use std::io::{BufReader, Read};
use std::process::{Child, ChildStderr, Command, Stdio};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender, TryRecvError};

use super::launcher::{self, RELOAD_EXIT_CODE};
use super::process::{detach, kill_group};
use crate::core::agent::{Agent, AgentStatus, Notice};
use crate::core::commands::{AppCommandsExt, CommandArgs};
use crate::core::turn::AgentSystems;

/// Lines of a failed build shown, from its first error.
const ERROR_LINES: usize = 60;
/// Lines of build output kept while it runs.
const KEPT_LINES: usize = 2000;

/// `/reload`, the rebuild in flight and the restart.
pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "reload",
            "Rebuild with the plugins in plugins.toml and restart",
            reload,
        )
        .add_observer(on_cancel_reload)
        .add_systems(
            Update,
            (drain_reload, finish_reload)
                .chain()
                .after(AgentSystems::Settle),
        )
        .add_systems(
            Last,
            stop_reload_on_exit
                .in_set(OnAppExitSystems)
                .run_if(on_message::<AppExit>),
        );
    }
}

/// The rebuild started by `/reload`. Dropping it while the build runs
/// kills the build.
#[derive(Resource)]
pub struct ReloadBuild {
    child: Child,
    lines: Receiver<String>,
    output: VecDeque<String>,
    latest: Option<String>,
    progress: Option<(u32, u32)>,
    exited: bool,
    ready: bool,
}

impl ReloadBuild {
    /// cargo's compilation units done and in total, once it reported them.
    /// Before that the build is resolving dependencies.
    pub fn progress(&self) -> Option<(u32, u32)> {
        self.progress
    }

    /// The build's latest line other than cargo's counter: the launcher's
    /// `Resolving dependencies…`, or cargo's own, such as
    /// `Downloaded serde v1.0.228`.
    pub fn latest(&self) -> Option<&str> {
        self.latest.as_deref()
    }

    /// Whether the build succeeded and the restart waits for every agent
    /// to be idle.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    fn start(launcher: &std::ffi::OsStr) -> std::io::Result<Self> {
        let mut command = Command::new(launcher);
        command
            .arg("build")
            .env("CARGO_TERM_PROGRESS_WHEN", "always")
            .env("CARGO_TERM_PROGRESS_WIDTH", "80")
            .env("CARGO_TERM_COLOR", "never")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        // Its own session and process group, so cancelling stops cargo and
        // rustc too, and a git prompt for a plugin cannot reach the screen.
        detach(&mut command);
        let mut child = command.spawn()?;
        let (sender, lines) = crossbeam_channel::unbounded();
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || forward(stderr, sender));
        }
        Ok(Self {
            child,
            lines,
            output: VecDeque::new(),
            latest: None,
            progress: None,
            exited: false,
            ready: false,
        })
    }

    fn take(&mut self, line: String) {
        if let Some(progress) = cargo_progress(&line) {
            self.progress = Some(progress);
        } else if !line.trim().is_empty() {
            if self.output.len() == KEPT_LINES {
                self.output.pop_front();
            }
            self.latest = Some(line.trim().to_owned());
            self.output.push_back(line);
        }
    }

    /// The output from the first error on, or its end when no line starts
    /// with `error`.
    fn errors(&self) -> String {
        let start = self
            .output
            .iter()
            .position(|line| line.starts_with("error"))
            .unwrap_or_else(|| self.output.len().saturating_sub(ERROR_LINES));
        self.output
            .iter()
            .skip(start)
            .take(ERROR_LINES)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
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

/// Stops the running rebuild, if any.
#[derive(Event, Clone, Copy, Debug)]
pub struct CancelReload;

/// Sends each `\r`- or `\n`-separated segment of the build's stderr; cargo
/// redraws its progress bar after `\r`.
fn forward(stderr: ChildStderr, lines: Sender<String>) {
    let mut segment = Vec::new();
    for byte in BufReader::new(stderr).bytes() {
        let Ok(byte) = byte else {
            break;
        };
        if byte != b'\r' && byte != b'\n' {
            segment.push(byte);
            continue;
        }
        if !segment.is_empty()
            && lines
                .send(String::from_utf8_lossy(&segment).into_owned())
                .is_err()
        {
            return;
        }
        segment.clear();
    }
    if !segment.is_empty() {
        lines
            .send(String::from_utf8_lossy(&segment).into_owned())
            .ok();
    }
}

/// `done/total` of cargo's progress bar:
/// `    Building [=====>      ] 37/41: crate_a, crate_b`.
fn cargo_progress(line: &str) -> Option<(u32, u32)> {
    let (_, counts) = line
        .trim_start()
        .strip_prefix("Building [")?
        .split_once("] ")?;
    let counts = counts.split(':').next()?;
    let (done, total) = counts.trim().split_once('/')?;
    Some((done.parse().ok()?, total.parse().ok()?))
}

fn reload(
    In(_): In<CommandArgs>,
    agents: Query<&AgentStatus, With<Agent>>,
    build: Option<Res<ReloadBuild>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let notice = if agents.iter().any(|status| *status != AgentStatus::Idle) {
        "A turn is running. Press Esc to stop it, then /reload.".to_owned()
    } else if build.is_some() {
        "A rebuild is already running; Esc cancels it.".to_owned()
    } else if let Some(launcher) = launcher::executable() {
        match ReloadBuild::start(&launcher) {
            Ok(build) => {
                commands.insert_resource(build);
                "Rebuilding the agent…".to_owned()
            }
            Err(failure) => format!("Could not start the rebuild: {failure}"),
        }
    } else {
        "/reload needs the rig launcher: start the agent with `rig`.".to_owned()
    };
    notices.write(Notice::new(notice));
}

/// Reads the build's output; on its exit, reports a failure or marks the
/// build ready.
fn drain_reload(
    build: Option<ResMut<ReloadBuild>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(mut build) = build else {
        return;
    };
    if build.ready {
        return;
    }
    loop {
        match build.lines.try_recv() {
            Ok(line) => build.take(line),
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => break,
        }
    }
    let status = match build.child.try_wait() {
        Ok(Some(status)) => status,
        Ok(None) => return,
        Err(failure) => {
            notices.write(Notice::new(format!("The rebuild failed: {failure}")));
            commands.remove_resource::<ReloadBuild>();
            return;
        }
    };
    build.exited = true;
    if status.success() {
        build.ready = true;
        notices.write(Notice::new("Build ready; restarting.".to_owned()));
    } else {
        notices.write(Notice::new(format!(
            "The rebuild failed ({status}); this build keeps running.\n{}",
            build.errors()
        )));
        commands.remove_resource::<ReloadBuild>();
    }
}

/// Exits with [`RELOAD_EXIT_CODE`] once the build is ready and every agent
/// is idle. The session is saved on exit.
fn finish_reload(
    build: Option<Res<ReloadBuild>>,
    agents: Query<&AgentStatus, With<Agent>>,
    mut exit: MessageWriter<AppExit>,
) {
    if build.is_some_and(|build| build.ready)
        && agents.iter().all(|status| *status == AgentStatus::Idle)
    {
        exit.write(AppExit::from_code(RELOAD_EXIT_CODE));
    }
}

/// Drops a running rebuild when the app exits, which kills it with cargo
/// and rustc, instead of leaving that to the end of the process.
fn stop_reload_on_exit(world: &mut World) {
    world.remove_resource::<ReloadBuild>();
}

fn on_cancel_reload(
    _: On<CancelReload>,
    build: Option<Res<ReloadBuild>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if build.is_some_and(|build| !build.ready) {
        commands.remove_resource::<ReloadBuild>();
        notices.write(Notice::new("Rebuild cancelled.".to_owned()));
    }
}
