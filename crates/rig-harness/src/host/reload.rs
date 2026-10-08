//! `/reload`: rebuild the agent through the `rig` launcher and restart on
//! the new build. The build runs as a child process whose stderr a std
//! thread forwards line by line. Until cargo's own `done/total` counter
//! appears, the build is resolving dependencies and its latest line is the
//! progress. Quitting during the build kills it with cargo and rustc.
//!
//! The restart does not wait for running turns: it exits as a crash would,
//! after giving running tool calls that may change something a few seconds
//! to finish, and the new build carries the turns on from the session
//! logs.

use std::collections::VecDeque;
use std::io::{BufReader, Read};
use std::process::{Child, ChildStderr, Command, Stdio};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_reflect::prelude::*;
use crossbeam_channel::{Receiver, Sender, TryRecvError};

use rig::harness_protocol::RELOAD_EXIT_CODE;

use super::launcher;
use super::process::{detach, kill_group};
use crate::core::agent::Notice;
use crate::core::calls::Wake;
use crate::core::commands::{AppCommandsExt, CommandArgs};
use crate::core::turn::PollCalls;

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
        .add_message::<ReloadFailed>()
        .add_observer(on_cancel_reload)
        .add_systems(
            Update,
            (drain_reload, finish_reload).chain().after(PollCalls),
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

    /// Whether the build succeeded and the agent restarts on it.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    fn start(launcher: &std::ffi::OsStr, wake: Wake) -> std::io::Result<Self> {
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
            std::thread::spawn(move || forward(stderr, sender, wake));
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

/// A `/reload` build failed. Views show its output until the user
/// dismisses it; it is logged too.
#[derive(Message, Clone, Debug)]
pub struct ReloadFailed {
    /// The build's output from its first error on.
    pub output: String,
}

/// Stops the running rebuild, if any.
#[derive(Event, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Event, Clone, Debug, Default)]
pub struct CancelReload;

/// Sends each `\r`- or `\n`-separated segment of the build's stderr, and
/// wakes the loop for each and at the end; cargo redraws its progress bar
/// after `\r`.
fn forward(stderr: ChildStderr, lines: Sender<String>, wake: Wake) {
    let mut segment = Vec::new();
    for byte in BufReader::new(stderr).bytes() {
        let Ok(byte) = byte else {
            break;
        };
        if byte != b'\r' && byte != b'\n' {
            segment.push(byte);
            continue;
        }
        if !segment.is_empty() {
            if lines
                .send(String::from_utf8_lossy(&segment).into_owned())
                .is_err()
            {
                return;
            }
            wake.wake();
        }
        segment.clear();
    }
    if !segment.is_empty() {
        lines
            .send(String::from_utf8_lossy(&segment).into_owned())
            .ok();
    }
    wake.wake();
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
    build: Option<Res<ReloadBuild>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let notice = if build.is_some() {
        "A rebuild is already running; Esc cancels it.".to_owned()
    } else if let Some(launcher) = launcher::executable() {
        match ReloadBuild::start(&launcher, wake.clone()) {
            Ok(build) => {
                commands.insert_resource(build);
                "Rebuilding the agent…".to_owned()
            }
            Err(failure) => format!("Could not start the rebuild: {failure}"),
        }
    } else {
        "/reload needs the rig launcher: start the agent with `rig`.".to_owned()
    };
    notices.write(Notice::info(None, notice));
}

/// Reads the build's output; on its exit, reports a failure or marks the
/// build ready.
fn drain_reload(
    build: Option<ResMut<ReloadBuild>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut failures: MessageWriter<ReloadFailed>,
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
            notices.write(Notice::error(
                None,
                format!("The rebuild failed: {failure}"),
            ));
            commands.remove_resource::<ReloadBuild>();
            return;
        }
    };
    build.exited = true;
    if status.success() {
        build.ready = true;
        notices.write(Notice::info(None, "Build ready; restarting.".to_owned()));
    } else {
        let output = build.errors();
        error!("the rebuild failed ({status}):\n{output}");
        notices.write(Notice::error(
            None,
            format!("The rebuild failed ({status}); this build keeps running."),
        ));
        failures.write(ReloadFailed { output });
        commands.remove_resource::<ReloadBuild>();
    }
}

/// Exits with [`RELOAD_EXIT_CODE`] once the build is ready, at most once.
/// The exit leaves the running turns to the new build.
fn finish_reload(
    build: Option<Res<ReloadBuild>>,
    mut exiting: Local<bool>,
    mut exit: MessageWriter<AppExit>,
) {
    if !*exiting && build.is_some_and(|build| build.ready) {
        *exiting = true;
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
        notices.write(Notice::info(None, "Rebuild cancelled.".to_owned()));
    }
}
