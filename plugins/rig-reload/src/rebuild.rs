//! The rebuild's child process, its output and the restart.

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStderr, Command, Stdio};

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use rig_harness::harness_protocol::{RELOAD_EXIT_CODE, first_errors};
use rig_harness::prelude::*;
use rig_tools::process::{detach, kill_group};

use crate::ReloadStatus;

/// Lines of a failed build in the note to the model, from its first error.
const NOTE_LINES: usize = 40;

pub(crate) fn add(app: &mut App) {
    app.add_systems(
        Update,
        (start_queued_reload, drain_reload, finish_reload)
            .chain()
            .after(PollCalls),
    );
}

/// Stops the running build.
pub(crate) fn stop(commands: &mut Commands) {
    commands.remove_resource::<ReloadBuild>();
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
