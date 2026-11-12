//! The rebuild's child process, whose stderr a std thread forwards line by
//! line, and the restart.

use std::io::{BufRead, BufReader};
use std::process::{ChildStderr, Command, Stdio};
use std::sync::{Mutex, PoisonError};

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use rig_harness::harness_protocol;
use rig_harness::harness_protocol::{UPDATE_SAYS, first_errors, without_secrets};
use rig_harness::host::launcher::Invoked;
use rig_harness::prelude::*;
use rig_tools::process::{Process, STOP_GRACE};

use crate::reload::{LauncherFile, Rebuild, ReloadStatus};

/// Lines of a failed build in the note to the model, from its first error.
const MESSAGE_LINES: usize = 40;

pub(crate) fn add(app: &mut App) {
    app.add_systems(
        Update,
        (start_queued_reload, drain_reload, finish_reload)
            .chain()
            .after(PollTasks),
    );
}

/// Stops the running build, then says it was cancelled. What an update
/// left is read once it stopped, so an install that landed as it stopped
/// is told: the new rig stays installed, for the next start or reload.
pub(crate) fn cancel(commands: &mut Commands) {
    commands.queue(|world: &mut World| {
        let notice = world
            .remove_resource::<ReloadBuild>()
            .map_or("Rebuild cancelled.", |mut build| build.cancel());
        world.write_message(Notice::info(None, notice));
    });
}

/// The rebuild's process. Dropping it while the build runs stops the
/// build, as the app's runner does when it drops the app on exit.
#[derive(Resource)]
pub(crate) struct ReloadBuild {
    /// In a mutex only to be `Sync`, as a resource is.
    process: Mutex<Process>,
    /// The agent that asked, whose model is told when the build fails.
    agent: Entity,
    rebuild: Rebuild,
    lines: Receiver<String>,
    /// This build's output so far. Not read back from the build log, which
    /// a build queued behind this one, or an older one when this one fails
    /// before it starts its log, may have written.
    output: Vec<String>,
    /// The launcher's file as this build started, which `rig update`
    /// replaces once it installed the new rig.
    installed: LauncherFile,
}

impl ReloadBuild {
    fn start(
        launcher: &std::ffi::OsStr,
        rebuild: Rebuild,
        agent: Entity,
        wake: Wake,
    ) -> std::io::Result<Self> {
        let path = std::path::PathBuf::from(launcher);
        let installed = LauncherFile::new(path.clone(), harness_protocol::Stamp::of(&path)?);
        let mut command = Command::new(launcher);
        command
            .arg(rebuild.command())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        // Not through `Processes`, whose sandbox may lack the network or
        // the home cargo needs; it keeps the secrets from cargo and every
        // build script all the same.
        without_secrets(&mut command);
        // Its own session and process group, so cancelling stops cargo and
        // rustc too, and a git prompt for a plugin cannot reach the screen.
        let mut process = Process::start_with_stdio(command)?;
        let (sender, lines) = crossbeam_channel::unbounded();
        if let Some(stderr) = process.take_stderr() {
            std::thread::spawn(move || forward(stderr, sender, wake));
        }
        Ok(Self {
            process: Mutex::new(process),
            agent,
            rebuild,
            lines,
            output: Vec::new(),
            installed,
        })
    }

    /// Whether this is `rig update` and it replaced the launcher's file
    /// since it started, so a failure is its agent build's.
    fn installed(&self) -> bool {
        self.rebuild == Rebuild::Update && self.installed.replaced()
    }

    /// Stops the build; what its cancel says.
    fn cancel(&mut self) -> &'static str {
        let process = self.process.get_mut();
        process
            .unwrap_or_else(PoisonError::into_inner)
            .stop(STOP_GRACE)
            .ok();
        match self.rebuild {
            Rebuild::Update if self.installed.replaced() => {
                "Update cancelled during the agent build: the new rig is installed, and the \
                 next start or /reload runs it."
            }
            Rebuild::Update => "Update cancelled.",
            Rebuild::Build => "Rebuild cancelled.",
        }
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
    /// Asks `rig build` to stop (`SIGTERM`) before it is killed: cargo runs
    /// in a process group of its own, which the launcher kills only when
    /// asked.
    fn drop(&mut self) {
        let process = self
            .process
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        process.stop(STOP_GRACE).ok();
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
    let ReloadStatus::Queued { agent, rebuild } = *status else {
        return;
    };
    if !turns.is_empty() {
        return;
    }
    *status = ReloadStatus::Idle;
    let Some(launcher) = harness_protocol::launcher() else {
        return;
    };
    let notice = match ReloadBuild::start(&launcher, rebuild, agent, wake.clone()) {
        Ok(build) => {
            commands.insert_resource(build);
            *status = ReloadStatus::Building { latest: None };
            match rebuild {
                Rebuild::Build => "Rebuilding the agent…".to_owned(),
                Rebuild::Update => "Updating rig with `rig update`, then building the agent \
                                    with it; after a change this takes 2-3 minutes…"
                    .to_owned(),
            }
        }
        Err(failure) => format!("Could not start the rebuild: {failure}"),
    };
    notices.write(Notice::info(None, notice));
}

/// Reads the build's output; on its exit, reports a failure or marks the
/// build ready.
fn drain_reload(
    build: Option<ResMut<ReloadBuild>>,
    (invoked, launcher): (Option<Res<Invoked>>, Option<Res<LauncherFile>>),
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
            Ok(line) if line.starts_with(UPDATE_SAYS) => {
                notices.write(Notice::info(None, line.trim_end()));
                build.output.push(line);
            }
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
    let process = build
        .process
        .get_mut()
        .unwrap_or_else(PoisonError::into_inner);
    let exit = match process.try_wait() {
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
    commands.remove_resource::<ReloadBuild>();
    if exit.success() {
        // Only this launcher restarts a run nobody sits at without its
        // prompt, and only with its own build: once the launcher was
        // replaced, the run does not reload and goes on, or ends, as any
        // run does, its last line written.
        let headless = invoked.is_some_and(|invoked| !invoked.interactive());
        if headless && launcher.is_some_and(|launcher| launcher.replaced()) {
            *status = ReloadStatus::Idle;
            // A warning reaches a `-p` run's stderr whatever its prompt.
            notices.write(match build.rebuild {
                Rebuild::Build => Notice::warn(
                    None,
                    "rig was replaced since this run started, so it does not reload; the next \
                     run starts the new version.",
                ),
                Rebuild::Update => {
                    Notice::info(None, "rig updated; the next run starts the new version.")
                }
            });
            return;
        }
        *status = ReloadStatus::Ready;
        let ready = match build.rebuild {
            Rebuild::Build => "Build ready; restarting.",
            Rebuild::Update => "rig updated; restarting on the new rig.",
        };
        notices.write(Notice::info(None, ready));
        return;
    }
    *status = ReloadStatus::Failed;
    let first = first_errors(build.output.iter().map(String::as_str), 1)
        .first()
        .map(|line| format!(": {}", line.trim()))
        .unwrap_or_default();
    if build.rebuild == Rebuild::Update && !build.installed() {
        // Before any build: cargo's error, not the build log, says why;
        // without one, `rig update`'s own line said it already.
        if !first.is_empty() {
            notices.write(Notice::error(
                None,
                format!("The update failed{first}; this rig keeps running."),
            ));
        }
        let text = format!(
            "Updating rig (/update) failed before it was installed; this rig keeps \
             running.\n\n`rig update` exited with {exit}. Its output from the first error:\n{}",
            build.errors(MESSAGE_LINES)
        );
        launcher::inform_build_failure(&mut commands, build.agent, text);
        return;
    }
    let (what, command) = match build.rebuild {
        Rebuild::Build => ("/reload", "rig build"),
        Rebuild::Update => ("/update", "rig update"),
    };
    // The notice is the one log line: the whole output is in the build log
    // it names.
    notices.write(Notice::error(
        None,
        format!(
            "The rebuild failed ({exit}){first}; this build keeps running. Whole output: {}",
            Home::from_env().build_log().display()
        ),
    ));
    let text = launcher::build_failure_message(
        what,
        &format!(
            "`{command}` exited with {exit}. Its output from the first error:\n{}",
            build.errors(MESSAGE_LINES)
        ),
    );
    launcher::inform_build_failure(&mut commands, build.agent, text);
}

/// Exits to reload once the build is ready and no turn runs, at most once.
fn finish_reload(
    status: Res<ReloadStatus>,
    turns: Query<(), With<TurnOf>>,
    mut exiting: Local<bool>,
    mut commands: Commands,
) {
    if !*exiting && *status == ReloadStatus::Ready && turns.is_empty() {
        *exiting = true;
        commands.trigger(Exit(ExitReason::Reload));
    }
}

#[cfg(all(test, unix))]
mod tests;
