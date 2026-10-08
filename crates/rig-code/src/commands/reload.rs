//! `/reload`: rebuild the agent with the `rig` launcher in the background,
//! show cargo's progress, and restart into the new build.

use std::io::Read;
use std::process::{Command, ExitStatus};
use std::time::Duration;

use bevy::prelude::*;
use bevy::tasks::IoTaskPool;
use bevy::tasks::futures::check_ready;

use crate::core::{AgentStatus, CallTask, DataDir, Notice, RELOAD_EXIT_CODE, RunCommand, Work};
use crate::tools::child::LoggedChild;

/// How often the build log is read.
const READ_EVERY: Duration = Duration::from_millis(100);
/// Lines of compiler output shown when the build fails.
const ERROR_LINES: usize = 60;

/// A `/reload` build in flight. Despawning it kills the build.
#[derive(Component)]
#[require(BuildProgress, BuildOutput)]
pub struct Build;

/// Cargo's unit counter for the running build.
#[derive(Component, Default, Debug)]
pub struct BuildProgress {
    /// Units finished.
    pub done: usize,
    /// Units in the build.
    pub total: usize,
    /// The crates being compiled, as cargo names them.
    pub current: String,
}

/// The build's output lines, and an unfinished line.
#[derive(Component, Default)]
pub(super) struct BuildOutput {
    lines: Vec<String>,
    partial: Vec<u8>,
}

/// New bytes of the build log on their way from the task.
#[derive(Component)]
pub(super) struct BuildRx(async_channel::Receiver<Vec<u8>>);

type BuildTask = CallTask<std::io::Result<ExitStatus>>;

/// Starts a build unless a turn is running, a build already is, or the
/// agent was not started by the launcher.
pub(super) fn reload(
    run: On<RunCommand>,
    mut commands: Commands,
    agents: Query<(&AgentStatus, Option<&Work>)>,
    builds: Query<(), With<Build>>,
    data: Res<DataDir>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = run.agent;
    if agents
        .iter()
        .any(|(status, work)| status.is_busy() || work.is_some_and(|work| !work.is_empty()))
    {
        notices.write(Notice::error(
            agent,
            "a turn is running; press Esc to stop it first",
        ));
        return;
    }
    if !builds.is_empty() {
        notices.write(Notice::error(agent, "a build is already running"));
        return;
    }
    let Some(launcher) = std::env::var_os("RIG_LAUNCHER").filter(|value| !value.is_empty()) else {
        notices.write(Notice::error(
            agent,
            "/reload needs the rig launcher; this agent was started without it",
        ));
        return;
    };
    let mut command = Command::new(launcher);
    command.arg("build");
    let log = data.0.join("logs").join("build.log");
    let (sender, receiver) = async_channel::unbounded();
    let task = IoTaskPool::get().spawn(async move {
        let mut child = LoggedChild::spawn(command, &log)?;
        let mut reader = std::fs::File::open(&log)?;
        loop {
            let status = child.wait(READ_EVERY).await?;
            let mut chunk = Vec::new();
            reader.read_to_end(&mut chunk)?;
            if !chunk.is_empty() {
                // The receiver is gone only when the build was cancelled.
                let _ = sender.try_send(chunk);
            }
            if let Some(status) = status {
                return Ok::<_, std::io::Error>(status);
            }
        }
    });
    commands.spawn((Build, CallTask(task), BuildRx(receiver)));
    notices.write(Notice::info(agent, "building the agent…"));
}

/// Reads the build's output into its progress and lines, and finishes it:
/// on success the app exits with the reload code, on failure the errors
/// are shown and this binary keeps running.
pub(super) fn poll_build(
    mut commands: Commands,
    mut builds: Query<(
        Entity,
        &mut BuildProgress,
        &mut BuildOutput,
        &BuildRx,
        &mut BuildTask,
    )>,
    mut notices: MessageWriter<Notice>,
    mut exit: MessageWriter<AppExit>,
) {
    for (build, mut progress, mut output, chunks, mut task) in &mut builds {
        while let Ok(chunk) = chunks.0.try_recv() {
            output.partial.extend(chunk);
            read_segments(&mut output, &mut progress);
        }
        let Some(result) = check_ready(&mut task.0) else {
            continue;
        };
        commands.entity(build).despawn();
        match result {
            Ok(status) if status.success() => {
                notices.write(Notice::info(None, "build finished; restarting"));
                exit.write(AppExit::from_code(RELOAD_EXIT_CODE));
            }
            Ok(status) => {
                let tail = std::mem::take(&mut output.partial);
                output
                    .lines
                    .push(String::from_utf8_lossy(&tail).into_owned());
                notices.write(Notice::error(
                    None,
                    format!(
                        "/reload: the build failed ({status}); this build keeps running. \
                         Esc closes this.\n{}",
                        errors(&output.lines, status.code() == Some(2))
                    ),
                ));
            }
            Err(error) => {
                notices.write(Notice::error(None, format!("/reload: {error}")));
            }
        }
    }
}

/// Moves the finished segments of `output.partial` into lines, and cargo's
/// `Building [...] N/M: names` segments into `progress`.
fn read_segments(output: &mut BuildOutput, progress: &mut BuildProgress) {
    while let Some(end) = output
        .partial
        .iter()
        .position(|byte| matches!(byte, b'\r' | b'\n'))
    {
        let segment: Vec<u8> = output.partial.drain(..=end).collect();
        let text = String::from_utf8_lossy(&segment);
        // Compiler output keeps its indentation; the bar is indented too.
        let text = text.trim_end();
        if let Some(bar) = text.trim_start().strip_prefix("Building [") {
            if let Some((done, total, current)) = bar.split_once(']').and_then(|(_, counter)| {
                let (counts, names) = counter.split_once(':').unwrap_or((counter, ""));
                let (done, total) = counts.trim().split_once('/')?;
                Some((done.parse().ok()?, total.parse().ok()?, names.trim()))
            }) {
                *progress = BuildProgress {
                    done,
                    total,
                    current: current.to_owned(),
                };
            }
        } else if !text.trim_start().is_empty() {
            output.lines.push(text.to_owned());
        }
    }
}

/// The lines worth showing for a failed build: from the first error on,
/// or everything for a launcher configuration error.
fn errors(lines: &[String], configuration: bool) -> String {
    let start = if configuration {
        0
    } else {
        lines
            .iter()
            .position(|line| line.starts_with("error"))
            .unwrap_or(0)
    };
    let shown: Vec<&str> = lines
        .iter()
        .skip(start)
        .take(ERROR_LINES)
        .map(String::as_str)
        .collect();
    shown.join("\n")
}
