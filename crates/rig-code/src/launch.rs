//! Working with the `rig` launcher: `/reload`, the startup marker and the
//! launcher's start notice.
//!
//! `/reload` runs `$RIG_CODE_LAUNCHER build` on the IO task pool, which
//! regenerates the agent project from the plugin list and builds it. Cargo's
//! `N/M` counter feeds [`BuildProgress`]. A failed build is shown as a notice
//! and the running binary carries on; a successful one exits with
//! [`RELOAD_EXIT_CODE`] once every agent is idle, and the launcher starts
//! the new binary on the same session.

use std::{
    collections::VecDeque,
    ffi::OsString,
    io::Read,
    process::{Command, Stdio},
};

use bevy::{
    prelude::*,
    tasks::{IoTaskPool, Task, futures::check_ready},
};
use crossbeam_channel::{Receiver, Sender};

use crate::{
    RELOAD_EXIT_CODE,
    ecs::{
        AgentSystems, Notice, RigAppExt,
        agent::{Agent, AgentStatus},
        command::CommandInput,
        paths,
    },
};

/// Lines of build output kept to explain a failure.
const KEPT_LINES: usize = 200;

/// The progress of a running `/reload` build, present while it runs and
/// after it succeeded until the app exits.
#[derive(Resource, Clone, Debug, Default)]
pub struct BuildProgress {
    /// Crates built so far.
    pub done: usize,
    /// Crates to build, `0` until cargo reports it.
    pub total: usize,
    /// The crates being built now, as cargo names them.
    pub current: String,
    /// The build succeeded; the app reloads once every agent is idle.
    pub finished: bool,
}

/// One reading of cargo's progress bar: done, total, and the crates being
/// built.
type Count = (usize, usize, String);

/// The running build.
#[derive(Resource)]
struct Rebuild {
    agent: Entity,
    task: Task<Built>,
    counts: Receiver<Count>,
}

/// How the build ended: success, and the output lines that are not
/// progress.
struct Built {
    success: bool,
    output: Vec<String>,
}

/// `/reload`, the startup marker and the start notice.
pub struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "reload",
            "Rebuild the agent with the plugin list and restart it on the same session",
            reload,
        )
        .add_systems(
            Update,
            (
                start_notice.run_if(run_once),
                // After the agent loop, so a turn started this frame is seen
                // before the reload exit.
                watch_build
                    .run_if(resource_exists::<BuildProgress>)
                    .after(AgentSystems::Collect),
            ),
        )
        .add_systems(Last, mark_ready.run_if(run_once));
    }
}

fn reload(
    In(input): In<CommandInput>,
    agents: Query<&AgentStatus>,
    progress: Option<Res<BuildProgress>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = input.agent;
    if agents.iter().any(|status| *status != AgentStatus::Idle) {
        notices.write(Notice::error(
            agent,
            "A turn is running. Press Esc to stop it, then /reload.",
        ));
        return;
    }
    if progress.is_some() {
        notices.write(Notice::error(agent, "A build is already running."));
        return;
    }
    let Some(launcher) = std::env::var_os("RIG_CODE_LAUNCHER") else {
        notices.write(Notice::error(
            agent,
            "This agent was started without the launcher; start it with `rig` to use /reload.",
        ));
        return;
    };
    let (sender, counts) = crossbeam_channel::unbounded();
    let task = IoTaskPool::get().spawn(async move { build(launcher, &sender) });
    commands.insert_resource(Rebuild {
        agent,
        task,
        counts,
    });
    commands.init_resource::<BuildProgress>();
    notices.write(Notice::info(agent, "Rebuilding the agent..."));
}

/// Run `launcher build`, sending each progress reading and keeping the last
/// other lines. This blocks the IO pool thread it runs on until cargo exits.
fn build(launcher: OsString, counts: &Sender<Count>) -> Built {
    let child = Command::new(launcher)
        .arg("build")
        .env("CARGO_TERM_PROGRESS_WHEN", "always")
        .env("CARGO_TERM_PROGRESS_WIDTH", "120")
        .env("CARGO_TERM_COLOR", "never")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            return Built {
                success: false,
                output: vec![format!("cannot start the launcher: {error}")],
            };
        }
    };
    let mut output = VecDeque::new();
    let mut segment = Vec::new();
    let mut keep = |segment: &[u8]| {
        let line = String::from_utf8_lossy(segment);
        if let Some(count) = progress(&line) {
            let _ = counts.send(count);
            return;
        }
        let line = line.trim_end();
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with("Compiling ") {
            return;
        }
        if output.len() == KEPT_LINES {
            output.pop_front();
        }
        output.push_back(line.to_owned());
    };
    if let Some(mut stderr) = child.stderr.take() {
        let mut buffer = [0; 4096];
        while let Ok(read) = stderr.read(&mut buffer) {
            if read == 0 {
                break;
            }
            for &byte in buffer.iter().take(read) {
                if byte == b'\n' || byte == b'\r' {
                    keep(&segment);
                    segment.clear();
                } else {
                    segment.push(byte);
                }
            }
        }
        keep(&segment);
    }
    let success = child.wait().is_ok_and(|status| status.success());
    Built {
        success,
        output: output.into(),
    }
}

/// Read cargo's progress bar, `Building [====>   ] 37/41: bevy_ecs, rig-core`.
fn progress(line: &str) -> Option<Count> {
    let (_, rest) = line.split_once("] ")?;
    let (counter, current) = rest.split_once(": ").unwrap_or((rest, ""));
    let (done, total) = counter.trim().split_once('/')?;
    Some((
        done.parse().ok()?,
        total.parse().ok()?,
        current.trim().to_owned(),
    ))
}

/// Follow the build: update the progress, report a failure, and exit with
/// the reload code once the build succeeded and every agent is idle.
fn watch_build(
    rebuild: Option<ResMut<Rebuild>>,
    mut progress: ResMut<BuildProgress>,
    agents: Query<&AgentStatus>,
    mut notices: MessageWriter<Notice>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
) {
    let Some(mut rebuild) = rebuild else {
        if progress.finished && agents.iter().all(|status| *status == AgentStatus::Idle) {
            exit.write(AppExit::from_code(RELOAD_EXIT_CODE));
        }
        return;
    };
    for (done, total, current) in rebuild.counts.try_iter() {
        progress.done = done;
        progress.total = total;
        progress.current = current;
    }
    let Some(built) = check_ready(&mut rebuild.task) else {
        return;
    };
    let agent = rebuild.agent;
    commands.remove_resource::<Rebuild>();
    if built.success {
        progress.finished = true;
        notices.write(Notice::info(
            agent,
            "Build done. Restarting once every agent is idle.",
        ));
    } else {
        commands.remove_resource::<BuildProgress>();
        notices.write(Notice::error(
            agent,
            format!(
                "The build failed; this binary keeps running.\n{}",
                built.output.join("\n")
            ),
        ));
    }
}

/// Show the launcher's message for this start, such as a rollback, to every
/// agent.
fn start_notice(agents: Query<Entity, With<Agent>>, mut notices: MessageWriter<Notice>) {
    let Some(text) = std::env::var_os("RIG_CODE_NOTICE") else {
        return;
    };
    let text = text.to_string_lossy();
    for agent in &agents {
        notices.write(Notice::error(agent, text.clone()));
    }
}

/// Tell the launcher this binary survived its first frame: the session was
/// restored and the screen drawn.
fn mark_ready() {
    let marker = paths::session_dir().join("ready");
    if let Err(error) = std::fs::write(&marker, "") {
        error!("cannot write {}: {error}", marker.display());
    }
}
