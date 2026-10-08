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
    io::Read,
    process::{Child, ChildStderr, Command, Stdio},
    sync::{Arc, Mutex},
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
    child: Arc<Mutex<Child>>,
}

/// Quitting while the build runs stops it, so no cargo is left holding the
/// agent project's build lock. On unix the launcher leads its own process
/// group, which is killed whole, cargo and rustc included. The lock is busy
/// only while the task waits for a build that already closed its output.
impl Drop for Rebuild {
    fn drop(&mut self) {
        let Ok(mut child) = self.child.try_lock() else {
            return;
        };
        if !matches!(child.try_wait(), Ok(None)) {
            return;
        }
        #[cfg(unix)]
        if let Ok(group) = libc::pid_t::try_from(child.id()) {
            // SAFETY: `kill` takes plain numbers; a negative pid names the
            // process group the launcher leads.
            unsafe { libc::kill(-group, libc::SIGKILL) };
        }
        let _ = child.kill();
        let _ = child.wait();
    }
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
    let mut command = Command::new(launcher);
    command
        .arg("build")
        .env("CARGO_TERM_PROGRESS_WHEN", "always")
        .env("CARGO_TERM_PROGRESS_WIDTH", "120")
        .env("CARGO_TERM_COLOR", "never")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            notices.write(Notice::error(
                agent,
                format!("Cannot start the launcher: {error}"),
            ));
            return;
        }
    };
    let stderr = child.stderr.take();
    let child = Arc::new(Mutex::new(child));
    let (sender, counts) = crossbeam_channel::unbounded();
    let waited = child.clone();
    let task = IoTaskPool::get().spawn(async move { build(stderr, &waited, &sender) });
    commands.insert_resource(Rebuild {
        agent,
        task,
        counts,
        child,
    });
    commands.init_resource::<BuildProgress>();
    notices.write(Notice::info(agent, "Rebuilding the agent..."));
}

/// Follow `launcher build` through its `stderr`, sending each progress
/// reading and keeping the last other lines. This blocks the IO pool thread
/// it runs on until the build exits.
fn build(stderr: Option<ChildStderr>, child: &Mutex<Child>, counts: &Sender<Count>) -> Built {
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
    if let Some(mut stderr) = stderr {
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
    let success = child
        .lock()
        .is_ok_and(|mut child| child.wait().is_ok_and(|status| status.success()));
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
