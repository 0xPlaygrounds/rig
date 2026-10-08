//! `/reload`: rebuild the agent with the launcher and restart into the new
//! binary. The build runs as `rig build` in the background while the agent
//! keeps working. Its progress comes from cargo's progress bar and its
//! errors from cargo's JSON messages. Once it succeeds and every agent is
//! idle, the app saves and exits with [`RELOAD_EXIT_CODE`], and the launcher
//! starts the new binary, which restores the session.

use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::process::Command;

use async_channel::{Receiver, Sender, TryRecvError};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use serde::Deserialize;

use super::process::Group;
use super::session::Dirs;
use crate::RELOAD_EXIT_CODE;
use crate::core::SessionDir;
use crate::core::agent::{Agent, Status};
use crate::core::registry::{AppExt, CommandInput, Notice};

/// How many lines of build errors a notice shows. The log has all of them.
const ERROR_LINES: usize = 40;

/// The `/reload` command, the build in flight, the reload once it is done,
/// and the readiness signal the launcher waits for.
#[derive(Default)]
pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "reload",
            "Rebuild with the plugins in plugins.toml and restart",
            reload,
        )
        .add_systems(Update, (poll_build, finish_reload).chain())
        .add_systems(PostStartup, launcher_notice)
        .add_systems(Last, ready.run_if(run_once));
    }
}

/// How far the build is, from cargo's progress bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Crates compiled.
    pub done: usize,
    /// Crates to compile in all.
    pub total: usize,
    /// The crates compiling now, as cargo lists them.
    pub building: String,
}

/// The `/reload` build: `rig build` running in the background. Dropping it
/// kills the build.
#[derive(Component)]
pub struct BuildJob {
    agent: Entity,
    child: Group,
    output: Receiver<Output>,
    /// The latest progress cargo reported.
    pub progress: Option<Progress>,
    /// The build succeeded; the app reloads once every agent is idle.
    pub built: bool,
    errors: Vec<String>,
    messages: Vec<String>,
}

/// A line of the build's output.
enum Output {
    /// A JSON message from cargo on stdout.
    Json(String),
    /// Text on stderr, split on carriage returns too, so each redraw of
    /// the progress bar arrives on its own.
    Text(String),
}

impl BuildJob {
    /// Starts `<launcher> build` with its output read on two threads.
    fn start(agent: Entity, launcher: OsString) -> std::io::Result<Self> {
        let mut child = Group::spawn(Command::new(launcher).arg("build"))?;
        let (stdout, stderr) = child.take_output();
        let (sender, output) = async_channel::unbounded();
        // On an error, dropping `child` kills the build.
        read_stdout(stdout, sender.clone())?;
        read_stderr(stderr, sender)?;
        Ok(Self {
            agent,
            child,
            output,
            progress: None,
            built: false,
            errors: Vec::new(),
            messages: Vec::new(),
        })
    }

    /// Takes in one line of output. Returns whether the progress changed.
    fn take(&mut self, output: Output) -> bool {
        match output {
            Output::Json(line) => {
                if let Ok(CargoMessage {
                    message: Some(diagnostic),
                }) = serde_json::from_str(&line)
                    && diagnostic.level == "error"
                    && let Some(rendered) = diagnostic.rendered
                {
                    self.errors.push(rendered.trim_end().to_owned());
                }
                false
            }
            Output::Text(text) => match parse_progress(&text) {
                Some(progress) => {
                    let changed = self.progress.as_ref() != Some(&progress);
                    self.progress = Some(progress);
                    changed
                }
                None => {
                    self.messages.push(text);
                    false
                }
            },
        }
    }

    /// What went wrong: the compiler's errors, or else the end of the
    /// build's other output, such as the launcher's own error.
    fn failure(&self) -> String {
        if self.errors.is_empty() {
            let start = self.messages.len().saturating_sub(ERROR_LINES);
            self.messages.get(start..).unwrap_or_default().join("\n")
        } else {
            self.errors.join("\n")
        }
    }
}

/// The part of a cargo JSON message the build reads: a compiler
/// diagnostic, present when the reason is `compiler-message`.
#[derive(Deserialize)]
struct CargoMessage {
    message: Option<Diagnostic>,
}

#[derive(Deserialize)]
struct Diagnostic {
    level: String,
    rendered: Option<String>,
}

/// Sends each line of the build's stdout.
fn read_stdout(
    pipe: Option<impl Read + Send + 'static>,
    sender: Sender<Output>,
) -> std::io::Result<()> {
    let Some(pipe) = pipe else {
        return Ok(());
    };
    std::thread::Builder::new()
        .name("rig-code-build-stdout".to_owned())
        .spawn(move || {
            for line in BufReader::new(pipe).lines() {
                let Ok(line) = line else { break };
                if sender.send_blocking(Output::Json(line)).is_err() {
                    break;
                }
            }
        })
        .map(drop)
}

/// Sends each segment of the build's stderr between line feeds or carriage
/// returns.
fn read_stderr(
    pipe: Option<impl Read + Send + 'static>,
    sender: Sender<Output>,
) -> std::io::Result<()> {
    let Some(pipe) = pipe else {
        return Ok(());
    };
    std::thread::Builder::new()
        .name("rig-code-build-stderr".to_owned())
        .spawn(move || {
            let mut segment = Vec::new();
            let send = |segment: &mut Vec<u8>| {
                let text = String::from_utf8_lossy(segment).trim_end().to_owned();
                segment.clear();
                text.is_empty() || sender.send_blocking(Output::Text(text)).is_ok()
            };
            for byte in BufReader::new(pipe).bytes() {
                let Ok(byte) = byte else { break };
                if byte != b'\n' && byte != b'\r' {
                    segment.push(byte);
                } else if !send(&mut segment) {
                    return;
                }
            }
            send(&mut segment);
        })
        .map(drop)
}

/// Reads cargo's progress bar, `Building [====>   ] 37/41: rig-core, …`.
fn parse_progress(text: &str) -> Option<Progress> {
    let rest = text.trim_start().strip_prefix("Building [")?;
    let (_, rest) = rest.split_once("] ")?;
    let (counts, building) = rest.split_once(':').unwrap_or((rest, ""));
    let (done, total) = counts.trim().split_once('/')?;
    Some(Progress {
        done: done.parse().ok()?,
        total: total.parse().ok()?,
        building: building.trim().to_owned(),
    })
}

/// `/reload` starts a build, when every agent is idle, no build is running
/// and the agent was started by the launcher.
fn reload(
    input: In<CommandInput>,
    statuses: Query<&Status>,
    jobs: Query<(), With<BuildJob>>,
    mut commands: Commands,
) {
    let agent = input.agent;
    if statuses.iter().any(|status| *status != Status::Idle) {
        commands.trigger(Notice::error(
            agent,
            "A turn is running. Press Esc to stop it, then /reload.",
        ));
        return;
    }
    if !jobs.is_empty() {
        commands.trigger(Notice::error(agent, "A build is already running."));
        return;
    }
    let Some(launcher) = std::env::var_os("RIG_LAUNCHER") else {
        commands.trigger(Notice::error(
            agent,
            "Start the agent with `rig` to use /reload.",
        ));
        return;
    };
    match BuildJob::start(agent, launcher) {
        Ok(job) => {
            commands.spawn(job);
            commands.trigger(Notice::info(agent, "Building the agent…"));
        }
        Err(error) => commands.trigger(Notice::error(
            agent,
            format!("Cannot start the build: {error}"),
        )),
    }
}

/// Collects the build's output, and reports how it ended. A failed build is
/// dropped and the running agent carries on.
fn poll_build(mut jobs: Query<(Entity, &mut BuildJob)>, mut commands: Commands) {
    for (entity, mut job) in &mut jobs {
        if job.built {
            continue;
        }
        let mut progressed = false;
        let job_ref = job.bypass_change_detection();
        let closed = loop {
            match job_ref.output.try_recv() {
                Ok(output) => progressed |= job_ref.take(output),
                Err(TryRecvError::Empty) => break false,
                Err(TryRecvError::Closed) => break true,
            }
        };
        if progressed {
            job.set_changed();
        }
        if !closed {
            continue;
        }
        let status = match job.child.try_wait() {
            Ok(None) => continue,
            Ok(Some(status)) => status.success(),
            Err(_) => false,
        };
        job.child.stop();
        if status {
            job.built = true;
            commands.trigger(Notice::info(
                job.agent,
                "Build finished. Reloading once every agent is idle.",
            ));
            continue;
        }
        let failure = job.failure();
        error!("the /reload build failed:\n{failure}");
        let shown: Vec<&str> = failure.lines().take(ERROR_LINES).collect();
        let mut text = format!(
            "The build failed; this agent keeps running.\n{}",
            shown.join("\n")
        );
        if failure.lines().count() > ERROR_LINES {
            text.push_str("\n… more in agent.log");
        }
        commands.trigger(Notice::error(job.agent, text));
        commands.entity(entity).despawn();
    }
}

/// Exits for the reload once a build succeeded and every agent is idle,
/// naming the session the new binary restores. The exit saves state.
fn finish_reload(
    jobs: Query<(Entity, &BuildJob)>,
    statuses: Query<&Status>,
    dirs: Res<Dirs>,
    session: Res<SessionDir>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
) {
    let Some((entity, job)) = jobs.iter().find(|(_, job)| job.built) else {
        return;
    };
    if statuses.iter().any(|status| *status != Status::Idle) {
        return;
    }
    match fs::write(&dirs.resume, &session.id) {
        Ok(()) => {
            exit.write(AppExit::from_code(RELOAD_EXIT_CODE));
        }
        Err(error) => {
            commands.trigger(Notice::error(
                job.agent,
                format!("Cannot reload: writing the resume file failed: {error}"),
            ));
            commands.entity(entity).despawn();
        }
    }
}

/// Shows the launcher's message, such as a rollback, to the first agent.
fn launcher_notice(agents: Query<Entity, With<Agent>>, mut commands: Commands) {
    let Some(text) = std::env::var("RIG_NOTICE")
        .ok()
        .filter(|text| !text.is_empty())
    else {
        return;
    };
    if let Some(agent) = agents.iter().next() {
        commands.trigger(Notice::error(agent, text));
    }
}

/// After the first frame: tells the launcher this binary works, and forgets
/// the resume file, so the next fresh start begins a new session.
fn ready(dirs: Res<Dirs>) -> Result {
    if let Some(path) = std::env::var_os("RIG_READY_FILE") {
        fs::write(path, b"")?;
    }
    match fs::remove_file(&dirs.resume) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error.into()),
        _ => Ok(()),
    }
}
