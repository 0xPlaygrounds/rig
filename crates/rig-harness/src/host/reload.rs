//! `/reload`: rebuild the agent through the `rig` launcher and restart on
//! the new build. The build runs as a child process whose stderr a std
//! thread forwards line by line. Until cargo's own `done/total` counter
//! appears, the build is resolving dependencies and its latest line is the
//! progress. Quitting during the build kills it with cargo and rustc.
//!
//! `/reload` typed while a turn runs is queued ([`ReloadQueued`]): the
//! build starts once no turn runs, and `/reload cancel` drops it. The
//! model can ask for the same with the `reload` tool, offered when the
//! launcher started the agent, so it applies its own plugin changes; the
//! user sees a notice and can cancel. A turn started while the build runs
//! delays the restart until it ends. A build that fails leaves this build
//! running, and the launcher rolls back a new build that fails to start.
//!
//! A failed build's first errors go into the conversation of the agent that
//! asked, as a note for its model ([`super::launcher::build_failure_note`]);
//! `rig build` keeps the whole output in `RIG_HOME/build.log`.

use std::collections::VecDeque;
use std::io::{BufReader, Read};
use std::process::{Child, ChildStderr, Command, Stdio};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use crossbeam_channel::{Receiver, Sender, TryRecvError};

use rig::harness_protocol::{Home, RELOAD_EXIT_CODE, first_errors};
use rig_tools::process::{detach, kill_group};
use schemars::JsonSchema;
use serde::Deserialize;

use super::launcher;
use rig_core::tool::{ToolExecutionError, ToolResult};
use rig_ecs::agent::{Notice, TurnOf};
use rig_ecs::calls::Wake;
use rig_ecs::commands::{AppCommandsExt, CommandArgs};
use rig_ecs::tools::{AppToolsExt, Footprint, ToolCalled, ToolOptions, ToolOutput};
use rig_ecs::turn::PollCalls;

/// The tool that asks for a reload once the turn ends.
pub const RELOAD_TOOL: &str = "reload";

const RELOAD_DESCRIPTION: &str = "Rebuild the agent with the plugins in plugins.toml and \
    restart on the new build, to apply your own plugin changes. Nothing happens mid-turn: the \
    build starts once this turn and every other running turn have ended, so finish your \
    edits first and end your turn after this call. The user sees a notice and can cancel it. \
    After the restart the session, with this conversation, carries on. If the build fails, \
    this build keeps running and its first errors arrive in your conversation as a note.";

/// Lines of a failed build shown, from its first error.
const ERROR_LINES: usize = 60;
/// Lines of a failed build in the note to the model, from its first error.
const NOTE_LINES: usize = 40;
/// Lines of build output kept while it runs.
const KEPT_LINES: usize = 2000;

/// `/reload`, the rebuild in flight and the restart.
pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        // Only an agent the launcher started can rebuild itself, so only
        // its model is offered the tool.
        if launcher::executable().is_some() {
            app.add_open_tool(
                RELOAD_TOOL,
                RELOAD_DESCRIPTION,
                ToolOptions {
                    rules: &[],
                    // Never run again after the restart it causes.
                    footprint: Footprint::Independent,
                },
                on_reload_tool,
            );
        }
        app.add_command(
            "reload",
            "Rebuild with the plugins in plugins.toml and restart once no turn runs; \
             `/reload cancel` cancels",
            reload,
        )
        .add_message::<ReloadFailed>()
        .add_observer(on_cancel_reload)
        .add_systems(
            Update,
            (start_queued_reload, drain_reload, finish_reload)
                .chain()
                .after(PollCalls),
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
    /// The agent that asked, whose model is told when the build fails.
    agent: Entity,
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

    fn start(launcher: &std::ffi::OsStr, agent: Entity, wake: Wake) -> std::io::Result<Self> {
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
            agent,
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

/// A reload asked for while a turn ran, by `/reload` or the `reload`
/// tool: its build starts once no turn runs. `/reload cancel` or
/// [`CancelReload`] drops it.
#[derive(Resource, Clone, Copy, Debug)]
pub struct ReloadQueued {
    agent: Entity,
}

impl ReloadQueued {
    /// The agent that asked, whose model is told when the build fails.
    pub fn agent(&self) -> Entity {
        self.agent
    }
}

/// A `/reload` build failed. Views show its output until the user
/// dismisses it; it is logged too.
#[derive(Message, Clone, Debug)]
pub struct ReloadFailed {
    /// The build's output from its first error on.
    pub output: String,
}

/// Stops the running rebuild, or drops the queued one, if any.
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

/// What asking for a reload did.
enum Asked {
    /// The build started.
    Started,
    /// The build waits for the running turns.
    Queued,
    /// Nothing changed, for the reason given.
    Refused(String),
}

/// Starts the rebuild for `agent`, or queues it while a turn runs.
fn ask_reload(
    agent: Entity,
    turns_run: bool,
    build: Option<&ReloadBuild>,
    queued: Option<&ReloadQueued>,
    wake: &Wake,
    commands: &mut Commands,
) -> Asked {
    if build.is_some() {
        return Asked::Refused("A rebuild is already running; Esc cancels it.".to_owned());
    }
    if queued.is_some() {
        return Asked::Refused(
            "A reload is already queued for when no turn runs; /reload cancel cancels it."
                .to_owned(),
        );
    }
    let Some(launcher) = launcher::executable() else {
        return Asked::Refused(
            "/reload needs the rig launcher: start the agent with `rig`.".to_owned(),
        );
    };
    if turns_run {
        commands.insert_resource(ReloadQueued { agent });
        return Asked::Queued;
    }
    match ReloadBuild::start(&launcher, agent, wake.clone()) {
        Ok(build) => {
            commands.insert_resource(build);
            Asked::Started
        }
        Err(failure) => Asked::Refused(format!("Could not start the rebuild: {failure}")),
    }
}

fn reload(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    build: Option<Res<ReloadBuild>>,
    queued: Option<Res<ReloadQueued>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if args.args == "cancel" {
        commands.trigger(CancelReload);
        return;
    }
    let notice = match ask_reload(
        args.agent,
        !turns.is_empty(),
        build.as_deref(),
        queued.as_deref(),
        &wake,
        &mut commands,
    ) {
        Asked::Started => "Rebuilding the agent…".to_owned(),
        Asked::Queued => "A turn is running: the agent rebuilds and restarts once no turn \
                          runs. /reload cancel cancels it."
            .to_owned(),
        Asked::Refused(why) => why,
    };
    notices.write(Notice::info(None, notice));
}

/// The arguments of a `reload` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReloadArgs {
    /// What the reload applies, in a few words, shown to the user.
    reason: Option<String>,
}

/// The `reload` tool: asks for a reload once no turn runs, with a notice
/// naming the agent and its reason, and answers whether it is queued.
fn on_reload_tool(
    called: On<ToolCalled<ReloadArgs>>,
    build: Option<Res<ReloadBuild>>,
    queued: Option<Res<ReloadQueued>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let (call, agent) = (called.call, called.agent);
    let reason = called
        .args
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
        .map(|reason| format!(" ({reason})"))
        .unwrap_or_default();
    // The caller's own turn runs: the reload always waits for it.
    let output = match ask_reload(
        agent,
        true,
        build.as_deref(),
        queued.as_deref(),
        &wake,
        &mut commands,
    ) {
        Asked::Started | Asked::Queued => {
            notices.write(Notice::info(
                None,
                format!(
                    "The model of agent {} asked to reload{reason}: the agent rebuilds and \
                     restarts once no turn runs. /reload cancel cancels it.",
                    called.caller.short()
                ),
            ));
            ToolResult::success(
                "Reload queued: the build starts once your turn and every other running \
                 turn have ended. End your turn now with a short summary for the user; the \
                 conversation carries on after the restart. A failed build keeps this build \
                 running and its errors arrive as a note."
                    .into(),
            )
        }
        Asked::Refused(why) => ToolResult::failed(ToolExecutionError::other(format!(
            "{why} Nothing was queued."
        ))),
    };
    commands.entity(call).insert_if_new(ToolOutput(output));
}

/// Starts the queued reload's build once no turn runs.
fn start_queued_reload(
    queued: Option<Res<ReloadQueued>>,
    build: Option<Res<ReloadBuild>>,
    turns: Query<(), With<TurnOf>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(queued) = queued else {
        return;
    };
    if !turns.is_empty() || build.is_some() {
        return;
    }
    commands.remove_resource::<ReloadQueued>();
    let Some(launcher) = launcher::executable() else {
        return;
    };
    let notice = match ReloadBuild::start(&launcher, queued.agent, wake.clone()) {
        Ok(build) => {
            commands.insert_resource(build);
            "No turn runs: rebuilding the agent…".to_owned()
        }
        Err(failure) => format!("Could not start the rebuild: {failure}"),
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
        let output = build.errors(ERROR_LINES);
        // The notice is the one log line: the whole output is in the
        // build log it names.
        let first = first_errors(build.output.iter().map(String::as_str), 1)
            .first()
            .map(|line| format!(": {}", line.trim()))
            .unwrap_or_default();
        notices.write(Notice::error(
            None,
            format!(
                "The rebuild failed ({status}){first}; this build keeps running. Whole output: \
                 {}",
                Home::from_env().build_log().display()
            ),
        ));
        launcher::note_build_failure(
            &mut commands,
            build.agent,
            launcher::build_failure_note(
                "/reload",
                &format!(
                    "`rig build` exited with {status}. Its output from the first error:\n{}",
                    build.errors(NOTE_LINES)
                ),
            ),
        );
        failures.write(ReloadFailed { output });
        commands.remove_resource::<ReloadBuild>();
    }
}

/// Exits with [`RELOAD_EXIT_CODE`] once the build is ready and no turn
/// runs, at most once.
fn finish_reload(
    build: Option<Res<ReloadBuild>>,
    turns: Query<(), With<TurnOf>>,
    mut exiting: Local<bool>,
    mut exit: MessageWriter<AppExit>,
) {
    if !*exiting && build.is_some_and(|build| build.ready) && turns.is_empty() {
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
    queued: Option<Res<ReloadQueued>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if queued.is_some() {
        commands.remove_resource::<ReloadQueued>();
        notices.write(Notice::info(None, "Queued reload cancelled.".to_owned()));
    } else if build.is_some_and(|build| !build.ready) {
        commands.remove_resource::<ReloadBuild>();
        notices.write(Notice::info(None, "Rebuild cancelled.".to_owned()));
    } else {
        notices.write(Notice::info(None, "No reload to cancel.".to_owned()));
    }
}
