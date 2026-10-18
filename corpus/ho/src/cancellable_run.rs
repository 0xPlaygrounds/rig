//! Long commands with live output and per-call cancel (spec
//! `H09-cancellable-run.md`): the `run` tool, `/cancel-call` and `ctrl+k`.
//!
//! `run` is an open tool. Each call starts `sh -c` through the shared
//! launcher ([`Processes`]) as a process group of its own. A thread hands
//! the command's output lines to a [`Running::stream`] on the call's job
//! entity, and each line goes into the call's [`RunOutput`], which keeps
//! the last 4 KB. From it the plugin keeps the call's [`ToolProgress`]
//! (all of it, at most 10 times a second, which every view shows)
//! and its [`IfStopped`]: the output so far and `[interrupted]`, which the
//! kernel answers with when the turn is interrupted or stopped.
//!
//! A cancel (`/cancel-call`, `ctrl+k`) or the timeout sends the group
//! SIGTERM, then SIGKILL three seconds later (a [`WakeAt`] on the job).
//! Once the command ended the call is answered with its last 4 KB of output
//! and how it ended, so a cancel lets the turn carry on with the model
//! deciding. The job is [`JobOf`] its call and ends when the call settles,
//! which kills a command still running.

use std::cmp::Reverse;
use std::thread;
use std::time::Instant;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc::unbounded;
use rig_harness::prelude::*;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::{
    AppKeyBindingsExt, AppToolRenderersExt, CallState, KeyBinding, KeyPress, ToolCallView,
    run_command, text_rows,
};
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The tool's name.
pub const RUN: &str = "run";
/// The command that cancels a call.
pub const CANCEL: &str = "cancel-call";
/// The end of the output a result keeps.
const KEPT_BYTES: usize = 4 * 1024;
/// How long a stopped command gets between SIGTERM and SIGKILL.
const GRACE: Duration = Duration::from_secs(3);
/// Seconds a command runs unless the call says otherwise.
const DEFAULT_TIMEOUT: u64 = 30 * 60;
/// How often the terminal view's progress changes at most, and the loop's
/// pace while a command runs.
const REDRAW: Duration = Duration::from_millis(100);

/// The `run` tool, `/cancel-call` and `ctrl+k`.
#[derive(Default)]
pub struct CancellableRunPlugin;

impl Plugin for CancellableRunPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Processes>()
            .add_open_tool(
                RUN,
                "Run a shell command with `sh -c` in the working directory, stdin empty. \
                 The user watches its output live and may cancel it. Returns the last 4 KB of \
                 stdout and stderr and how it ended: its exit code, `[timed out after N s]` \
                 after timeout_s (default 1800), or `[cancelled by user after N s]`. \
                 Use it for builds, test suites and other commands that take long. `run` \
                 calls of one reply run at the same time; put commands that depend on each \
                 other in one call. When the user cancels one, decide from its output so far \
                 whether to go on.",
                ToolOptions {
                    footprint: Some(Touches::Independent),
                },
                start,
            )
            .add_command(
                CANCEL,
                "Cancel the agent's running `run` command, or pick one: /cancel-call [id]",
                cancel_call,
            )
            .add_observer(take_event)
            .add_observer(on_due)
            .add_observer(end_job)
            // Every view gets the output so far, at most 10 times a second.
            .add_systems(
                PostUpdate,
                show_progress(RunOutput::all).run_if(on_real_timer(REDRAW)),
            );
    }
}

/// The view half: the terminal view's look of a `run` call and `ctrl+k`,
/// named in `plugins.toml` as the entry's `view`.
#[derive(Default)]
pub struct CancellableRunView;

impl Plugin for CancellableRunView {
    fn build(&self, app: &mut App) {
        app.add_tool_renderer(RUN, look).add_key_binding(
            KeyBinding::new(KeyPress::ctrl('k'), "Cancel a running command"),
            run_command(CANCEL),
        );
    }
}

/// The arguments of `run`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RunArgs {
    /// The command line.
    command: String,
    /// Seconds before the command is stopped (default 1800).
    timeout_s: Option<u64>,
}

/// A `run` call's command and its output so far, on the call while it
/// runs. Not saved.
#[derive(Component, Debug)]
pub struct RunOutput {
    /// The command line.
    pub command: String,
    /// The process group's leader.
    pub pid: u32,
    /// When it started.
    pub started: Instant,
    /// The last 4 KB of its output, line by line.
    pub text: String,
    /// Whether earlier output was dropped from `text`.
    pub cut: bool,
    /// The seconds it may run.
    timeout: u64,
    /// Why it is being stopped, once it is.
    stop: Option<Stop>,
}

impl RunOutput {
    /// Adds a line of output.
    fn push(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
        let mut excess = self.text.len().saturating_sub(KEPT_BYTES);
        if excess > 0 {
            while !self.text.is_char_boundary(excess) {
                excess += 1;
            }
            self.text.drain(..excess);
            self.cut = true;
        }
    }

    /// The output so far, then `ending`.
    fn result(&self, ending: &str) -> Outcome {
        let cut = if self.cut {
            "[earlier output cut]\n"
        } else {
            ""
        };
        Outcome::from(ToolResult::success(
            format!("{cut}{}{ending}", self.text).into(),
        ))
    }

    /// All of it, without the last line's newline.
    fn all(&self) -> String {
        self.text.trim_end_matches('\n').to_owned()
    }
}

/// Why a command is being stopped.
#[derive(Clone, Copy, Debug)]
enum Stop {
    /// The user cancelled it after this long.
    Cancelled(Duration),
    /// It ran past its timeout.
    TimedOut,
}

/// A `run` call's command: its output stream and its process group.
#[derive(Component)]
#[relationship(relationship_target = Job)]
pub struct JobOf(pub Entity);

/// The command a `run` call runs, despawned with the call.
#[derive(Component)]
#[relationship_target(relationship = JobOf, linked_spawn)]
pub struct Job(Entity);

/// A signal to a job's process group.
#[derive(Clone, Copy)]
enum Signal {
    Terminate,
    Kill,
}

/// Signals a job's process group, which is killed when the job is dropped,
/// as when its call settled. It acts only while the command is not reaped.
#[derive(Component)]
struct Group(Box<dyn Fn(Signal) + Send + Sync>);

impl Drop for Group {
    fn drop(&mut self) {
        (self.0)(Signal::Kill);
    }
}

/// What a job's thread reports.
pub enum RunEvent {
    /// A line of output.
    Line(String),
    /// The command ended: its exit code, `None` for a signal, or why
    /// waiting for it failed.
    Ended(Result<Option<i32>, String>),
}

fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Starts a `run` call's command and its job.
fn start(
    called: On<ToolCalled<RunArgs>>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let call = called.call;
    let RunArgs { command, timeout_s } = &called.args;
    let timeout = timeout_s.unwrap_or(DEFAULT_TIMEOUT).max(1);
    let spawned = processes
        .launcher()
        .spawn("sh", ["-c", command.as_str()], Default::default());
    let mut process = match spawned {
        Ok(process) => process,
        Err(error) => {
            let failed = ToolExecutionError::other(format!("could not run sh: {error}"));
            commands
                .entity(call)
                .insert_if_new(Outcome::from(ToolResult::failed(failed)));
            return;
        }
    };
    let pid = process.id();
    let group = process.group();
    let (events, received) = unbounded();
    thread::spawn(move || {
        // Until every process holding the output closed it.
        for line in process.lines().iter() {
            let line = line.text().trim_end_matches(['\n', '\r']).to_owned();
            events.unbounded_send(RunEvent::Line(line)).ok();
        }
        let ended = loop {
            match process.wait_for(GRACE) {
                Ok(Some(status)) => break Ok(status.code()),
                Ok(None) => {}
                Err(error) => break Err(error.to_string()),
            }
        };
        // What the command left running in its group goes with it.
        process.kill().ok();
        events.unbounded_send(RunEvent::Ended(ended)).ok();
    });
    let output = RunOutput {
        command: command.clone(),
        pid,
        started: Instant::now(),
        text: String::new(),
        cut: false,
        timeout,
        stop: None,
    };
    let if_stopped = IfStopped(output.result("[interrupted]"));
    commands
        .entity(call)
        .insert((output, ToolProgress::default(), if_stopped));
    commands.spawn((
        Name::new(format!("run job {pid}")),
        JobOf(call),
        Group(Box::new(move |signal| match signal {
            Signal::Terminate => group.terminate(),
            Signal::Kill => group.kill(),
        })),
        WakeAt::after(Duration::from_secs(timeout)),
        KeepAwake(REDRAW),
        Running::stream(pool(), &wake, received),
    ));
}

/// Takes each line into its call's output, and answers the call once its
/// command ended.
fn take_event(
    event: On<Insert<Done<RunEvent>>>,
    jobs: Query<(&Done<RunEvent>, &JobOf)>,
    mut calls: Query<(&mut RunOutput, &mut IfStopped)>,
    mut commands: Commands,
) {
    let Ok((Done(event), &JobOf(call))) = jobs.get(event.entity) else {
        return;
    };
    let Ok((mut output, mut if_stopped)) = calls.get_mut(call) else {
        return;
    };
    match event {
        RunEvent::Line(line) => {
            output.push(line);
            if_stopped.0 = output.result("[interrupted]");
        }
        RunEvent::Ended(ended) => {
            let ending = match (output.stop, ended) {
                (Some(Stop::Cancelled(after)), _) => {
                    format!("[cancelled by user after {:.1} s]", after.as_secs_f64())
                }
                (Some(Stop::TimedOut), _) => format!("[timed out after {} s]", output.timeout),
                (None, Ok(Some(code))) => format!("[exit code {code}]"),
                (None, Ok(None)) => "[killed by a signal]".to_owned(),
                (None, Err(why)) => format!("[waiting for the command failed: {why}]"),
            };
            commands
                .entity(call)
                .try_insert_if_new(output.result(&ending));
        }
    }
}

/// Starts stopping a command: SIGTERM now, SIGKILL after [`GRACE`].
fn stop(output: &mut RunOutput, why: Stop, job: Entity, group: &Group, commands: &mut Commands) {
    output.stop = Some(why);
    (group.0)(Signal::Terminate);
    commands.entity(job).try_insert(WakeAt::after(GRACE));
}

/// A job's time is up: past its timeout it is stopped, past the grace of
/// a stop its group is killed.
fn on_due(
    due: On<Add<Due>>,
    jobs: Query<(&JobOf, &Group)>,
    mut calls: Query<&mut RunOutput>,
    mut commands: Commands,
) {
    let Ok((&JobOf(call), group)) = jobs.get(due.entity) else {
        return;
    };
    let Ok(mut output) = calls.get_mut(call) else {
        return;
    };
    if output.stop.is_some() {
        (group.0)(Signal::Kill);
    } else {
        stop(
            &mut output,
            Stop::TimedOut,
            due.entity,
            group,
            &mut commands,
        );
    }
}

/// Ends a call's job once the call settled: at once when something else
/// answered it while its command ran, such as an interrupt or a stopped
/// turn.
fn end_job(settled: On<Add<Settled>>, calls: Query<&Job>, mut commands: Commands) {
    if let Ok(&Job(job)) = calls.get(settled.entity) {
        commands.entity(job).try_despawn();
    }
}

/// Keeps each running call's [`ToolProgress`] at what `shown` shows of
/// its output.
fn show_progress(
    shown: fn(&RunOutput) -> String,
) -> impl FnMut(Query<(&RunOutput, &mut ToolProgress), Changed<RunOutput>>) {
    move |mut calls| {
        for (output, mut progress) in &mut calls {
            progress.set_if_neq(ToolProgress(shown(output)));
        }
    }
}

/// The terminal view's look of a `run` call: the command, then its output
/// while it runs, or its result.
fn look(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let command = view.argument("command").unwrap_or_default();
    let mut lines = vec![view.header(RUN, command)];
    match &view.state {
        CallState::Running(Some(progress)) if view.result.is_none() => {
            lines.extend(text_rows(progress, Style::new().dim()));
            lines.push(Line::from("    ctrl+k cancels").yellow());
        }
        _ => lines.extend(view.result_lines()),
    }
    lines
}

/// `/cancel-call [id]`: cancels the agent's newest running `run` call, the
/// one named, or offers a picker of them when there are several.
fn cancel_call(
    In(args): In<CommandArgs>,
    mut calls: Query<(Entity, &ToolRequest, &ForAgent, &mut RunOutput, &Job), Without<Outcome>>,
    groups: Query<&Group>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut running: Vec<_> = calls
        .iter()
        .filter(|&(_, _, &ForAgent(agent), output, _)| agent == args.agent && output.stop.is_none())
        .collect();
    running.sort_by_key(|(.., output, _)| Reverse(output.started));
    let id = args.args.trim();
    let chosen = match (id, running.as_slice()) {
        ("", []) => {
            notices.write(Notice::info(args.agent, "No `run` command is running."));
            return;
        }
        ("", [one]) => Some(one),
        ("", several) => {
            let picker = several.iter().fold(
                Picker::new("Cancel which command?").for_agent(args.agent),
                |picker, (_, request, _, output, _)| {
                    let secs = output.started.elapsed().as_secs();
                    let label = format!("{} ({secs} s)", output.command);
                    picker.item(label, format!("{CANCEL} {}", request.call.id))
                },
            );
            commands.spawn(picker);
            return;
        }
        (id, _) => running
            .iter()
            .find(|(_, request, ..)| request.call.id.to_string() == id),
    };
    let Some(&(call, .., &Job(job))) = chosen else {
        notices.write(Notice::error(
            args.agent,
            format!("No running `run` call `{id}`."),
        ));
        return;
    };
    if let (Ok((.., mut output, _)), Ok(group)) = (calls.get_mut(call), groups.get(job)) {
        let after = output.started.elapsed();
        stop(
            &mut output,
            Stop::Cancelled(after),
            job,
            group,
            &mut commands,
        );
    }
}
