//! Long commands with live output and per-call cancel (spec
//! `H09-cancellable-run.md`): the `run` tool, `/cancel-call` and `ctrl+k`.
//!
//! `run` is an open tool. Each call starts `sh -c` as every child process
//! starts ([`Processes`]), in a process group of its own, and a thread
//! supervises it: it splits the output into lines, sends them through a
//! channel that a [`Running::stream`] on the call's job entity turns into
//! [`Done<RunEvent>`]s, and stops the group with SIGTERM, then SIGKILL after
//! three seconds, when the call is cancelled or times out. Each line goes
//! into the call's [`RunProgress`], which the terminal view's panel shows
//! and a `--print` run writes to stderr; the progress is not saved, only the
//! call's result is. Once the command ended, the call is answered with the
//! last 4 KB of its output and how it ended, so a cancel lets the turn carry
//! on with the model deciding what next. The job is [`JobOf`] its call and
//! ends when the call settles, which kills a command still running: an
//! interrupt answers the call itself, as "interrupted by the user", without
//! the output so far, since the kernel gives a plugin no turn there.

use std::cmp::Reverse;
use std::collections::VecDeque;
use std::io::Read as _;
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
use std::thread;
use std::time::Instant;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc::{UnboundedSender, unbounded};
use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Stylize;
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    AppKeyBindingsExt, Focused, KeyBinding, KeyPress, PanelCanvas, Placement, RequestRedraw,
    TuiPanel, TuiSystems, run_command,
};
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The tool's name.
pub const RUN: &str = "run";
/// The command that cancels a call.
pub const CANCEL: &str = "cancel-call";
/// Lines of a running call the fronts show.
pub const SHOWN_LINES: usize = 5;
/// The end of the output a result keeps.
const KEPT_BYTES: usize = 4 * 1024;
/// How long a stopped command gets between SIGTERM and SIGKILL.
const GRACE: Duration = Duration::from_secs(3);
/// Seconds a command runs unless the call says otherwise.
const DEFAULT_TIMEOUT: u64 = 30 * 60;
/// How often the panel is redrawn at most, and the loop's pace while a
/// command runs.
const REDRAW: Duration = Duration::from_millis(100);
/// How often the supervisor checks on the command while no output comes.
const POLL: Duration = Duration::from_millis(20);
/// How long output may still come once the command exited.
const PIPE_GRACE: Duration = Duration::from_millis(100);

/// The `run` tool, `/cancel-call`, `ctrl+k`, the output panel and the
/// `--print` progress lines.
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
                 after timeout_s (default 1800), or `[cancelled by user after N s]`.",
                ToolOptions {
                    rules: &[
                        "Use `run` for builds, test suites and other commands that take long. \
                         `run` calls of one reply run at the same time; put commands that \
                         depend on each other in one call. When the user cancels one, decide \
                         from its output so far whether to go on.",
                    ],
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
            .add_observer(end_job);
        app.add_key_binding(
            KeyBinding::new(KeyPress::ctrl('k'), "Cancel a running command"),
            run_command(CANCEL),
        );
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(rig_print::FRONT) {
            app.insert_resource(PrintProgress);
        }
        if app.runs_front(rig_tui::FRONT) {
            app.add_systems(Startup, spawn_panel).add_systems(
                PostUpdate,
                (
                    size_panel.in_set(TuiSystems::Prepare),
                    redraw.run_if(on_real_timer(REDRAW)),
                    draw_panel.in_set(TuiSystems::Draw),
                ),
            );
        }
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

/// A `run` call's output so far, on the call while it runs. Not saved.
#[derive(Component, Debug)]
pub struct RunProgress {
    /// The command line.
    pub command: String,
    /// The process group's leader.
    pub pid: u32,
    /// When it started.
    pub started: Instant,
    /// Its last [`SHOWN_LINES`] lines.
    pub lines: VecDeque<String>,
    /// The last 4 KB of its output.
    pub output: String,
    /// Whether earlier output was dropped from `output`.
    pub cut: bool,
    /// The seconds it may run.
    timeout: u64,
}

impl RunProgress {
    /// Adds a line of output.
    fn push(&mut self, line: &str) {
        if self.lines.len() == SHOWN_LINES {
            self.lines.pop_front();
        }
        self.lines.push_back(line.to_owned());
        self.output.push_str(line);
        self.output.push('\n');
        let mut excess = self.output.len().saturating_sub(KEPT_BYTES);
        if excess > 0 {
            while !self.output.is_char_boundary(excess) {
                excess += 1;
            }
            self.output.drain(..excess);
            self.cut = true;
        }
    }
}

/// On a `run` call the user cancelled: how long it had run.
#[derive(Component, Debug)]
struct Cancelled(Duration);

/// A `run` call's command: its output stream and its process group.
#[derive(Component)]
#[relationship(relationship_target = Job)]
pub struct JobOf(pub Entity);

/// The command a `run` call runs, despawned with the call.
#[derive(Component)]
#[relationship_target(relationship = JobOf, linked_spawn)]
pub struct Job(Entity);

/// What a job and its supervisor share.
#[derive(Default)]
struct Control {
    /// The leader's pid while it is not reaped, else 0.
    leader: AtomicU32,
    /// Set when the user cancels the call.
    stop: AtomicBool,
}

/// A job's process group, killed when the job is dropped, as an interrupt
/// does by despawning its call.
#[derive(Component)]
struct Process(Arc<Control>);

impl Drop for Process {
    fn drop(&mut self) {
        let leader = self.0.leader.swap(0, Ordering::SeqCst);
        if leader != 0 {
            signal(leader, libc::SIGKILL);
        }
    }
}

/// What a job's supervisor reports.
pub enum RunEvent {
    /// A line of output.
    Line(String),
    /// The command ended.
    Ended(End),
}

/// How a command ended.
pub enum End {
    /// It exited with this code, or by a signal when `None`.
    Exited(Option<i32>),
    /// It ran past its timeout and was stopped.
    TimedOut,
    /// Waiting for it failed.
    Failed(String),
}

/// A `--print` run: progress lines go to stderr.
#[derive(Resource)]
struct PrintProgress;

fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

fn answer(text: String) -> Outcome {
    Outcome::from(ToolResult::success(text.into()))
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
    let (child, output) = match launch(&processes, command) {
        Ok(started) => started,
        Err(error) => {
            let failed = ToolResult::failed(ToolExecutionError::other(format!(
                "could not run sh: {error}"
            )));
            commands.entity(call).insert_if_new(Outcome::from(failed));
            return;
        }
    };
    let pid = child.id();
    let control = Arc::new(Control::default());
    control.leader.store(pid, Ordering::SeqCst);
    let (events, received) = unbounded();
    let supervised = Arc::clone(&control);
    let limit = Duration::from_secs(timeout);
    thread::spawn(move || supervise(child, &output, &supervised, limit, &events));
    commands.entity(call).insert(RunProgress {
        command: command.clone(),
        pid,
        started: Instant::now(),
        lines: VecDeque::new(),
        output: String::new(),
        cut: false,
        timeout,
    });
    commands.spawn((
        Name::new(format!("run job {pid}")),
        JobOf(call),
        Process(control),
        KeepAwake(REDRAW),
        Running::stream(pool(), &wake, received),
    ));
}

/// Starts `command` with `sh -c`, stdout and stderr on one pipe so their
/// lines keep their order.
fn launch(processes: &Processes, command: &str) -> std::io::Result<(Child, std::io::PipeReader)> {
    let (reader, writer) = std::io::pipe()?;
    let errors = writer.try_clone()?;
    let mut sh = processes.launcher().command("sh");
    sh.arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(writer)
        .stderr(errors);
    let child = sh.spawn()?;
    // The parent's write ends must close, or reading never ends.
    drop(sh);
    Ok((child, reader))
}

/// Sends `signal` to the process group `leader` leads.
fn signal(leader: u32, signal: libc::c_int) {
    if let Ok(group) = i32::try_from(leader) {
        // SAFETY: `kill` takes plain integers; a negative pid names the
        // process group the command leads (the launcher starts it in a
        // session of its own).
        unsafe {
            libc::kill(-group, signal);
        }
    }
}

/// Watches the command until it ends: sends its output line by line, stops
/// it with SIGTERM, then SIGKILL after [`GRACE`], once cancelled or past
/// `timeout`, and reports how it ended.
fn supervise(
    mut child: Child,
    output: &std::io::PipeReader,
    control: &Control,
    timeout: Duration,
    events: &UnboundedSender<RunEvent>,
) {
    let pid = child.id();
    let chunks = drain(output);
    let mut lines = Lines::default();
    let started = Instant::now();
    let mut stopping: Option<(Instant, bool)> = None;
    let exited = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status.code()),
            Err(error) => break Err(error.to_string()),
            Ok(None) => {}
        }
        match stopping {
            None if control.stop.load(Ordering::Relaxed) => {
                signal(pid, libc::SIGTERM);
                stopping = Some((Instant::now(), false));
            }
            None if started.elapsed() >= timeout => {
                signal(pid, libc::SIGTERM);
                stopping = Some((Instant::now(), true));
            }
            Some((since, _)) if since.elapsed() >= GRACE => signal(pid, libc::SIGKILL),
            _ => {}
        }
        match chunks.recv_timeout(POLL) {
            Ok(chunk) => lines.push(&chunk, events),
            Err(RecvTimeoutError::Timeout) => {}
            // The command closed its output but still runs.
            Err(RecvTimeoutError::Disconnected) => thread::sleep(POLL),
        }
    };
    // Reaped: the pid may be reused once its group is empty.
    control.leader.store(0, Ordering::SeqCst);
    // Something the command left running may hold the pipe; the rest of
    // its group goes then.
    if !lines.read_rest(&chunks, events) {
        signal(pid, libc::SIGKILL);
        lines.read_rest(&chunks, events);
    }
    lines.finish(events);
    let end = match (exited, stopping) {
        (Ok(_), Some((_, true))) => End::TimedOut,
        (Ok(code), _) => End::Exited(code),
        (Err(why), _) => End::Failed(why),
    };
    events.unbounded_send(RunEvent::Ended(end)).ok();
}

/// Reads `pipe` on a thread of its own and sends it on in chunks, until
/// every writer closed it or the receiver is gone.
fn drain(pipe: &std::io::PipeReader) -> Receiver<Vec<u8>> {
    let (sender, chunks) = sync_channel(16);
    let Ok(mut pipe) = pipe.try_clone() else {
        return chunks;
    };
    thread::spawn(move || {
        let mut buffer = [0; 8192];
        loop {
            let read = match pipe.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            let chunk = buffer.iter().take(read).copied().collect();
            if sender.send(chunk).is_err() {
                break;
            }
        }
    });
    chunks
}

/// Output cut into lines as it comes.
#[derive(Default)]
struct Lines(Vec<u8>);

impl Lines {
    /// Adds `chunk` and sends each line it completes.
    fn push(&mut self, chunk: &[u8], events: &UnboundedSender<RunEvent>) {
        for &byte in chunk {
            if byte == b'\n' {
                self.send(events);
            } else {
                self.0.push(byte);
            }
        }
    }

    /// Sends the line so far, without a trailing `\r`.
    fn send(&mut self, events: &UnboundedSender<RunEvent>) {
        let line = String::from_utf8_lossy(&self.0);
        let line = line.strip_suffix('\r').unwrap_or(&line).to_owned();
        events.unbounded_send(RunEvent::Line(line)).ok();
        self.0.clear();
    }

    /// Reads the rest of the output for at most [`PIPE_GRACE`]: whether it
    /// ended, every process holding the pipe gone.
    fn read_rest(
        &mut self,
        chunks: &Receiver<Vec<u8>>,
        events: &UnboundedSender<RunEvent>,
    ) -> bool {
        let deadline = Instant::now() + PIPE_GRACE;
        loop {
            match chunks.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(chunk) => self.push(&chunk, events),
                Err(RecvTimeoutError::Disconnected) => return true,
                Err(RecvTimeoutError::Timeout) => return false,
            }
        }
    }

    /// Sends a last line that did not end in a newline.
    fn finish(&mut self, events: &UnboundedSender<RunEvent>) {
        if !self.0.is_empty() {
            self.send(events);
        }
    }
}

/// Takes each line into its call's progress, and answers the call once
/// its command ended.
fn take_event(
    event: On<Insert<Done<RunEvent>>>,
    jobs: Query<(&Done<RunEvent>, &JobOf)>,
    mut calls: Query<(&mut RunProgress, Option<&Cancelled>)>,
    print: Option<Res<PrintProgress>>,
    mut commands: Commands,
) {
    let Ok((Done(event), &JobOf(call))) = jobs.get(event.entity) else {
        return;
    };
    let Ok((mut progress, cancelled)) = calls.get_mut(call) else {
        return;
    };
    match event {
        RunEvent::Line(line) => {
            if print.is_some() {
                eprintln!("{RUN} │ {line}");
            }
            progress.push(line);
        }
        RunEvent::Ended(end) => {
            let result = answer(result_text(&progress, end, cancelled));
            commands.entity(call).try_insert_if_new(result);
        }
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

/// A call's result: the end of its output and how it ended.
fn result_text(progress: &RunProgress, end: &End, cancelled: Option<&Cancelled>) -> String {
    let ending = match (cancelled, end) {
        (Some(Cancelled(after)), _) => {
            format!("[cancelled by user after {:.1} s]", after.as_secs_f64())
        }
        (None, End::TimedOut) => format!("[timed out after {} s]", progress.timeout),
        (None, End::Exited(Some(code))) => format!("[exit code {code}]"),
        (None, End::Exited(None)) => "[killed by a signal]".to_owned(),
        (None, End::Failed(why)) => format!("[waiting for the command failed: {why}]"),
    };
    let cut = if progress.cut {
        "[earlier output cut]\n"
    } else {
        ""
    };
    format!("{cut}{}{ending}", progress.output)
}

/// The running `run` calls of an agent, newest first.
type RunningCalls<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static ToolRequest,
        &'static ForAgent,
        &'static RunProgress,
        &'static Job,
    ),
    (Without<Outcome>, Without<Cancelled>),
>;

/// `/cancel-call [id]`: cancels the agent's running `run` call, the one
/// named, or offers a picker of them when there are several.
fn cancel_call(
    In(args): In<CommandArgs>,
    calls: RunningCalls,
    processes: Query<&Process>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut running: Vec<_> = calls
        .iter()
        .filter(|&(_, _, &ForAgent(agent), ..)| agent == args.agent)
        .collect();
    running.sort_by_key(|(.., progress, _)| Reverse(progress.started));
    let id = args.args.trim();
    let chosen = match (id, running.as_slice()) {
        ("", []) => {
            notices.write(Notice::info(args.agent, "No `run` command is running."));
            return;
        }
        ("", [one]) => Some(*one),
        ("", several) => {
            let picker = several.iter().fold(
                Picker::new("Cancel which command?").for_agent(args.agent),
                |picker, (_, request, _, progress, _)| {
                    let secs = progress.started.elapsed().as_secs();
                    let label = format!("{} ({secs} s)", progress.command);
                    picker.item(label, format!("{CANCEL} {}", request.call.id))
                },
            );
            commands.spawn(picker);
            return;
        }
        (id, _) => running
            .iter()
            .find(|(_, request, ..)| request.call.id.to_string() == id)
            .copied(),
    };
    let Some((call, _, _, progress, &Job(job))) = chosen else {
        notices.write(Notice::error(
            args.agent,
            format!("No running `run` call `{id}`."),
        ));
        return;
    };
    let Ok(Process(control)) = processes.get(job) else {
        return;
    };
    control.stop.store(true, Ordering::Relaxed);
    commands
        .entity(call)
        .insert(Cancelled(progress.started.elapsed()));
}

/// The terminal view's panel of the running commands' output.
#[derive(Component)]
struct RunPanel;

fn spawn_panel(mut commands: Commands) {
    let panel = TuiPanel::new(Placement::Bottom(Constraint::Length(0)));
    commands.spawn((RunPanel, panel));
}

/// The running `run` calls of the agent shown, oldest first.
fn shown<'a>(
    calls: &'a Query<(&RunProgress, &ForAgent), Without<Outcome>>,
    focused: &Query<(), With<Focused>>,
) -> Vec<&'a RunProgress> {
    let mut shown: Vec<_> = calls
        .iter()
        .filter(|&(_, &ForAgent(agent))| focused.contains(agent))
        .map(|(progress, _)| progress)
        .collect();
    shown.sort_by_key(|progress| progress.started);
    shown
}

/// Gives the panel the rows the shown commands need.
fn size_panel(
    calls: Query<(&RunProgress, &ForAgent), Without<Outcome>>,
    focused: Query<(), With<Focused>>,
    mut panels: Query<&mut TuiPanel, With<RunPanel>>,
) {
    let rows: usize = shown(&calls, &focused)
        .iter()
        .map(|progress| 1 + progress.lines.len())
        .sum();
    let rows = u16::try_from(rows).unwrap_or(u16::MAX);
    for mut panel in &mut panels {
        panel.set_if_neq(TuiPanel::new(Placement::Bottom(Constraint::Length(rows))));
    }
}

/// Asks for a frame when output came since the last one, at most every
/// [`REDRAW`].
fn redraw(changed: Query<(), Changed<RunProgress>>, mut redraws: MessageWriter<RequestRedraw>) {
    if !changed.is_empty() {
        redraws.write(RequestRedraw);
    }
}

/// Draws each shown command: its line, then its last lines.
fn draw_panel(
    calls: Query<(&RunProgress, &ForAgent), Without<Outcome>>,
    focused: Query<(), With<Focused>>,
    mut panels: Query<&mut PanelCanvas, With<RunPanel>>,
) {
    let mut lines = Vec::new();
    for progress in shown(&calls, &focused) {
        let secs = progress.started.elapsed().as_secs();
        let header = format!("● {RUN} {} ({secs} s, ctrl+k cancels)", progress.command);
        lines.push(Line::from(header).yellow());
        lines.extend(
            progress
                .lines
                .iter()
                .map(|line| Line::from(format!("  {line}")).dim()),
        );
    }
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.clone()));
    }
}
