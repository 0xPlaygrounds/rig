//! Interactive terminal sessions (held-out spec 08): the agent drives live
//! programs, such as a REPL, a debugger or an installer, through a real
//! pseudo-terminal across turns, with `term_open`, `term_send`,
//! `term_read` and `term_close`, and the user watches and types into them
//! in a panel of the terminal view (`/term`).
//!
//! A terminal is an entity [`TerminalOf`] its agent, so it ends with the
//! agent; its [`Terminal`] holds the program, which a thread reads into a
//! screen as it writes, escape sequences interpreted. The program is `sh -c
//! <command>`, started as every child process starts ([`Processes`]), as
//! the leader of a session of its own whose controlling terminal is the
//! pty, so Ctrl+C reaches the job in front and closing kills everything it
//! started. Every terminal is killed on the way out. The names of an
//! agent's open terminals are saved with the session ([`TerminalNames`]);
//! terminals are not restored, and the agent is told with the user's next
//! message which ones ended.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

mod keys;
mod panel;
mod pty;
mod screen;
#[cfg(test)]
mod tests;

pub use panel::TerminalsPanel;
use pty::{GRACE, Pty, Waited};

/// The most terminals an agent has open at once.
pub const MAX_OPEN: usize = 4;
/// The lines of scrollback `term_read` returns.
const SCROLLBACK_LINES: usize = 500;
/// The longest a `term_send` or `term_open` waits.
const MAX_WAIT: Duration = Duration::from_secs(600);
/// How long `term_send` waits unless told.
const DEFAULT_WAIT: Duration = Duration::from_secs(10);
/// Who decides the plugin's calls.
const NAME: &str = "terminal-sessions";

/// A terminal the agent opened: its name, its command and the program.
#[derive(Component)]
pub struct Terminal {
    /// Its name, unique among its agent's terminals.
    pub name: String,
    /// The command it runs.
    pub command: String,
    pty: Arc<Pty>,
}

/// A terminal's program is killed when the terminal goes, unless
/// `term_close` ends it gently.
impl Drop for Terminal {
    fn drop(&mut self) {
        if !self.pty.closing.load(Ordering::Acquire) {
            self.pty.kill_all();
        }
    }
}

/// A change of a terminal's screen, as its entity hears it.
pub struct Output;

/// The agent a terminal belongs to.
#[derive(Component, Debug)]
#[relationship(relationship_target = Terminals)]
pub struct TerminalOf(pub Entity);

/// An agent's terminals, which go with it.
#[derive(Component, Debug)]
#[relationship_target(relationship = TerminalOf, linked_spawn)]
pub struct Terminals(Vec<Entity>);

/// The names of an agent's terminals, kept with the session: those open,
/// and those that ended with the last run that the agent was not told of.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default)]
pub struct TerminalNames {
    /// Open now.
    pub open: Vec<String>,
    /// Ended with the run before; told with the user's next message.
    pub untold: Vec<String>,
}

/// The tools, the panel and the terminals' lifecycle.
#[derive(Default)]
pub struct TerminalSessionsPlugin;

impl Plugin for TerminalSessionsPlugin {
    fn build(&self, app: &mut App) {
        let options = |footprint| ToolOptions {
            footprint: Some(footprint),
        };
        let writes = Touches::Writes(Scope::Anything);
        app.init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .register_required_components::<Agent, TerminalNames>()
            .add_open_tool(
                "term_open",
                "Start a command in a pseudo-terminal of its own in the working directory, \
                 such as a REPL, a debugger or an interactive installer. Returns the \
                 terminal's name and its first screen. It keeps running across turns until \
                 term_close. Use it for programs that need a live terminal across several \
                 steps; use `shell` for commands that just run and end. At most 4 terminals \
                 are open: close those you no longer need.",
                options(writes.clone()),
                on_open,
            )
            .add_open_tool(
                "term_send",
                "Type into a terminal and return what the program printed since, once it waits \
                 for input (quiet for 300 ms), `wait_for` matches, or the timeout passes. \
                 `<Enter>`, `<Tab>`, `<Esc>`, `<Up>`, `<Down>`, `<Left>`, `<Right>`, `<BS>`, \
                 `<C-c>`, `<C-d>` (any `<C-x>`) and `<M-x>` are keys; `<lt>` is `<`.",
                options(writes.clone()),
                on_send,
            )
            .add_open_tool(
                "term_read",
                "A terminal's screen as drawn now, or the last 500 lines of its scrollback.",
                options(Touches::Reads(Scope::Anything)),
                on_read,
            )
            .add_open_tool(
                "term_close",
                "End a terminal's program: hang up, then kill it and what it started after 2 s.",
                options(writes),
                on_close,
            )
            .add_observer(name_opened)
            .add_observer(name_closed)
            .add_observer(ended_before)
            .add_systems(AgentPass, tell_ended.in_set(PrepareSystems::Edit))
            .add_systems(
                Last,
                kill_on_exit
                    .in_set(OnAppExitSystems)
                    .run_if(on_message::<AppExit>),
            );
        panel::add(app);
    }
}

/// The arguments of `term_open`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    /// The command line, run with `sh -c`, such as `python3` or `gdb ./app`.
    command: String,
    /// A short name for the terminal; `t1`, `t2`, … when left out.
    name: Option<String>,
    /// Its width in columns (default 120).
    cols: Option<u16>,
    /// Its height in rows (default 40).
    rows: Option<u16>,
}

/// The arguments of `term_send`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SendArgs {
    /// The terminal's name.
    name: String,
    /// The keystrokes: text as typed, `<Enter>`, `<C-c>` and the like as keys.
    input: String,
    /// A regex; return as soon as the output since the send matches it.
    wait_for: Option<String>,
    /// The longest to wait, in seconds (default 10).
    timeout: Option<f64>,
}

/// What `term_read` reads.
#[derive(Deserialize, JsonSchema, Clone, Copy, Default)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// The screen as drawn now.
    #[default]
    Screen,
    /// The last 500 lines, the screen's included.
    Scrollback,
}

/// The arguments of `term_read`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
    /// The terminal's name.
    name: String,
    /// `screen` (default) or `scrollback`.
    mode: Option<Mode>,
}

/// The arguments of `term_close`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct CloseArgs {
    /// The terminal's name.
    name: String,
}

fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Decides `call` now: its result's text, or why it failed.
fn decide(
    calls: &mut Query<&mut Decision, ToolCalls>,
    call: Entity,
    result: Result<String, String>,
) {
    if let Ok(mut decision) = calls.get_mut(call) {
        decision.decide(Decider::plugin(NAME), tool_result(result));
    }
}

fn tool_result(result: Result<String, String>) -> ToolResult {
    match result {
        Ok(text) => ToolResult::success(text.into()),
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    }
}

/// The terminals of `agent`: entity, terminal.
fn of_agent<'a>(
    terminals: &'a Query<(Entity, &Terminal, &TerminalOf)>,
    agent: Entity,
) -> impl Iterator<Item = (Entity, &'a Terminal)> {
    terminals
        .iter()
        .filter(move |(_, _, of)| of.0 == agent)
        .map(|(entity, terminal, _)| (entity, terminal))
}

/// Why a terminal was not found, naming those open.
fn not_found(
    terminals: &Query<(Entity, &Terminal, &TerminalOf)>,
    agent: Entity,
    name: &str,
) -> String {
    let names: Vec<String> = of_agent(terminals, agent)
        .map(|(_, t)| format!("`{}`", t.name))
        .collect();
    if names.is_empty() {
        format!("No terminal `{name}` is open; open one with term_open.")
    } else {
        format!("No terminal `{name}`; open: {}.", names.join(", "))
    }
}

/// The note that the user typed in the terminal since the agent looked,
/// once.
fn user_note(state: &mut pty::State) -> &'static str {
    if std::mem::take(&mut state.user_typed) {
        "[the user typed in this terminal]\n"
    } else {
        ""
    }
}

/// What a call on a terminal whose program exited says: the code and the
/// final screen.
fn exited(name: &str, pty: &Pty) -> String {
    let mut state = pty.lock();
    let note = user_note(&mut state);
    let code = state.exit.unwrap_or(-1);
    format!(
        "{note}Terminal `{name}` exited with code {code}. Final screen:\n{}",
        state.screen.text()
    )
}

fn on_open(
    called: On<ToolCalled<OpenArgs>>,
    terminals: Query<(Entity, &Terminal, &TerminalOf)>,
    dirs: Query<&WorkDir>,
    (root, workspace, processes, wake): (
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Processes>,
        Res<Wake>,
    ),
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let (call, agent, args) = (called.call, called.agent, &called.args);
    if let Some(host) = workspace.0.host() {
        let why = format!("Terminals run on this machine, and the project is on {host}.");
        decide(&mut calls, call, Err(why));
        return;
    }
    let mut open: Vec<&str> = of_agent(&terminals, agent)
        .filter(|(_, terminal)| terminal.pty.lock().exit.is_none())
        .map(|(_, terminal)| terminal.name.as_str())
        .collect();
    open.sort_unstable();
    if open.len() >= MAX_OPEN {
        let why = format!(
            "At most {MAX_OPEN} terminals are open at once; close one of: {}.",
            open.join(", ")
        );
        decide(&mut calls, call, Err(why));
        return;
    }
    let taken = |name: &str| of_agent(&terminals, agent).any(|(_, t)| t.name == name);
    let name = match &args.name {
        Some(name) if taken(name) => {
            let why = format!("A terminal named `{name}` is open already.");
            decide(&mut calls, call, Err(why));
            return;
        }
        Some(name) if !name.trim().is_empty() => name.trim().to_owned(),
        _ => (1..)
            .map(|n| format!("t{n}"))
            .find(|name| !taken(name))
            .unwrap_or_default(),
    };
    let size = (
        args.cols.unwrap_or(120).clamp(20, 500),
        args.rows.unwrap_or(40).clamp(5, 200),
    );
    let dir = dirs.get(agent).map_or(&root.0, |dir| &dir.0);
    let (ticks, changes) = futures::channel::mpsc::unbounded();
    let pty = match Pty::start(&processes.launcher(), &args.command, dir, size, ticks) {
        Ok(pty) => pty,
        Err(error) => {
            decide(
                &mut calls,
                call,
                Err(format!("Could not start `{}`: {error}", args.command)),
            );
            return;
        }
    };
    let changes = futures::StreamExt::map(changes, |()| Output);
    commands.spawn((
        Name::new(format!("terminal {name}")),
        Terminal {
            name: name.clone(),
            command: args.command.clone(),
            pty: Arc::clone(&pty),
        },
        TerminalOf(agent),
        Running::stream(pool(), &wake, changes),
    ));
    let first = async move {
        let text = blocking(move || {
            let waited = pty.wait(None, DEFAULT_WAIT, true);
            if let Waited::Exited = waited {
                return Ok(exited(&name, &pty));
            }
            let mut state = pty.lock();
            state.since.clear();
            let still = matches!(waited, Waited::TimedOut)
                .then_some(" (still printing)")
                .unwrap_or_default();
            Ok(format!(
                "Opened terminal `{name}`{still}. Screen:\n{}",
                state.screen.text()
            ))
        })
        .await;
        tool_result(text.map_err(|error: ToolExecutionError| error.to_string()))
    };
    commands
        .entity(call)
        .insert(Running::answering(pool(), &wake, first));
}

fn on_send(
    called: On<ToolCalled<SendArgs>>,
    terminals: Query<(Entity, &Terminal, &TerminalOf)>,
    wake: Res<Wake>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let (call, agent, args) = (called.call, called.agent, &called.args);
    let Some((entity, terminal)) = of_agent(&terminals, agent).find(|(_, t)| t.name == args.name)
    else {
        decide(
            &mut calls,
            call,
            Err(not_found(&terminals, agent, &args.name)),
        );
        return;
    };
    if terminal.pty.lock().exit.is_some() {
        decide(&mut calls, call, Ok(exited(&terminal.name, &terminal.pty)));
        commands.entity(entity).despawn();
        return;
    }
    let wait_for = match args.wait_for.as_deref().map(regex::Regex::new).transpose() {
        Ok(pattern) => pattern,
        Err(error) => {
            decide(
                &mut calls,
                call,
                Err(format!("`wait_for` is no regex: {error}")),
            );
            return;
        }
    };
    let timeout = args
        .timeout
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .map_or(DEFAULT_WAIT, Duration::from_secs_f64)
        .min(MAX_WAIT);
    let (pty, name, bytes) = (
        Arc::clone(&terminal.pty),
        terminal.name.clone(),
        keys::parse(&args.input),
    );
    let sending = async move {
        let text = blocking(move || {
            pty.send(&bytes, false)
                .map_err(ToolExecutionError::from_error)?;
            let waited = pty.wait(wait_for.as_ref(), timeout, false);
            let mut state = pty.lock();
            let note = user_note(&mut state);
            let output = screen::Screen::plain(&state.since, state.screen.cols());
            let end = match waited {
                Waited::Quiet | Waited::Matched => String::new(),
                Waited::TimedOut => format!(
                    "\n[timed out after {:.1} s, program still running]",
                    timeout.as_secs_f64()
                ),
                Waited::Exited => {
                    let code = state.exit.unwrap_or(-1);
                    format!("\n[terminal `{name}` exited with code {code}]")
                }
            };
            Ok(format!("{note}{output}{end}"))
        })
        .await;
        tool_result(text.map_err(|error: ToolExecutionError| error.to_string()))
    };
    commands
        .entity(call)
        .insert(Running::answering(pool(), &wake, sending));
}

fn on_read(
    called: On<ToolCalled<ReadArgs>>,
    terminals: Query<(Entity, &Terminal, &TerminalOf)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let (call, agent, args) = (called.call, called.agent, &called.args);
    let Some((entity, terminal)) = of_agent(&terminals, agent).find(|(_, t)| t.name == args.name)
    else {
        decide(
            &mut calls,
            call,
            Err(not_found(&terminals, agent, &args.name)),
        );
        return;
    };
    if terminal.pty.lock().exit.is_some() {
        decide(&mut calls, call, Ok(exited(&terminal.name, &terminal.pty)));
        commands.entity(entity).despawn();
        return;
    }
    let mut state = terminal.pty.lock();
    let note = user_note(&mut state);
    let text = match args.mode.unwrap_or_default() {
        Mode::Screen => state.screen.text(),
        Mode::Scrollback => state.screen.tail(SCROLLBACK_LINES),
    };
    drop(state);
    decide(&mut calls, call, Ok(format!("{note}{text}")));
}

fn on_close(
    called: On<ToolCalled<CloseArgs>>,
    terminals: Query<(Entity, &Terminal, &TerminalOf)>,
    wake: Res<Wake>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let (call, agent, args) = (called.call, called.agent, &called.args);
    let Some((entity, terminal)) = of_agent(&terminals, agent).find(|(_, t)| t.name == args.name)
    else {
        decide(
            &mut calls,
            call,
            Err(not_found(&terminals, agent, &args.name)),
        );
        return;
    };
    let (pty, name) = (Arc::clone(&terminal.pty), terminal.name.clone());
    // This call's thread ends it, gently; the terminal goes at once.
    pty.closing.store(true, Ordering::Release);
    commands.entity(entity).despawn();
    let closing = async move {
        let text = blocking(move || {
            pty.close(GRACE);
            let code = pty.lock().exit;
            Ok(match code {
                Some(code) => format!("Closed terminal `{name}`; it exited with code {code}."),
                None => format!("Closed terminal `{name}`."),
            })
        })
        .await;
        tool_result(text.map_err(|error: ToolExecutionError| error.to_string()))
    };
    commands
        .entity(call)
        .insert(Running::answering(pool(), &wake, closing));
}

/// Keeps a new terminal's name with the session.
fn name_opened(
    added: On<Add<Terminal>>,
    terminals: Query<(&Terminal, &TerminalOf)>,
    mut agents: Query<&mut TerminalNames>,
) {
    if let Ok((terminal, &TerminalOf(agent))) = terminals.get(added.entity)
        && let Ok(mut names) = agents.get_mut(agent)
    {
        names.open.push(terminal.name.clone());
    }
}

/// Forgets a closed terminal's name.
fn name_closed(
    removed: On<Remove<Terminal>>,
    terminals: Query<(&Terminal, &TerminalOf)>,
    mut agents: Query<&mut TerminalNames>,
) {
    if let Ok((terminal, &TerminalOf(agent))) = terminals.get(removed.entity)
        && let Ok(mut names) = agents.get_mut(agent)
    {
        names.open.retain(|name| *name != terminal.name);
    }
}

/// A restored agent's saved names of open terminals name terminals that
/// ended with the run before: none is restored.
fn ended_before(
    inserted: On<Insert<TerminalNames>>,
    mut agents: Query<(&mut TerminalNames, Option<&Terminals>)>,
) {
    if let Ok((mut names, live)) = agents.get_mut(inserted.entity)
        && live.is_none_or(|live| live.is_empty())
        && !names.open.is_empty()
    {
        let names = &mut *names;
        names.untold.append(&mut names.open);
    }
}

/// Tells the agent, with the user's next message, which terminals ended
/// with the run before.
fn tell_ended(
    mut messages: Query<(&mut Incoming, &ForAgent), (Added<Admitted>, IncomingMessages)>,
    mut agents: Query<&mut TerminalNames>,
) {
    for (mut message, &ForAgent(agent)) in &mut messages {
        if message.author.kind != AuthorKind::User {
            continue;
        }
        let Ok(mut names) = agents.get_mut(agent) else {
            continue;
        };
        if names.untold.is_empty() {
            continue;
        }
        let ended = std::mem::take(&mut names.untold).join("`, `");
        let text = format!(
            "The terminals `{ended}` ended when rig last stopped; terminals are not restored. \
             Open new ones with term_open if you still need them."
        );
        let attachment = Attachment {
            label: "terminals".to_owned(),
            content: UserContent::text(text),
        };
        message.parts.insert(0, MessagePart::Attachment(attachment));
    }
}

/// Kills every terminal's program and what it started, on the way out.
fn kill_on_exit(terminals: Query<&Terminal>) {
    for terminal in &terminals {
        terminal.pty.kill_all();
    }
}
