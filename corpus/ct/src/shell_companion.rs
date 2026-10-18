//! CT-18: Shell companion (conversation-over-time fresh held-out spec).
//!
//! `rig shell-init zsh|bash|fish` prints a snippet for the shell's rc file.
//! It records each command's text, exit code and directory, and for
//! commands run through its `rig-run` wrapper the output's tail, in a
//! directory of the shell's own. `?? <question>` (or `??` alone, "Why did
//! this fail?", or a key binding) runs `rig shell-ask --state DIR …`, which
//! hands the question and the last command to the session that serves the
//! project over `RIG_HOME/shell/<project>.sock`, prints the reply as it
//! streams, tool calls as one-line progress, and "waiting for approval in
//! rig" while a question holds the turn. `?? --cmd …` puts the reply's
//! suggested command in the shell's edit buffer and never runs it.
//!
//! A running session (any run that is not one of this plugin's commands)
//! serves its project's socket. Each request becomes a user message titled
//! `shell`, with the command in a fenced block, its tail capped and
//! redacted (rig-core's key shapes and the `[shell-companion] redact`
//! strings), and any redaction plugin's Prepare system sees it as it sees
//! every message. Requests wait in a queue of their own and go one at a
//! time, each once the agent is idle, so each has a turn and reply of its
//! own. With no session serving the project, `rig shell-ask` answers in a
//! one-shot session of its own, says so, and exits once it answered.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::redact;
use serde::{Deserialize, Serialize};

pub mod client;
pub mod snippet;
mod socket;

use socket::{Arrived, Listening};

/// The command that prints the shell snippet.
pub const SHELL_INIT: &str = "shell-init";

/// The command the snippet's `??` runs.
pub const SHELL_ASK: &str = "shell-ask";

/// What a message from the shell is titled with.
pub const TITLE: &str = "shell";

/// What a bare `??` asks.
pub const WHY: &str = "Why did this fail?";

/// The shell companion: the session's socket, `rig shell-init` and `rig
/// shell-ask`.
#[derive(Default)]
pub struct ShellCompanionPlugin;

impl Plugin for ShellCompanionPlugin {
    fn build(&self, app: &mut App) {
        app.add_front(
            Front::new(SHELL_INIT)
                .on_command(
                    SHELL_INIT,
                    "zsh|bash|fish",
                    "Print the shell companion's rc snippet",
                )
                .sessionless(),
        )
        .add_front(Front::new(SHELL_ASK).on_command(
            SHELL_ASK,
            "--state DIR [--cmd] [question…]",
            "Ask the project's session about the shell's last command",
        ))
        .add_setting::<ShellSettings>("shell-companion");
    }

    fn finish(&self, app: &mut App) {
        let words = app
            .world()
            .get_resource::<Invoked>()
            .and_then(|invoked| invoked.command().map(|(_, words)| words.to_vec()))
            .unwrap_or_default();
        if app.runs_front(SHELL_INIT) {
            app.add_systems(Startup, move |mut commands: Commands| {
                let shell = words.first().map_or("", String::as_str);
                match snippet::for_shell(shell) {
                    Some(text) => {
                        print!("{text}");
                        commands.trigger(Exit(ExitReason::Quit));
                    }
                    None => {
                        eprintln!("rig shell-init: name a shell: zsh, bash or fish");
                        commands.trigger(Exit(ExitReason::Failed(2)));
                    }
                }
            });
            return;
        }
        app.init_resource::<Server>()
            .add_systems(Update, dispatch.before(AgentLoop))
            .add_systems(PostUpdate, stream_text)
            .add_observer(arrived)
            .add_observer(delivered)
            .add_observer(settled)
            .add_observer(tool_started)
            .add_observer(asked);
        if app.runs_front(SHELL_ASK) {
            client::build(app, &words);
        } else {
            app.add_systems(Startup, listen);
        }
    }
}

/// The `[shell-companion]` settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ShellSettings {
    /// Strings taken out of every request before the session sees it, as
    /// rig-core's `redact` takes out what looks like a key.
    pub redact: Vec<String>,
    /// The most output lines a request keeps, its last.
    pub tail_lines: usize,
    /// The most output bytes a request keeps, its last.
    pub tail_bytes: usize,
}

impl Default for ShellSettings {
    fn default() -> Self {
        Self {
            redact: Vec::new(),
            tail_lines: 40,
            tail_bytes: 4000,
        }
    }
}

/// What the shell sends: the question and its last command.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShellAsk {
    /// What the user asked; empty for "why did this fail?".
    pub question: String,
    /// Whether the user asks for a command for the edit buffer.
    pub cmd: bool,
    /// The last command's text.
    pub command: String,
    /// Its exit code, if recorded.
    pub exit: Option<i32>,
    /// The directory it ran in.
    pub cwd: String,
    /// The tail of its output, for a command run through `rig-run`.
    pub tail: String,
}

/// What the session sends back, one JSON line each.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// The request waits behind `ahead` others.
    Queued {
        /// How many go first.
        ahead: usize,
    },
    /// The request went to the agent.
    Sent,
    /// Reply text, as it streams.
    Text {
        /// The fragment.
        text: String,
    },
    /// A tool call started, as one line.
    Tool {
        /// The tool and its main argument.
        line: String,
    },
    /// A question holds the turn, to answer in the rig window.
    Approval,
    /// The turn answered.
    Done {
        /// The whole answer.
        answer: String,
        /// For `--cmd`, the command the answer suggests.
        command: Option<String>,
    },
    /// The turn failed or stopped, or the request never reached the agent.
    Failed {
        /// Why.
        why: String,
    },
}

/// Triggered as a request from the shell is answered, whichever way it
/// came.
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub struct ShellReplied {
    /// The agent that answered.
    pub agent: Entity,
    /// What it ended with.
    pub event: Event,
}

/// Where a request's events go.
enum Sink {
    /// A client connected to the socket.
    Socket(std::sync::mpsc::Sender<String>),
    /// This process's stdout: a one-shot `rig shell-ask` run, with the
    /// snippet's state directory.
    Local(PathBuf),
}

impl Sink {
    fn send(&self, event: &Event) {
        match self {
            Self::Socket(lines) => {
                if let Ok(line) = serde_json::to_string(event) {
                    lines.send(line).ok();
                }
            }
            Self::Local(state) => {
                let mut out = std::io::stdout();
                client::render(event, &mut out).ok();
                if let Event::Done { command, .. } = event {
                    client::suggest(state, command.as_deref());
                }
            }
        }
    }
}

/// A request and where its answer goes.
struct Request {
    ask: ShellAsk,
    sink: Sink,
}

/// The request being answered.
struct InFlight {
    request: Request,
    agent: Entity,
    message: Entity,
    turn: Option<Entity>,
}

/// The socket, the queue of requests and the one in flight.
#[derive(Resource, Default)]
struct Server {
    listening: Option<Listening>,
    queue: VecDeque<Request>,
    in_flight: Option<InFlight>,
    deltas: MessageCursor<StreamDelta>,
}

/// On the entity reading the socket.
#[derive(Component)]
struct Connections;

/// On the entity that marks a one-shot run: it exits once it answered.
#[derive(Component)]
struct OneShot;

/// The longest socket path the system takes (`sun_path` less its nul).
const MAX_SOCKET_PATH: usize = 107;

/// The socket a session serving `project` listens on: under `home`, or,
/// when that path is longer than a socket path may be, under the user's
/// runtime directory (else the temporary one), keyed by `home` too.
pub fn socket_path(home: &Path, project: &Path) -> PathBuf {
    let path = home_socket_path(home, project);
    if path.as_os_str().len() <= MAX_SOCKET_PATH {
        return path;
    }
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let name = path.file_name().map(PathBuf::from).unwrap_or_default();
    runtime
        .join(format!(
            "rig-shell-{:016x}",
            fnv(home.as_os_str().as_encoded_bytes())
        ))
        .join(name)
}

/// FNV-1a: stable across builds, unlike std's hasher.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The socket path under `home`, whatever its length.
fn home_socket_path(home: &Path, project: &Path) -> PathBuf {
    let project = std::fs::canonicalize(project).unwrap_or_else(|_| project.to_path_buf());
    let hash = fnv(project.as_os_str().as_encoded_bytes());
    home.join("shell").join(format!("{hash:016x}.sock"))
}

/// Serves the project's socket, unless another session does.
fn listen(
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut server: ResMut<Server>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let path = socket_path(home.0.root(), &root.0);
    match socket::listen(&path) {
        Ok((listening, arrivals)) => {
            server.listening = Some(listening);
            let pool = IoTaskPool::get_or_init(TaskPool::default);
            commands.spawn((Connections, Running::stream(pool, &wake, arrivals)));
        }
        Err(why) => {
            notices.write(Notice::info(
                None,
                format!("shell companion: not serving: {why}"),
            ));
        }
    }
}

/// Queues each request a client sends.
fn arrived(
    inserted: On<Insert<Done<Arrived>>>,
    arrivals: Query<&Done<Arrived>>,
    mut server: ResMut<Server>,
) {
    let Ok(Done(arrival)) = arrivals.get(inserted.entity) else {
        return;
    };
    let Some((ask, lines)) = arrival.take() else {
        return;
    };
    queue(
        &mut server,
        Request {
            ask,
            sink: Sink::Socket(lines),
        },
    );
}

/// Puts `request` at the end of the queue and tells its asker.
fn queue(server: &mut Server, request: Request) {
    let ahead = server.queue.len() + usize::from(server.in_flight.is_some());
    request.sink.send(&Event::Queued { ahead });
    server.queue.push_back(request);
}

/// What the dispatcher reads of the agents.
#[derive(SystemParam)]
struct Target<'w, 's> {
    shown: Option<Res<'w, ShownAgent>>,
    primary: MainAgentQuery<'w, 's>,
    connected: Query<'w, 's, (), With<Connection>>,
    turns: Query<'w, 's, &'static TurnOf>,
    inbox: Query<'w, 's, &'static ForAgent, (With<Incoming>, Without<Settled>)>,
}

impl Target<'_, '_> {
    /// The agent a request goes to, once it can take one: connected, with
    /// no turn running and nothing in its inbox.
    fn idle(&self) -> Option<Entity> {
        let agent = self
            .shown
            .as_ref()
            .and_then(|shown| shown.0)
            .or_else(|| main_agent(&self.primary))?;
        let busy = self.turns.iter().any(|&TurnOf(of)| of == agent)
            || self.inbox.iter().any(|&ForAgent(of)| of == agent);
        (self.connected.contains(agent) && !busy).then_some(agent)
    }
}

/// Sends the next request once nothing is in flight and the agent is
/// idle, so each request has its own turn.
fn dispatch(
    mut server: ResMut<Server>,
    target: Target,
    settings: Res<ShellSettings>,
    mut commands: Commands,
) {
    if server.in_flight.is_some() || server.queue.is_empty() {
        return;
    }
    let Some(agent) = target.idle() else {
        return;
    };
    let Some(request) = server.queue.pop_front() else {
        return;
    };
    let text = message(&request.ask, &settings);
    let author = Author::user().titled(TITLE);
    let incoming = Incoming::new(text, DeliveryMode::Steer).with_author(author);
    let message = commands.spawn((incoming, ForAgent(agent))).id();
    request.sink.send(&Event::Sent);
    server.in_flight = Some(InFlight {
        request,
        agent,
        message,
        turn: None,
    });
}

/// The message the agent reads for `ask`: the question, then the command
/// in a fenced block, its output's tail capped and redacted.
pub fn message(ask: &ShellAsk, settings: &ShellSettings) -> String {
    let clean = |text: &str| {
        let mut text = redact::text(text).into_owned();
        for secret in settings.redact.iter().filter(|secret| !secret.is_empty()) {
            text = text.replace(secret.as_str(), redact::REDACTED);
        }
        text
    };
    let question = match ask.question.trim() {
        "" => WHY,
        question => question,
    };
    let mut block = format!("$ {}\n", clean(&ask.command));
    match ask.exit {
        Some(code) => block.push_str(&format!("# exit code: {code}\n")),
        None => block.push_str("# exit code: unknown\n"),
    }
    block.push_str(&format!("# cwd: {}\n", ask.cwd));
    let tail = clean(&tail(&ask.tail, settings));
    if !tail.trim().is_empty() {
        block.push_str("# output (last lines):\n");
        block.push_str(tail.trim_end());
        block.push('\n');
    }
    let fence = "`".repeat(longest_backticks(&block).max(2) + 1);
    let mut text = format!(
        "[from my shell] {}\n\n{fence}shell\n{block}{fence}",
        clean(question)
    );
    if ask.cmd {
        text.push_str(
            "\n\nAnswer with the one shell command to run, in a ```sh block: it goes in my \
             shell's edit buffer, and is never run for me.",
        );
    }
    text
}

/// The last lines and bytes of `output` the settings keep.
fn tail(output: &str, settings: &ShellSettings) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let kept = lines
        .get(lines.len().saturating_sub(settings.tail_lines)..)
        .unwrap_or_default()
        .join("\n");
    let mut start = kept.len().saturating_sub(settings.tail_bytes);
    while !kept.is_char_boundary(start) {
        start += 1;
    }
    kept.get(start..).unwrap_or_default().to_owned()
}

/// The longest run of backticks in `text`.
fn longest_backticks(text: &str) -> usize {
    text.split(|c| c != '`')
        .map(str::len)
        .max()
        .unwrap_or_default()
}

/// The command an answer suggests: its first fenced block, else its first
/// `$ ` line.
pub fn suggested_command(answer: &str) -> Option<String> {
    let mut lines = answer.lines();
    let mut fenced = Vec::new();
    while let Some(line) = lines.next() {
        if line.trim_start().starts_with("```") {
            for line in lines.by_ref() {
                if line.trim_start().starts_with("```") {
                    break;
                }
                fenced.push(line.trim().trim_start_matches("$ "));
            }
            break;
        }
    }
    let command = fenced.join("\n");
    let command = command.trim();
    if !command.is_empty() {
        return Some(command.to_owned());
    }
    answer
        .lines()
        .find_map(|line| line.trim().strip_prefix("$ "))
        .map(str::to_owned)
}

/// Sends the request's events, first the text streamed before them, so
/// the shell sees things in the order they happened.
#[derive(SystemParam)]
struct Events<'w, 's> {
    server: ResMut<'w, Server>,
    deltas: Res<'w, Messages<StreamDelta>>,
    calls: Query<'w, 's, &'static CallOf>,
}

impl Events<'_, '_> {
    /// Sends the text the turn in flight streamed since the last event.
    fn flush(&mut self) {
        let Server {
            deltas, in_flight, ..
        } = &mut *self.server;
        for delta in deltas.read(&self.deltas) {
            let Some(in_flight) = in_flight.as_ref() else {
                continue;
            };
            let ours = self
                .calls
                .get(delta.effect)
                .is_ok_and(|&CallOf(turn)| Some(turn) == in_flight.turn);
            if let (true, StreamEvent::Text { text, .. }) = (ours, &delta.event) {
                in_flight
                    .request
                    .sink
                    .send(&Event::Text { text: text.clone() });
            }
        }
    }

    /// Sends `event` for the turn in flight if it is `turn`.
    fn send(&mut self, turn: Entity, event: &Event) {
        self.flush();
        if let Some(in_flight) = &self.server.in_flight
            && in_flight.turn == Some(turn)
        {
            in_flight.request.sink.send(event);
        }
    }
}

fn stream_text(mut events: Events) {
    events.flush();
}

/// The turn the request's message went into.
fn delivered(
    added: On<Add<DeliveredIn>>,
    messages: Query<&DeliveredIn>,
    mut server: ResMut<Server>,
) {
    let Ok(&DeliveredIn(turn)) = messages.get(added.entity) else {
        return;
    };
    if let Some(in_flight) = &mut server.in_flight
        && in_flight.message == added.entity
    {
        in_flight.turn = Some(turn);
    }
}

/// The turn in flight ended, or its message settled without a turn.
fn settled(
    added: On<Add<Settled>>,
    outcomes: Query<&TurnOutcome>,
    one_shots: Query<Entity, With<OneShot>>,
    mut events: Events,
    mut commands: Commands,
) {
    let entity = added.entity;
    events.flush();
    let Some(in_flight) = events.server.in_flight.as_ref() else {
        return;
    };
    let event = if in_flight.turn == Some(entity) {
        match outcomes.get(entity) {
            Ok(TurnOutcome::Answered(message)) => {
                let answer = match message {
                    message::Message::Assistant(reply) => reply.text(),
                    _ => String::new(),
                };
                let command = in_flight
                    .request
                    .ask
                    .cmd
                    .then(|| suggested_command(&answer))
                    .flatten();
                Event::Done { answer, command }
            }
            Ok(TurnOutcome::Failed(why)) => Event::Failed { why: why.clone() },
            Ok(TurnOutcome::Stopped(why)) => Event::Failed {
                why: why.clone().unwrap_or_else(|| "stopped in rig".to_owned()),
            },
            Err(_) => return,
        }
    } else if in_flight.message == entity && in_flight.turn.is_none() {
        Event::Failed {
            why: "the session took the message without a turn".to_owned(),
        }
    } else {
        return;
    };
    let Some(in_flight) = events.server.in_flight.take() else {
        return;
    };
    in_flight.request.sink.send(&event);
    if matches!(in_flight.request.sink, Sink::Local(_)) && !one_shots.is_empty() {
        let exit = match &event {
            Event::Failed { .. } => ExitReason::Failed(1),
            _ => ExitReason::Quit,
        };
        commands.trigger(Exit(exit));
    }
    commands.trigger(ShellReplied {
        agent: in_flight.agent,
        event,
    });
}

/// A tool call of the turn in flight, as one line.
fn tool_started(
    started: On<Add<Started>>,
    calls: Query<(&ToolRequest, &CallOf)>,
    mut events: Events,
) {
    let Ok((request, &CallOf(turn))) = calls.get(started.entity) else {
        return;
    };
    let function = &request.call.function;
    let argument = function
        .arguments
        .values()
        .find_map(serde_json::Value::as_str)
        .unwrap_or_default();
    let mut line = format!("{} {argument}", function.name.as_str());
    line.truncate(line.trim_end().len());
    events.send(turn, &Event::Tool { line });
}

/// A question that holds a call of the turn in flight.
fn asked(added: On<Add<Ask>>, holds: Query<&HoldOf>, calls: Query<&CallOf>, mut events: Events) {
    let Ok(&HoldOf(held)) = holds.get(added.entity) else {
        return;
    };
    let turn = calls.get(held).map_or(held, |&CallOf(turn)| turn);
    events.send(turn, &Event::Approval);
}

#[cfg(test)]
mod tests;
