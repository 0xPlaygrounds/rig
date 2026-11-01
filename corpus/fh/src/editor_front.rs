//! FH-21: an editor front (Neovim, VS Code). `rig --editor` speaks a JSON
//! protocol over stdio, one object per line ([`FromEditor`] in,
//! [`ToEditor`] out); other editor windows join the same session over
//! the session directory's `editor.sock`, the same protocol, and any other
//! transport hands its lines to [`connect`]. Every
//! connected editor sees the transcript as it streams, the questions and
//! pickers, and answers them.
//!
//! Open buffers win. A `read` of a file an editor has open is answered,
//! once every gate had its say, from the buffer. An `edit` or `write` of an open file is held by an
//! open `editor_review` effect: the editors get its hunks, the first
//! editor that answers applies the hunks it accepted to its buffer, and
//! the others are told the review is closed. The call's result says
//! `applied`, `rejected by user` or `partially applied: hunk 2 rejected`,
//! followed by the editor's diagnostics of the file, awaited at most 2 s.
//! `/auto-accept` has the editor apply and save every edit unreviewed.
//! Files no editor has open are edited on disk as usual, and the editors
//! are told which changed.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::BufRead;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::StreamExt;
use futures::channel::mpsc::{UnboundedSender, unbounded};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::redact;
use rig_harness::rig_core::tool::FilesChanged;
use rig_tools::fs::{place, resolve};
use rig_tools::{Edit, EditArgs, Read, ReadArgs, Write, WriteArgs};
use serde::{Deserialize, Serialize};
use similar::{DiffTag, TextDiff};

/// The front's name and its option, `rig --editor`.
pub const FRONT: &str = "editor";

/// How long an applied edit waits for the editor's diagnostics.
pub const DIAGNOSTICS_WAIT: Duration = Duration::from_secs(2);

/// Adds the editor front, its buffer-aware file tools and `/auto-accept`.
#[derive(Default)]
pub struct EditorFrontPlugin;

impl Plugin for EditorFrontPlugin {
    fn build(&self, app: &mut App) {
        let front = Front::new(FRONT).on_flag(FRONT, "Serve an editor extension over stdio");
        app.add_front(front.shows_pickers())
            .register_type::<AutoAccept>()
            .init_resource::<AutoAccept>()
            .init_resource::<Buffers>()
            .init_resource::<Inbox>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ShownAgent>()
            .configure_sets(AgentPass, EditorSystems.in_set(GateSystems::After))
            .add_systems(
                AgentPass,
                (from_buffers, review_edits).chain().in_set(EditorSystems),
            )
            .add_systems(Update, receive.after(PollTasks).before(AgentLoop))
            .add_systems(PostUpdate, (forward, close_questions))
            .add_observer(take_line)
            .add_observer(attach)
            .add_observer(hello)
            .add_observer(show_ask)
            .add_observer(show_picker)
            .add_observer(close_picker)
            .add_observer(withdraw_review)
            .add_observer(diagnostics_timeout)
            .add_observer(changed_on_disk)
            .add_command(
                "auto-accept",
                "Apply edits to the editor's buffers and save them without review: \
                 /auto-accept [on|off]",
                auto_accept,
            );
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(FRONT) {
            app.add_systems(Startup, serve_stdio);
        }
    }
}

/// The systems that answer `read` from open buffers and hold `edit` and
/// `write` of open files for review, in [`GateSystems::After`] so they
/// never answer or hold a call a gate refused.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EditorSystems;

/// Whether edits to open buffers are applied and saved without review;
/// kept with the session.
#[derive(Resource, Reflect, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[reflect(Resource, Saved)]
pub struct AutoAccept(pub bool);

/// The files the editors have open, by path resolved from the
/// [`ProjectRoot`]: the buffer's text,
/// unsaved changes included, and the editor that sent it.
#[derive(Resource, Default, Debug)]
pub struct Buffers(pub HashMap<PathBuf, Buffer>);

/// An open buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    /// Its text.
    pub text: String,
    /// The editor that last sent it.
    pub client: Entity,
}

/// A selection the user sends with a message.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// The file.
    pub path: String,
    /// Its first line, from 1.
    pub start: u32,
    /// Its last line.
    pub end: u32,
    /// The selected text.
    pub text: String,
}

/// What an editor sends, one JSON object per line, tagged by `type`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FromEditor {
    /// A buffer opened or changed: its whole text.
    Buffer {
        /// The file.
        path: String,
        /// The buffer's text.
        text: String,
    },
    /// A buffer closed.
    Close {
        /// The file.
        path: String,
    },
    /// A message to the agent, with the selection if any; a line starting
    /// with `/` is a command, such as `/auto-accept`.
    Send {
        /// The text.
        text: String,
        /// The selection sent with it.
        #[serde(default)]
        selection: Option<Selection>,
    },
    /// A question about the symbol under the cursor.
    Symbol {
        /// The file.
        path: String,
        /// The line, from 1.
        line: u32,
        /// The column, from 1.
        column: u32,
        /// The symbol.
        name: String,
        /// What the user asks, if anything.
        #[serde(default)]
        question: Option<String>,
    },
    /// Talk to another agent, by its id.
    Agent {
        /// The agent's id.
        id: String,
    },
    /// Stop the agent's turn.
    Interrupt,
    /// An answer to a question or picker: a choice by position, typed
    /// text, or neither to close it.
    Pick {
        /// The question's id.
        id: u64,
        /// The choice.
        #[serde(default)]
        choice: Option<usize>,
        /// The typed text.
        #[serde(default)]
        text: Option<String>,
    },
    /// A review answered: whether each hunk was accepted, and so applied
    /// to the buffer. A missing hunk counts as rejected.
    Review {
        /// The review's id.
        id: u64,
        /// One entry per hunk, in order.
        accept: Vec<bool>,
    },
    /// The editor's diagnostics of the file a review changed.
    Diagnostics {
        /// The review's id.
        id: u64,
        /// One line per diagnostic.
        items: Vec<String>,
    },
}

/// What the agent sends an editor, one JSON object per line, tagged by
/// `type`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToEditor {
    /// On connecting: the agents, the one this editor talks to, and
    /// whether edits are auto-accepted.
    Hello {
        /// Each agent's id and name.
        agents: Vec<(String, String)>,
        /// The agent this editor talks to.
        agent: Option<String>,
        /// Whether edits are applied without review.
        auto_accept: bool,
    },
    /// An agent's whole conversation, on connecting.
    Transcript {
        /// The agent.
        agent: String,
        /// Its messages.
        messages: Vec<Message>,
    },
    /// A fragment a model streamed.
    Delta {
        /// The agent.
        agent: Option<String>,
        /// `text`, `reasoning` or `arguments`.
        kind: String,
        /// The fragment.
        text: String,
    },
    /// A message added to a conversation.
    Message {
        /// The agent.
        agent: String,
        /// The message.
        message: Message,
    },
    /// A notice.
    Notice {
        /// The agent, if any.
        agent: Option<String>,
        /// `info`, `warn` or `error`.
        level: String,
        /// The text.
        text: String,
    },
    /// A question or picker to show, answered with [`FromEditor::Pick`].
    Question {
        /// Its id.
        id: u64,
        /// The title.
        title: String,
        /// The body.
        body: String,
        /// The choices.
        choices: Vec<String>,
        /// Whether typed text answers it.
        free_text: bool,
    },
    /// A question or picker closed, answered elsewhere or withdrawn.
    Closed {
        /// Its id.
        id: u64,
    },
    /// An edit of an open buffer to review, answered with
    /// [`FromEditor::Review`]; with `auto`, applied and saved at once.
    Review {
        /// Its id.
        id: u64,
        /// The file.
        path: String,
        /// The hunks.
        hunks: Vec<Hunk>,
        /// Whether to apply and save every hunk without asking.
        auto: bool,
    },
    /// A review answered elsewhere or withdrawn.
    ReviewClosed {
        /// Its id.
        id: u64,
    },
    /// Send the diagnostics of the file a review changed.
    DiagnosticsRequest {
        /// The review's id.
        id: u64,
        /// The file.
        path: String,
    },
    /// A file changed on disk: reload it when it is opened.
    FileChanged {
        /// The file.
        path: String,
    },
    /// A line that could not be read.
    Error {
        /// Why.
        text: String,
    },
}

/// A run of changed lines.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    /// Where its old lines start in the buffer, from 0.
    pub start: usize,
    /// The lines it replaces, with their line ends.
    pub old: Vec<String>,
    /// The lines it puts there.
    pub new: Vec<String>,
}

/// One end of a connection to an editor: send it lines with `send`, read
/// its lines from `receive`.
pub struct EditorLink {
    /// The editor's lines to the agent; dropping it disconnects.
    pub send: UnboundedSender<String>,
    /// The agent's lines to the editor.
    pub receive: mpsc::Receiver<String>,
}

/// A connected editor.
#[derive(Component, Debug)]
pub struct EditorClient {
    out: mpsc::Sender<String>,
    /// The agent it talks to.
    pub agent: Option<Entity>,
}

impl EditorClient {
    fn send(&self, message: &ToEditor) {
        send(&self.out, message);
    }
}

/// Sends `message` as a line on `out`, redacted as the session log is.
fn send(out: &mpsc::Sender<String>, message: &ToEditor) {
    if let Ok(line) = serde_json::to_string(message) {
        out.send(redact::json_text(&line).into_owned()).ok();
    }
}

/// On the editor served over stdio: when it goes, the app exits.
#[derive(Component)]
struct Stdio;

/// Lines the editors sent this frame, in order.
#[derive(Resource, Default)]
struct Inbox(Vec<(Entity, Option<String>)>);

/// An open `editor_review` effect holding an `edit` or `write` call.
#[derive(Component, Debug)]
pub struct EditReview {
    /// The call.
    pub call: Entity,
    /// The file.
    pub path: PathBuf,
    /// The buffer's text before the edit.
    pub before: String,
    /// The hunks offered.
    pub hunks: Vec<Hunk>,
    /// Whether it was applied without review.
    pub auto: bool,
}

/// A hold on an applied edit's call while the editor's diagnostics come.
#[derive(Component, Debug)]
struct DiagnosticsWait {
    review: u64,
    call: Entity,
    result: String,
}

/// Connects an editor: every line it sends is read as a [`FromEditor`],
/// and it is sent every [`ToEditor`]. It answers questions about every
/// agent while it is connected.
pub fn connect(world: &mut World) -> EditorLink {
    connect_client(world).1
}

/// [`connect`], with the editor's entity.
fn connect_client(world: &mut World) -> (Entity, EditorLink) {
    let (send, lines) = unbounded::<String>();
    let (out, receive) = mpsc::channel();
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let lines = lines.map(Some).chain(futures::stream::once(async { None }));
    let agent = world.get_resource::<ShownAgent>().and_then(|shown| shown.0);
    let client = world.spawn((
        EditorClient { out, agent },
        Answerer::default(),
        Running::stream(pool, &wake, lines),
    ));
    (client.id(), EditorLink { send, receive })
}

/// Connects the editor that reads `reader` and writes `writer`, on
/// threads of their own; returns its entity.
fn serve(
    world: &mut World,
    reader: impl BufRead + Send + 'static,
    mut writer: impl std::io::Write + Send + 'static,
) -> Entity {
    let (client, EditorLink { send, receive }) = connect_client(world);
    std::thread::spawn(move || {
        for line in reader.lines().map_while(Result::ok) {
            if send.unbounded_send(line).is_err() {
                break;
            }
        }
    });
    std::thread::spawn(move || {
        for line in receive {
            if writeln!(writer, "{line}")
                .and_then(|()| writer.flush())
                .is_err()
            {
                break;
            }
        }
    });
    client
}

/// On the listener of the session's `editor.sock`.
#[derive(Component)]
struct EditorSocket;

/// Serves the editor that started `rig --editor` over stdin and stdout,
/// and other editor windows over the session's `editor.sock`.
fn serve_stdio(world: &mut World) {
    let stdin = std::io::BufReader::new(std::io::stdin());
    let client = serve(world, stdin, std::io::stdout());
    world.entity_mut(client).insert(Stdio);
    let Some(paths) = world.get_resource::<SessionPaths>() else {
        return;
    };
    let socket = paths.0.path().join("editor.sock");
    std::fs::remove_file(&socket).ok();
    let Ok(listener) = UnixListener::bind(&socket) else {
        return;
    };
    let (accepted, streams) = unbounded::<UnixStream>();
    std::thread::spawn(move || {
        for stream in listener.incoming().map_while(Result::ok) {
            if accepted.unbounded_send(stream).is_err() {
                break;
            }
        }
    });
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    world.spawn((EditorSocket, Running::stream(pool, &wake, streams)));
}

/// Connects each editor window that opens the session's `editor.sock`.
fn attach(
    accepted: On<Insert<Done<UnixStream>>>,
    streams: Query<&Done<UnixStream>, With<EditorSocket>>,
    mut commands: Commands,
) {
    let Ok(Done(stream)) = streams.get(accepted.entity) else {
        return;
    };
    if let Ok(reader) = stream.try_clone() {
        let writer = stream.try_clone();
        commands.queue(move |world: &mut World| {
            if let Ok(writer) = writer {
                serve(world, std::io::BufReader::new(reader), writer);
            }
        });
    }
}

/// Keeps each line an editor sent, or its going, for [`receive`].
fn take_line(
    inserted: On<Insert<Done<Option<String>>>>,
    lines: Query<&Done<Option<String>>, With<EditorClient>>,
    mut inbox: ResMut<Inbox>,
) {
    if let Ok(Done(line)) = lines.get(inserted.entity) {
        inbox.0.push((inserted.entity, line.clone()));
    }
}

/// Greets an editor as it connects with the agents and their
/// conversations, so a reopened editor shows the session.
fn hello(
    added: On<Add<EditorClient>>,
    clients: Query<&EditorClient>,
    agents: Query<(&AgentId, &Name, &Conversation)>,
    auto: Res<AutoAccept>,
) {
    let Ok(client) = clients.get(added.entity) else {
        return;
    };
    let agent = client
        .agent
        .and_then(|agent| agents.get(agent).ok())
        .map(|(id, ..)| id.0.clone());
    let list = agents
        .iter()
        .map(|(id, name, _)| (id.0.clone(), name.to_string()))
        .collect();
    client.send(&ToEditor::Hello {
        agents: list,
        agent,
        auto_accept: auto.0,
    });
    for (id, _, conversation) in &agents {
        client.send(&ToEditor::Transcript {
            agent: id.0.clone(),
            messages: conversation
                .messages()
                .iter()
                .map(|message| (**message).clone())
                .collect(),
        });
    }
}

/// Sends `message` to every editor.
fn broadcast(clients: &Query<&EditorClient>, message: &ToEditor) {
    for client in clients {
        client.send(message);
    }
}

/// The agent's id, for the editors.
fn id_of(ids: &Query<&AgentId>, agent: Entity) -> Option<String> {
    ids.get(agent).ok().map(|id| id.0.clone())
}

/// The review's id, for the editors.
fn id(entity: Entity) -> u64 {
    entity.to_bits()
}

/// Carries out what the editors sent, in order: buffers, messages,
/// answers to questions and reviews, diagnostics.
#[allow(clippy::too_many_arguments)]
fn receive(
    mut inbox: ResMut<Inbox>,
    mut clients: Query<(&mut EditorClient, Has<Stdio>)>,
    mut buffers: ResMut<Buffers>,
    agents: Query<(Entity, &AgentId)>,
    primaries: MainAgentQuery,
    mut reviews: Query<(&EditReview, &mut Decision), (PluginEffects, Without<Outcome>)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    waits: Query<(Entity, &DiagnosticsWait)>,
    (root, mut shown): (Res<ProjectRoot>, ResMut<ShownAgent>),
    mut commands: Commands,
) {
    for (from, line) in std::mem::take(&mut inbox.0) {
        let Some(line) = line else {
            buffers.0.retain(|_, buffer| buffer.client != from);
            if clients.get(from).is_ok_and(|(_, stdio)| stdio) {
                commands.trigger(Exit(ExitReason::Quit));
            }
            commands.entity(from).try_despawn();
            continue;
        };
        let Ok((client, _)) = clients.get(from) else {
            continue;
        };
        let reply = client.out.clone();
        let reply = |message: &ToEditor| send(&reply, message);
        let agent = client.agent.or_else(|| main_agent(&primaries));
        let message = match serde_json::from_str::<FromEditor>(&line) {
            Ok(message) => message,
            Err(error) => {
                let text = format!("unreadable line: {error}");
                reply(&ToEditor::Error { text });
                continue;
            }
        };
        match message {
            FromEditor::Buffer { path, text } => {
                let buffer = Buffer { text, client: from };
                buffers.0.insert(resolve(Some(&root.0), &path), buffer);
            }
            FromEditor::Close { path } => {
                buffers.0.remove(&resolve(Some(&root.0), &path));
            }
            FromEditor::Send { text, selection } => {
                let Some(agent) = agent else { continue };
                let mut incoming = Incoming::new(text, DeliveryMode::Queue);
                if let Some(selection) = selection {
                    let label = format!("{}:{}-{}", selection.path, selection.start, selection.end);
                    let content = UserContent::text(format!(
                        "The user's selection, {label}:\n{}",
                        selection.text
                    ));
                    let selection = Attachment { label, content };
                    incoming.parts.insert(0, MessagePart::Attachment(selection));
                }
                commands.spawn((incoming, ForAgent(agent)));
            }
            FromEditor::Symbol {
                path,
                line,
                column,
                name,
                question,
            } => {
                let Some(agent) = agent else { continue };
                let question = question
                    .unwrap_or_else(|| "What is it, and where is it defined and used?".to_owned());
                let text = format!("About `{name}` at {path}:{line}:{column}: {question}");
                commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue));
            }
            FromEditor::Agent { id } => {
                let found = agents.iter().find(|(_, agent)| agent.0 == id);
                match (found, clients.get_mut(from)) {
                    (Some((agent, _)), Ok((mut client, _))) => {
                        client.agent = Some(agent);
                        shown.0 = Some(agent);
                    }
                    _ => reply(&ToEditor::Error {
                        text: format!("no agent {id}"),
                    }),
                }
            }
            FromEditor::Interrupt => {
                if let Some(agent) = agent {
                    commands.trigger(StopTurn::new(agent));
                }
            }
            FromEditor::Pick { id, choice, text } => {
                let Some(entity) = Entity::try_from_bits(id) else {
                    continue;
                };
                let chosen = choice.map(Chosen::item).or(text.map(Chosen::text));
                // A picker's command runs for the shown agent: this window's.
                if agent.is_some() {
                    shown.0 = agent;
                }
                commands.trigger(Pick::person(entity, chosen));
            }
            FromEditor::Review { id, accept } => {
                let review = Entity::try_from_bits(id).filter(|review| reviews.contains(*review));
                let Some((review, Ok((edit, mut decision)))) =
                    review.map(|review| (review, reviews.get_mut(review)))
                else {
                    // Answered by another editor first, or withdrawn.
                    reply(&ToEditor::ReviewClosed { id });
                    continue;
                };
                if !decision.decide(
                    Decider::Person,
                    Outcome::custom(serde_json::json!({ "accept": accept })),
                ) {
                    reply(&ToEditor::ReviewClosed { id });
                    continue;
                }
                commands.entity(review).remove::<HoldOf>();
                for (other, _) in &clients {
                    other.send(&ToEditor::ReviewClosed { id });
                }
                let accepted = |at: usize| accept.get(at).copied().unwrap_or(false);
                let after = apply_hunks(&edit.before, &edit.hunks, accepted);
                if let Some(buffer) = buffers.0.get_mut(&edit.path) {
                    buffer.text.clone_from(&after);
                }
                let result = summary(edit, accepted);
                let path = edit.path.display().to_string();
                if after == edit.before {
                    if let Ok(mut call) = calls.get_mut(edit.call) {
                        call.decide(
                            Decider::plugin("fh-editor-front"),
                            ToolResult::success(result.into()),
                        );
                    }
                    continue;
                }
                for (other, _) in &clients {
                    let request = ToEditor::DiagnosticsRequest {
                        id,
                        path: path.clone(),
                    };
                    other.send(&request);
                }
                commands.spawn((
                    Name::new(format!("waiting for the editor's diagnostics of {path}")),
                    DiagnosticsWait {
                        review: id,
                        call: edit.call,
                        result,
                    },
                    HoldOf(edit.call),
                    WakeAt::after(DIAGNOSTICS_WAIT),
                ));
            }
            FromEditor::Diagnostics { id, items } => {
                for (wait, waiting) in waits.iter().filter(|(_, wait)| wait.review == id) {
                    let mut result = waiting.result.clone();
                    if items.is_empty() {
                        result.push_str("\nThe editor reports no diagnostics.");
                    } else {
                        result.push_str("\nThe editor's diagnostics:");
                        for item in &items {
                            let _ = write!(result, "\n- {item}");
                        }
                    }
                    if let Ok(mut call) = calls.get_mut(waiting.call) {
                        call.decide(
                            Decider::plugin("fh-editor-front"),
                            ToolResult::success(result.into()),
                        );
                    }
                    commands.entity(wait).try_despawn();
                }
            }
        }
    }
}

/// An applied edit whose diagnostics did not come in time is answered
/// without them.
fn diagnostics_timeout(
    due: On<Add<Due>>,
    waits: Query<&DiagnosticsWait>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok(wait) = waits.get(due.entity) else {
        return;
    };
    if let Ok(mut call) = calls.get_mut(wait.call) {
        let result = format!(
            "{}\nThe editor sent no diagnostics within {} s.",
            wait.result,
            DIAGNOSTICS_WAIT.as_secs()
        );
        call.decide(
            Decider::plugin("fh-editor-front"),
            ToolResult::success(result.into()),
        );
    }
    commands.entity(due.entity).try_despawn();
}

/// What became of a review, for the model: `applied`, `rejected by user`
/// or `partially applied: hunks 2,3 rejected`.
fn summary(review: &EditReview, accepted: impl Fn(usize) -> bool) -> String {
    let rejected: Vec<String> = (0..review.hunks.len())
        .filter(|&at| !accepted(at))
        .map(|at| (at + 1).to_string())
        .collect();
    let path = review.path.display();
    let state = match rejected.len() {
        0 => "applied".to_owned(),
        n if n == review.hunks.len() => "rejected by user".to_owned(),
        1 => format!("partially applied: hunk {} rejected", rejected.join(",")),
        _ => format!("partially applied: hunks {} rejected", rejected.join(",")),
    };
    let saved = if review.auto {
        "applied to the editor's buffer and saved"
    } else {
        "applied to the editor's unsaved buffer, not to the file on disk"
    };
    format!("{state}\n{path} is open in the editor: what was accepted is {saved}.")
}

/// The lines of `text`, each with its line end.
fn lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

/// The hunks turning `before` into `after`.
pub fn diff(before: &str, after: &str) -> Vec<Hunk> {
    let (old, new) = (lines(before), lines(after));
    let take = |lines: &[&str], range: std::ops::Range<usize>| -> Vec<String> {
        let taken = lines.get(range).unwrap_or_default();
        taken.iter().map(|line| (*line).to_owned()).collect()
    };
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut last_end = None;
    for op in TextDiff::from_lines(before, after).ops() {
        let (tag, old_range, new_range) = op.as_tag_tuple();
        if tag == DiffTag::Equal {
            continue;
        }
        let joins = last_end == Some(old_range.start);
        last_end = Some(old_range.end);
        match hunks.last_mut() {
            Some(hunk) if joins => {
                hunk.old.extend(take(&old, old_range));
                hunk.new.extend(take(&new, new_range));
            }
            _ => hunks.push(Hunk {
                start: old_range.start,
                old: take(&old, old_range),
                new: take(&new, new_range),
            }),
        }
    }
    hunks
}

/// `before` with the hunks `accepted` says applied.
pub fn apply_hunks(before: &str, hunks: &[Hunk], accepted: impl Fn(usize) -> bool) -> String {
    let old = lines(before);
    let mut out = String::with_capacity(before.len());
    let mut at = 0;
    for (index, hunk) in hunks.iter().enumerate() {
        old.get(at..hunk.start)
            .unwrap_or_default()
            .iter()
            .for_each(|line| out.push_str(line));
        let lines = if accepted(index) {
            &hunk.new
        } else {
            &hunk.old
        };
        lines.iter().for_each(|line| out.push_str(line));
        at = hunk.start + hunk.old.len();
    }
    old.get(at..)
        .unwrap_or_default()
        .iter()
        .for_each(|line| out.push_str(line));
    out
}

/// The one path a call's footprint names, unless the file tools refuse it
/// as outside the agent's [`WorkDir`] ([`place`]): the tool's own refusal
/// then stands, as its handler runs.
fn one_path<'a>(touches: &'a Touches, dir: Option<&WorkDir>) -> Option<&'a Path> {
    let path = match touches {
        Touches::Reads(Scope::Paths(paths)) | Touches::Writes(Scope::Paths(paths)) => {
            match paths.as_slice() {
                [path] => path,
                _ => return None,
            }
        }
        _ => return None,
    };
    let placed = dir.is_none_or(|dir| place(Some(&dir.0), true, &path.to_string_lossy()).is_ok());
    placed.then_some(path.as_path())
}

/// Answers each `read` of an open file that no gate decided from its
/// buffer, unsaved changes included, as `read` shows a file.
fn from_buffers(
    mut calls: Query<
        (&ToolRequest, &ForAgent, &Footprint, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    dirs: Query<&WorkDir>,
    buffers: Res<Buffers>,
) {
    for (request, &ForAgent(agent), Footprint(touches), mut decision) in &mut calls {
        let function = &request.call.function;
        if function.name.as_str() != Read::NAME {
            continue;
        }
        let buffer = one_path(touches, dirs.get(agent).ok()).and_then(|path| buffers.0.get(path));
        let args =
            serde_json::from_value::<ReadArgs>(serde_json::Value::from(function.arguments.clone()));
        if let (Some(buffer), Ok(args)) = (buffer, args) {
            decision.decide(
                Decider::plugin("fh-editor-front"),
                ToolResult::success(args.show(&buffer.text).into()),
            );
        }
    }
}

/// Holds each `edit` and `write` of an open file with a review the
/// editors answer, unless a gate decided it already or no editor is
/// connected. A call a gate holds, such as one waiting for the user's
/// approval, is not [`ReadyToStart`], so it is reviewed only once the gate
/// lets it go.
fn review_edits(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &Footprint, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    dirs: Query<&WorkDir>,
    buffers: Res<Buffers>,
    auto: Res<AutoAccept>,
    clients: Query<&EditorClient>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), Footprint(touches), mut decision) in &mut calls {
        let function = &request.call.function;
        let name = function.name.as_str();
        if decision.is_decided() || clients.is_empty() || !matches!(name, Edit::NAME | Write::NAME)
        {
            continue;
        }
        let Some((path, buffer)) = one_path(touches, dirs.get(agent).ok())
            .and_then(|path| Some((path, buffers.0.get(path)?)))
        else {
            continue;
        };
        let shown = path.display().to_string();
        let after = if name == Edit::NAME {
            serde_json::from_value::<EditArgs>(serde_json::Value::from(function.arguments.clone()))
                .map_err(ToolExecutionError::from_error)
                .and_then(|args| rig_tools::edit::apply(&shown, &buffer.text, &args.edits))
                .map(|applied| applied.text)
        } else {
            serde_json::from_value::<WriteArgs>(serde_json::Value::from(function.arguments.clone()))
                .map_err(ToolExecutionError::from_error)
                .map(|args| args.content)
        };
        let after = match after {
            Ok(after) => after,
            Err(error) => {
                decision.decide(
                    Decider::plugin("fh-editor-front"),
                    ToolResult::failed(error),
                );
                continue;
            }
        };
        let hunks = diff(&buffer.text, &after);
        if hunks.is_empty() {
            let text = format!("{shown} is open in the editor and already reads so.");
            decision.decide(
                Decider::plugin("fh-editor-front"),
                ToolResult::success(text.into()),
            );
            continue;
        }
        let kind = EffectKind::Custom {
            kind: "editor_review".into(),
            payload: serde_json::json!({ "path": shown, "hunks": hunks, "auto": auto.0 }),
        };
        let review = commands
            .spawn((
                EffectRequest {
                    kind,
                    handler: None,
                },
                EditReview {
                    call,
                    path: path.to_owned(),
                    before: buffer.text.clone(),
                    hunks: hunks.clone(),
                    auto: auto.0,
                },
                Name::new(format!("waiting for the editor's review of {shown}")),
                ForAgent(agent),
                HoldOf(call),
            ))
            .id();
        broadcast(
            &clients,
            &ToEditor::Review {
                id: id(review),
                path: shown,
                hunks,
                auto: auto.0,
            },
        );
    }
}

/// A review withdrawn before any editor answered it, such as by an
/// interrupt, closes in every editor.
fn withdraw_review(
    removed: On<Remove<EditReview>>,
    reviews: Query<&Decision>,
    clients: Query<&EditorClient>,
) {
    if reviews.get(removed.entity).is_ok_and(Decision::is_decided) {
        return;
    }
    let id = id(removed.entity);
    broadcast(&clients, &ToEditor::ReviewClosed { id });
}

/// Tells the editors of each file a tool changed on disk.
fn changed_on_disk(
    settled: On<Add<Settled>>,
    calls: Query<&Published, With<ToolRequest>>,
    clients: Query<&EditorClient>,
) {
    let Ok(Published(facts)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(FilesChanged(changes))) = facts.get::<FilesChanged>() else {
        return;
    };
    for change in changes {
        let path = change.path.clone();
        broadcast(&clients, &ToEditor::FileChanged { path });
    }
}

/// Shows each question in the editors.
fn show_ask(added: On<Add<Ask>>, asks: Query<&Ask>, clients: Query<&EditorClient>) {
    if let Ok(ask) = asks.get(added.entity) {
        let question = ToEditor::Question {
            id: id(added.entity),
            title: ask.title.clone(),
            body: ask.body.clone(),
            choices: ask
                .choices
                .iter()
                .map(|choice| choice.label.clone())
                .collect(),
            free_text: ask.free_text,
        };
        broadcast(&clients, &question);
    }
}

/// Shows each picker in the editors, again when it changes.
fn show_picker(added: On<Insert<Picker>>, pickers: Query<&Picker>, clients: Query<&EditorClient>) {
    if let Ok(picker) = pickers.get(added.entity) {
        let question = ToEditor::Question {
            id: id(added.entity),
            title: picker.title.clone(),
            body: picker.body.clone(),
            choices: picker.items.iter().map(|item| item.label.clone()).collect(),
            free_text: picker.free_text,
        };
        broadcast(&clients, &question);
    }
}

/// Closes a picker in the editors as it goes.
fn close_picker(removed: On<Remove<Picker>>, clients: Query<&EditorClient>) {
    let id = id(removed.entity);
    broadcast(&clients, &ToEditor::Closed { id });
}

/// Closes each question in the editors once answered, wherever.
fn close_questions(
    asks: Query<Entity, (With<Ask>, Added<Outcome>)>,
    clients: Query<&EditorClient>,
) {
    for ask in &asks {
        broadcast(&clients, &ToEditor::Closed { id: id(ask) });
    }
}

/// Sends the editors what the models stream, each message committed and
/// each notice.
fn forward(
    mut deltas: MessageReader<StreamDelta>,
    mut committed: MessageReader<ConversationChanged>,
    mut notices: MessageReader<Notice>,
    ids: Query<&AgentId>,
    clients: Query<&EditorClient>,
) {
    for delta in deltas.read() {
        let (kind, text) = match &delta.event {
            StreamEvent::Text { text, .. } => ("text", text),
            StreamEvent::Reasoning { text, .. } => ("reasoning", text),
            StreamEvent::Arguments { json, .. } => ("arguments", json),
            _ => continue,
        };
        let message = ToEditor::Delta {
            agent: delta.agent.and_then(|agent| id_of(&ids, agent)),
            kind: kind.to_owned(),
            text: text.clone(),
        };
        broadcast(&clients, &message);
    }
    for committed in committed.read() {
        if let Change::Push { entries } = &committed.change.change
            && let Some(agent) = id_of(&ids, committed.agent)
        {
            for entry in entries {
                let (agent, message) = (agent.clone(), Message::clone(&entry.message));
                broadcast(&clients, &ToEditor::Message { agent, message });
            }
        }
    }
    for notice in notices.read() {
        let message = ToEditor::Notice {
            agent: notice.agent.and_then(|agent| id_of(&ids, agent)),
            level: format!("{:?}", notice.level).to_lowercase(),
            text: notice.text.clone(),
        };
        broadcast(&clients, &message);
    }
}

/// `/auto-accept [on|off]`: whether edits to open buffers are applied and
/// saved without review; without an argument, toggles it.
fn auto_accept(
    In(args): In<CommandArgs>,
    mut auto: ResMut<AutoAccept>,
    mut notices: MessageWriter<Notice>,
) {
    let on = match args.args.trim() {
        "on" => true,
        "off" => false,
        "" => !auto.0,
        _ => {
            notices.write(Notice::error(
                args.agent,
                "Use /auto-accept, /auto-accept on or off.",
            ));
            return;
        }
    };
    auto.0 = on;
    let state = if on { "on" } else { "off" };
    notices.write(Notice::info(args.agent, format!("Auto-accept is {state}.")));
}

#[cfg(test)]
mod tests;
