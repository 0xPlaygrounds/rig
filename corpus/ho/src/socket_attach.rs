//! Plugin corpus (held-out): H23: Attach to a running session over a socket
//! (spec `H23-socket-attach.md`).
//!
//! A running session listens on `RIG_HOME/sockets/<session-id>.sock` (mode
//! 0600, in a directory of mode 0700), removed on exit; sockets no session
//! listens on are removed at startup. The protocol is JSON lines.
//!
//! - **Server to client:** on connect, `backlog` (the focused agent's last
//!   50 messages), then live events: `delivery`, `text` and `reasoning`
//!   (streamed), `tool_start`, `tool_end`, `notice`, `picker` (a plugin's
//!   [`Picker`] or [`Ask`]), `closed` (answered in any front), `turn_end`,
//!   `error` (to the one client whose line failed) and, on the session's
//!   exit, `ended` ("session ended").
//! - **Client to server:** `{"text": "…"}`, a line the user typed, to the
//!   focused agent, or `@<agent name> …` to that one; a `/` line runs as a
//!   command in the session. `{"pick": id, "choice": n | "text" | null}`
//!   answers an open picker; the first answer from any front wins.
//!
//! A client more than 1 MB behind is dropped with a notice: the app queues
//! each line and never waits on a client.
//!
//! `rig --attach` makes the run a thin client of the newest live session,
//! or of `--attach-session=ID`: it prints what arrives as plain lines and
//! sends what is typed; while a picker is open, the typed line answers it.
//! Ctrl-D detaches and leaves the session running.

use std::sync::Arc;

use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::AppExit;
use rig_harness::prelude::*;
use rig_harness::rig_ecs::StopTurnsOnExit;
use serde::Deserialize;
use serde_json::{Value, json};

mod client;
mod server;

use server::{Client, Listening, Request, Sent};

/// The option that makes the run a thin client.
pub const ATTACH: &str = "attach";

/// The option naming the session a thin client attaches to.
pub const ATTACH_SESSION: &str = "attach-session";

/// How many of the focused agent's messages a client gets on connect.
const BACKLOG: usize = 50;

/// What a client is told when the session exits.
const ENDED: &str = "session ended";

/// The session's socket, and the thin client of `rig --attach`.
#[derive(Default)]
pub struct SocketAttachPlugin;

impl Plugin for SocketAttachPlugin {
    fn build(&self, app: &mut App) {
        app.add_front(
            Front::new(ATTACH)
                .on_flag(
                    ATTACH,
                    "Attach to a running session: watch it and type into it",
                )
                .sessionless(),
        )
        .add_flag(Flag::new(ATTACH_SESSION, "The session `--attach` attaches to").value("SESSION"));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(ATTACH) {
            client::build(app);
            return;
        }
        app.init_resource::<Server>()
            .add_systems(Startup, listen)
            .add_systems(PostUpdate, (stream_text, relay_notices))
            // After the turns a quit stops, so clients see them end.
            .add_systems(
                Last,
                close.after(StopTurnsOnExit).run_if(on_message::<AppExit>),
            )
            .add_observer(handle)
            .add_observer(delivered)
            .add_observer(turn_ended)
            .add_observer(tool_started)
            .add_observer(tool_ended)
            .add_observer(asked)
            .add_observer(ask_closed)
            .add_observer(picker_opened)
            .add_observer(picker_closed);
    }
}

/// The socket, its clients, and how far they got in what model calls
/// streamed.
#[derive(Resource, Default)]
struct Server {
    listening: Option<Listening>,
    clients: Vec<Client>,
    deltas: MessageCursor<StreamDelta>,
    next: u64,
}

/// What a connection hands the app, on the entity reading the socket.
#[derive(Component)]
struct Socket;

/// Listens on the session's socket.
fn listen(
    home: Res<RigHome>,
    paths: Option<Res<SessionPaths>>,
    wake: Res<Wake>,
    mut server: ResMut<Server>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(id) = paths.and_then(|paths| paths.id()) else {
        notices.write(Notice::error(
            None,
            "attach: this run has no session to serve",
        ));
        return;
    };
    match server::listen(home.0.root(), id.as_str()) {
        Ok((listening, requests)) => {
            server.listening = Some(listening);
            let pool = IoTaskPool::get_or_init(TaskPool::default);
            commands.spawn((Socket, ShowsPickers, Running::stream(pool, &wake, requests)));
        }
        Err(why) => {
            notices.write(Notice::error(None, format!("attach: cannot listen: {why}")));
        }
    }
}

/// Sends events to every client, dropping those that fell behind. Every
/// event first sends the fragments model calls streamed before it, so the
/// stream keeps the order things happened in: the fragments are messages,
/// the rest observers.
#[derive(SystemParam)]
struct Events<'w, 's> {
    server: ResMut<'w, Server>,
    deltas: Res<'w, Messages<StreamDelta>>,
    names: Query<'w, 's, &'static Name>,
    commands: Commands<'w, 's>,
}

impl Events<'_, '_> {
    /// The name of `agent`.
    fn name(&self, agent: Entity) -> Option<String> {
        self.names
            .get(agent)
            .ok()
            .map(|name| name.as_str().to_owned())
    }

    /// Sends `event` to every client.
    fn send(&mut self, event: &Value) {
        self.flush();
        self.broadcast(event);
    }

    /// Sends the fragments streamed since the last event.
    fn flush(&mut self) {
        let fragments: Vec<Value> = {
            let Server { deltas, .. } = &mut *self.server;
            deltas
                .read(&self.deltas)
                .filter_map(|delta| {
                    let (kind, text) = match &delta.event {
                        StreamEvent::Text { text, .. } => ("text", text),
                        StreamEvent::Reasoning { text, .. } => ("reasoning", text),
                        _ => return None,
                    };
                    let agent = delta.agent.and_then(|agent| self.names.get(agent).ok());
                    Some(json!({ "type": kind, "agent": agent.map(Name::as_str), "text": text }))
                })
                .collect()
        };
        for fragment in &fragments {
            self.broadcast(fragment);
        }
    }

    /// Queues `event` for every client; drops, with a notice, those more
    /// than 1 MB behind, and forgets those gone.
    fn broadcast(&mut self, event: &Value) {
        if self.server.clients.is_empty() {
            return;
        }
        let line: Arc<str> = format!("{event}\n").into();
        let mut behind = Vec::new();
        self.server
            .clients
            .retain(|client| match client.send(&line) {
                Sent::Queued => true,
                Sent::Behind => {
                    behind.push(client.id);
                    false
                }
                Sent::Gone => false,
            });
        for id in behind {
            let text = format!("Attached client {id} fell 1 MB behind and was dropped.");
            self.commands.write_message(Notice::info(None, text));
        }
    }
}

/// A line a client sends.
#[derive(Deserialize)]
#[serde(untagged)]
enum Line {
    /// What the user typed.
    Text { text: String },
    /// An answer to the picker or question `pick`: a choice's position,
    /// typed text, or `null` to close it.
    Pick { pick: u64, choice: Value },
}

/// The agents, by entity and name.
type Agents<'w, 's> = Query<'w, 's, (Entity, &'static Name), With<Agent>>;

/// The agent the user looks at, which a client's line goes to.
#[derive(SystemParam)]
struct Focused<'w, 's> {
    shown: Option<Res<'w, ShownAgent>>,
    agents: MainAgentQuery<'w, 's>,
}

impl Focused<'_, '_> {
    fn agent(&self) -> Option<Entity> {
        self.shown
            .as_ref()
            .and_then(|shown| shown.0)
            .or_else(|| main_agent(&self.agents))
    }
}

/// What answering a picker reads: the open questions and the pickers.
type Choosable<'w, 's> = (
    Query<'w, 's, &'static Ask, (Without<Outcome>, Without<Settled>)>,
    Query<'w, 's, &'static Picker>,
);

/// Handles what the socket hands over: a new client gets the backlog, a
/// line is sent to its agent or answers a picker.
fn handle(
    handed: On<Insert<Done<Request>>>,
    requests: Query<&Done<Request>>,
    focused: Focused,
    agents: Agents,
    conversations: Query<&Conversation>,
    choosable: Choosable,
    mut events: Events,
) {
    let Ok(Done(request)) = requests.get(handed.entity) else {
        return;
    };
    match request {
        Request::Connect(stream) => {
            // A new client gets what streams from now on.
            events.flush();
            let focus = focused.agent();
            let backlog = backlog(
                focus.and_then(|agent| events.name(agent)),
                focus.and_then(|agent| conversations.get(agent).ok()),
            );
            let id = events.server.next;
            events.server.next += 1;
            let opened = events.server.listening.as_ref().map(|listening| {
                let stream = stream.try_clone()?;
                Client::open(id, stream, listening.requests.clone())
            });
            if let Some(Ok(client)) = opened
                && client.send(&format!("{backlog}\n").into()) == Sent::Queued
            {
                events.server.clients.push(client);
            }
        }
        Request::Line { client, line } => {
            let reply = match serde_json::from_str::<Line>(line) {
                Ok(Line::Text { text }) => deliver(text, &focused, &agents, &mut events.commands),
                Ok(Line::Pick { pick, choice }) => {
                    choose(pick, &choice, &choosable, &mut events.commands)
                }
                Err(_) => Err(r#"expected {"text": …} or {"pick": id, "choice": …}"#.to_owned()),
            };
            if let Err(why) = reply
                && let Some(client) = events.server.clients.iter().find(|c| c.id == *client)
            {
                client.send(&format!("{}\n", json!({ "type": "error", "text": why })).into());
            }
        }
        Request::Gone(client) => events.server.clients.retain(|c| c.id != *client),
    }
}

/// The `backlog` event: the last [`BACKLOG`] messages of the focused agent.
fn backlog(agent: Option<String>, conversation: Option<&Conversation>) -> Value {
    let messages = conversation.map_or(&[][..], |conversation| conversation.messages());
    let first = messages.len().saturating_sub(BACKLOG);
    let items: Vec<Value> = messages
        .iter()
        .skip(first)
        .map(|message| item(message))
        .collect();
    json!({ "type": "backlog", "agent": agent, "items": items })
}

/// A message as a client shows it: who wrote it and its text.
fn item(message: &message::Message) -> Value {
    match message {
        message::Message::System { content } => json!({ "role": "system", "text": content }),
        message::Message::User { content } => {
            let mut texts = Vec::new();
            let mut tools = Vec::new();
            for part in content {
                match part {
                    UserContent::Text(text) => texts.push(text.text.clone()),
                    UserContent::ToolResult(result) => tools.push(result.name.as_str().to_owned()),
                    _ => {}
                }
            }
            if texts.is_empty() && !tools.is_empty() {
                json!({ "role": "tool", "text": format!("results of {}", tools.join(", ")) })
            } else {
                json!({ "role": "user", "text": texts.join("\n") })
            }
        }
        message::Message::Assistant(reply) => {
            let calls: Vec<&str> = reply
                .tool_calls()
                .map(|call| call.function.name.as_str())
                .collect();
            json!({
                "role": "assistant",
                "text": reply.text(),
                "tools": calls,
            })
        }
    }
}

/// Sends `text` as the user's to the focused agent, or to the agent
/// `@name` names.
fn deliver(
    text: String,
    focused: &Focused,
    agents: &Agents,
    commands: &mut Commands,
) -> Result<(), String> {
    let named = text.strip_prefix('@').and_then(|rest| {
        let (name, text) = rest.split_once(char::is_whitespace)?;
        Some((name.to_owned(), text.trim_start().to_owned()))
    });
    let (agent, text) = match named {
        Some((name, text)) => {
            let agent = agents.iter().find(|(_, agent)| agent.as_str() == name);
            let (agent, _) = agent.ok_or_else(|| format!("no agent named {name}"))?;
            (agent, text)
        }
        None => (focused.agent().ok_or("no agent to send to")?, text),
    };
    commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    Ok(())
}

/// Answers the open picker or question `id` with `choice`.
fn choose(
    id: u64,
    choice: &Value,
    (asks, pickers): &Choosable,
    commands: &mut Commands,
) -> Result<(), String> {
    let open = Entity::try_from_bits(id).and_then(|entity| {
        if let Ok(ask) = asks.get(entity) {
            return Some((entity, ask.choices.len(), ask.free_text));
        }
        let picker = pickers.get(entity).ok()?;
        Some((entity, picker.items.len(), picker.free_text))
    });
    let (entity, count, free_text) = open.ok_or_else(|| format!("no open picker {id}"))?;
    let index = choice
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .filter(|&index| index < count);
    let chosen = match (index, choice) {
        (Some(index), _) => Some(Chosen::item(index)),
        (None, Value::String(text)) if free_text => Some(Chosen::text(text.clone())),
        (None, Value::Null) => None,
        _ => return Err(format!("no such choice in picker {id}")),
    };
    commands.trigger(Pick::person(entity, chosen));
    Ok(())
}

fn delivered(
    started: On<Add<Started>>,
    messages: Query<(&Incoming, &ForAgent)>,
    mut events: Events,
) {
    if let Ok((message, &ForAgent(agent))) = messages.get(started.entity) {
        let event = json!({
            "type": "delivery",
            "agent": events.name(agent),
            "author": message.author,
            "text": message.text(),
        });
        events.send(&event);
    }
}

fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut events: Events,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    // The answer streamed already; a large one is not sent twice.
    let (how, why) = match outcome {
        TurnOutcome::Answered(_) => ("answered", None),
        TurnOutcome::Failed(why) => ("failed", Some(why.clone())),
        TurnOutcome::Stopped(why) => ("stopped", why.clone()),
    };
    let event =
        json!({ "type": "turn_end", "agent": events.name(agent), "outcome": how, "reason": why });
    events.send(&event);
}

fn tool_started(
    started: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    mut events: Events,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(started.entity) {
        let event = json!({
            "type": "tool_start",
            "agent": events.name(agent),
            "id": request.call.id,
            "name": request.call.function.name.as_str(),
            "arguments": request.call.function.arguments,
        });
        events.send(&event);
    }
}

fn tool_ended(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, &Outcome)>,
    mut events: Events,
) {
    if let Ok((request, &ForAgent(agent), outcome)) = calls.get(settled.entity) {
        let result = outcome.tool_result();
        let event = json!({
            "type": "tool_end",
            "agent": events.name(agent),
            "id": request.call.id,
            "name": request.call.function.name.as_str(),
            "output": result.output().render(),
        });
        events.send(&event);
    }
}

/// Sends what model calls streamed since the last event.
fn stream_text(mut events: Events) {
    events.flush();
}

fn relay_notices(mut notices: MessageReader<Notice>, mut events: Events) {
    for notice in notices.read() {
        let level = match notice.level {
            NoticeLevel::Info => "info",
            NoticeLevel::Warn => "warn",
            NoticeLevel::Error => "error",
        };
        let event = json!({
            "type": "notice",
            "agent": notice.agent.and_then(|agent| events.name(agent)),
            "level": level,
            "text": notice.text,
        });
        events.send(&event);
    }
}

/// Offers each question a plugin asks, by its entity's bits.
fn asked(added: On<Add<Ask>>, asks: Query<(&Ask, Option<&ForAgent>)>, mut events: Events) {
    if let Ok((ask, agent)) = asks.get(added.entity) {
        let labels: Vec<&str> = ask
            .choices
            .iter()
            .map(|choice| choice.label.as_str())
            .collect();
        let event = json!({
            "type": "picker",
            "id": added.entity.to_bits(),
            "agent": agent.and_then(|agent| events.name(agent.0)),
            "title": ask.title,
            "body": ask.body,
            "items": labels,
            "free_text": ask.free_text,
        });
        events.send(&event);
    }
}

/// Says a question was answered, here or in another front.
fn ask_closed(answered: On<Add<Outcome>>, asks: Query<(), With<Ask>>, mut events: Events) {
    if asks.contains(answered.entity) {
        events.send(&json!({ "type": "closed", "id": answered.entity.to_bits() }));
    }
}

/// Offers each picker a plugin opens or replaces, by its entity's bits.
fn picker_opened(inserted: On<Insert<Picker>>, pickers: Query<&Picker>, mut events: Events) {
    if let Ok(picker) = pickers.get(inserted.entity) {
        let labels: Vec<&str> = picker
            .items
            .iter()
            .map(|item| item.label.as_str())
            .collect();
        let event = json!({
            "type": "picker",
            "id": inserted.entity.to_bits(),
            "agent": picker.agent.and_then(|agent| events.name(agent)),
            "title": picker.title,
            "body": picker.body,
            "items": labels,
            "free_text": picker.free_text,
        });
        events.send(&event);
    }
}

/// Says a picker closed, here or in another front.
fn picker_closed(removed: On<Remove<Picker>>, mut events: Events) {
    events.send(&json!({ "type": "closed", "id": removed.entity.to_bits() }));
}

/// The session exits: every client is told and closed, and the socket
/// removed.
fn close(mut events: Events) {
    events.send(&json!({ "type": "ended", "text": ENDED }));
    let server = &mut *events.server;
    server.listening = None;
    server::finish_all(std::mem::take(&mut server.clients));
}

#[cfg(test)]
mod tests;
