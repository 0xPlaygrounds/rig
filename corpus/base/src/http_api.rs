//! Plugin corpus: 46. HTTP API front (spec `46-http-api.md`).
//!
//! Drives the agent over local HTTP while the terminal view keeps working:
//! `rig --serve=127.0.0.1:7777`, or `addr` in the `[serve]` settings,
//! starts a server whose random bearer token is shown as a notice and
//! written to `RIG_HOME/serve-token`.
//!
//! - `POST /agents/{id}/messages {text, mode}` delivers a message as if
//!   typed (slash commands run), `POST /agents/{id}/interrupt` stops a turn.
//! - `GET /agents` lists the agents with their status,
//!   `GET /agents/{id}/conversation` returns the messages.
//! - `GET /events` streams Server-Sent Events, in the order things
//!   happened: `turn_start`, `text_delta`, `reasoning_delta`, `tool_start`,
//!   `tool_end`, `turn_end`, `notice`, `approval_request` and
//!   `approval_closed`.
//! - `POST /approvals/{id} {choice}` answers any plugin's open [`Ask`] or
//!   [`Picker`] with [`Pick`], as the terminal view does; a `null` choice
//!   closes it.
//!
//! An agent is named by its id or the id's first characters. A client
//! lagging more than 256 events behind is dropped. Binding to anything but
//! localhost needs `allow_remote = true` in `[serve]`.

use std::sync::mpsc::SyncSender;

use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::activity::Activity;
use rig_harness::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

mod server;

use server::{Call, Reply, Request};

/// The option naming the address to serve on, such as `127.0.0.1:7777`.
pub const SERVE: &str = "serve";

/// The `[serve]` settings: `addr = "127.0.0.1:7777"`, `allow_remote`.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ServeSettings {
    /// The address to serve on; `--serve=ADDR` overrides it. Port 0 picks
    /// a free one.
    pub addr: Option<String>,
    /// Whether an address that is not localhost may be bound.
    pub allow_remote: bool,
}

/// The HTTP API. Serves nothing without `--serve` or `[serve] addr`.
#[derive(Default)]
pub struct HttpApiPlugin;

/// The running server's address and bearer token.
#[derive(Resource, Clone, Debug)]
pub struct HttpApi {
    /// The bound address.
    pub addr: std::net::SocketAddr,
    /// The token every request carries as `Authorization: Bearer <token>`.
    pub token: String,
}

impl Plugin for HttpApiPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(Flag::new(SERVE, "Serve the HTTP API on ADDR").value("ADDR"))
            .add_setting::<ServeSettings>("serve");
        let world = app.world();
        let flag = world
            .get_resource::<Invoked>()
            .and_then(|invoked| invoked.flag_value(SERVE).map(str::to_owned));
        let settings = world.resource::<ServeSettings>();
        let Some(addr) = flag.or_else(|| settings.addr.clone()) else {
            return;
        };
        let allow_remote = settings.allow_remote;
        let token_file = RigHome::of(world).root().join("serve-token");
        app.init_resource::<Subscribers>()
            .add_systems(
                Startup,
                move |wake: Res<Wake>,
                      mut commands: Commands,
                      mut notices: MessageWriter<Notice>| {
                    let (api, listener, requests) =
                        match server::start(&addr, allow_remote, &token_file) {
                            Ok(started) => started,
                            Err(why) => {
                                notices.write(Notice::error(None, why));
                                return;
                            }
                        };
                    let file = token_file.display();
                    let text = format!(
                        "HTTP API on http://{}, bearer token {} (in {file})",
                        api.addr, api.token
                    );
                    notices.write(Notice::info(None, text));
                    commands.insert_resource(api);
                    let pool = IoTaskPool::get_or_init(TaskPool::default);
                    commands.spawn((listener, Running::stream(pool, &wake, requests)));
                    commands.spawn(Answerer::default());
                },
            )
            .add_systems(PostUpdate, (stream_text, relay_notices))
            .add_observer(answer_requests)
            .add_observer(turn_started)
            .add_observer(turn_ended)
            .add_observer(tool_started)
            .add_observer(tool_ended)
            .add_observer(asked)
            .add_observer(ask_answered)
            .add_observer(picker_opened)
            .add_observer(picker_closed);
    }
}

/// The event streams' senders, and how far they got in what model calls
/// streamed.
#[derive(Resource, Default)]
struct Subscribers {
    clients: Vec<SyncSender<String>>,
    deltas: MessageCursor<StreamDelta>,
}

/// Sends events to every client, dropping those that lag behind or went
/// away. Every event first sends the fragments model calls streamed before
/// it, so a stream keeps the order things happened in: the fragments are
/// messages, the rest observers.
#[derive(SystemParam)]
struct Events<'w, 's> {
    subscribers: ResMut<'w, Subscribers>,
    deltas: Res<'w, Messages<StreamDelta>>,
    ids: Query<'w, 's, &'static AgentId>,
}

impl Events<'_, '_> {
    /// The id of `agent`.
    fn agent(&self, agent: Entity) -> Option<String> {
        self.ids.get(agent).ok().map(|id| id.0.clone())
    }

    /// Sends the event `kind` with `data`.
    fn send(&mut self, kind: &str, data: &Value) {
        self.flush();
        send(&mut self.subscribers.clients, kind, data);
    }

    /// Sends the fragments streamed since the last event, as `text_delta`
    /// and `reasoning_delta`.
    fn flush(&mut self) {
        let Subscribers { clients, deltas } = &mut *self.subscribers;
        for delta in deltas.read(&self.deltas) {
            let (kind, text) = match &delta.event {
                StreamEvent::Text { text, .. } => ("text_delta", text),
                StreamEvent::Reasoning { text, .. } => ("reasoning_delta", text),
                _ => continue,
            };
            let agent = delta.agent.and_then(|agent| self.ids.get(agent).ok());
            let data = json!({
                "agent": agent.map(|id| &id.0),
                "call": delta.effect.to_bits(),
                "text": text,
            });
            send(clients, kind, &data);
        }
    }
}

/// Sends the event `kind` with `data` to `clients`, dropping those that lag
/// behind or went away.
fn send(clients: &mut Vec<SyncSender<String>>, kind: &str, data: &Value) {
    if clients.is_empty() {
        return;
    }
    let event = format!("event: {kind}\ndata: {data}\n\n");
    clients.retain(|client| client.try_send(event.clone()).is_ok());
}

/// The agents as requests name them.
type AgentRow<'a> = (
    Entity,
    &'a AgentId,
    &'a Name,
    &'a Conversation,
    Option<&'a Activity>,
    Has<ActiveTurn>,
    Option<&'a SpawnedBy>,
);

/// What answering a choice reads.
type Choosable<'w, 's> = (
    Query<'w, 's, &'static Ask, (Without<Outcome>, Without<Settled>)>,
    Query<'w, 's, &'static Picker>,
);

/// Answers each request a connection hands over.
fn answer_requests(
    handed: On<Insert<Done<Request>>>,
    requests: Query<&Done<Request>>,
    agents: Query<AgentRow>,
    choosable: Choosable,
    mut events: Events,
    mut commands: Commands,
) {
    match requests.get(handed.entity) {
        Ok(Done(Request::Subscribe(client))) => {
            // A new client gets what streams from now on.
            events.flush();
            events.subscribers.clients.push(client.clone());
        }
        Ok(Done(Request::Call(call, reply))) => {
            let answer = answer(call, &agents, &events.ids, &choosable, &mut commands);
            reply.send(answer).ok();
        }
        Err(_) => {}
    }
}

/// The reply to `call`.
fn answer(
    call: &Call,
    agents: &Query<AgentRow>,
    ids: &Query<&AgentId>,
    choosable: &Choosable,
    commands: &mut Commands,
) -> Reply {
    let find = |wanted: &str| {
        agents
            .iter()
            .find(|(_, id, ..)| id.0 == wanted || id.short() == wanted)
    };
    let missing = |wanted: &str| Reply::error(404, format!("no agent {wanted}"));
    match call {
        Call::Agents => {
            let listed: Vec<Value> = agents
                .iter()
                .map(|(_, id, name, _, activity, busy, parent)| {
                    let status = activity.map_or_else(
                        || if busy { "busy" } else { "idle" }.to_owned(),
                        |activity| activity.status.to_string(),
                    );
                    json!({
                        "id": id.0,
                        "name": name.as_str(),
                        "status": status,
                        "busy": busy,
                        "spawned_by": parent.and_then(|parent| ids.get(parent.0).ok()).map(|id| &id.0),
                    })
                })
                .collect();
            Reply::new(200, json!({ "agents": listed }))
        }
        Call::Conversation(wanted) => match find(wanted) {
            Some((_, id, _, conversation, ..)) => Reply::new(
                200,
                json!({ "agent": id.0, "messages": conversation.messages() }),
            ),
            None => missing(wanted),
        },
        Call::Deliver { agent, text, mode } => match find(agent) {
            Some((entity, ..)) => {
                commands.trigger(Deliver::new(entity, text.clone(), *mode));
                Reply::new(202, json!({ "accepted": true }))
            }
            None => missing(agent),
        },
        Call::Interrupt(agent) => match find(agent) {
            Some((entity, .., true, _)) => {
                commands.trigger(StopTurn::new(entity));
                Reply::new(202, json!({ "accepted": true }))
            }
            Some(_) => Reply::error(409, "no turn runs"),
            None => missing(agent),
        },
        Call::Answer { id, choice } => choose(*id, choice, choosable, commands),
    }
}

/// Answers the open [`Ask`] or [`Picker`] `id` with `choice`: a choice's
/// position, id, label or command, a line of text where one answers it, or
/// `null` to close it.
fn choose(id: u64, choice: &Value, (asks, pickers): &Choosable, commands: &mut Commands) -> Reply {
    let open = Entity::try_from_bits(id).and_then(|entity| {
        if let Ok(ask) = asks.get(entity) {
            let names = ask.choices.iter().map(|choice| [&choice.id, &choice.label]);
            return Some((entity, names.collect::<Vec<_>>(), ask.free_text));
        }
        let picker = pickers.get(entity).ok()?;
        let names = picker.items.iter().map(|item| [&item.label, &item.command]);
        Some((entity, names.collect(), picker.free_text))
    });
    let Some((entity, names, free_text)) = open else {
        return Reply::error(404, format!("no open question {id}"));
    };
    let index = match choice {
        Value::Number(index) => index
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .filter(|&index| index < names.len()),
        Value::String(wanted) => names.iter().position(|names| names.contains(&wanted)),
        _ => None,
    };
    let chosen = match (index, choice) {
        (Some(index), _) => Some(Chosen::item(index)),
        (None, Value::String(text)) if free_text => Some(Chosen::text(text.clone())),
        (None, Value::Null) => None,
        _ => return Reply::error(400, "no such choice"),
    };
    commands.trigger(Pick::person(entity, chosen));
    Reply::new(202, json!({ "accepted": true }))
}

fn turn_started(started: On<Add<TurnOf>>, turns: Query<&TurnOf>, mut events: Events) {
    if let Ok(&TurnOf(agent)) = turns.get(started.entity) {
        let data = json!({ "agent": events.agent(agent) });
        events.send("turn_start", &data);
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
    let agent = events.agent(agent);
    let data = match outcome {
        TurnOutcome::Answered(message) => {
            json!({ "agent": agent, "outcome": "answered", "answer": message.final_answer() })
        }
        TurnOutcome::Failed(why) => json!({ "agent": agent, "outcome": "failed", "error": why }),
        TurnOutcome::Stopped(why) => json!({ "agent": agent, "outcome": "stopped", "reason": why }),
    };
    events.send("turn_end", &data);
}

fn tool_started(
    started: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    mut events: Events,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(started.entity) {
        let data = json!({ "agent": events.agent(agent), "call": request.call });
        events.send("tool_start", &data);
    }
}

fn tool_ended(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, &Outcome)>,
    mut events: Events,
) {
    if let Ok((request, &ForAgent(agent), outcome)) = calls.get(settled.entity) {
        let data = json!({
            "agent": events.agent(agent),
            "id": request.call.id,
            "name": request.call.function.name.as_str(),
            "result": outcome.tool_result(),
        });
        events.send("tool_end", &data);
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
        let data = json!({
            "agent": notice.agent.and_then(|agent| events.agent(agent)),
            "level": level,
            "text": notice.text,
        });
        events.send("notice", &data);
    }
}

/// Offers each question a plugin asks, by its entity's bits.
fn asked(added: On<Add<Ask>>, asks: Query<(&Ask, Option<&ForAgent>)>, mut events: Events) {
    if let Ok((ask, agent)) = asks.get(added.entity) {
        let data = json!({
            "id": added.entity.to_bits(),
            "kind": "ask",
            "agent": agent.and_then(|agent| events.agent(agent.0)),
            "title": ask.title,
            "body": ask.body,
            "choices": ask.choices,
            "free_text": ask.free_text,
        });
        events.send("approval_request", &data);
    }
}

/// Says a question was answered, here or in another front.
fn ask_answered(answered: On<Add<Outcome>>, asks: Query<(), With<Ask>>, mut events: Events) {
    if asks.contains(answered.entity) {
        let data = json!({ "id": answered.entity.to_bits() });
        events.send("approval_closed", &data);
    }
}

/// Offers each picker a plugin opens or replaces, by its entity's bits.
fn picker_opened(inserted: On<Insert<Picker>>, pickers: Query<&Picker>, mut events: Events) {
    if let Ok(picker) = pickers.get(inserted.entity) {
        let choices: Vec<Value> = picker
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| json!({ "id": index, "label": item.label }))
            .collect();
        let data = json!({
            "id": inserted.entity.to_bits(),
            "kind": "picker",
            "agent": picker.agent.and_then(|agent| events.agent(agent)),
            "title": picker.title,
            "body": picker.body,
            "choices": choices,
            "free_text": picker.free_text,
        });
        events.send("approval_request", &data);
    }
}

/// Says a picker closed, here or in the terminal view.
fn picker_closed(removed: On<Remove<Picker>>, mut events: Events) {
    let data = json!({ "id": removed.entity.to_bits() });
    events.send("approval_closed", &data);
}

#[cfg(test)]
mod tests;
