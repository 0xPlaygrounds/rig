//! Plugin corpus: 46. HTTP API front (spec `46-http-api.md`).
//!
//! Drives the agent over local HTTP while the terminal view keeps working:
//! `RIG_SERVE=127.0.0.1:7777` starts a server whose random bearer token is
//! shown as a notice and written to `RIG_HOME/serve-token`.
//!
//! - `POST /agents/{id}/messages {text, mode}` delivers a message as if
//!   typed (slash commands run), `POST /agents/{id}/interrupt` stops a turn.
//! - `GET /agents` lists the agents with their status,
//!   `GET /agents/{id}/conversation` returns the messages.
//! - `GET /events` streams Server-Sent Events: `turn_start`, `text_delta`,
//!   `tool_start`, `tool_end`, `turn_end`, `notice` and `approval_request`.
//! - `POST /approvals/{id} {choice}` answers a picker a plugin opened.
//!
//! An agent is named by its id or the id's first characters. A client
//! lagging more than 256 events behind is dropped. Binding to anything but
//! localhost needs `RIG_SERVE_ALLOW_REMOTE=1`.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;

use crossbeam_channel::Sender;
use rig_activity::Activity;
use rig_harness::prelude::*;
use rig_tui::{PickItem, PickRequest};
use serde_json::{Value, json};

mod server;

use server::{Call, Reply, Request, Server};

/// The address to serve on, such as `127.0.0.1:7777`.
pub const SERVE_ENV: &str = "RIG_SERVE";

/// `1` or `true` lets the server bind an address that is not localhost.
pub const ALLOW_REMOTE_ENV: &str = "RIG_SERVE_ALLOW_REMOTE";

/// The pickers kept for an answer; older ones are forgotten.
const PENDING_PICKERS: usize = 32;

/// Where and how to serve.
#[derive(Clone, Debug)]
pub struct Serve {
    /// The address, such as `127.0.0.1:7777`; port 0 picks a free one.
    pub addr: String,
    /// Whether an address that is not localhost may be bound.
    pub allow_remote: bool,
    /// Where the token is written, if anywhere.
    pub token_file: Option<PathBuf>,
}

impl Serve {
    /// From [`SERVE_ENV`] and [`ALLOW_REMOTE_ENV`], with the token in
    /// `RIG_HOME/serve-token`; `None` without [`SERVE_ENV`].
    pub fn from_env() -> Option<Self> {
        let addr = std::env::var(SERVE_ENV)
            .ok()
            .filter(|addr| !addr.is_empty())?;
        let allow_remote =
            std::env::var(ALLOW_REMOTE_ENV).is_ok_and(|allow| allow == "1" || allow == "true");
        Some(Self {
            addr,
            allow_remote,
            token_file: Some(Home::from_env().root().join("serve-token")),
        })
    }
}

/// The HTTP API front. Serves nothing without a [`Serve`].
pub struct HttpApiPlugin {
    /// Where to serve; [`Serve::from_env`] by default.
    pub serve: Option<Serve>,
}

impl Default for HttpApiPlugin {
    fn default() -> Self {
        Self {
            serve: Serve::from_env(),
        }
    }
}

/// The running server's address and bearer token.
#[derive(Resource, Clone, Debug)]
pub struct HttpApi {
    /// The bound address.
    pub addr: SocketAddr,
    /// The token every request carries as `Authorization: Bearer <token>`.
    pub token: String,
}

impl Plugin for HttpApiPlugin {
    fn build(&self, app: &mut App) {
        let Some(serve) = self.serve.clone() else {
            return;
        };
        app.add_message::<PickRequest>()
            .init_resource::<Subscribers>()
            .init_resource::<Streamed>()
            .init_resource::<Pickers>()
            .add_systems(
                Startup,
                move |wake: Res<Wake>,
                      mut commands: Commands,
                      mut notices: MessageWriter<Notice>| {
                    match server::start(&serve, wake.clone()) {
                        Ok((addr, token, server)) => {
                            let file = serve.token_file.as_ref().map_or_else(String::new, |path| {
                                format!(" (in {})", path.display())
                            });
                            notices.write(Notice::info(
                                None,
                                format!("HTTP API on http://{addr}, bearer token {token}{file}"),
                            ));
                            commands.insert_resource(HttpApi { addr, token });
                            commands.insert_resource(server);
                        }
                        Err(why) => {
                            notices.write(Notice::error(None, why));
                        }
                    }
                },
            )
            .add_systems(PreUpdate, answer_requests.run_if(resource_exists::<Server>))
            .add_systems(PostUpdate, (stream_text, relay_notices, offer_pickers))
            .add_observer(turn_started)
            .add_observer(turn_ended)
            .add_observer(tool_started)
            .add_observer(tool_ended)
            .add_observer(reply_done)
            .add_observer(forget_call);
    }
}

/// The event streams' senders.
#[derive(Resource, Default)]
struct Subscribers(Vec<Sender<String>>);

impl Subscribers {
    /// Sends the event `kind` with `data` to every client, dropping those
    /// that lag behind or went away.
    fn send(&mut self, kind: &str, data: &Value) {
        if self.0.is_empty() {
            return;
        }
        let event = format!("event: {kind}\ndata: {data}\n\n");
        self.0
            .retain(|subscriber| subscriber.try_send(event.clone()).is_ok());
    }
}

/// How much of each model call's streamed text went out as deltas.
#[derive(Resource, Default)]
struct Streamed(HashMap<Entity, usize>);

/// A picker another plugin opened, waiting for an answer.
struct Picker {
    agent: Entity,
    items: Vec<PickItem>,
}

/// The pickers open, by id, oldest first, and the next id.
#[derive(Resource, Default)]
struct Pickers {
    open: VecDeque<(u64, Picker)>,
    next: u64,
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

/// Answers the requests the connections handed over since the last frame.
fn answer_requests(
    server: Res<Server>,
    agents: Query<AgentRow>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
    mut pickers: ResMut<Pickers>,
    mut commands: Commands,
) {
    for request in server.requests.try_iter() {
        match request {
            Request::Subscribe(sender) => subscribers.0.push(sender),
            Request::Call(call, reply) => {
                let answer = answer(call, &agents, &ids, &mut pickers, &mut commands);
                reply.send(answer).ok();
            }
        }
    }
}

/// The reply to `call`.
fn answer(
    call: Call,
    agents: &Query<AgentRow>,
    ids: &Query<&AgentId>,
    pickers: &mut Pickers,
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
                    let parent = parent.and_then(|parent| ids.get(parent.0).ok());
                    json!({
                        "id": id.0,
                        "name": name.as_str(),
                        "status": status,
                        "busy": busy,
                        "spawned_by": parent.map(|parent| parent.0.clone()),
                    })
                })
                .collect();
            Reply::new(200, json!({ "agents": listed }))
        }
        Call::Conversation(wanted) => match find(&wanted) {
            Some((_, id, _, conversation, ..)) => Reply::new(
                200,
                json!({ "agent": id.0, "messages": conversation.messages() }),
            ),
            None => missing(&wanted),
        },
        Call::Deliver { agent, text, mode } => match find(&agent) {
            Some((entity, ..)) => {
                commands.trigger(Deliver::new(entity, text, mode));
                Reply::new(202, json!({ "accepted": true }))
            }
            None => missing(&agent),
        },
        Call::Interrupt(agent) => match find(&agent) {
            Some((entity, .., true, _)) => {
                commands.trigger(Interrupt::new(entity));
                Reply::new(202, json!({ "accepted": true }))
            }
            Some(_) => Reply::error(409, "no turn runs"),
            None => missing(&agent),
        },
        Call::Answer { id, choice } => {
            let Some(at) = pickers.open.iter().position(|(open, _)| *open == id) else {
                return Reply::error(404, format!("no open picker {id}"));
            };
            let chosen = pickers.open.get(at).and_then(|(_, picker)| {
                let item = match &choice {
                    Value::Number(index) => index
                        .as_u64()
                        .and_then(|index| usize::try_from(index).ok())
                        .and_then(|index| picker.items.get(index)),
                    Value::String(wanted) => picker
                        .items
                        .iter()
                        .find(|item| &item.label == wanted || &item.command == wanted),
                    _ => None,
                };
                item.map(|item| RunCommand {
                    entity: picker.agent,
                    line: item.command.clone(),
                })
            });
            let Some(run) = chosen else {
                return Reply::error(400, "no such choice");
            };
            pickers.open.remove(at);
            commands.trigger(run);
            Reply::new(202, json!({ "accepted": true }))
        }
    }
}

/// The id of the agent whose turn made the call `of` names.
fn agent_of(of: &CallOf, turns: &Query<&TurnOf>, ids: &Query<&AgentId>) -> Option<String> {
    let TurnOf(agent) = turns.get(of.0).ok()?;
    ids.get(*agent).ok().map(|id| id.0.clone())
}

fn turn_started(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    if let Ok(TurnOf(agent)) = turns.get(started.entity)
        && let Ok(id) = ids.get(*agent)
    {
        subscribers.send("turn_start", &json!({ "agent": id.0 }));
    }
}

fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(id) = ids.get(agent) else {
        return;
    };
    let data = match outcome {
        TurnOutcome::Answered(message) => {
            json!({ "agent": id.0, "outcome": "answered", "answer": final_answer(message) })
        }
        TurnOutcome::Failed(why) => json!({ "agent": id.0, "outcome": "failed", "error": why }),
        TurnOutcome::Stopped(why) => json!({ "agent": id.0, "outcome": "stopped", "reason": why }),
    };
    subscribers.send("turn_end", &data);
}

fn tool_started(
    started: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    if let Ok((request, ForAgent(agent))) = calls.get(started.entity) {
        let agent = ids.get(*agent).ok().map(|id| id.0.clone());
        let data = json!({ "agent": agent, "call": request.call });
        subscribers.send("tool_start", &data);
    }
}

fn tool_ended(
    ended: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, &Outcome)>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    if let Ok((request, ForAgent(agent), outcome)) = calls.get(ended.entity) {
        let data = json!({
            "agent": ids.get(*agent).ok().map(|id| id.0.clone()),
            "id": request.call.id,
            "name": request.call.function.name.as_str(),
            "result": outcome.tool_result(),
        });
        subscribers.send("tool_end", &data);
    }
}

/// Sends the text each streaming model call added since the last frame.
fn stream_text(
    calls: Query<(Entity, &Partial, &CallOf), (With<TurnCall>, Changed<Partial>)>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
    mut streamed: ResMut<Streamed>,
    mut subscribers: ResMut<Subscribers>,
) {
    for (call, partial, of) in &calls {
        let sent = streamed.0.entry(call).or_default();
        if let Some(delta) = partial.text.get(*sent..).filter(|delta| !delta.is_empty()) {
            let data = json!({ "agent": agent_of(of, &turns, &ids), "text": delta });
            subscribers.send("text_delta", &data);
            *sent = partial.text.len();
        }
    }
}

/// Sends what a settled model call's reply has beyond what was streamed:
/// its reply often arrives before [`stream_text`] saw its last fragments.
fn reply_done(
    done: On<Add<Settled>>,
    calls: Query<(&Outcome, &CallOf), With<TurnCall>>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
    mut streamed: ResMut<Streamed>,
    mut subscribers: ResMut<Subscribers>,
) {
    let Some((response, of)) = calls
        .get(done.entity)
        .ok()
        .and_then(|(outcome, of)| Some((outcome.completion()?, of)))
    else {
        return;
    };
    let text: String = response
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let sent = streamed
        .0
        .insert(done.entity, text.len())
        .unwrap_or_default();
    if let Some(delta) = text.get(sent..).filter(|delta| !delta.is_empty()) {
        let data = json!({ "agent": agent_of(of, &turns, &ids), "text": delta });
        subscribers.send("text_delta", &data);
    }
}

fn forget_call(gone: On<Remove<TurnCall>>, mut streamed: ResMut<Streamed>) {
    streamed.0.remove(&gone.entity);
}

fn relay_notices(
    mut notices: MessageReader<Notice>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    for notice in notices.read() {
        let agent = notice.agent.and_then(|agent| ids.get(agent).ok());
        let level = match notice.level {
            NoticeLevel::Info => "info",
            NoticeLevel::Error => "error",
        };
        let data = json!({
            "agent": agent.map(|id| id.0.clone()),
            "level": level,
            "text": notice.text,
        });
        subscribers.send("notice", &data);
    }
}

/// Offers each picker a plugin opens as an approval request with an id.
fn offer_pickers(
    mut requests: MessageReader<PickRequest>,
    ids: Query<&AgentId>,
    mut pickers: ResMut<Pickers>,
    mut subscribers: ResMut<Subscribers>,
) {
    for request in requests.read() {
        let id = pickers.next;
        pickers.next += 1;
        let choices: Vec<&str> = request
            .items
            .iter()
            .map(|item| item.label.as_str())
            .collect();
        let data = json!({
            "id": id,
            "agent": ids.get(request.agent).ok().map(|id| id.0.clone()),
            "title": request.title,
            "choices": choices,
            "selected": request.selected,
        });
        subscribers.send("approval_request", &data);
        if pickers.open.len() == PENDING_PICKERS {
            pickers.open.pop_front();
        }
        pickers.open.push_back((
            id,
            Picker {
                agent: request.agent,
                items: request.items.clone(),
            },
        ));
    }
}

#[cfg(test)]
mod tests;
