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
//! - `GET /events` streams Server-Sent Events: `turn_start`, `text_delta`,
//!   `tool_start`, `tool_end`, `turn_end`, `notice`, `approval_request`
//!   and `approval_closed`.
//! - `POST /approvals/{id} {choice}` answers any plugin's open [`Ask`], or
//!   a [`Picker`] the terminal view would show.
//!
//! An agent is named by its id or the id's first characters. A client
//! lagging more than 256 events behind is dropped. Binding to anything but
//! localhost needs `allow_remote = true` in `[serve]`.

use crossbeam_channel::Sender;
use rig_activity::Activity;
use rig_harness::prelude::*;
use rig_tui::Focused;
use serde::Deserialize;
use serde_json::{Value, json};

mod server;

use server::{Call, Reply, Request, Server};

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
            .register_required_components::<TurnCall, Streamed>()
            .add_systems(
                Startup,
                move |wake: Res<Wake>,
                      mut commands: Commands,
                      mut notices: MessageWriter<Notice>| {
                    match server::start(&addr, allow_remote, &token_file, wake.clone()) {
                        Ok((api, server)) => {
                            notices.write(Notice::info(
                                None,
                                format!(
                                    "HTTP API on http://{}, bearer token {} (in {})",
                                    api.addr,
                                    api.token,
                                    token_file.display()
                                ),
                            ));
                            commands.insert_resource(api);
                            commands.insert_resource(server);
                            commands.spawn(Answerer::default());
                        }
                        Err(why) => {
                            notices.write(Notice::error(None, why));
                        }
                    }
                },
            )
            .add_systems(PreUpdate, answer_requests.run_if(resource_exists::<Server>))
            .add_systems(PostUpdate, (stream_text, relay_notices))
            .add_observer(turn_started)
            .add_observer(turn_ended)
            .add_observer(reply_settled)
            .add_observer(tool_started)
            .add_observer(tool_ended)
            .add_observer(asked)
            .add_observer(ask_answered)
            .add_observer(picker_opened)
            .add_observer(picker_closed);
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

/// On each turn's model call: how much of its [`Partial`] text went out
/// as deltas.
#[derive(Component, Default)]
struct Streamed(usize);

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
    Query<'w, 's, Entity, With<Focused>>,
);

/// Answers the requests the connections handed over since the last frame.
fn answer_requests(
    server: Res<Server>,
    agents: Query<AgentRow>,
    ids: Query<&AgentId>,
    choosable: Choosable,
    mut subscribers: ResMut<Subscribers>,
    mut commands: Commands,
) {
    for request in server.requests.try_iter() {
        match request {
            Request::Subscribe(sender) => subscribers.0.push(sender),
            Request::Call(call, reply) => {
                let answer = answer(call, &agents, &ids, &choosable, &mut commands);
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
                        "spawned_by": parent.and_then(|parent| agent_id(ids, parent.0)),
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
        Call::Answer { id, choice } => choose(id, &choice, choosable, commands),
    }
}

/// Answers the open [`Ask`] or [`Picker`] `id` with `choice`: a choice's
/// position, id or label, or a line of text where one answers it.
fn choose(
    id: u64,
    choice: &Value,
    (asks, pickers, focused): &Choosable,
    commands: &mut Commands,
) -> Reply {
    let accepted = Reply::new(202, json!({ "accepted": true }));
    let no_choice = Reply::error(400, "no such choice");
    let entity = Entity::try_from_bits(id);
    if let Some((entity, ask)) = entity.and_then(|entity| Some((entity, asks.get(entity).ok()?))) {
        let chosen = match choice {
            Value::Number(index) => index
                .as_u64()
                .and_then(|index| ask.choices.get(usize::try_from(index).ok()?)),
            Value::String(wanted) => ask
                .choices
                .iter()
                .find(|choice| &choice.id == wanted || &choice.label == wanted),
            _ => None,
        };
        let answer = match (chosen, choice) {
            (Some(chosen), _) => Answer::Choice(chosen.id.clone()),
            (None, Value::String(text)) if ask.free_text => Answer::Text(text.clone()),
            _ => return no_choice,
        };
        commands.entity(entity).insert_if_new(answer.outcome());
        return accepted;
    }
    let Some((entity, picker)) =
        entity.and_then(|entity| Some((entity, pickers.get(entity).ok()?)))
    else {
        return Reply::error(404, format!("no open question {id}"));
    };
    let index = match choice {
        Value::Number(index) => index
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .filter(|&index| index < picker.items.len()),
        Value::String(wanted) => picker
            .items
            .iter()
            .position(|item| &item.label == wanted || &item.command == wanted),
        _ => None,
    };
    let chosen = match (index, choice) {
        (Some(index), _) => Chosen::Item(index),
        (None, Value::String(text)) if picker.free_text => Chosen::Text(text.clone()),
        _ => return no_choice,
    };
    // What the terminal view does on a pick, which it keeps to itself.
    let item = index.and_then(|index| picker.items.get(index));
    if let (Some(item), Some(agent)) = (item, picker.agent.or(focused.iter().next()))
        && !item.command.is_empty()
    {
        commands.trigger(RunCommand::new(agent, item.command.clone()));
    }
    commands.trigger(Picked {
        entity,
        chosen: Some(chosen),
    });
    commands.entity(entity).try_despawn();
    accepted
}

/// The id of `agent`.
fn agent_id(ids: &Query<&AgentId>, agent: Entity) -> Option<String> {
    ids.get(agent).ok().map(|id| id.0.clone())
}

fn turn_started(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    if let Ok(&TurnOf(agent)) = turns.get(started.entity) {
        let data = json!({ "agent": agent_id(&ids, agent) });
        subscribers.send("turn_start", &data);
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
    let agent = agent_id(&ids, agent);
    let data = match outcome {
        TurnOutcome::Answered(message) => {
            json!({ "agent": agent, "outcome": "answered", "answer": final_answer(message) })
        }
        TurnOutcome::Failed(why) => json!({ "agent": agent, "outcome": "failed", "error": why }),
        TurnOutcome::Stopped(why) => json!({ "agent": agent, "outcome": "stopped", "reason": why }),
    };
    subscribers.send("turn_end", &data);
}

fn tool_started(
    started: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(started.entity) {
        let data = json!({ "agent": agent_id(&ids, agent), "call": request.call });
        subscribers.send("tool_start", &data);
    }
}

fn tool_ended(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, &Outcome)>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    if let Ok((request, &ForAgent(agent), outcome)) = calls.get(settled.entity) {
        let data = json!({
            "agent": agent_id(&ids, agent),
            "id": request.call.id,
            "name": request.call.function.name.as_str(),
            "result": outcome.tool_result(),
        });
        subscribers.send("tool_end", &data);
    }
}

/// Sends the text each turn's model call streamed since the last frame.
fn stream_text(
    mut calls: Query<(&Partial, &mut Streamed, &ForAgent), Changed<Partial>>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    for (partial, mut streamed, &ForAgent(agent)) in &mut calls {
        send_delta(
            &partial.text,
            &mut streamed,
            agent_id(&ids, agent),
            &mut subscribers,
        );
    }
}

/// Sends what a settled call's reply has beyond what [`stream_text`] sent:
/// the kernel takes the call's [`Partial`] away when it commits the reply,
/// often in the frame its last fragments came, before `PostUpdate`.
fn reply_settled(
    settled: On<Add<Settled>>,
    mut calls: Query<(&Outcome, &mut Streamed, &ForAgent)>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    let Ok((outcome, mut streamed, &ForAgent(agent))) = calls.get_mut(settled.entity) else {
        return;
    };
    let Some(response) = outcome.completion() else {
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
    send_delta(
        &text,
        &mut streamed,
        agent_id(&ids, agent),
        &mut subscribers,
    );
}

/// Sends `text` past what `streamed` says went out, as a `text_delta`.
fn send_delta(
    text: &str,
    streamed: &mut Streamed,
    agent: Option<String>,
    subscribers: &mut Subscribers,
) {
    if let Some(delta) = text.get(streamed.0..).filter(|delta| !delta.is_empty()) {
        subscribers.send("text_delta", &json!({ "agent": agent, "text": delta }));
        streamed.0 = text.len();
    }
}

fn relay_notices(
    mut notices: MessageReader<Notice>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    for notice in notices.read() {
        let level = match notice.level {
            NoticeLevel::Info => "info",
            NoticeLevel::Error => "error",
        };
        let data = json!({
            "agent": notice.agent.and_then(|agent| agent_id(&ids, agent)),
            "level": level,
            "text": notice.text,
        });
        subscribers.send("notice", &data);
    }
}

/// Offers each question a plugin asks, by its entity's bits.
fn asked(
    added: On<Add<Ask>>,
    asks: Query<(&Ask, Option<&ForAgent>)>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
    if let Ok((ask, agent)) = asks.get(added.entity) {
        let data = json!({
            "id": added.entity.to_bits(),
            "kind": "ask",
            "agent": agent.and_then(|agent| agent_id(&ids, agent.0)),
            "title": ask.title,
            "body": ask.body,
            "choices": ask.choices,
            "free_text": ask.free_text,
        });
        subscribers.send("approval_request", &data);
    }
}

/// Says a question was answered, here or in another front.
fn ask_answered(
    answered: On<Add<Outcome>>,
    asks: Query<(), With<Ask>>,
    mut subscribers: ResMut<Subscribers>,
) {
    if asks.contains(answered.entity) {
        let data = json!({ "id": answered.entity.to_bits() });
        subscribers.send("approval_closed", &data);
    }
}

/// Offers each picker a plugin opens or replaces, by its entity's bits.
fn picker_opened(
    inserted: On<Insert<Picker>>,
    pickers: Query<&Picker>,
    ids: Query<&AgentId>,
    mut subscribers: ResMut<Subscribers>,
) {
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
            "agent": picker.agent.and_then(|agent| agent_id(&ids, agent)),
            "title": picker.title,
            "body": picker.body,
            "choices": choices,
            "free_text": picker.free_text,
        });
        subscribers.send("approval_request", &data);
    }
}

/// Says a picker closed, here or in the terminal view.
fn picker_closed(removed: On<Remove<Picker>>, mut subscribers: ResMut<Subscribers>) {
    let data = json!({ "id": removed.entity.to_bits() });
    subscribers.send("approval_closed", &data);
}

#[cfg(test)]
mod tests;
