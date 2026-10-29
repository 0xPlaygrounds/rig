use std::io::{BufRead, Write};

use async_channel::Receiver;
use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use rig_basics::approval::NeedsApproval;
use rig_harness::prelude::*;
use serde_json::{Value, json};

#[derive(Resource)]
struct Input(Receiver<Value>);
#[derive(Resource, Default)]
struct Read(MessageCursor<StreamDelta>);
#[derive(Resource, Default)]
struct State {
    ready: bool,
    busy: bool,
    pending: Option<String>,
    delivered: bool,
    requires_action: bool,
    failure: Option<String>,
    notices: MessageCursor<Notice>,
    recalled: MessageCursor<Recalled>,
}

fn write(value: Value) {
    let mut stdout = std::io::stdout().lock();
    if writeln!(stdout, "{value}")
        .and_then(|()| stdout.flush())
        .is_err()
    {
        eprintln!("ACP worker output closed");
    }
}

pub(super) fn install(app: &mut App) {
    // Setup precedes the first frame, including restoration of unfinished work.
    let setup = std::io::stdin()
        .lock()
        .lines()
        .next()
        .transpose()
        .ok()
        .flatten()
        .and_then(|line| serde_json::from_str::<Value>(&line).ok());
    let servers = setup
        .as_ref()
        .and_then(|v| v.get("servers"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let runtime = app.world().resource::<TokioRuntime>().0.clone();
    match runtime.block_on(super::mcp::connect(&servers)) {
        Ok(tools) => {
            for tool in tools {
                let (definition, handler, _) = tool.into_parts();
                app.world_mut()
                    .spawn_tool(definition, handler, ToolOptions::default());
            }
        }
        Err(error) => {
            write(json!({"error":error.to_string()}));
            app.add_systems(Startup, |mut commands: Commands| {
                commands.trigger(Exit(ExitReason::Failed(1)));
            });
            return;
        }
    }
    let (sender, receiver) = async_channel::unbounded();
    let wake = app.world().resource::<Wake>().clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if let Ok(value) = serde_json::from_str(&line) {
                if sender.send_blocking(value).is_err() {
                    break;
                }
                wake.wake();
            }
        }
        drop(sender);
        wake.wake();
    });
    app.insert_resource(Input(receiver))
        .init_resource::<Read>()
        .init_resource::<State>()
        .add_systems(First, requests.run_if(resource_exists::<SessionRestored>))
        .add_systems(PostUpdate, (ready, stream, idle).chain())
        .add_observer(tool_added)
        .add_observer(started)
        .add_observer(settled);
    app.world_mut().spawn(Answerer::default());
}

fn ready(world: &mut World) {
    if world.resource::<State>().ready || !world.contains_resource::<SessionRestored>() {
        return;
    }
    let busy = primary(world).is_some_and(|agent| world.get::<ActiveTurn>(agent).is_some());
    let mut state = world.resource_mut::<State>();
    state.ready = true;
    state.busy = busy;
    write(json!({"ready":true,"busy":busy,"configOptions":options(world)}));
}

fn options(world: &mut World) -> Value {
    let agent = primary(world);
    let chosen = agent
        .and_then(|a| world.get::<ModelChoice>(a))
        .map(|m| m.0.clone());
    let reasoning = agent
        .and_then(|a| world.get::<ReasoningSetting>(a))
        .copied()
        .unwrap_or_default();
    let models = world.resource::<Models>();
    let mut available = models.0.reachable();
    if let Some(chosen) = chosen.as_deref()
        && let Ok(model) = models.0.catalog().resolve(chosen)
    {
        available.retain(|entry| entry.reference() != model.spec.reference());
        available.insert(0, model.spec);
    }
    let entries: Vec<_> = available
        .iter()
        .map(|m| json!({"value":m.reference(),"name":m.display_name}))
        .collect();
    let mut options = vec![];
    if !entries.is_empty() {
        let current = available.first().map(|model| model.reference());
        if let Some(current) = current {
            options.push(json!({"id":"model","name":"Model","category":"model","type":"select","currentValue":current,"options":entries}));
            if let Ok(model) = models.0.catalog().resolve(&current) {
                let choices = model.spec.reasoning.choices();
                let current = choices
                    .iter()
                    .find(|c| c.reasoning == reasoning.0)
                    .map(|c| c.name)
                    .unwrap_or("default");
                options.push(json!({"id":"reasoning","name":"Reasoning","category":"thought_level","type":"select","currentValue":current,"options":choices.iter().map(|c|json!({"value":c.name,"name":c.label()})).collect::<Vec<_>>()}));
            }
        }
    }
    Value::Array(options)
}

fn primary(world: &mut World) -> Option<Entity> {
    let mut query = world.query_filtered::<(Entity, &AgentId, Has<SpawnedBy>), With<Agent>>();
    query
        .iter(world)
        .min_by_key(|(_, id, child)| (*child, id.0.clone()))
        .map(|(entity, _, _)| entity)
}

fn requests(world: &mut World) {
    loop {
        let received = world.resource::<Input>().0.try_recv();
        let value = match received {
            Ok(value) => value,
            Err(async_channel::TryRecvError::Empty) => break,
            Err(async_channel::TryRecvError::Closed) => {
                world.trigger(Exit(ExitReason::Quit));
                break;
            }
        };
        let id = value.get("id").cloned().unwrap_or(Value::Null);
        let result = request(world, &value);
        match result {
            Ok(result) => write(json!({"id":id,"result":result})),
            Err(error) => write(json!({"id":id,"error":error.to_string()})),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("invalid model or reasoning: {0}")]
    Model(Box<dyn std::error::Error + Send + Sync>),
}

fn request(world: &mut World, value: &Value) -> Result<Value, Error> {
    let agent = primary(world).ok_or(Error::Invalid("no main agent"))?;
    match value.get("method").and_then(Value::as_str) {
        Some("prompt") => {
            if world.resource::<State>().busy || world.get::<ActiveTurn>(agent).is_some() {
                return Err(Error::Invalid("session is busy"));
            }
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or(Error::Invalid("prompt must contain non-whitespace text"))?;
            if !text.trim_start().starts_with('/') && world.get::<ModelChoice>(agent).is_none() {
                let model = world
                    .resource::<Models>()
                    .0
                    .reachable()
                    .first()
                    .map(|m| m.reference())
                    .ok_or(Error::Invalid("no reachable model; sign in with rig first"))?;
                world.trigger(SetModel {
                    entity: agent,
                    model,
                });
                world.flush();
            }
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .ok_or(Error::Invalid("missing text"))?;
            let text = text.to_owned();
            let mut state = world.resource_mut::<State>();
            state.busy = true;
            state.delivered = false;
            state.requires_action = false;
            state.failure = None;
            state.pending = Some(text);
            Ok(json!({}))
        }
        Some("run") => {
            let text = world
                .resource_mut::<State>()
                .pending
                .take()
                .ok_or(Error::Invalid("no accepted prompt"))?;
            world.resource_mut::<State>().delivered = true;
            if let Some(line) = text.trim_start().strip_prefix('/') {
                world.trigger(RunCommand::new(agent, line));
                return Ok(json!({}));
            }
            if world.get::<Connection>(agent).is_none() {
                world.resource_mut::<State>().busy = false;
                write(
                    json!({"event":"end","reason":"error","error":"model is unavailable; check Rig authentication"}),
                );
                return Err(Error::Invalid("model is unavailable"));
            }
            world.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            Ok(json!({}))
        }
        Some("cancel") => {
            if world.resource_mut::<State>().pending.take().is_some() {
                world.resource_mut::<State>().busy = false;
                write(json!({"event":"end","reason":"cancelled"}));
            }
            world.trigger(StopTurn::new(agent));
            Ok(json!({}))
        }
        Some("config") => {
            if world.resource::<State>().busy || world.get::<ActiveTurn>(agent).is_some() {
                return Err(Error::Invalid("session is busy"));
            }
            let selected = value
                .get("value")
                .and_then(Value::as_str)
                .ok_or(Error::Invalid("missing value"))?;
            match value.get("configId").and_then(Value::as_str) {
                Some("model") => {
                    world
                        .resource::<Models>()
                        .0
                        .catalog()
                        .resolve(selected)
                        .map_err(|e| Error::Model(Box::new(e)))?;
                    world.trigger(SetModel {
                        entity: agent,
                        model: selected.to_owned(),
                    });
                }
                Some("reasoning") => {
                    let choice = world
                        .get::<ModelChoice>(agent)
                        .ok_or(Error::Invalid("choose a model first"))?;
                    let model = world
                        .resource::<Models>()
                        .0
                        .catalog()
                        .resolve(&choice.0)
                        .map_err(|e| Error::Model(Box::new(e)))?;
                    let effort = model
                        .spec
                        .reasoning
                        .named(selected)
                        .map_err(|e| Error::Model(Box::new(e)))?;
                    world.trigger(SetReasoning {
                        entity: agent,
                        effort: ReasoningSetting(effort.reasoning),
                    });
                }
                _ => return Err(Error::Invalid("unknown configuration option")),
            }
            world.flush();
            Ok(json!({"configOptions":options(world)}))
        }
        Some("answer") => {
            let entity = value
                .get("ask")
                .and_then(Value::as_u64)
                .and_then(Entity::try_from_bits)
                .ok_or(Error::Invalid("invalid question"))?;
            let choice = value
                .get("choice")
                .and_then(Value::as_str)
                .unwrap_or("deny");
            let ask = world
                .get::<Ask>(entity)
                .ok_or(Error::Invalid("question has ended"))?;
            if !ask.choices.iter().any(|c| c.id == choice) {
                return Err(Error::Invalid("invalid answer"));
            }
            let mut decision = world
                .get_mut::<Decision>(entity)
                .ok_or(Error::Invalid("question has ended"))?;
            decision.decide(
                Decider::Person,
                Answer {
                    choices: vec![choice.to_owned()],
                    text: None,
                }
                .outcome(),
            );
            Ok(json!({}))
        }
        _ => Err(Error::Invalid("unknown worker request")),
    }
}

#[derive(SystemParam)]
struct Out<'w, 's> {
    deltas: Res<'w, Messages<StreamDelta>>,
    read: ResMut<'w, Read>,
    primary: MainAgentQuery<'w, 's>,
    calls: Query<'w, 's, (), With<TurnCall>>,
}
impl Out<'_, '_> {
    fn stream(&mut self) {
        let agent = main_agent(&self.primary);
        for delta in self
            .read
            .0
            .read(&self.deltas)
            .filter(|d| d.agent == agent && self.calls.contains(d.effect))
        {
            let (kind, text) = match &delta.event {
                StreamEvent::Text { text, .. } => ("text", text),
                StreamEvent::Reasoning { text, .. } => ("thought", text),
                _ => continue,
            };
            write(
                json!({"event":kind,"message":format!("{}-{kind}",delta.effect.to_bits()),"text":text}),
            );
        }
    }
    fn emit(&mut self, value: Value) {
        self.stream();
        write(value);
    }
}
fn stream(mut out: Out) {
    out.stream();
}

fn started(
    event: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    asks: Query<(&Ask, Option<&HoldOf>)>,
    approval: Query<(), With<NeedsApproval>>,
    model_calls: Query<&ForAgent, With<TurnCall>>,
    mut state: ResMut<State>,
    mut out: Out,
) {
    let resuming = calls
        .get(event.entity)
        .is_ok_and(|(_, agent)| Some(agent.0) == main_agent(&out.primary))
        || model_calls
            .get(event.entity)
            .is_ok_and(|agent| Some(agent.0) == main_agent(&out.primary));
    if resuming && state.requires_action {
        state.requires_action = false;
        out.emit(json!({"event":"state","state":"running"}));
    }
    if let Ok((request, agent)) = calls.get(event.entity) {
        if Some(agent.0) == main_agent(&out.primary) {
            out.emit(json!({"event":"tool","id":format!("tool-{}",event.entity.to_bits()),"title":request.call.function.name.as_str(),"input":request.call.function.arguments_value(),"status":"in_progress"}));
        }
    } else if let Ok((ask, hold)) = asks.get(event.entity) {
        let tool = hold
            .and_then(|hold| calls.get(hold.0).ok())
            .filter(|_| hold.is_some_and(|h| approval.contains(h.0)));
        out.emit(json!({"event":"ask","ask":event.entity.to_bits(),"title":ask.title,"body":ask.body,"tool":tool.and_then(|_|hold.map(|h|format!("tool-{}",h.0.to_bits()))),"choices":ask.choices}));
    }
}
fn settled(
    event: On<Add<Settled>>,
    effects: Query<(&ForAgent, Option<&Outcome>, Option<&TurnOutcome>)>,
    calls: Query<&ToolRequest>,
    mut state: ResMut<State>,
    mut out: Out,
) {
    let Ok((agent, outcome, turn)) = effects.get(event.entity) else {
        return;
    };
    if Some(agent.0) != main_agent(&out.primary) {
        return;
    }
    if let (Ok(tool), Some(outcome)) = (calls.get(event.entity), outcome) {
        let result = outcome.tool_result();
        out.emit(json!({"event":"tool","id":format!("tool-{}",event.entity.to_bits()),"title":tool.call.function.name.as_str(),"status":if result.is_success(){"completed"}else{"failed"},"output":result.output().render()}));
    } else if let Some(turn) = turn {
        state.busy = false;
        let (reason, error) = match turn {
            TurnOutcome::Answered(_) => ("end_turn", None),
            TurnOutcome::Stopped(why) => ("cancelled", why.clone()),
            TurnOutcome::Failed(why) => ("error", Some(why.clone())),
        };
        out.emit(json!({"event":"end","reason":reason,"error":error}));
    }
}

fn tool_added(event: On<Add<ToolRequest>>, calls: Query<(&ToolRequest, &ForAgent)>, mut out: Out) {
    if let Ok((request, agent)) = calls.get(event.entity)
        && Some(agent.0) == main_agent(&out.primary)
    {
        out.emit(json!({"event":"tool","id":format!("tool-{}",event.entity.to_bits()),"title":request.call.function.name.as_str(),"input":request.call.function.arguments_value(),"status":"pending"}));
    }
}

fn idle(world: &mut World) {
    let agent = primary(world);
    let notices = world.resource_scope(|world, mut state: Mut<State>| {
        let mut messages = Vec::new();
        for notice in state.notices.read(world.resource::<Messages<Notice>>()) {
            if notice.agent.is_none_or(|a| Some(a) == agent) {
                messages.push((notice.text.clone(), notice.level == NoticeLevel::Error));
            }
        }
        for recalled in state.recalled.read(world.resource::<Messages<Recalled>>()) {
            if Some(recalled.agent) == agent
                && let Some(why) = &recalled.why
            {
                messages.push((why.clone(), true));
            }
        }
        messages
    });
    if !world.resource::<State>().busy || !world.resource::<State>().delivered {
        return;
    }
    for (text, error) in notices {
        write(json!({"event":"text","message":uuid::Uuid::new_v4().to_string(),"text":text}));
        if error {
            world.resource_mut::<State>().failure = Some(text);
        }
    }
    let active = world
        .query_filtered::<Entity, With<TurnOf>>()
        .iter(world)
        .next()
        .is_some()
        || world
            .query_filtered::<Entity, (With<Incoming>, Without<Settled>)>()
            .iter(world)
            .next()
            .is_some()
        || world
            .query_filtered::<Entity, (
                With<Effect>,
                Without<Settled>,
                Without<ExitRequest>,
                Or<(Without<Started>, With<Outcome>)>,
            )>()
            .iter(world)
            .next()
            .is_some()
        || world
            .query::<&Running>()
            .iter(world)
            .any(|running| !running.is_stream());
    let waiting = world
        .query_filtered::<(&HoldOf, &Ask), (With<Started>, Without<Settled>)>()
        .iter(world)
        .any(|(hold, _)| world.get::<NeedsApproval>(hold.0).is_some());
    let executing = world
        .query_filtered::<Entity, (
            With<ForAgent>,
            With<Started>,
            Without<Outcome>,
            Or<(With<ToolRequest>, With<TurnCall>)>,
        )>()
        .iter(world)
        .next()
        .is_some();
    let blocked = waiting && !executing;
    if active && blocked != world.resource::<State>().requires_action {
        world.resource_mut::<State>().requires_action = blocked;
        write(json!({"event":"state","state":if blocked {"requires_action"}else{"running"}}));
    }
    if !active {
        let mut state = world.resource_mut::<State>();
        state.busy = false;
        let error = state.failure.take();
        write(
            json!({"event":"end","reason":if error.is_some(){"error"}else{"end_turn"},"error":error}),
        );
    }
}
