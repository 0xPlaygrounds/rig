//! Driving the agents from a program's thread. [`Harness`] is a cloneable
//! resource: the `run_steel` task keeps a clone and spawns agents, sends
//! them requests, awaits their replies and calls their tools. Each call is a
//! job sent into the world, which runs it in the next frame (the call asks
//! for one through [`Wake`]) and answers on a oneshot channel. Tool calls
//! made through it are tool call effects ([`ToolRequest`]), like the
//! model's own: gated, started and recorded the same way. Made for a call
//! ([`Harness::within`]), they are that call's children (`ChildOf`): they
//! end with it, so an interrupt that ends the call ends a tool call it held
//! too, and whatever held it.
//!
//! Each request it sends has a [`Receipt`], an entity that holds the reply
//! until it is read, [`RequestOf`] the call it was sent for and
//! [`ForAgent`] the agent it went to, so it goes when either goes.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use futures::channel::oneshot;
use rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_core::tool::ToolResult;
use rig_ecs::agent::{Agent, AgentId, TurnOutcome};
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_ecs::prelude::{Author, RequestId};
use rig_ecs::requests::RequestOf;
use rig_ecs::spawn::{AgentSpec, Origin, spawn};
use rig_ecs::stage::{
    AgentLoop, AgentWork, Effect, ForAgent, Outcome, Settled, despawn_open_children,
    follow_successor,
};
use rig_ecs::tasks::{PollTasks, Wake};
use rig_ecs::tools::ToolRequest;

/// The [`Harness`] resource and the jobs, replies and tool calls behind it.
pub(crate) struct HarnessPlugin;

impl Plugin for HarnessPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Jobs>()
            .add_systems(PreStartup, insert_harness)
            .add_systems(Update, run_jobs.before(PollTasks).before(AgentLoop))
            .add_observer(end_replies)
            .add_observer(answer_call)
            .add_observer(follow_successor::<HarnessCall>)
            .add_observer(despawn_open_children::<HarnessCall>);
    }
}

/// A job the world runs for a [`Harness`] call.
type Job = Box<dyn FnOnce(&mut World) + Send>;

/// Where a call's answer goes: its value, or why it failed.
type Answer<T> = oneshot::Sender<Result<T, String>>;

/// Why a call failed when the app stopped, or dropped the call, before it
/// answered.
const CLOSED: &str = "the app stopped before it answered";

/// Why a tool call was not made for a call that ended.
const ENDED: &str = "the call it was made for ended";

/// A handle that drives the agents from async code. Clone it out of the
/// world; every call is answered in a later frame, so awaiting one never
/// blocks the app's loop. Available from `PreStartup` on.
#[derive(Resource, Clone)]
pub struct Harness {
    jobs: Sender<Job>,
    wake: Wake,
    call: Option<Entity>,
}

impl Harness {
    /// This handle doing the work of `call`, such as an open tool call:
    /// tool calls through it are its children, recorded under it and ended
    /// with it, and the model calls of agents it spawns are recorded under
    /// it. Once `call` ended, every call through it fails and does nothing.
    pub fn within(&self, call: Entity) -> Self {
        Self {
            call: Some(call),
            ..self.clone()
        }
    }

    /// Spawns an agent from `spec`, a child of the agent `parent` when
    /// given, as a [`Spawner`](rig_ecs::spawn::Spawner) does, and returns
    /// its id. It is idle until a message is sent to it. Refused past the
    /// depth limit, or for a tool in `spec.tools` the parent may not call.
    pub async fn spawn_agent(
        &self,
        spec: AgentSpec,
        parent: Option<AgentId>,
    ) -> Result<AgentId, String> {
        let call = self.call;
        self.ask(move |world, answer: Answer<AgentId>| {
            answer.send(spawn_agent(world, spec, parent, call)).ok();
        })
        .await
    }

    /// Delivers `text` to the agent `to` as a request from `author`, queued
    /// behind its current work when it is busy, and returns the request's
    /// id: `author.request` when given, a new one otherwise. A request id
    /// already sent is not delivered again, so it doubles as an idempotency
    /// key.
    pub async fn send(
        &self,
        to: AgentId,
        text: String,
        author: Author,
    ) -> Result<RequestId, String> {
        let call = self.call;
        self.ask(move |world, answer: Answer<RequestId>| {
            answer.send(send(world, &to, text, author, call)).ok();
        })
        .await
    }

    /// How the turn that answered `request` ended: the first turn of its
    /// agent to end once the request was sent, which reads it before it
    /// ends. Resolves at once when that turn already ended. The outcome is
    /// kept for one `reply`; another for the same request fails.
    pub async fn reply(&self, request: RequestId) -> Result<TurnOutcome, String> {
        self.ask(move |world, answer: Answer<TurnOutcome>| {
            await_reply(world, request, answer);
        })
        .await
    }

    /// Calls the tool `name` with `args` as the agent `agent`, which must be
    /// allowed it, on the one dispatch path, and returns its result. A tool
    /// that is unknown or refuses the arguments answers with an error
    /// result, as a model's call would get.
    pub async fn call_tool(
        &self,
        agent: AgentId,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult, String> {
        let call = self.call;
        let name = name.to_owned();
        self.ask(move |world, answer: Answer<ToolResult>| {
            call_tool(world, &agent, name, args, call, answer);
        })
        .await
    }

    /// Sends `job` into the world and wakes its loop; the job answers on
    /// the channel it is given. It does not run once the call this handle
    /// works for ended.
    async fn ask<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut World, Answer<T>) + Send + 'static,
    ) -> Result<T, String> {
        let (answer, answered) = oneshot::channel();
        let call = self.call;
        let job = move |world: &mut World| match call {
            Some(call) if !open(world, call) => {
                answer.send(Err(ENDED.to_owned())).ok();
            }
            _ => job(world, answer),
        };
        let sent = self.jobs.send(Box::new(job)).is_ok();
        self.wake.wake();
        if !sent {
            return Err(CLOSED.to_owned());
        }
        answered.await.unwrap_or_else(|_| Err(CLOSED.to_owned()))
    }
}

/// The channel [`Harness`] calls come in on.
#[derive(Resource)]
struct Jobs {
    sender: Sender<Job>,
    receiver: Receiver<Job>,
}

impl Default for Jobs {
    fn default() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self { sender, receiver }
    }
}

/// Inserts the [`Harness`], once the loop's [`Wake`] is set.
fn insert_harness(jobs: Res<Jobs>, wake: Res<Wake>, mut commands: Commands) {
    commands.insert_resource(Harness {
        jobs: jobs.sender.clone(),
        wake: wake.clone(),
        call: None,
    });
}

/// Runs the jobs [`Harness`] calls sent since the last frame.
fn run_jobs(world: &mut World) {
    let Some(jobs) = world
        .get_resource::<Jobs>()
        .map(|jobs| jobs.receiver.clone())
    else {
        return;
    };
    while let Ok(job) = jobs.try_recv() {
        job(world);
    }
}

/// Whether `call` is there and not settled.
fn open(world: &World, call: Entity) -> bool {
    world
        .get_entity(call)
        .is_ok_and(|call| !call.contains::<Settled>())
}

/// The agent with the id `id`.
fn find(world: &mut World, id: &AgentId) -> Result<Entity, String> {
    let mut agents = world.query_filtered::<(Entity, &AgentId), With<Agent>>();
    id.find_in(agents.iter(world))
        .ok_or_else(|| format!("no agent has the id `{}`", id.0))
}

fn spawn_agent(
    world: &mut World,
    spec: AgentSpec,
    parent: Option<AgentId>,
    call: Option<Entity>,
) -> Result<AgentId, String> {
    let under = call
        .and_then(|call| world.get::<Effect>(call))
        .map(Effect::id);
    let origin = match parent {
        Some(parent) => Origin::ChildOf(find(world, &parent)?),
        None => Origin::Root,
    };
    let id = AgentId::default();
    let spec = AgentSpec {
        id: Some(id.clone()),
        under: under.or(spec.under),
        ..spec
    };
    spawn(world, origin, spec).map_err(|why| why.to_string())?;
    Ok(id)
}

/// A request sent through a [`Harness`], [`ForAgent`] the agent it went to:
/// its [`Harness::reply`], until read.
#[derive(Component)]
struct Receipt {
    request: RequestId,
    state: ReplyState,
}

/// A request's outcome, or who waits for it.
enum ReplyState {
    Waiting(Vec<Answer<TurnOutcome>>),
    Ended(TurnOutcome),
}

fn send(
    world: &mut World,
    to: &AgentId,
    text: String,
    author: Author,
    call: Option<Entity>,
) -> Result<RequestId, String> {
    let agent = find(world, to)?;
    if text.trim().is_empty() {
        return Err("the message is empty".to_owned());
    }
    let request = author
        .request
        .clone()
        .unwrap_or_else(|| RequestId(uuid::Uuid::new_v4().to_string()));
    let mut receipts = world.query::<&Receipt>();
    if receipts
        .iter(world)
        .any(|receipt| receipt.request == request)
    {
        return Ok(request);
    }
    let receipt = Receipt {
        request: request.clone(),
        state: ReplyState::Waiting(Vec::new()),
    };
    let mut receipt = world.spawn((receipt, ForAgent(agent)));
    if let Some(call) = call {
        receipt.insert(RequestOf(call));
    }
    let author = Author {
        request: Some(request.clone()),
        ..author
    };
    world.trigger(Deliver::new(agent, text, DeliveryMode::Queue).with_author(author));
    Ok(request)
}

fn await_reply(world: &mut World, request: RequestId, answer: Answer<TurnOutcome>) {
    let mut receipts = world.query::<(Entity, &mut Receipt)>();
    let found = receipts
        .iter_mut(world)
        .find(|(_, receipt)| receipt.request == request);
    let Some((entity, mut receipt)) = found else {
        answer
            .send(Err(format!("no request `{}` was sent", request.0)))
            .ok();
        return;
    };
    if let ReplyState::Waiting(waiting) = &mut receipt.state {
        waiting.push(answer);
        return;
    }
    // Read once: the receipt goes with its reply.
    let Ok(mut entity) = world.get_entity_mut(entity) else {
        return;
    };
    if let Some(Receipt {
        state: ReplyState::Ended(outcome),
        ..
    }) = entity.take::<Receipt>()
    {
        answer.send(Ok(outcome)).ok();
    }
    entity.despawn();
}

/// Ends every request waiting on the agent whose turn ended with that
/// turn's outcome: a turn reads every message queued for it before it
/// ends, so it answers them all. Whoever waits for a receipt that goes
/// first is told the app stopped before it answered.
fn end_replies(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    work: Query<&AgentWork>,
    mut receipts: Query<&mut Receipt>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(work) = work.get(agent) else {
        return;
    };
    for receipt in work.iter() {
        let Ok(mut receipt) = receipts.get_mut(receipt) else {
            continue;
        };
        if let ReplyState::Waiting(waiting) = &mut receipt.state {
            for answer in waiting.drain(..) {
                answer.send(Ok(outcome.clone())).ok();
            }
            receipt.state = ReplyState::Ended(outcome.clone());
        }
    }
}

/// On a tool call made through a [`Harness`]: where its result goes.
#[derive(Component)]
struct HarnessCall(Option<Answer<ToolResult>>);

fn call_tool(
    world: &mut World,
    agent: &AgentId,
    name: String,
    args: serde_json::Value,
    within: Option<Entity>,
    answer: Answer<ToolResult>,
) {
    let name = ToolName::new(name).map_err(|error| error.to_string());
    let (agent, name) = match find(world, agent).and_then(|agent| Ok((agent, name?))) {
        Ok(found) => found,
        Err(error) => {
            answer.send(Err(error)).ok();
            return;
        }
    };
    let call = ToolCall::from_wire("", ToolFunction::new(name, args));
    let call = (
        ToolRequest { call },
        ForAgent(agent),
        HarnessCall(Some(answer)),
    );
    // Spawned with its parent, which its effect takes its parent from.
    match within {
        Some(within) => world.spawn((call, ChildOf(within))),
        None => world.spawn(call),
    };
}

/// Answers a [`Harness`] tool call with its result once it settled.
fn answer_call(settled: On<Add<Settled>>, mut calls: Query<(&Outcome, &mut HarnessCall)>) {
    let Ok((outcome, mut call)) = calls.get_mut(settled.entity) else {
        return;
    };
    if let Some(answer) = call.0.take() {
        answer.send(Ok(outcome.tool_result())).ok();
    }
}
