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

use std::collections::HashMap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use futures::channel::oneshot;
use rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_core::tool::ToolResult;
use rig_ecs::agent::{
    Agent, AgentId, EffectParent, SpawnedBy, SystemPrompt, ToolAccess, TurnOutcome,
};
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_ecs::model::{ModelChoice, Models, ReasoningSetting};
use rig_ecs::prelude::{Author, RequestId};
use rig_ecs::stage::{
    AgentLoop, Effect, ForAgent, Outcome, Settled, despawn_open_children, follow_successor,
};
use rig_ecs::tasks::{PollTasks, Wake};
use rig_ecs::tools::{ToolDef, ToolRequest};

/// The [`Harness`] resource and the jobs, replies and tool calls behind it.
pub(crate) struct HarnessPlugin;

impl Plugin for HarnessPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Jobs>()
            .init_resource::<Replies>()
            .add_systems(PreStartup, insert_harness)
            .add_systems(Update, run_jobs.before(PollTasks).before(AgentLoop))
            .add_observer(end_replies)
            .add_observer(drop_replies)
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

/// What [`Harness::spawn_agent`] starts. Each setting left out is taken
/// from the parent, or is the default without one; the effort is the
/// parent's only when the model is too.
#[derive(Clone, Debug, Default)]
pub struct AgentSpec {
    /// The agent's name, shown in views.
    pub name: String,
    /// A catalog model as `vendor/model`.
    pub model: Option<String>,
    /// The agent's own part of its system prompt.
    pub system_prompt: Option<String>,
    /// The tools it may call, by name: each a registered tool its parent
    /// may call, or every tool without a parent.
    pub tools: Option<Vec<String>>,
}

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
    /// it.
    pub fn within(&self, call: Entity) -> Self {
        Self {
            call: Some(call),
            ..self.clone()
        }
    }

    /// Spawns an agent from `spec`, [`SpawnedBy`] the agent `parent` when
    /// given, and returns its id. It is idle until a message is sent to it.
    /// Refused for a tool in `spec.tools` the parent may not call.
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
        self.ask(move |world, answer: Answer<RequestId>| {
            answer.send(send(world, &to, text, author)).ok();
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
    /// the channel it is given.
    async fn ask<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut World, Answer<T>) + Send + 'static,
    ) -> Result<T, String> {
        let (answer, answered) = oneshot::channel();
        let sent = self
            .jobs
            .send(Box::new(move |world: &mut World| job(world, answer)))
            .is_ok();
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
    let effect = call
        .and_then(|call| world.get::<Effect>(call))
        .map(Effect::id);
    let parent = parent.map(|parent| find(world, &parent)).transpose()?;
    let inherited = parent.map(|parent| {
        (
            world.get::<ModelChoice>(parent).cloned(),
            world
                .get::<ReasoningSetting>(parent)
                .copied()
                .unwrap_or_default(),
            world.get::<SystemPrompt>(parent).cloned(),
            world.get::<ToolAccess>(parent).cloned(),
        )
    });
    let (parent_model, parent_effort, parent_prompt, parent_access) = inherited.unwrap_or_default();
    let models = world.get_resource_or_init::<Models>();
    let (model, effort) = ModelChoice::inherit(
        &models.0,
        (parent_model.as_ref(), parent_effort),
        spec.model.as_deref(),
        None,
    )?;
    let prompt = spec
        .system_prompt
        .map(SystemPrompt)
        .or(parent_prompt)
        .unwrap_or_default();
    let parent_access = parent_access.unwrap_or_default();
    let access = match &spec.tools {
        Some(asked) => {
            let mut tools = world.query::<&ToolDef>();
            let registered = tools.iter(world).map(|def| def.0.name.as_str());
            parent_access
                .grant(asked, registered)
                .map_err(|refused| refused.to_string())?
        }
        None => parent_access,
    };
    let id = AgentId::default();
    let mut agent = world.spawn((Agent, id.clone(), Name::new(spec.name), prompt, access));
    // The parent first, so the agent's log opens naming it.
    if let Some(parent) = parent {
        agent.insert(SpawnedBy(parent));
    }
    if let Some(effect) = effect {
        agent.insert(EffectParent(effect));
    }
    agent.insert(effort);
    if let Some(model) = model {
        agent.insert(model);
    }
    Ok(id)
}

/// The [`Harness::reply`] of each request, by id, until the reply is read
/// or its agent is gone.
#[derive(Resource, Default)]
struct Replies(HashMap<RequestId, Reply>);

/// A request sent through a [`Harness`].
struct Reply {
    /// The agent it went to.
    agent: Entity,
    /// Its outcome, or who waits for it.
    state: ReplyState,
}

enum ReplyState {
    Waiting(Vec<Answer<TurnOutcome>>),
    Ended(TurnOutcome),
}

fn send(
    world: &mut World,
    to: &AgentId,
    text: String,
    author: Author,
) -> Result<RequestId, String> {
    let agent = find(world, to)?;
    if text.trim().is_empty() {
        return Err("the message is empty".to_owned());
    }
    let request = author
        .request
        .clone()
        .unwrap_or_else(|| RequestId(uuid::Uuid::new_v4().to_string()));
    let mut replies = world.get_resource_or_init::<Replies>();
    if replies.0.contains_key(&request) {
        return Ok(request);
    }
    replies.0.insert(
        request.clone(),
        Reply {
            agent,
            state: ReplyState::Waiting(Vec::new()),
        },
    );
    let author = Author {
        request: Some(request.clone()),
        ..author
    };
    world.trigger(Deliver::new(agent, text, DeliveryMode::Queue).with_author(author));
    Ok(request)
}

fn await_reply(world: &mut World, request: RequestId, answer: Answer<TurnOutcome>) {
    let mut replies = world.get_resource_or_init::<Replies>();
    let Some(reply) = replies.0.get_mut(&request) else {
        answer
            .send(Err(format!("no request `{}` was sent", request.0)))
            .ok();
        return;
    };
    if let ReplyState::Waiting(waiting) = &mut reply.state {
        waiting.push(answer);
        return;
    }
    // Read once: the outcome is not kept after its reply.
    if let Some(Reply {
        state: ReplyState::Ended(outcome),
        ..
    }) = replies.0.remove(&request)
    {
        answer.send(Ok(outcome)).ok();
    }
}

/// Ends every request waiting on the agent whose turn ended with that
/// turn's outcome: a turn reads every message queued for it before it
/// ends, so it answers them all.
fn end_replies(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut replies: ResMut<Replies>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    for reply in replies.0.values_mut() {
        if reply.agent != agent {
            continue;
        }
        if let ReplyState::Waiting(waiting) = &mut reply.state {
            for answer in waiting.drain(..) {
                answer.send(Ok(outcome.clone())).ok();
            }
            reply.state = ReplyState::Ended(outcome.clone());
        }
    }
}

/// Drops the replies of a despawned agent: whoever waits for one is told
/// the app stopped before it answered.
fn drop_replies(gone: On<Remove<Agent>>, mut replies: ResMut<Replies>) {
    replies.0.retain(|_, reply| reply.agent != gone.entity);
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
    let open = |call: Entity| {
        world
            .get_entity(call)
            .is_ok_and(|call| !call.contains::<Settled>())
    };
    if within.is_some_and(|call| !open(call)) {
        answer.send(Err(ENDED.to_owned())).ok();
        return;
    }
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
