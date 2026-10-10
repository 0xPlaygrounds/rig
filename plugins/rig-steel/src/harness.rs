//! Driving the agents from a program's thread. [`Harness`] is a cloneable
//! resource: the `run_steel` task keeps a clone and spawns agents, sends
//! them requests, awaits their replies and calls their tools. Each call is a
//! job sent into the world, which runs it in the next frame (the call asks
//! for one through [`Wake`]) and answers on a oneshot channel. Tool calls
//! made through it go through the one recorded dispatch path
//! ([`ToolStarter`]), like the model's own.

use std::collections::HashMap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use futures::channel::oneshot;
use rig_core::effect::EffectId;
use rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_core::tool::ToolResult;
use rig_ecs::agent::{
    Agent, AgentId, EffectParent, SpawnedBy, SystemPrompt, ToolAccess, TurnEnded, TurnOutcome,
};
use rig_ecs::calls::{PollCalls, Wake};
use rig_ecs::inbox::{Deliver, DeliveryMode, Origin, RequestId};
use rig_ecs::model::{Effort, ModelChoice, Models};
use rig_ecs::tools::ToolOutput;
use rig_ecs::turn::{ToolStarter, tool_name};

/// The [`Harness`] resource and the jobs, replies and tool calls behind it.
pub(crate) struct HarnessPlugin;

impl Plugin for HarnessPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Jobs>()
            .init_resource::<Replies>()
            .add_systems(PreStartup, insert_harness)
            .add_systems(Update, run_jobs.before(PollCalls))
            .add_observer(end_replies)
            .add_observer(drop_replies)
            .add_observer(answer_call);
    }
}

/// A job the world runs for a [`Harness`] call.
type Job = Box<dyn FnOnce(&mut World) + Send>;

/// Where a call's answer goes: its value, or why it failed.
type Answer<T> = oneshot::Sender<Result<T, String>>;

/// Why a call failed when the app stopped, or dropped the call, before it
/// answered.
const CLOSED: &str = "the app stopped before it answered";

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
    /// The tools it may call, by name.
    pub tools: Option<Vec<String>>,
}

/// A handle that drives the agents from async code. Clone it out of the
/// world; every call is answered in a later frame, so awaiting one never
/// blocks the app's loop. Available from `PreStartup` on.
#[derive(Resource, Clone)]
pub struct Harness {
    jobs: Sender<Job>,
    wake: Wake,
    parent: Option<EffectId>,
}

impl Harness {
    /// This handle with `parent`, such as the open tool call whose work it
    /// does, as the effect that tool calls through it and the model calls
    /// of agents it spawns are recorded under.
    pub fn within(&self, parent: EffectId) -> Self {
        Self {
            parent: Some(parent),
            ..self.clone()
        }
    }

    /// Spawns an agent from `spec`, [`SpawnedBy`] the agent `parent` when
    /// given, and returns its id. It is idle until a message is sent to it.
    pub async fn spawn_agent(
        &self,
        spec: AgentSpec,
        parent: Option<AgentId>,
    ) -> Result<AgentId, String> {
        let effect = self.parent;
        self.ask(move |world, answer: Answer<AgentId>| {
            answer.send(spawn_agent(world, spec, parent, effect)).ok();
        })
        .await
    }

    /// Delivers `text` to the agent `to` as a request from `origin`, queued
    /// behind its current work when it is busy, and returns the request's
    /// id: `origin.request` when given, a new one otherwise. A request id
    /// already sent is not delivered again, so it doubles as an idempotency
    /// key.
    pub async fn send(
        &self,
        to: AgentId,
        text: String,
        origin: Origin,
    ) -> Result<RequestId, String> {
        self.ask(move |world, answer: Answer<RequestId>| {
            answer.send(send(world, &to, text, origin)).ok();
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
        let effect = self.parent;
        let name = name.to_owned();
        self.ask(move |world, answer: Answer<ToolResult>| {
            call_tool(world, &agent, name, args, effect, answer);
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
        parent: None,
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
    world
        .query_filtered::<(Entity, &AgentId), With<Agent>>()
        .iter(world)
        .find(|(_, of)| *of == id)
        .map(|(agent, _)| agent)
        .ok_or_else(|| format!("no agent has the id `{}`", id.0))
}

fn spawn_agent(
    world: &mut World,
    spec: AgentSpec,
    parent: Option<AgentId>,
    effect: Option<EffectId>,
) -> Result<AgentId, String> {
    let parent = parent.map(|parent| find(world, &parent)).transpose()?;
    let inherited = parent.map(|parent| {
        (
            world.get::<ModelChoice>(parent).cloned(),
            world.get::<Effort>(parent).copied().unwrap_or_default(),
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
    let access = spec
        .tools
        .map(ToolAccess::Only)
        .or(parent_access)
        .unwrap_or_default();
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
    origin: Origin,
) -> Result<RequestId, String> {
    let agent = find(world, to)?;
    if text.trim().is_empty() {
        return Err("the message is empty".to_owned());
    }
    let request = origin
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
    world.trigger(Deliver {
        entity: agent,
        text,
        origin: Origin {
            request: Some(request.clone()),
            ..origin
        },
        mode: DeliveryMode::Queue,
        attachments: Vec::new(),
    });
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
fn end_replies(end: On<TurnEnded>, mut replies: ResMut<Replies>) {
    for reply in replies.0.values_mut() {
        if reply.agent != end.entity {
            continue;
        }
        if let ReplyState::Waiting(waiting) = &mut reply.state {
            for answer in waiting.drain(..) {
                answer.send(Ok(end.outcome.clone())).ok();
            }
            reply.state = ReplyState::Ended(end.outcome.clone());
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
    effect: Option<EffectId>,
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
    // A failure drops `answer`, which the caller reads as closed.
    world
        .run_system_cached_with(start_call, (agent, call, effect, answer))
        .ok();
}

/// Starts a [`Harness`] tool call as an entity of its own, outside any
/// turn, on the path a model's call takes.
fn start_call(
    In((agent, call, parent, answer)): In<(Entity, ToolCall, Option<EffectId>, Answer<ToolResult>)>,
    starter: ToolStarter,
    mut commands: Commands,
) {
    let run = starter.run(call, parent);
    let entity = commands
        .spawn((tool_name(&run), run.clone(), HarnessCall(Some(answer))))
        .id();
    starter.start(&mut commands, entity, agent, &run);
}

/// Answers a [`Harness`] tool call with its output, then despawns it.
fn answer_call(
    done: On<Add<ToolOutput>>,
    mut calls: Query<(&ToolOutput, &mut HarnessCall)>,
    mut commands: Commands,
) {
    let Ok((ToolOutput(result), mut call)) = calls.get_mut(done.entity) else {
        return;
    };
    if let Some(answer) = call.0.take() {
        answer.send(Ok(result.clone())).ok();
    }
    commands.entity(done.entity).try_despawn();
}
