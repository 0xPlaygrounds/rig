//! The turn loop. A user message spawns a turn entity, [`TurnOf`] its
//! agent; the turn's model call and tool calls are entities [`CallOf`] the
//! turn, and observers of their [`Done`] outputs and [`ToolOutput`]s carry
//! the turn on. A reply's read-only tool calls run side by side and the
//! others alone, in order, as their tools' [`Footprint`]s say; their
//! results go back in call order. rig-core's turn-failure
//! rule decides when a reply ends the turn instead. A failed model call is
//! retried after a [`Backoff`] or recovered from as rig-core's [`RETRY`]
//! policy decides; each turn counts its attempts in its [`Recovery`].
//! Plugins extend a turn: [`PrepareRequest`] before each model request,
//! which may rewrite what it sends or hold it with calls of their own, such
//! as a [`ModelRequest`] on the agent's model; and [`ModelFailed`] after a
//! call fails in a way a retry does not fix. Despawning the turn ends it,
//! and announces its [`TurnEnded`] on the agent.

use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::tracing::Instrument;
use bevy_log::{info, info_span, warn};
use bevy_reflect::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{AsyncComputeTaskPool, IoTaskPool, TaskPool};
use bevy_time::DelayedCommandsExt;
use crossbeam_channel::{Receiver, Sender};
use futures::StreamExt;
use rig_core::completion::message::turn_failure;
use rig_core::completion::{
    CompletionRequest, CompletionResponse, GenerationOptions, Message, ToolDefinition,
};
use rig_core::effect::Outcome;
use rig_core::effect::family::Completion;
use rig_core::effect::{EffectId, EffectKind, Family};
use rig_core::error::retry::{RetryPolicy, Verdict};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolCall, UserContent};
use rig_core::serve::{Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};
use rig_core::tool::{ToolErrorKind, ToolExecutionError, ToolResult};
use rig_core::transcript::{arguments_refusal, close_pending_with};
use serde::{Deserialize, Serialize};
use web_time::Instant;

use super::agent::{
    ActiveTurn, Agent, AgentId, CallOf, Calls, Condensed, Conversation, EffectParent, Ending, Halt,
    Interrupt, LastUsage, Notice, Partial, Queued, Retry, SystemPrompt, ToolAccess, ToolCallRun,
    TurnEnded, TurnOf, TurnOutcome,
};
use super::calls::{Done, Running, Wake};
use super::effects::Effects;
use super::inbox::{Inbox, Pending};
use super::journal::{Commit, SessionLog, commit_message};
use super::model::{Connection, Effort};
use super::prompt::{PromptSection, SectionOf, ToolRules, system_prompt};
use super::tools::{Footprint, OpenCall, Serves, ToolDef, ToolOutput, run_tool_call};

/// The notice when the agent has no model to call.
pub(crate) const NO_MODEL: &str = "No model is connected; pick one first.";

/// How failed model calls are retried: rig-core's default, four retries in
/// a row per turn (a reply resets the count).
pub const RETRY: RetryPolicy = RetryPolicy::DEFAULT;

/// A turn's recovery so far: the failed calls retried since its last reply.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
pub struct Recovery {
    /// Retries since the last reply.
    pub retries: u32,
}

/// A wait before the turn's next model call, on a call entity of the turn,
/// so interrupting the turn cancels it like any other call. The call is
/// sent again by a delayed command on Bevy's clock, which despawns this
/// entity first.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, from_reflect = false)]
pub struct Backoff {
    /// Which retry this wait is for, from 1 to [`RETRY`]'s `max_retries`.
    pub attempt: u32,
    /// When the call is sent again.
    #[reflect(ignore)]
    pub until: Instant,
}

impl Backoff {
    /// Whole seconds left to wait, rounded up.
    pub fn seconds_left(&self) -> u64 {
        let left = self.until.saturating_duration_since(Instant::now());
        left.as_secs() + u64::from(left.subsec_nanos() > 0)
    }
}

/// What a model call's task returns.
pub type ModelReply = Result<CompletionResponse, ErrorReport>;

/// A streaming model call of the turn it is a [`CallOf`]; its [`Running`]
/// task ends as a [`Done<ModelReply>`](Done).
#[derive(Component)]
pub struct ModelCall {
    effect: EffectId,
    feed: Receiver<Delta>,
}

/// A streamed fragment for [`Partial`].
enum Delta {
    Text(String),
    Reasoning(String),
}

/// Sends the conversation of the turn's agent to its model.
#[derive(EntityEvent, Reflect, Clone, Copy, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct CallModel {
    /// The turn.
    pub entity: Entity,
}

/// Triggered on a turn right before its model request is sent, with
/// everything it sends, which observers may change: the system prompt, the
/// messages (the agent's conversation, after its [`Condensed`] summary if
/// any, with what was delivered meanwhile), the tools offered and the
/// generation options. A changed system prompt or tool list misses the
/// provider's prompt cache. The request is checked against the model only
/// afterwards, and a tool left out is still refused only if the agent's
/// [`ToolAccess`] does not allow it. Observers may also spawn calls
/// [`CallOf`] the turn, such as a [`ModelRequest`]: while the turn has
/// calls afterwards, nothing is sent, and the plugin that spawned them
/// triggers [`CallModel`] again once they are done. Observers run in no
/// set order, so each does its own part only.
///
/// The request goes to the turn's model: a [`Connection`] on the turn,
/// such as one a plugin inserts on `On<Add<TurnOf>>`, else the agent's. The
/// options are made for that model, with the agent's reasoning setting
/// when the model takes it; the agent's model still decides its
/// [`LastUsage`] and context window.
#[derive(EntityEvent, Clone, Debug)]
pub struct PrepareRequest {
    /// The turn.
    pub entity: Entity,
    /// The turn's agent.
    pub agent: Entity,
    /// The system prompt: the agent's [`SystemPrompt`], the rules of the
    /// tools offered and the prompt sections.
    pub preamble: String,
    /// The messages the request sends, oldest first.
    pub messages: Vec<Message>,
    /// The tools offered, by name.
    pub tools: Vec<ToolDefinition>,
    /// The generation options, such as `temperature` or `max_tokens`.
    pub options: GenerationOptions,
}

/// Triggered on a turn whose model call failed in a way sending the same
/// request again does not fix, such as a request longer than the model's
/// window ([`ErrorReport::is_context_overflow`]). An observer that takes
/// the turn over, such as by shrinking the conversation and triggering
/// [`CallModel`], sets `handled`; otherwise the turn fails, and the user's
/// last message is taken out.
#[derive(EntityEvent, Clone, Debug)]
pub struct ModelFailed {
    /// The turn.
    pub entity: Entity,
    /// The turn's agent.
    pub agent: Entity,
    /// Why the call failed.
    pub report: ErrorReport,
    /// Whether an observer took the turn over.
    pub handled: bool,
}

/// A plugin's model call on the agent's model, such as a summary of its
/// conversation. Spawned [`CallOf`] a turn, it is dispatched and recorded
/// like the turn's own calls, with the agent's connection, and ends with a
/// [`Done<ModelReply>`](Done) on its entity, which is the plugin's to take.
/// Interrupting the turn cancels it.
#[derive(Component, Reflect, Clone, Debug, Serialize, Deserialize)]
#[reflect(opaque, Component, Clone, Debug, Serialize, Deserialize)]
pub struct ModelRequest {
    /// The request.
    pub request: CompletionRequest,
}

/// Model calls wait on the network: they run on the IO pool.
fn model_pool() -> &'static IoTaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Tool calls run on the async compute pool; their blocking work runs on
/// threads of its own (see `rig_tools::blocking`).
fn tool_pool() -> &'static AsyncComputeTaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// Sends the conversation to the model again as it stands, when it ends
/// in a message the model has not answered, also a halted one.
pub(crate) fn on_retry(
    retry: On<Retry>,
    mut agents: Query<(&mut Conversation, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = retry.entity;
    let Ok((mut conversation, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::info(agent, "A turn is running."));
        return;
    }
    if !conversation.resume() {
        notices.write(Notice::info(
            agent,
            "Nothing to retry: the model answered the last message.",
        ));
        return;
    }
    let turn = commands.spawn((Name::new("turn"), TurnOf(agent))).id();
    commands.trigger(CallModel { entity: turn });
}

/// Announces a turn's end with [`TurnEnded`], however its entity went
/// away: with the [`Ending`] it was given, or as stopped. The event goes
/// once the agent is idle, after the relationship dropped its
/// [`ActiveTurn`]. A turn the exit stops is not ended: it is left for the
/// restart to carry on, as after a crash.
pub(crate) fn on_turn_despawn(
    end: On<Remove<TurnOf>>,
    turns: Query<(&TurnOf, Option<&Ending>)>,
    agents: Query<&AgentId>,
    exiting: Option<Res<Exiting>>,
    mut commands: Commands,
) {
    let Ok((&TurnOf(agent), ending)) = turns.get(end.entity) else {
        return;
    };
    if exiting.is_some() {
        return;
    }
    let outcome = ending.map_or(TurnOutcome::Stopped, |ending| ending.0.clone());
    if let Ok(id) = agents.get(agent) {
        let how = match &outcome {
            TurnOutcome::Answered(_) => "answered",
            TurnOutcome::Failed(_) => "failed",
            TurnOutcome::Stopped => "stopped",
        };
        info!(agent = %id.0, "turn ended: {how}");
    }
    commands.trigger(TurnEnded {
        entity: agent,
        outcome,
    });
}

/// Ends `turn` as `outcome` says.
fn end_turn(commands: &mut Commands, turn: Entity, outcome: TurnOutcome) {
    commands.entity(turn).insert(Ending(outcome)).despawn();
}

/// Stops a running turn. The text a streaming reply had sent is kept as
/// the model's message, every tool call of the last reply gets a result,
/// real or "interrupted", and despawning the turn cancels its calls. The
/// outcome is written to the log at once, so a log that ends mid-turn
/// always means a crash or a restart.
pub(crate) fn on_interrupt(
    interrupt: On<Interrupt>,
    mut agents: Query<(&mut Conversation, &ActiveTurn)>,
    turns: Query<&Calls>,
    runs: Query<(&ToolCallRun, Option<&ToolOutput>)>,
    partials: Query<&Partial, With<ModelCall>>,
    mut commit: Commit,
    log: Res<SessionLog>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = interrupt.entity;
    let Ok((mut conversation, active)) = agents.get_mut(agent) else {
        return;
    };
    let turn = active.turn();
    let calls: Vec<Entity> = turns
        .get(turn)
        .map(|calls| calls.iter().collect())
        .unwrap_or_default();
    let aborted: String = partials
        .iter_many(calls.iter().copied())
        .flatten()
        .map(|partial| partial.text.as_str())
        .collect();
    if !aborted.trim().is_empty() {
        commit.message(agent, &mut conversation, Message::assistant(aborted));
    }
    if let Some(results) = stopped_results(
        runs.iter_many(calls.iter().copied()).flatten(),
        "interrupted by the user",
    ) {
        commit.message(agent, &mut conversation, results);
    }
    commit.halt(agent, &mut conversation, Halt::Stopped);
    log.flush();
    commands.entity(turn).despawn();
    notices.write(Notice::info(agent, "Interrupted."));
}

/// The results of a stopped reply's tool calls, in call order: each
/// finished call's result, and an error saying `why` for the others. `None`
/// when the reply made no calls.
fn stopped_results<'a>(
    runs: impl Iterator<Item = (&'a ToolCallRun, Option<&'a ToolOutput>)>,
    why: &str,
) -> Option<Message> {
    let why = ToolResult::failed(ToolExecutionError::other(why));
    let results: Vec<_> = runs
        .map(|(run, done)| {
            run.call
                .answer(done.map_or(&why, |ToolOutput(result)| result))
        })
        .collect();
    (!results.is_empty()).then(|| Message::tool_results(results))
}

/// How long exit waits for running calls to be cancelled. A cancellation
/// waits for a pool thread to drop the call's future, which records its
/// effect as cancelled before the last flush; a plugin tool that blocks in
/// `poll` must not stall the exit, so whatever misses this is dropped.
const EXIT_GRACE: Duration = Duration::from_secs(1);

type Cancelling = Vec<Pin<Box<dyn Future<Output = ()>>>>;

/// Inserted when the app exits: the turns it stops are left for the
/// restart to carry on, so they end without a [`TurnEnded`] and are not
/// logged as halted.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct Exiting;

/// On exit (`/quit` or a signal; `/reload` and switching sessions wait for
/// idle agents) leaves the running turns for the restart to carry on, as
/// after a crash: every [`Running`] task, a plugin's too, is cancelled, the
/// tool results that came in are logged, and the restart answers the rest.
pub(crate) fn stop_turns_on_exit(world: &mut World) {
    world.insert_resource(Exiting);
    let mut cancelling = Cancelling::new();
    take_running(world, &mut cancelling);
    let deadline = Instant::now() + EXIT_GRACE;
    loop {
        cancelling.retain_mut(|cancel| check_ready(cancel).is_none());
        // On the web the calls run on this thread: waiting cannot help them.
        if cancelling.is_empty() || cfg!(target_family = "wasm") || Instant::now() >= deadline {
            break;
        }
        #[cfg(not(target_family = "wasm"))]
        std::thread::sleep(Duration::from_millis(2));
    }
    let turns: Vec<(Entity, Entity)> = world
        .query::<(Entity, &TurnOf)>()
        .iter(world)
        .map(|(turn, of)| (turn, of.0))
        .collect();
    let mut outputs = world.query::<(&ToolCallRun, &ToolOutput)>();
    for (turn, agent) in turns {
        let calls: Vec<Entity> = world
            .get::<Calls>(turn)
            .map(|calls| calls.iter().collect())
            .unwrap_or_default();
        let results: Vec<_> = outputs
            .iter_many(world, calls)
            .flatten()
            .map(|(run, ToolOutput(result))| run.call.answer(result))
            .collect();
        if !results.is_empty() {
            commit_message(world, agent, Message::tool_results(results));
        }
        world.despawn(turn);
    }
    world.flush();
}

/// Takes every [`Running`] task off its entity, a plugin's too, to be
/// cancelled.
fn take_running(world: &mut World, cancelling: &mut Cancelling) {
    let calls: Vec<Entity> = world
        .query_filtered::<Entity, With<Running>>()
        .iter(world)
        .collect();
    for call in calls {
        if let Ok(mut call) = world.get_entity_mut(call)
            && let Some(running) = call.take::<Running>()
        {
            let task = running.into_task();
            cancelling.push(Box::pin(async move {
                task.cancel().await;
            }));
        }
    }
}

/// Delivers what waits for the turn's agent, then lets [`PrepareRequest`]
/// observers see what the request sends. Unless they gave the turn calls
/// to wait for, the request is sent.
pub(crate) fn on_call_model(
    call: On<CallModel>,
    turns: Query<&TurnOf>,
    mut agents: Query<(
        &mut Conversation,
        &mut Inbox,
        Option<&Condensed>,
        Option<&Connection>,
    )>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let turn = call.entity;
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Ok((mut conversation, mut inbox, condensed, connection)) = agents.get_mut(agent) else {
        return;
    };
    let spec = connection.map(|connection| &*connection.spec);
    let (conversation, inbox) = (&mut *conversation, &mut *inbox);
    let waiting = inbox.notes.drain(..).chain(inbox.steering.drain(..));
    commit.pending(agent, spec, waiting, conversation, &mut notices);
    let messages = match condensed {
        Some(condensed) => condensed.request(conversation.messages()),
        None => conversation.messages().to_vec(),
    };
    commands.queue(move |world: &mut World| {
        if let Err(why) = request(world, turn, messages) {
            if let Err(error) = world.run_system_cached_with(fail_turn, (turn, why)) {
                warn!("could not end a failed turn: {error}");
            }
        }
    });
}

/// Prepares the turn's request with `messages`, lets [`PrepareRequest`]
/// observers change it, and sends it unless they gave the turn calls to
/// wait for; or why the turn fails.
fn request(world: &mut World, turn: Entity, messages: Vec<Message>) -> Result<(), String> {
    let prepared = world.run_system_cached_with(prepare_request, (turn, messages));
    let Some(mut prepare) = prepared.map_err(|error| error.to_string())?? else {
        return Ok(());
    };
    world.trigger_ref(&mut prepare);
    // The observers' calls exist once their commands are applied.
    world.flush();
    if world.get::<Calls>(turn).is_some() {
        return Ok(());
    }
    let sent = world.run_system_cached_with(send_request, prepare);
    sent.map_err(|error| error.to_string())?
}

/// What the turn's request sends with `messages`, on the turn's model: its
/// system prompt, the tools offered and the options; `None` for a turn
/// that is gone. It asks for the provider's prompt cache where the model
/// has one; the preamble and tools come first and do not change between
/// calls, so each call reads the prefix the last one wrote.
fn prepare_request(
    In((turn, messages)): In<(Entity, Vec<Message>)>,
    turns: Query<(&TurnOf, Option<&Connection>)>,
    agents: Query<(
        &AgentId,
        Option<&Connection>,
        &Effort,
        &SystemPrompt,
        &ToolAccess,
    )>,
    tools: Query<(&ToolDef, &ToolRules)>,
    sections: Query<(&PromptSection, Option<&SectionOf>)>,
) -> Result<Option<PrepareRequest>, String> {
    let Ok((&TurnOf(agent), routed)) = turns.get(turn) else {
        return Ok(None);
    };
    let Ok((id, own, effort, prompt, access)) = agents.get(agent) else {
        return Ok(None);
    };
    let spec = &routed.or(own).ok_or(NO_MODEL)?.spec;
    // Sorted by name, so the tools, and the prompt with their rules, are
    // the same on every call and stay cached.
    let mut offered: Vec<(&ToolDef, &ToolRules)> = tools
        .iter()
        .filter(|(def, _)| spec.tools && access.allows(def.0.name.as_str()))
        .collect();
    offered.sort_by(|a, b| a.0.0.name.as_str().cmp(b.0.0.name.as_str()));
    let rules = offered.iter().map(|(_, rules)| *rules);
    let sections = sections
        .iter()
        .filter(|(_, of)| of.is_none_or(|of| of.0 == agent));
    let sections = sections.map(|(section, _)| section);
    // A model the turn is routed to may not take the agent's setting.
    let options = Some(spec.default_options(effort.0))
        .filter(|options| spec.validate(options).is_ok())
        .unwrap_or_else(|| spec.default_options(None));
    Ok(Some(PrepareRequest {
        entity: turn,
        agent,
        preamble: system_prompt(&prompt.0, rules, sections),
        messages,
        tools: offered.iter().map(|(def, _)| def.0.clone()).collect(),
        options: options.cache_key(&id.0),
    }))
}

/// Sends what `prepare` says to the turn's model, once the options are
/// checked against it; or why it cannot be sent.
fn send_request(
    In(prepare): In<PrepareRequest>,
    turns: Query<Option<&Connection>, With<TurnOf>>,
    agents: Query<(&AgentId, Option<&Connection>, Option<&EffectParent>)>,
    effects: Res<Effects>,
    wake: Res<Wake>,
    mut commands: Commands,
) -> Result<(), String> {
    let PrepareRequest {
        entity: turn,
        agent,
        preamble,
        mut messages,
        tools,
        options,
    } = prepare;
    let (Ok(routed), Ok((id, own, effect_parent))) = (turns.get(turn), agents.get(agent)) else {
        return Ok(());
    };
    let connection = routed.or(own).ok_or(NO_MODEL)?;
    let validated = connection.spec.validate(&options);
    validated.map_err(|refusal| refusal.to_string())?;
    let prompt = messages.pop().ok_or("The conversation is empty.")?;
    let request = CompletionRequest::new(prompt)
        .messages(messages)
        .preamble(preamble)
        .tools(tools)
        .options(options);
    let (effect, reply) = effects.dispatch(
        &id.0,
        effect_parent.map(|parent| parent.0),
        connection.handler.0.clone(),
        EffectKind::Completion {
            request,
            stream: true,
        },
    );
    let (sender, feed) = crossbeam_channel::unbounded();
    let span = info_span!(
        "model_call",
        agent = %id.0,
        effect = %effect,
        model = %connection.spec.reference()
    );
    let reply = effects
        .caught(effect, stream_reply(reply, sender, wake.clone()))
        .instrument(span);
    commands.spawn((
        Name::new("model call"),
        ModelCall { effect, feed },
        Running::spawn(model_pool(), &wake, reply),
        Partial::default(),
        CallOf(turn),
    ));
    Ok(())
}

/// Dispatches a plugin's [`ModelRequest`] on the one dispatch path, to the
/// model of a [`Connection`] on the call, else on its turn, else its
/// turn's agent's. Without a model it is done at once, failed.
pub(crate) fn on_model_request(
    add: On<Add<ModelRequest>>,
    calls: Query<(&CallOf, &ModelRequest, Option<&Connection>)>,
    turns: Query<(&TurnOf, Option<&Connection>)>,
    agents: Query<(&AgentId, Option<&Connection>, Option<&EffectParent>)>,
    effects: Res<Effects>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let call = add.entity;
    let Ok((&CallOf(turn), ModelRequest { request }, own)) = calls.get(call) else {
        return;
    };
    let Ok((&TurnOf(agent), routed)) = turns.get(turn) else {
        return;
    };
    let Ok((id, agents, effect_parent)) = agents.get(agent) else {
        return;
    };
    let Some(connection) = own.or(routed).or(agents) else {
        let failed = ErrorReport::new(ErrorKind::HandlerUnavailable, NO_MODEL);
        commands
            .entity(call)
            .insert(Done::<ModelReply>(Err(failed)));
        return;
    };
    let (effect, reply) = effects.dispatch(
        &id.0,
        effect_parent.map(|parent| parent.0),
        connection.handler.0.clone(),
        EffectKind::Completion {
            request: request.clone(),
            stream: true,
        },
    );
    let span = info_span!("model_request", agent = %id.0, effect = %effect);
    let work = async move {
        reply
            .await
            .into_outcome()
            .await
            .and_then(Completion::unwrap)
    };
    let work = effects.caught(effect, work).instrument(span);
    commands
        .entity(call)
        .insert(Running::spawn(model_pool(), &wake, work));
}

/// Removes the user's last message when no model answered it, so the next
/// message does not follow an unanswered one. Tool results stay: the model
/// asked for them.
fn drop_unanswered(
    agent: Entity,
    conversation: &mut Conversation,
    commit: &mut Commit,
    notices: &mut MessageWriter<Notice>,
) {
    let unanswered = conversation.messages().last().is_some_and(|message| {
        matches!(message, Message::User { content }
            if !content.iter().any(|item| matches!(item, UserContent::ToolResult(_))))
    });
    if unanswered {
        commit.retract(agent, conversation);
        notices.write(Notice::info(
            agent,
            "Your last message was taken out of the conversation; send it again.",
        ));
    }
}

/// Streams the reply, feeding text and reasoning to the view and waking
/// the loop for each, and returns the response the provider ended it with.
async fn stream_reply(
    reply: impl Future<Output = Reply>,
    feed: Sender<Delta>,
    wake: Wake,
) -> ModelReply {
    let mut events = reply.await.into_stream();
    while let Some(item) = events.next().await {
        let delta = match item {
            Ok(Relayed::Done(response)) => return Ok(*response),
            Err(report) => return Err(report),
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => Delta::Text(text),
            Ok(Relayed::Item(Item::Event(StreamEvent::Reasoning { text, .. }))) => {
                Delta::Reasoning(text)
            }
            Ok(_) => continue,
        };
        // The view may be gone; the reply still finishes.
        feed.send(delta).ok();
        wake.wake();
    }
    Err(stream_truncated())
}

/// Moves streamed text and reasoning into each model call's [`Partial`].
pub(crate) fn stream_partials(mut calls: Query<(&ModelCall, &mut Partial)>) {
    for (call, mut partial) in &mut calls {
        for delta in call.feed.try_iter() {
            match delta {
                Delta::Text(text) => partial.text.push_str(&text),
                Delta::Reasoning(text) => partial.reasoning.push_str(&text),
            }
        }
    }
}

/// Takes a finished reply: appends it, then lets rig-core's turn-failure
/// rule decide. A failed reply runs none of its tool calls and ends the
/// turn; a reply without tool calls ends it too; otherwise each of its
/// tool calls starts that no earlier call holds back, and the rest are
/// [`Queued`].
pub(crate) fn on_model_done(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &ModelCall, &Done<ModelReply>)>,
    mut turns: Query<(&TurnOf, &mut Recovery)>,
    mut agents: Query<(
        (&mut Conversation, &mut Inbox),
        Option<&Connection>,
        &mut LastUsage,
    )>,
    starter: ToolStarter,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let call = done.entity;
    let Ok((&CallOf(turn), model_call, Done(reply))) = calls.get(call) else {
        return;
    };
    commands.entity(call).despawn();
    let Ok((&TurnOf(agent), mut recovery)) = turns.get_mut(turn) else {
        return;
    };
    let Ok(((mut conversation, mut inbox), connection, mut last)) = agents.get_mut(agent) else {
        return;
    };
    let spec = connection.map(|connection| &*connection.spec);
    let response = match reply {
        Ok(response) => {
            if response.usage.context_tokens().is_some() {
                *last = LastUsage(Some(response.usage));
            }
            recovery.retries = 0;
            response
        }
        Err(report) => {
            recover(
                (agent, turn),
                report,
                &mut recovery,
                &mut commands,
                &mut notices,
            );
            return;
        }
    };
    if let Some(message) = response.message() {
        commit.message(agent, &mut conversation, message);
    }
    let tool_calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
    let failure = turn_failure(
        &response.choice,
        Some(&response.stop()),
        response.finish_reason().as_ref(),
    );
    if let Some(failure) = failure {
        // Every call in the history gets a result, though none ran.
        if !tool_calls.is_empty() {
            let results = close_pending_with(&tool_calls, &format!("not run: {failure}"));
            commit.message(agent, &mut conversation, results);
        }
        notices.write(Notice::error(agent, format!("The turn failed: {failure}.")));
        drop_unanswered(agent, &mut conversation, &mut commit, &mut notices);
        end_turn(
            &mut commands,
            turn,
            TurnOutcome::Failed(failure.to_string()),
        );
        return;
    }
    if tool_calls.is_empty() {
        // What was sent meanwhile carries the turn on: steering first,
        // else everything queued, as one step. Notes go along but carry
        // nothing on.
        let inbox = &mut *inbox;
        let next: Vec<Pending> = match inbox.steering.is_empty() {
            true => inbox.queued.drain(..).collect(),
            false => inbox.steering.drain(..).collect(),
        };
        if !next.is_empty() {
            let waiting = inbox.notes.drain(..).chain(next);
            commit.pending(agent, spec, waiting, &mut conversation, &mut notices);
            commands.trigger(CallModel { entity: turn });
            return;
        }
        let outcome = match conversation.messages().last() {
            Some(answer @ Message::Assistant(_)) => TurnOutcome::Answered(answer.clone()),
            _ => TurnOutcome::Failed("the model ended the turn without a message".to_owned()),
        };
        end_turn(&mut commands, turn, outcome);
        return;
    }
    let parent = Some(model_call.effect);
    starter.spawn_calls(&mut commands, (agent, turn), tool_calls, parent);
}

/// Carries the turn of `agent` on after its model call failed for
/// `report`, as the [`RETRY`] policy decides: waits and calls again; ends
/// the turn, keeping the user's message when nothing is wrong with it; or
/// hands the turn to [`ModelFailed`] observers.
fn recover(
    (agent, turn): (Entity, Entity),
    report: &ErrorReport,
    recovery: &mut Recovery,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let why = match RETRY.verdict(report, recovery.retries, now()) {
        Verdict::Retry(delay) => {
            recovery.retries += 1;
            let backoff = Backoff {
                attempt: recovery.retries,
                until: Instant::now() + delay,
            };
            notices.write(Notice::info(
                agent,
                format!(
                    "The model call failed: {report}. Retrying in {}s ({}/{}).",
                    backoff.seconds_left(),
                    backoff.attempt,
                    RETRY.max_retries
                ),
            ));
            let wait = commands
                .spawn((Name::new("retry wait"), backoff, CallOf(turn)))
                .id();
            // An interrupt despawns the turn and its wait meanwhile; a call
            // for a turn that is gone does nothing.
            let mut delayed = commands.delayed();
            let mut later = delayed.duration(delay);
            later.entity(wait).try_despawn();
            later.trigger(CallModel { entity: turn });
            return;
        }
        Verdict::GaveUp(why) => {
            notices.write(Notice::error(
                agent,
                format!(
                    "The model call failed: {report}. Not retrying: {why}. Your message is \
                     kept for a retry."
                ),
            ));
            end_turn(commands, turn, TurnOutcome::Failed(report.to_string()));
            return;
        }
        Verdict::Overflow => {
            format!("The conversation does not fit the model's context window: {report}")
        }
        Verdict::Final => format!("The model call failed: {report}"),
    };
    let report = report.clone();
    commands.queue(move |world: &mut World| {
        let mut failed = ModelFailed {
            entity: turn,
            agent,
            report,
            handled: false,
        };
        world.trigger_ref(&mut failed);
        if failed.handled {
            return;
        }
        if let Err(error) = world.run_system_cached_with(fail_turn, (turn, why)) {
            warn!("could not end a failed turn: {error}");
        }
    });
}

/// Ends the turn as failed for `why`, taking out the user's message: the
/// same request would fail again.
fn fail_turn(
    In((turn, why)): In<(Entity, String)>,
    turns: Query<&TurnOf>,
    mut agents: Query<&mut Conversation>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    notices.write(Notice::error(agent, why.clone()));
    drop_unanswered(agent, &mut conversation, &mut commit, &mut notices);
    end_turn(&mut commands, turn, TurnOutcome::Failed(why));
}

/// Takes a tool call's [`ToolOutput`], however it came: records it as an
/// open call's outcome, and cancels the tool's own work if it still runs.
/// Then starts each queued call of the reply that no earlier unfinished
/// call holds back, or, once every call finished, appends their results in
/// call order and calls the model again.
pub(crate) fn on_tool_done(
    done: On<Add<ToolOutput>>,
    mut ended: Query<(&ToolOutput, Option<&mut OpenCall>)>,
    of: Query<&CallOf>,
    turns: Query<(&TurnOf, &Calls)>,
    mut agents: Query<&mut Conversation>,
    runs: Query<(&ToolCallRun, Option<&ToolOutput>, Has<Queued>)>,
    starter: ToolStarter,
    mut commit: Commit,
    mut commands: Commands,
) {
    if let Ok((ToolOutput(result), Some(mut open))) = ended.get_mut(done.entity) {
        let result = result.clone();
        open.0.settle(Ok(Outcome::ToolResult { result }));
    }
    commands
        .entity(done.entity)
        .try_remove::<(OpenCall, Running)>();
    let Ok(&CallOf(turn)) = of.get(done.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), calls)) = turns.get(turn) else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    // The calls not finished yet, in call order, each holding back the
    // later ones it must not run beside.
    let mut unfinished: Vec<Footprint> = Vec::new();
    let mut results = Vec::new();
    for call in calls.iter() {
        let Ok((run, output, queued)) = runs.get(call) else {
            continue;
        };
        if let Some(ToolOutput(result)) = output {
            results.push(run.call.answer(result));
            continue;
        }
        if queued
            && unfinished
                .iter()
                .all(|&before| !run.footprint.waits_for(before))
        {
            commands.entity(call).remove::<Queued>();
            starter.start(&mut commands, call, agent, run);
        }
        unfinished.push(run.footprint);
    }
    if !unfinished.is_empty() {
        return;
    }
    commit.message(agent, &mut conversation, Message::tool_results(results));
    commands.entity(turn).despawn_related::<Calls>();
    commands.trigger(CallModel { entity: turn });
}

/// Starts tool calls on the one dispatch path: what that reads, the
/// registered tools and the calling agent. A plugin that calls a tool
/// outside a turn spawns the call's entity with a [`ToolCallRun`] from
/// [`run`](Self::run) and [`start`](Self::start)s it; it ends with a
/// [`ToolOutput`] like a model's call.
#[derive(SystemParam)]
pub struct ToolStarter<'w, 's> {
    tools: Query<
        'w,
        's,
        (
            Entity,
            &'static ToolDef,
            &'static Serves,
            &'static Footprint,
        ),
    >,
    agents: Query<'w, 's, (&'static AgentId, &'static ToolAccess)>,
    effects: Res<'w, Effects>,
    log: Res<'w, SessionLog>,
    wake: Res<'w, Wake>,
}

impl ToolStarter<'_, '_> {
    /// The registered tool `name`.
    fn tool(&self, name: &str) -> Option<(Entity, &ToolDef, &Serves, &Footprint)> {
        self.tools
            .iter()
            .find(|(_, def, ..)| def.0.name.as_str() == name)
    }

    /// The run of `call`, asked for by the effect `parent`. A tool that is
    /// not registered runs on its own.
    pub fn run(&self, call: ToolCall, parent: Option<EffectId>) -> ToolCallRun {
        let tool = self.tool(call.function.name.as_str());
        ToolCallRun {
            footprint: tool.map_or_else(Footprint::default, |(.., &footprint)| footprint),
            call,
            parent,
        }
    }

    /// Whether a call of the tool `name` left without a result by a restart
    /// starts again: an ordinary read-only tool. Any other such call is
    /// answered as interrupted.
    pub(crate) fn reruns(&self, name: &str) -> bool {
        self.tool(name).is_some_and(|(_, _, serves, footprint)| {
            matches!(serves, Serves::Handler(_)) && *footprint == Footprint::ReadOnly
        })
    }

    /// Spawns a call entity of `turn`, of `agent`, for each of `calls`,
    /// asked for by the effect `parent`, then starts each that no earlier
    /// call holds back; the rest are [`Queued`]. Every call exists before
    /// any starts: an open call may end at once.
    pub(crate) fn spawn_calls(
        &self,
        commands: &mut Commands,
        (agent, turn): (Entity, Entity),
        calls: Vec<ToolCall>,
        parent: Option<EffectId>,
    ) {
        let mut earlier: Vec<Footprint> = Vec::with_capacity(calls.len());
        let mut ready = Vec::new();
        for call in calls {
            let run = self.run(call, parent);
            let waits = earlier
                .iter()
                .any(|&before| run.footprint.waits_for(before));
            earlier.push(run.footprint);
            let mut entity = commands.spawn((tool_name(&run), CallOf(turn), run.clone()));
            if waits {
                entity.insert(Queued);
            } else {
                ready.push((entity.id(), run));
            }
        }
        for (entity, run) in ready {
            self.start(commands, entity, agent, &run);
        }
    }

    /// Starts `run`, the call entity `call` of `agent`. An ordinary tool's
    /// call runs on the one dispatch path, and its [`ToolOutput`] is
    /// inserted when it finishes. An open tool's call is recorded as
    /// started and its tool's observer gets [`ToolCalled`] with the parsed
    /// arguments. A call to a tool that is not registered, that the agent
    /// may not use, or with arguments that do not fit, is recorded like any
    /// other and answered with an error. Before a call
    /// that may change something, the session log is written, so the reply
    /// that asked for it is on disk first.
    ///
    /// [`ToolCalled`]: super::tools::ToolCalled
    pub fn start(&self, commands: &mut Commands, call: Entity, agent: Entity, run: &ToolCallRun) {
        let Ok((id, access)) = self.agents.get(agent) else {
            return;
        };
        let name = run.call.function.name.as_str();
        let tool = self.tool(name).filter(|_| access.allows(name));
        if !tool.is_some_and(|(.., footprint)| *footprint == Footprint::ReadOnly) {
            self.log.flush();
        }
        let refused = |kind, why| Err(ErrorReport::new(ErrorKind::Tool(kind), why));
        let opened = match tool {
            None => refused(
                ToolErrorKind::NotFound,
                format!("no tool named `{name}` is available"),
            ),
            Some((tool, def, serves, _)) => {
                match (arguments_refusal(&def.0.parameters, &run.call), serves) {
                    (Some(why), _) => refused(ToolErrorKind::InvalidArgs, why),
                    (None, Serves::Open(open)) => open(&run.call)
                        .map(|trigger| (tool, trigger))
                        .or_else(|why| refused(ToolErrorKind::InvalidArgs, why)),
                    (None, Serves::Handler(handler)) => {
                        let work = run_tool_call(&self.effects, &id.0, run, handler.0.clone());
                        let running =
                            Running::spawn_into::<ToolOutput, _>(tool_pool(), &self.wake, work);
                        commands.entity(call).insert(running);
                        return;
                    }
                }
            }
        };
        // A call no handler runs is recorded as opened, and settled at
        // once when it is refused.
        let args = run.call.function.raw_arguments();
        let mut effect = self.effects.open(&id.0, run.parent, name, args);
        let (tool, trigger) = match opened {
            Ok(opened) => opened,
            Err(report) => {
                let output = ToolOutput(ToolResult::failed(report.clone().into()));
                effect.settle(Err(report));
                commands.entity(call).insert(output);
                return;
            }
        };
        let (caller, id, run) = (id.clone(), effect.id(), run.clone());
        commands
            .entity(call)
            .queue_silenced(move |mut entity: EntityWorldMut| {
                if entity.world().get_entity(agent).is_ok() {
                    entity.insert(OpenCall(effect));
                    entity.world_scope(|world| {
                        trigger(world, [tool, call, agent], caller, id, run);
                    });
                }
            });
    }
}

/// The [`Name`] of a tool call's entity.
pub fn tool_name(run: &ToolCallRun) -> Name {
    Name::new(format!("tool call {}", run.call.function.name.as_str()))
}

/// The wall clock as std's `SystemTime`, which rig-core's retry policy
/// reads a `Retry-After` date against; read through `web_time` so it also
/// works on the web.
fn now() -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(super::journal::now_ms())
}
