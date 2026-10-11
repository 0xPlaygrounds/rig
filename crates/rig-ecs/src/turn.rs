//! The turn loop. A user message spawns a turn entity, [`TurnOf`] its
//! agent; the turn's model calls and tool calls are effect entities
//! [`CallOf`] the turn that move through the stages. The turn's own model
//! call is a [`TurnCall`]: the kernel builds its [`ModelRequest`] from the
//! agent when it is admitted, plugins edit it in Prepare, and Start fits
//! it to the model it goes to and sends it. Its reply's tool calls are
//! [`ToolRequest`] effects; a reply's read-only calls run side by side and
//! the others alone, in order, as their [`Footprint`]s say ([`Waiting`]),
//! and once every one settled their results go back in call order and the
//! next [`TurnCall`] is spawned. rig-core's turn-failure rule decides when
//! a reply ends the turn instead. A model call that failed gets the
//! [`RETRY`] policy's [`RetryVerdict`] in Arrive, which Review may read,
//! replace or act on by superseding the call; Commit carries out what is
//! left: a [`Backoff`] then a new call, or the turn's end. A call refused
//! before it was sent ends the turn and keeps the user's message. Each
//! turn counts its retries in its [`Recovery`]. Every end the kernel
//! decides is a [`TurnOutcome`] proposed on the turn, which Review may
//! replace, hold or carry on from; once nothing holds it and every call of
//! the turn settled, Commit ends the turn and marks it [`Settled`] for
//! the readers of how it ended. [`Interrupt`] ends it at once.

use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryData;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use bevy_log::{info, warn};
use bevy_reflect::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{IoTaskPool, TaskPool};
use bevy_time::DelayedCommandsExt;
use crossbeam_channel::{Receiver, Sender};
use futures::StreamExt;
use rig_core::completion::message::turn_failure;
use rig_core::completion::{CompletionRequest, CompletionResponse, GenerationOptions, Message};
use rig_core::effect::{self, EffectId, EffectKind};
use rig_core::error::retry::{RetryPolicy, Verdict};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolCall, UserContent};
use rig_core::serve::{Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};
use rig_core::tool::{ToolExecutionError, ToolResult};
use rig_core::transcript::close_pending_with;
use web_time::Instant;

use super::agent::{
    ActiveTurn, Agent, AgentId, CallOf, Calls, Condensed, Conversation, EffectParent, Halt,
    Interrupt, LastUsage, Notice, Partial, Retry, SystemPrompt, ToolAccess, Turn, TurnOf,
    TurnOutcome, TurnsStarted,
};
use super::calls::{Running, Wake};
use super::effects::Handler;
use super::inbox::{Inbox, Pending, Starting};
use super::journal::{Commit, SessionLog};
use super::model::{Connection, Effort};
use super::prompt::{
    PromptSection, SectionOf, ToolRules, swap_tool_rules, system_prompt, tool_rules,
};
use super::stage::{
    Admitted, Effect, ForAgent, HoldOf, Holds, Outcome, Settled, Stage, Started, SupersededBy,
    Supersedes, Waiting, cancel, failed_in,
};
use super::tools::{Footprint, ToolDef, ToolRequest};

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

/// The wait before a failed [`TurnCall`] is sent again, on the hold the
/// kernel puts on it ([`HoldOf`]). Once the wait passed on Bevy's clock, a
/// delayed command despawns the hold and spawns a new call that
/// [`Supersedes`] the failed one, unless the hold is gone by then, as
/// interrupting the turn makes it.
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

/// A model call: an effect entity whose request is `request`, sent to
/// `connection`, else to the model of the agent it is [`ForAgent`]. Its
/// [`Outcome`] is rig-core's `Outcome::Completion` (see
/// [`Outcome::completion`]). Prepare may change the request and the
/// connection; Start then fits the request to the model it goes to (an
/// option the model cannot take is left out, as rig-core's `ModelSpec::fit`
/// says, and the tools of a model that calls none) and sends it streamed.
/// A [`TurnCall`] or [`BuildFrom`] has its request built by the kernel when
/// it is admitted; a plugin's own call, such as a summary, brings its own
/// with [`ModelRequest::new`].
#[derive(Component, Reflect, Clone)]
#[reflect(opaque, Component, Clone)]
#[require(Effect)]
pub struct ModelRequest {
    /// The request.
    pub request: CompletionRequest,
    /// The model it goes to, when not its agent's, such as a fallback
    /// model; `Models::connect` makes one.
    pub connection: Option<Connection>,
    /// The tool rules the kernel wrote in the system prompt when it built
    /// the request, swapped at Start for those of the tools it carries.
    rules: Option<String>,
}

impl ModelRequest {
    /// A call of `request`, on its agent's model.
    pub fn new(request: CompletionRequest) -> Self {
        Self {
            request,
            connection: None,
            rules: None,
        }
    }

    /// This call, sent to `connection`.
    #[must_use]
    pub fn on(self, connection: Connection) -> Self {
        Self {
            connection: Some(connection),
            ..self
        }
    }

    /// The system prompt the request sends, its leading system message, to
    /// change in place; `None` when it has none.
    pub fn system_prompt_mut(&mut self) -> Option<&mut String> {
        match self.request.chat_history.first_mut() {
            Some(Message::System { content }) => Some(content),
            _ => None,
        }
    }
}

/// The turn's own model call, [`CallOf`] its turn: built from its agent
/// when admitted (unless spawned with a [`ModelRequest`]), after the
/// messages sent to the agent meanwhile are delivered, and its reply goes
/// into the agent's conversation and carries the turn on or ends it. Its
/// text streams into its [`Partial`].
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
#[require(Effect, Partial, Name = Name::new("model call"))]
pub struct TurnCall;

/// A model call whose request the kernel builds from the agent this names
/// when it is admitted, as a turn's request would be (system prompt,
/// conversation, tools, options), and that is not part of any turn, such
/// as a side question or a preview of the context. Prepare then edits it,
/// such as by adding a question; its [`Outcome`] is the plugin's to take.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
#[require(Effect)]
pub struct BuildFrom(pub Entity);

/// On a model call whose handler failed: what the [`RETRY`] policy says,
/// put by the kernel in Arrive with the turn's [`Recovery`]. Review may
/// read it, replace it, or take the failure over by superseding the call;
/// for a [`TurnCall`], Commit carries out what is left: `Retry` waits a
/// [`Backoff`] and sends a new call, `GaveUp` ends the turn and keeps the
/// user's message, `Overflow` and `Final` end it and take the message out.
/// A call refused before it was sent gets none.
#[derive(Component, Clone, Debug)]
pub struct RetryVerdict(pub Verdict);

/// The feed of a [`TurnCall`]'s streamed text, into its [`Partial`].
#[derive(Component)]
pub(crate) struct Feed(Receiver<Delta>);

/// A streamed fragment for [`Partial`].
pub(crate) enum Delta {
    Text(String),
    Reasoning(String),
}

/// Model calls wait on the network: they run on the IO pool.
pub(crate) fn model_pool() -> &'static IoTaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Spawns the next model call of `turn`, for `agent`.
fn spawn_turn_call(commands: &mut Commands, turn: Entity, agent: Entity) {
    commands.spawn((TurnCall, CallOf(turn), ForAgent(agent)));
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
        notices.write(Notice::turn_running(agent));
        return;
    }
    if !conversation.resume() {
        notices.write(Notice::info(
            agent,
            "Nothing to retry: the model answered the last message.",
        ));
        return;
    }
    let turn = commands.spawn(TurnOf(agent)).id();
    spawn_turn_call(&mut commands, turn, agent);
}

/// Fills in a new turn's [`Turn`]: its index from its agent's count, which
/// it bumps, and where it begins in the agent's conversation.
pub(crate) fn number_turn(mut world: DeferredWorld, context: HookContext) {
    let Some(&TurnOf(agent)) = world.get::<TurnOf>(context.entity) else {
        return;
    };
    let first_message = world.get::<Conversation>(agent).map_or(0, |conversation| {
        let messages = conversation.messages();
        let asks = matches!(messages.last(), Some(Message::User { .. }));
        messages.len().saturating_sub(usize::from(asks))
    });
    let index = match world.get_mut::<TurnsStarted>(agent) {
        Some(mut started) => {
            let index = started.0;
            started.0 = index.saturating_add(1);
            index
        }
        None => 0,
    };
    if let Some(mut turn) = world.get_mut::<Turn>(context.entity) {
        turn.index = index;
        turn.first_message = first_message;
    }
}

/// On a turn whose outcome the kernel proposed: what the kernel does when
/// the turn ends that way, unless the turn carries on.
#[derive(Component, Clone, Debug)]
pub(crate) struct Ending {
    /// The error shown to the user.
    notice: String,
    /// Whether the user's unanswered message is taken out.
    take_out: bool,
}

/// Stops a running turn at once. The text a streaming reply had sent is
/// kept as the model's message, every tool call of the last reply gets a
/// result, real or "interrupted", and every call of the turn that has not
/// settled is answered as interrupted and settled, whatever held it gone.
/// The turn ends [`TurnOutcome::Stopped`]. The outcome is written to the
/// log at once, so a log that ends mid-turn always means a crash or a
/// restart.
pub(crate) fn on_interrupt(
    interrupt: On<Interrupt>,
    mut agents: Query<(&mut Conversation, &ActiveTurn)>,
    mut turns: Query<(&mut Turn, Option<&Calls>)>,
    replies: Query<ReplyCall>,
    partials: Query<&Partial, (With<TurnCall>, Without<Settled>)>,
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
    let (reply, calls) = match turns.get_mut(turn) {
        Ok((mut state, calls)) => (state.reply.take(), calls),
        Err(_) => (None, None),
    };
    let aborted: String = partials
        .iter_many(calls.into_iter().flat_map(Calls::iter))
        .flatten()
        .map(|partial| partial.text.as_str())
        .collect();
    if !aborted.trim().is_empty() {
        commit.message(agent, &mut conversation, Message::assistant(aborted));
    }
    let reason = interrupt.reason.clone();
    let why = match &reason {
        Some(reason) => format!("interrupted: {reason}"),
        None => "interrupted by the user".to_owned(),
    };
    let reply = reply
        .zip(calls)
        .map(|(reply, calls)| reply_calls(reply, calls, &replies));
    if let Some(results) = stopped_results(reply.unwrap_or_default(), &why) {
        commit.message(agent, &mut conversation, results);
    }
    commit.halt(agent, &mut conversation, Halt::Stopped);
    log.flush();
    let calls: Vec<Entity> = calls.into_iter().flat_map(Calls::iter).collect();
    commands.queue(move |world: &mut World| {
        for call in calls {
            cancel(world, call, &why);
        }
        let holds: Vec<Entity> = world
            .get::<Holds>(turn)
            .map(|holds| holds.iter().collect())
            .unwrap_or_default();
        for hold in holds {
            world.try_despawn(hold).ok();
        }
        if let Ok(mut turn) = world.get_entity_mut(turn) {
            turn.remove::<(TurnOf, Ending)>().insert((
                TurnOutcome::Stopped(reason),
                ForAgent(agent),
                Settled,
            ));
        }
    });
    notices.write(Notice::info(agent, "Interrupted."));
}

/// The results of a stopped reply's tool calls, in call order: each
/// answered call's result, and an error saying `why` for the others. `None`
/// when the reply made no calls.
fn stopped_results(calls: Vec<ReplyCallItem>, why: &str) -> Option<Message> {
    let why = ToolResult::failed(ToolExecutionError::other(why));
    let results: Vec<_> = calls
        .into_iter()
        .map(|call| {
            let result = call
                .outcome
                .map_or_else(|| why.clone(), Outcome::tool_result);
            call.request.call.answer(&result)
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
/// restart to carry on, so they end without a [`TurnOutcome`] and are not
/// logged as halted.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct Exiting;

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
    if let Err(error) = world.run_system_cached(log_answered_results) {
        warn!("could not log the results of stopped turns: {error}");
    }
    let turns: Vec<Entity> = world
        .query_filtered::<Entity, With<TurnOf>>()
        .iter(world)
        .collect();
    for turn in turns {
        world.despawn(turn);
    }
    world.flush();
}

/// Logs the results the running replies' calls got so far, in call order,
/// for the restart, which answers the rest.
fn log_answered_results(
    turns: Query<(&TurnOf, &Turn, &Calls)>,
    replies: Query<ReplyCall>,
    mut agents: Query<&mut Conversation>,
    mut commit: Commit,
) {
    for (&TurnOf(agent), state, calls) in &turns {
        let Some(reply) = state.reply else {
            continue;
        };
        let results: Vec<_> = reply_calls(reply, calls, &replies)
            .into_iter()
            .filter_map(|call| Some(call.request.call.answer(&call.outcome?.tool_result())))
            .collect();
        if let Ok(mut conversation) = agents.get_mut(agent)
            && !results.is_empty()
        {
            commit.message(agent, &mut conversation, Message::tool_results(results));
        }
    }
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

/// An agent as the kernel builds a model request from it.
#[derive(QueryData)]
#[query_data(mutable)]
pub(crate) struct Builder {
    id: &'static AgentId,
    connection: Option<&'static Connection>,
    effort: &'static Effort,
    prompt: &'static SystemPrompt,
    access: &'static ToolAccess,
    conversation: &'static mut Conversation,
    inbox: &'static mut Inbox,
    condensed: Option<&'static Condensed>,
    parent: Option<&'static EffectParent>,
}

/// A model call whose request the kernel builds, as Admit finds it.
#[derive(QueryData)]
#[query_data(mutable)]
pub(crate) struct Unbuilt {
    entity: Entity,
    effect: &'static mut Effect,
    from: Option<&'static BuildFrom>,
    turn_call: Has<TurnCall>,
    of: Option<&'static CallOf>,
    agent: Option<&'static ForAgent>,
}

/// What a request is built from besides its agent: the tools and prompt
/// sections.
#[derive(SystemParam)]
pub(crate) struct Parts<'w, 's> {
    tools: Query<'w, 's, (&'static ToolDef, &'static ToolRules)>,
    sections: Query<'w, 's, (&'static PromptSection, Option<&'static SectionOf>)>,
}

/// Builds the [`ModelRequest`] of each [`TurnCall`] and [`BuildFrom`]
/// admitted this pass without one, in [`Stage::Admit`]: the agent's system
/// prompt with the rules of the tools it is offered and its prompt
/// sections, its conversation after its [`Condensed`] summary, the tools,
/// sorted by name so the prompt cache keeps them, and its model's options
/// with its reasoning setting. A turn's call first delivers what was sent
/// to the agent meanwhile. A call the kernel cannot build is answered with
/// why.
pub(crate) fn build_requests(
    mut unbuilt: Query<
        Unbuilt,
        (
            Or<(With<TurnCall>, With<BuildFrom>)>,
            Added<Admitted>,
            Without<ModelRequest>,
            Without<Outcome>,
        ),
    >,
    turns: Query<&TurnOf>,
    mut agents: Query<Builder>,
    parts: Parts,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for mut call in &mut unbuilt {
        let turn = call.of.map(|of| of.0);
        let agent = call
            .from
            .map(|from| from.0)
            .or(call.agent.map(|agent| agent.0))
            .or_else(|| turns.get(turn?).ok().map(|of| of.0));
        let mut built = commands.entity(call.entity);
        let Some((agent, mut builder)) =
            agent.and_then(|agent| Some((agent, agents.get_mut(agent).ok()?)))
        else {
            let why = ErrorReport::new(ErrorKind::Internal, "a model call needs an agent");
            built.try_insert_if_new(Outcome(Err(why)));
            continue;
        };
        if call.agent.is_none() {
            built.try_insert(ForAgent(agent));
        }
        let effect = &mut *call.effect;
        if effect.parent.is_none() {
            effect.parent = builder.parent.map(|parent| parent.0);
        }
        let spec = builder.connection.map(|connection| &*connection.spec);
        if call.turn_call {
            effect.purpose.get_or_insert("turn".into());
            if let Some(turn) = turn {
                commands.entity(turn).try_remove::<Starting>();
            }
            let inbox = &mut *builder.inbox;
            let waiting = inbox.notes.drain(..).chain(inbox.steering.drain(..));
            let conversation = &mut *builder.conversation;
            commit.pending(agent, spec, waiting, conversation, &mut notices);
        }
        let messages = match builder.condensed {
            Some(condensed) => condensed.request(builder.conversation.messages()),
            None => builder.conversation.messages().to_vec(),
        };
        if call.turn_call && messages.is_empty() {
            let why = ErrorReport::new(ErrorKind::Internal, "The conversation is empty.");
            commands
                .entity(call.entity)
                .try_insert_if_new(Outcome(Err(why)));
            continue;
        }
        let model = build((agent, &builder), spec, &parts, messages);
        commands.entity(call.entity).try_insert(model);
    }
}

/// The request `agent`, as `builder` reads it, sends with `messages` to the
/// model `spec`, if it has one.
fn build(
    (agent, builder): (Entity, &BuilderItem),
    spec: Option<&rig_core::catalog::ModelSpec>,
    parts: &Parts,
    messages: Vec<Message>,
) -> ModelRequest {
    let tools = spec.is_none_or(|spec| spec.tools);
    let mut offered: Vec<(&ToolDef, &ToolRules)> = parts
        .tools
        .iter()
        .filter(|(def, _)| tools && builder.access.allows(def.0.name.as_str()))
        .collect();
    offered.sort_by(|a, b| a.0.0.name.as_str().cmp(b.0.0.name.as_str()));
    let rules = tool_rules(offered.iter().map(|(_, rules)| *rules));
    let sections = parts
        .sections
        .iter()
        .filter(|(_, of)| of.is_none_or(|of| of.0 == agent))
        .map(|(section, _)| section);
    let preamble = system_prompt(&builder.prompt.0, &rules, sections);
    let effort = builder.effort.0;
    let options = spec.map_or_else(
        || {
            let options = GenerationOptions::default();
            match effort {
                Some(reasoning) => options.reasoning(reasoning),
                None => options,
            }
        },
        |spec| spec.default_options(effort),
    );
    let mut request = CompletionRequest::new(Message::system(preamble))
        .tools(offered.iter().map(|(def, _)| def.0.clone()).collect())
        .options(options.cache_key(&builder.id.0));
    request.chat_history.extend(messages);
    ModelRequest {
        request,
        connection: builder.connection.cloned(),
        rules: Some(rules),
    }
}

/// How Start starts a model call.
pub(crate) enum ModelStart {
    /// The kernel sends `kind` to the handler, of the model named.
    Run(Handler, EffectKind, String),
    /// Refused before it was sent.
    Refused(ErrorReport),
}

/// What starting a model call reads: the agents' models and the tools'
/// rules.
#[derive(SystemParam)]
pub(crate) struct ModelStarts<'w, 's> {
    agents: Query<'w, 's, &'static Connection, With<Agent>>,
    rules: Query<'w, 's, (&'static ToolDef, &'static ToolRules)>,
}

impl ModelStarts<'_, '_> {
    /// How `model`, a call for `agent`, starts: on its connection, else the
    /// agent's model, its request fitted to that model and, when the kernel
    /// built it, with the rules of the tools it carries; refused without a
    /// model.
    pub(crate) fn start(&self, model: &mut ModelRequest, agent: Option<Entity>) -> ModelStart {
        let own = agent.and_then(|agent| self.agents.get(agent).ok());
        let Some(connection) = model.connection.as_ref().or(own).cloned() else {
            return ModelStart::Refused(ErrorReport::new(ErrorKind::HandlerUnavailable, NO_MODEL));
        };
        let spec = &connection.spec;
        let request = &mut model.request;
        if !spec.tools {
            request.tools.clear();
        }
        spec.fit(request);
        if let Some(written) = &model.rules {
            let carried = request.tools.iter().filter_map(|tool| {
                let rules = self.rules.iter().find(|(def, _)| def.0.name == tool.name);
                rules.map(|(_, rules)| rules)
            });
            let carried = tool_rules(carried);
            if carried != *written
                && let Some(Message::System { content }) = request.chat_history.first_mut()
            {
                *content = swap_tool_rules(std::mem::take(content), written, &carried);
                model.rules = Some(carried);
            }
        }
        let kind = EffectKind::Completion {
            request: request.clone(),
            stream: true,
        };
        ModelStart::Run(connection.handler.clone(), kind, spec.reference())
    }
}

/// Streams the reply, feeding text and reasoning to `feed`, if any, and
/// waking the loop for each, and returns the response the provider ended
/// it with.
pub(crate) async fn stream_reply(
    reply: Reply,
    feed: Option<Sender<Delta>>,
    wake: Wake,
) -> Result<effect::Outcome, ErrorReport> {
    let mut events = reply.into_stream();
    while let Some(item) = events.next().await {
        let delta = match item {
            Ok(Relayed::Done(response)) => return Ok(effect::Outcome::Completion(*response)),
            Err(report) => return Err(report),
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => Delta::Text(text),
            Ok(Relayed::Item(Item::Event(StreamEvent::Reasoning { text, .. }))) => {
                Delta::Reasoning(text)
            }
            Ok(_) => continue,
        };
        if let Some(feed) = &feed {
            // The view may be gone; the reply still finishes.
            feed.send(delta).ok();
            wake.wake();
        }
    }
    Err(stream_truncated())
}

/// The channel a [`TurnCall`] streams into: the sender for its task, and
/// the [`Feed`] for its entity.
pub(crate) fn feed() -> (Sender<Delta>, Feed) {
    let (sender, receiver) = crossbeam_channel::unbounded();
    (sender, Feed(receiver))
}

/// Moves streamed text and reasoning into each model call's [`Partial`].
pub(crate) fn stream_partials(mut calls: Query<(&Feed, &mut Partial)>) {
    for (feed, mut partial) in &mut calls {
        for delta in feed.0.try_iter() {
            match delta {
                Delta::Text(text) => partial.text.push_str(&text),
                Delta::Reasoning(text) => partial.reasoning.push_str(&text),
            }
        }
    }
}

/// Puts the [`RETRY`] policy's [`RetryVerdict`] on each model call whose
/// handler just failed, in [`Stage::Arrive`], counting its turn's
/// [`Recovery`].
pub(crate) fn judge_failures(
    failed: Query<
        (Entity, &Effect, &Outcome, Option<&CallOf>),
        (With<ModelRequest>, Added<Outcome>, Without<Settled>),
    >,
    turns: Query<&Recovery>,
    mut commands: Commands,
) {
    for (call, effect, Outcome(outcome), of) in &failed {
        let Err(report) = outcome else {
            continue;
        };
        if !effect.ran() {
            continue;
        }
        let retries = of
            .and_then(|of| turns.get(of.0).ok())
            .map_or(0, |recovery| recovery.retries);
        let verdict = RETRY.verdict(report, retries, now());
        commands.entity(call).try_insert(RetryVerdict(verdict));
    }
}

/// Holds each failed [`TurnCall`] whose verdict, after Review, is still
/// `Retry` with a [`Backoff`], in [`Stage::Commit`] before the kernel
/// settles, and sends a new call that supersedes it once the wait passed,
/// if the hold is still there then. Not when a Review system failed, nor
/// for a turn that is failed or stopped.
pub(crate) fn back_off(
    failed: Query<
        (Entity, &Effect, &Outcome, &RetryVerdict, &CallOf),
        (
            With<TurnCall>,
            Without<Holds>,
            Without<Settled>,
            Without<SupersededBy>,
        ),
    >,
    mut turns: Query<(&TurnOf, &mut Recovery, Option<&TurnOutcome>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if failed_in(Stage::Review).is_some() {
        return;
    }
    for (call, effect, Outcome(outcome), RetryVerdict(verdict), &CallOf(turn)) in &failed {
        let (Err(report), Verdict::Retry(delay)) = (outcome, verdict) else {
            continue;
        };
        let Ok((&TurnOf(agent), mut recovery, ending)) = turns.get_mut(turn) else {
            continue;
        };
        if ending.and_then(TurnOutcome::stops).is_some() {
            continue;
        }
        recovery.retries += 1;
        let backoff = Backoff {
            attempt: recovery.retries,
            until: Instant::now() + *delay,
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
        let hold = commands
            .spawn((Name::new("retry wait"), backoff, HoldOf(call)))
            .id();
        let parent = effect.parent;
        commands
            .delayed()
            .duration(*delay)
            .queue(move |world: &mut World| {
                // Interrupting the turn despawned the call and its hold.
                if world.get_entity(hold).is_err() {
                    return;
                }
                world.despawn(hold);
                world.spawn((
                    TurnCall,
                    Supersedes(call),
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::child_of(parent),
                ));
            });
    }
}

/// Answers every admitted call of a failed or stopped turn that has not
/// settled, in [`Stage::Commit`] before the kernel settles: one not
/// answered yet as stopped, which cancels it if it runs, and whatever holds
/// them is despawned, so they settle in this pass and the turn can end. A
/// call spawned since the last pass is left for Admit, where a new
/// [`TurnCall`] carries the turn on.
pub(crate) fn stop_calls(
    turns: Query<(&TurnOutcome, &Calls), With<TurnOf>>,
    calls: Query<(Has<Outcome>, Option<&Holds>), (With<Admitted>, Without<Settled>)>,
    mut commands: Commands,
) {
    for (ending, of) in &turns {
        let Some(why) = ending.stops() else {
            continue;
        };
        for (call, (answered, holds)) in of
            .iter()
            .filter_map(|call| Some((call, calls.get(call).ok()?)))
        {
            for hold in holds.into_iter().flat_map(Holds::iter) {
                commands.entity(hold).try_despawn();
            }
            if !answered {
                let stopped = ErrorReport::new(ErrorKind::Cancelled, why.clone());
                commands
                    .entity(call)
                    .try_insert_if_new(Outcome(Err(stopped)));
            }
        }
    }
}

/// Takes each [`TurnCall`] settled this pass and not superseded, in
/// [`Stage::Commit`], its streamed [`Partial`] gone: a reply is appended,
/// then rig-core's turn-failure rule decides. A failed reply runs none of
/// its tool calls; a reply without tool calls is the turn's answer, unless
/// messages sent meanwhile carry it on; otherwise its tool calls are
/// spawned, all at once, and become the turn's [`Turn::reply`]. A failure
/// fails the turn as its verdict says; one refused before it was sent
/// keeps the user's message. Each end is a [`TurnOutcome`] proposed on the
/// turn, unless it has one: a failed or stopped turn takes in a reply that
/// still came, its tool calls not run, and goes no further.
pub(crate) fn commit_turn_calls(
    calls: Query<
        (
            Entity,
            &Effect,
            Option<&Outcome>,
            Option<&RetryVerdict>,
            &CallOf,
            Has<SupersededBy>,
        ),
        (With<TurnCall>, Added<Settled>),
    >,
    mut turns: Query<(&TurnOf, &mut Recovery, &mut Turn, Option<&TurnOutcome>)>,
    mut agents: Query<Answered>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, effect, outcome, verdict, &CallOf(turn), superseded) in &calls {
        // What it streamed is in the conversation now, or never goes there.
        commands.entity(call).try_remove::<(Partial, Feed)>();
        let Some(Outcome(outcome)) = outcome.filter(|_| !superseded) else {
            continue;
        };
        let Ok((&TurnOf(agent), mut recovery, mut state, ending)) = turns.get_mut(turn) else {
            continue;
        };
        let Ok(mut answered) = agents.get_mut(agent) else {
            continue;
        };
        let stopped = ending.and_then(TurnOutcome::stops);
        let propose = |commands: &mut Commands, outcome: TurnOutcome, kernel: Option<Ending>| {
            if ending.is_none() {
                let mut turn = commands.entity(turn);
                turn.try_insert(outcome);
                if let Some(kernel) = kernel {
                    turn.try_insert(kernel);
                }
            }
        };
        let report = match outcome {
            Ok(effect::Outcome::Completion(response)) => {
                recovery.retries = 0;
                let next = commit_reply(agent, response, &mut answered, &mut commit, &mut notices);
                match (next, stopped) {
                    (Next::End(outcome, kernel), _) => propose(&mut commands, outcome, kernel),
                    (Next::Call, None) => spawn_turn_call(&mut commands, turn, agent),
                    (Next::Run(calls), None) => {
                        state.reply = Some(effect.id());
                        spawn_reply(&mut commands, (agent, turn), calls, effect.id());
                    }
                    (Next::Run(calls), Some(why)) => {
                        let results = close_pending_with(&calls, &format!("not run: {why}"));
                        commit.message(agent, &mut answered.conversation, results);
                    }
                    (Next::Call, Some(_)) => {}
                }
                continue;
            }
            _ if stopped.is_some() => continue,
            Ok(other) => ErrorReport::new(
                ErrorKind::Internal,
                format!("the model answered with a {} outcome", other.family()),
            ),
            Err(report) => report.clone(),
        };
        let kept = |notice: String| {
            Some(Ending {
                notice,
                take_out: false,
            })
        };
        let mut propose = |outcome, kernel| propose(&mut commands, outcome, kernel);
        match verdict.map(|verdict| &verdict.0) {
            None => propose(
                TurnOutcome::Failed(report.message.clone()),
                kept(report.message),
            ),
            Some(Verdict::GaveUp(why)) => propose(
                TurnOutcome::Failed(report.to_string()),
                kept(format!(
                    "The model call failed: {report}. Not retrying: {why}. Your message is kept \
                     for a retry."
                )),
            ),
            Some(Verdict::Retry(_)) => propose(
                TurnOutcome::Failed(report.to_string()),
                kept(format!(
                    "The model call failed: {report}. Your message is kept for a retry."
                )),
            ),
            Some(verdict @ (Verdict::Overflow | Verdict::Final)) => {
                let why = match verdict {
                    Verdict::Overflow => format!(
                        "The conversation does not fit the model's context window: {report}"
                    ),
                    _ => format!("The model call failed: {report}"),
                };
                let ending = Ending {
                    notice: why.clone(),
                    take_out: true,
                };
                propose(TurnOutcome::Failed(why), Some(ending));
            }
        }
    }
}

/// What a turn does after a reply.
enum Next {
    /// It would end, as proposed, with what the kernel does then.
    End(TurnOutcome, Option<Ending>),
    /// It calls the model again, with the messages sent meanwhile.
    Call,
    /// It runs the reply's tool calls.
    Run(Vec<ToolCall>),
}

/// The messages that carry a turn on that would end: steering first, else
/// everything queued, as one step, with the notes; none when neither
/// waits.
fn carry_on(inbox: &mut Inbox) -> Option<Vec<Pending>> {
    let next: Vec<Pending> = match inbox.steering.is_empty() {
        true => inbox.queued.drain(..).collect(),
        false => inbox.steering.drain(..).collect(),
    };
    (!next.is_empty()).then(|| inbox.notes.drain(..).chain(next).collect())
}

/// Appends `response`, the reply to `agent`, and says what its turn does
/// next.
fn commit_reply(
    agent: Entity,
    response: &CompletionResponse,
    answered: &mut AnsweredItem,
    commit: &mut Commit,
    notices: &mut MessageWriter<Notice>,
) -> Next {
    if response.usage.context_tokens().is_some() {
        *answered.last = LastUsage(Some(response.usage));
    }
    if let Some(message) = response.message() {
        commit.message(agent, &mut answered.conversation, message);
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
            commit.message(agent, &mut answered.conversation, results);
        }
        let ending = Ending {
            notice: format!("The turn failed: {failure}."),
            take_out: true,
        };
        return Next::End(TurnOutcome::Failed(failure.to_string()), Some(ending));
    }
    if !tool_calls.is_empty() {
        return Next::Run(tool_calls);
    }
    // What was sent meanwhile carries the turn on. Notes go along but carry
    // nothing on.
    if let Some(next) = carry_on(&mut answered.inbox) {
        let spec = answered.connection.map(|connection| &*connection.spec);
        commit.pending(agent, spec, next, &mut answered.conversation, notices);
        return Next::Call;
    }
    match answered.conversation.messages().last() {
        Some(answer @ Message::Assistant(_)) => {
            Next::End(TurnOutcome::Answered(answer.clone()), None)
        }
        _ => {
            let why = "the model ended the turn without a message".to_owned();
            Next::End(TurnOutcome::Failed(why), None)
        }
    }
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

/// An agent as its model's reply finds it.
#[derive(QueryData)]
#[query_data(mutable)]
pub(crate) struct Answered {
    conversation: &'static mut Conversation,
    inbox: &'static mut Inbox,
    connection: Option<&'static Connection>,
    last: &'static mut LastUsage,
}

/// Spawns a call of `turn`, for `agent`, for each of `calls`, in order,
/// asked for by the model call `reply`. Every call exists before any is
/// gated.
pub(crate) fn spawn_reply(
    commands: &mut Commands,
    (agent, turn): (Entity, Entity),
    calls: Vec<ToolCall>,
    reply: EffectId,
) {
    for call in calls {
        commands.spawn((
            ToolRequest { call },
            Effect::child_of(Some(reply)),
            CallOf(turn),
            ForAgent(agent),
        ));
    }
}

/// A call of a model's reply, as the stages find it.
#[derive(QueryData)]
pub(crate) struct ReplyCall {
    effect: &'static Effect,
    request: &'static ToolRequest,
    outcome: Option<&'static Outcome>,
    successor: Option<&'static SupersededBy>,
    footprint: Option<&'static Footprint>,
    settled: Has<Settled>,
    started: Has<Started>,
    waiting: Has<Waiting>,
    entity: Entity,
}

/// The calls of the reply `reply` among `calls`, in the reply's order:
/// for each call the model asked for, the last effect that superseded it.
fn reply_calls<'a>(
    reply: EffectId,
    calls: &Calls,
    replies: &'a Query<ReplyCall>,
) -> Vec<ReplyCallItem<'a, 'a>> {
    let mut asked: Vec<ReplyCallItem<'a, 'a>> = Vec::new();
    for call in calls.iter() {
        let Ok(mut call) = replies.get(call) else {
            continue;
        };
        let first = |at: &ReplyCallItem| at.request.call.id == call.request.call.id;
        if call.effect.parent != Some(reply) || asked.iter().any(first) {
            continue;
        }
        for _ in 0..calls.len() {
            match call
                .successor
                .and_then(|by| replies.get(by.successor()).ok())
            {
                Some(successor) => call = successor,
                None => break,
            }
        }
        asked.push(call);
    }
    asked
}

/// Marks each call of a running reply [`Waiting`] while an earlier call it
/// must not run beside is not settled, and takes the mark off once none
/// is, in [`Stage::Admit`](super::stage::Stage::Admit). A held call holds
/// back the calls after it; a refused one, once it settled, does not.
pub(crate) fn wait_in_order(
    turns: Query<(&Turn, &Calls)>,
    replies: Query<ReplyCall>,
    mut commands: Commands,
) {
    for (state, calls) in &turns {
        let Some(reply) = state.reply else {
            continue;
        };
        let mut earlier: Vec<Footprint> = Vec::new();
        for call in reply_calls(reply, calls, &replies) {
            let footprint = call.footprint.copied().unwrap_or_default();
            let waits = earlier.iter().any(|&before| footprint.waits_for(before));
            let open = !call.started && call.outcome.is_none() && !call.settled;
            if open && waits != call.waiting {
                let mut entity = commands.entity(call.entity);
                match waits {
                    true => entity.try_insert(Waiting),
                    false => entity.try_remove::<Waiting>(),
                };
            }
            if !call.settled {
                earlier.push(footprint);
            }
        }
    }
}

/// Once every call of a turn's running reply settled, appends their
/// results in call order and spawns the turn's next [`TurnCall`], unless
/// the turn is failed or stopped, in [`Stage::Commit`], after the kernel
/// settled that pass's effects.
pub(crate) fn commit_replies(
    mut turns: Query<(Entity, &TurnOf, &mut Turn, &Calls, Option<&TurnOutcome>)>,
    replies: Query<ReplyCall>,
    mut agents: Query<&mut Conversation>,
    mut commit: Commit,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent), mut state, calls, ending) in &mut turns {
        let Some(reply) = state.reply else {
            continue;
        };
        let calls = reply_calls(reply, calls, &replies);
        if calls.iter().any(|call| !call.settled) {
            continue;
        }
        let Ok(mut conversation) = agents.get_mut(agent) else {
            continue;
        };
        state.reply = None;
        let unanswered = || ToolResult::failed(ToolExecutionError::other("the call had no result"));
        let results: Vec<_> = calls
            .into_iter()
            .map(|call| {
                let result = call.outcome.map_or_else(unanswered, Outcome::tool_result);
                call.request.call.answer(&result)
            })
            .collect();
        if !results.is_empty() {
            commit.message(agent, &mut conversation, Message::tool_results(results));
        }
        if ending.and_then(TurnOutcome::stops).is_none() {
            spawn_turn_call(&mut commands, turn, agent);
        }
    }
}

/// Ends each turn whose [`TurnOutcome`] nothing holds and whose calls all
/// settled, in [`Stage::Commit`], before this pass's proposals are made, so
/// Review sees each proposal first: what the kernel proposed shows its
/// error and may take the user's unanswered message out, the turn is no
/// longer [`TurnOf`] its agent, and it is [`ForAgent`] it and [`Settled`].
/// An `Answered` turn that messages were sent to meanwhile carries on with
/// them instead. Not when a Review system failed.
pub(crate) fn end_turns(
    turns: Query<
        (
            Entity,
            &TurnOf,
            &TurnOutcome,
            Option<&Ending>,
            Option<&Calls>,
        ),
        Without<Holds>,
    >,
    open: Query<(), (With<Effect>, Without<Settled>)>,
    mut agents: Query<(&AgentId, &mut Conversation, &mut Inbox, Option<&Connection>)>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if failed_in(Stage::Review).is_some() {
        return;
    }
    for (turn, &TurnOf(agent), outcome, ending, calls) in &turns {
        let Ok((id, mut conversation, mut inbox, connection)) = agents.get_mut(agent) else {
            continue;
        };
        if let TurnOutcome::Answered(_) = outcome
            && let Some(next) = carry_on(&mut inbox)
        {
            let spec = connection.map(|connection| &*connection.spec);
            commit.pending(agent, spec, next, &mut conversation, &mut notices);
            commands.entity(turn).try_remove::<(TurnOutcome, Ending)>();
            spawn_turn_call(&mut commands, turn, agent);
            continue;
        }
        if calls.is_some_and(|calls| calls.iter().any(|call| open.contains(call))) {
            continue;
        }
        if let Some(ending) = ending {
            notices.write(Notice::error(agent, ending.notice.clone()));
            if ending.take_out {
                drop_unanswered(agent, &mut conversation, &mut commit, &mut notices);
            }
        }
        let how = match outcome {
            TurnOutcome::Answered(_) => "answered",
            TurnOutcome::Failed(_) => "failed",
            TurnOutcome::Stopped(_) => "stopped",
        };
        info!(agent = %id.0, "turn ended: {how}");
        commands
            .entity(turn)
            .try_remove::<(TurnOf, Ending)>()
            .try_insert((ForAgent(agent), Settled));
    }
}

/// Takes the [`TurnOutcome`] off each turn whose [`TurnCall`] was admitted
/// this pass, in [`Stage::Admit`]: calling the model again carries the turn
/// on, whatever was proposed or decided.
pub(crate) fn carry_on_called(
    called: Query<&CallOf, (With<TurnCall>, Added<Admitted>)>,
    ending: Query<(), With<TurnOutcome>>,
    mut commands: Commands,
) {
    for &CallOf(turn) in &called {
        if ending.contains(turn) {
            commands.entity(turn).try_remove::<(TurnOutcome, Ending)>();
        }
    }
}

/// Whether the settled effect `entity` stays past its frame: a call of a
/// reply whose other calls still run, which keeps its result for them.
pub(crate) fn keeps_settled(world: &World, entity: Entity) -> bool {
    let reply = world
        .get::<CallOf>(entity)
        .and_then(|of| world.get::<Turn>(of.0))
        .and_then(|turn| turn.reply);
    let parent = world.get::<Effect>(entity).and_then(|effect| effect.parent);
    reply.is_some() && parent == reply && world.get::<ToolRequest>(entity).is_some()
}

/// The wall clock as std's `SystemTime`, which rig-core's retry policy
/// reads a `Retry-After` date against; read through `web_time` so it also
/// works on the web.
fn now() -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(super::journal::now_ms())
}
