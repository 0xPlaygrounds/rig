//! The turn loop. A user message spawns a turn entity, [`TurnOf`] its
//! agent; the turn's model call and tool calls are entities [`CallOf`] the
//! turn, and observers of their [`Done`] outputs and [`ToolOutput`]s carry
//! the turn on. A reply's read-only tool calls run side by side and the
//! others alone, in order, as their tools' [`Footprint`]s say; their
//! results go back in call order. rig-core's turn-failure
//! rule decides when a reply ends the turn instead. A failed model call is
//! retried or recovered from as [`recovery`] decides. A conversation near
//! the model's window is [`compaction`]-ed before the next call. Despawning
//! the turn ends it, and announces its [`TurnEnded`] on the agent.

use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::tracing::Instrument;
use bevy_log::{info, info_span};
use bevy_reflect::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{AsyncComputeTaskPool, IoTaskPool, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use futures::{FutureExt, StreamExt};
use rig_core::catalog::ModelSpec;
use rig_core::completion::message::turn_failure;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::effect::{EffectId, EffectKind};
use rig_core::error::ErrorReport;
use rig_core::error::retry::Verdict;
use rig_core::message::{ToolCall, ToolResult, UserContent};
use rig_core::serve::{ErasedHandler, Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};
use rig_core::tool::ToolErrorKind;
use rig_core::transcript::arguments_refusal;
use rig_memory::{Summarizer, SummaryState};
use web_time::Instant;

use super::agent::{
    ActiveTurn, Agent, AgentId, CallOf, Calls, Compact, Connection, Conversation, EffectParent,
    Effort, Ending, Interrupt, ModelChoice, Notice, Partial, Queued, Retry, SetEffort, SetModel,
    SettingsChosen, SystemPrompt, ToolAccess, ToolCallRun, TurnEnded, TurnOf, TurnOutcome,
};
use super::calls::{Done, Running, Wake};
use super::compaction::{
    self, CompactReason, Compacted, CompactionPolicy, MAX_COMPACTIONS, Summarize, Summarizing,
    Summary,
};
use super::effects::{Effects, Handler};
use super::inbox::{Delivery, Inbox, deliver_notes, deliver_queued, deliver_steering};
use super::journal::SessionLog;
use super::models::{self, ModelConnector};
use super::prompt::{PromptSection, ToolRules, system_prompt};
use super::recovery::{self, Backoff, RETRY, Recovery, RetryDue};
use super::tools::{
    Footprint, OpenCall, Refused, ToolCalled, ToolDef, ToolHandler, ToolOutput, failed, outcome_of,
    recorded_args, run_tool_call,
};
use super::usage::{self, Spending, TurnSpending};

/// The notice when the agent has no model to call.
const NO_MODEL: &str = "No model is connected; pick one first.";

/// The systems polling running calls, in `Update`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PollCalls;

/// What a model call's task returns.
pub type ModelReply = Result<CompletionResponse, ErrorReport>;

/// A streaming model call of the turn it is a [`CallOf`]; its task is a
/// [`Running<ModelReply>`](Running).
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
/// in a message the model has not answered.
pub(crate) fn on_retry(
    retry: On<Retry>,
    agents: Query<(&Conversation, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = retry.entity;
    let Ok((conversation, busy)) = agents.get(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::info(agent, "A turn is running."));
        return;
    }
    if !matches!(conversation.messages().last(), Some(Message::User { .. })) {
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
    mut agents: Query<(&AgentId, &mut Conversation, &ActiveTurn)>,
    turns: Query<&Calls>,
    runs: Query<(&ToolCallRun, Option<&ToolOutput>)>,
    partials: Query<&Partial, With<ModelCall>>,
    log: Res<SessionLog>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = interrupt.entity;
    let Ok((id, mut conversation, active)) = agents.get_mut(agent) else {
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
        log.commit(id, &mut conversation, Message::assistant(aborted), None);
    }
    if let Some(results) = stopped_results(
        runs.iter_many(calls.iter().copied()).flatten(),
        "interrupted by the user",
    ) {
        log.commit(id, &mut conversation, results, None);
    }
    log.halt(id, &conversation);
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
    let results: Vec<ToolResult> = runs
        .map(|(run, done)| match done {
            Some(ToolOutput(result)) => result.clone(),
            None => failed(&run.call, why.to_owned()),
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
/// after a crash: every running call is cancelled, the tool results that
/// came in are logged, and the restart answers the rest.
pub(crate) fn stop_turns_on_exit(world: &mut World) {
    world.insert_resource(Exiting);
    let log = world.get_resource::<SessionLog>().cloned();
    let mut cancelling = Cancelling::new();
    take_running::<ModelReply>(world, &mut cancelling);
    take_running::<Summary>(world, &mut cancelling);
    take_running::<ToolResult>(world, &mut cancelling);
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
    let mut outputs = world.query::<&ToolOutput>();
    for (turn, agent) in turns {
        let calls: Vec<Entity> = world
            .get::<Calls>(turn)
            .map(|calls| calls.iter().collect())
            .unwrap_or_default();
        let results: Vec<ToolResult> = outputs
            .iter_many(world, calls)
            .flatten()
            .map(|ToolOutput(result)| result.clone())
            .collect();
        if !results.is_empty()
            && let (Some(log), Some(id)) = (&log, world.get::<AgentId>(agent).cloned())
            && let Some(mut conversation) = world.get_mut::<Conversation>(agent)
        {
            log.commit(&id, &mut conversation, Message::tool_results(results), None);
        }
        world.despawn(turn);
    }
    world.flush();
}

/// Takes every [`Running<T>`] task off its call, to be cancelled.
fn take_running<T: Send + Sync + 'static>(world: &mut World, cancelling: &mut Cancelling) {
    let calls: Vec<Entity> = world
        .query_filtered::<Entity, With<Running<T>>>()
        .iter(world)
        .collect();
    for call in calls {
        if let Ok(mut call) = world.get_entity_mut(call)
            && let Some(Running(task)) = call.take::<Running<T>>()
        {
            cancelling.push(Box::pin(async move {
                task.cancel().await;
            }));
        }
    }
}

/// Chooses the agent's model: a known catalog model replaces the agent's
/// [`ModelChoice`], and [`on_model_chosen`] connects it.
pub(crate) fn on_set_model(
    set: On<SetModel>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    connector: Res<ModelConnector>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(busy) = agents.get(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, busy, &mut notices) {
        return;
    }
    match connector.resolve(&set.model) {
        Some(spec) => {
            commands
                .entity(set.entity)
                .insert(ModelChoice(spec.reference()));
            commands.trigger(SettingsChosen { entity: set.entity });
        }
        None => {
            notices.write(Notice::error(
                set.entity,
                format!("No catalog model `{}`. Use vendor/model.", set.model),
            ));
        }
    }
}

/// Connects an agent whose [`ModelChoice`] was inserted, by [`SetModel`] or by
/// restoring a session, so requests never re-resolve the provider. A
/// reasoning setting the new model does not take is reset. Only a change of
/// a connected agent's model is announced: spawning and restoring an agent
/// are silent, since views show its model anyway.
pub(crate) fn on_model_chosen(
    chosen: On<Insert<ModelChoice>>,
    agents: Query<(&ModelChoice, &Effort, Has<Connection>)>,
    mut effects: ResMut<Effects>,
    connector: Res<ModelConnector>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = chosen.entity;
    let Ok((choice, effort, switched)) = agents.get(agent) else {
        return;
    };
    let connection = connector
        .resolve(&choice.0)
        .ok_or_else(|| format!("the catalog has no model `{}`", choice.0))
        .and_then(|spec| {
            effects
                .model_handler(&spec, &connector)
                .map(|handler| Connection {
                    spec,
                    handler: Handler(handler),
                })
                .map_err(|error| error.to_string())
        });
    let connection = match connection {
        Ok(connection) => connection,
        Err(why) => {
            commands.entity(agent).remove::<Connection>();
            notices.write(Notice::error(
                agent,
                format!("Cannot use {}: {why}.", choice.0),
            ));
            return;
        }
    };
    let spec = &*connection.spec;
    if switched {
        notices.write(Notice::info(
            agent,
            format!("Model: {} ({}).", spec.display_name, choice.0),
        ));
    }
    if let Err(refusal) = models::check_effort(spec, effort.0) {
        commands.entity(agent).insert(Effort(None));
        notices.write(Notice::info(
            agent,
            format!("Reasoning reset to default: {refusal}."),
        ));
    }
    commands.entity(agent).insert(connection);
}

/// Sets the agent's reasoning setting after checking it against the model.
pub(crate) fn on_set_effort(
    set: On<SetEffort>,
    agents: Query<(Option<&Connection>, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((connection, busy)) = agents.get(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, busy, &mut notices) {
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::info(set.entity, NO_MODEL));
        return;
    };
    match models::check_effort(&connection.spec, set.effort.0) {
        Ok(()) => {
            commands.entity(set.entity).insert(set.effort);
            commands.trigger(SettingsChosen { entity: set.entity });
            notices.write(Notice::info(
                set.entity,
                format!("Reasoning: {}.", models::effort_label(set.effort.0)),
            ));
        }
        Err(refusal) => {
            notices.write(Notice::error(set.entity, format!("{refusal}.")));
        }
    }
}

/// Refuses a model or reasoning change, or a compaction, while the
/// agent's turn runs: the rest of the turn would go to a model, or use a
/// setting, it did not start with. Every sender of [`SetModel`],
/// [`SetEffort`] and [`Compact`] gets the same refusal.
fn refused_mid_turn(agent: Entity, busy: bool, notices: &mut MessageWriter<Notice>) -> bool {
    if busy {
        notices.write(Notice::info(agent, "A turn is running; stop it first."));
    }
    busy
}

/// Sends the conversation of the turn's agent to its model. When that
/// cannot be done the turn ends.
pub(crate) fn on_call_model(
    call: On<CallModel>,
    mut turns: Query<(&TurnOf, &mut Recovery)>,
    mut agents: Query<(
        (
            &AgentId,
            &mut Conversation,
            &mut Inbox,
            Option<&EffectParent>,
        ),
        &Compacted,
        &mut Spending,
        Option<&Connection>,
        &Effort,
        &SystemPrompt,
        &ToolAccess,
    )>,
    tools: Query<(&ToolDef, &ToolRules)>,
    sections: Query<&PromptSection>,
    policy: Res<CompactionPolicy>,
    effects: Res<Effects>,
    log: Res<SessionLog>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let turn = call.entity;
    let Ok((&TurnOf(agent), mut recovery)) = turns.get_mut(turn) else {
        return;
    };
    let Ok((
        (id, mut conversation, mut inbox, effect_parent),
        compacted,
        mut spent,
        connection,
        effort,
        prompt,
        access,
    )) = agents.get_mut(agent)
    else {
        return;
    };
    if let Some(connection) = connection
        && recovery.compactions < MAX_COMPACTIONS
        && must_summarize(
            agent,
            &mut conversation,
            (compacted, &policy),
            &mut spent,
            &connection.spec,
            &mut notices,
        )
    {
        recovery.compactions += 1;
        commands.trigger(Summarize {
            entity: turn,
            reason: CompactReason::Threshold,
        });
        return;
    }
    let to = Delivery {
        agent,
        id,
        spec: connection.map(|connection| &*connection.spec),
        log: &log,
    };
    deliver_notes(&to, &mut inbox, &mut conversation, &mut notices);
    deliver_steering(&to, &mut inbox, &mut conversation, &mut notices);
    let request = connection
        .ok_or_else(|| NO_MODEL.to_owned())
        .and_then(|connection| {
            // Sorted by name, so the tools, and the prompt with their
            // rules, are the same on every call and stay cached.
            let mut offered: Vec<(&ToolDef, &ToolRules)> = tools
                .iter()
                .filter(|(def, _)| connection.spec.tools && access.allows(def.0.name.as_str()))
                .collect();
            offered.sort_by(|a, b| a.0.0.name.as_str().cmp(b.0.0.name.as_str()));
            let preamble =
                system_prompt(&prompt.0, offered.iter().map(|(_, rules)| *rules), sections);
            let definitions = offered.iter().map(|(def, _)| def.0.clone()).collect();
            let messages = compacted.request(conversation.messages());
            prepare(messages, connection, effort, &id.0, preamble, definitions).map(|request| {
                (
                    connection.handler.erased(),
                    connection.spec.clone(),
                    request,
                )
            })
        });
    let (handler, spec, request) = match request {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, why.clone()));
            drop_unanswered(agent, id, &mut conversation, &log, &mut notices);
            end_turn(&mut commands, turn, TurnOutcome::Failed(why));
            return;
        }
    };
    let (effect, reply) = effects.dispatch(
        &id.0,
        effect_parent.map(|parent| parent.0),
        handler,
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
        model = %spec.reference()
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
}

/// Whether the next request leaves less than the reserve of the model's
/// window free and must be summarized first. It clears old tool outputs
/// before that, which costs no model call, and asks for a summary only when
/// clearing was not enough. The context in use is the last call's reported
/// one or the conversation's estimate, whichever is larger.
fn must_summarize(
    agent: Entity,
    conversation: &mut Mut<Conversation>,
    (compacted, policy): (&Compacted, &CompactionPolicy),
    spent: &mut Mut<Spending>,
    spec: &ModelSpec,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    let used = spent
        .context
        .unwrap_or(0)
        .max(compacted.estimate(conversation.messages()));
    if !compaction::over_threshold(used, spec) {
        return false;
    }
    let cleared = recovery::clearing().clear(compacted.live_mut(conversation.messages_mut()));
    let left = used.saturating_sub(cleared.tokens as u64);
    if cleared.results > 0 {
        spent.context = Some(left);
        notices.write(Notice::info(
            agent,
            format!(
                "The conversation nears the model's context window: cleared {} older tool \
                 outputs (about {} tokens).",
                cleared.results,
                usage::tokens(cleared.tokens as u64)
            ),
        ));
    }
    compaction::over_threshold(left, spec)
        && compacted
            .cut(
                conversation.messages(),
                policy,
                spec,
                &CompactReason::Threshold,
            )
            .is_some()
}

/// Removes the user's last message when no model answered it, so the next
/// message does not follow an unanswered one. Tool results stay: the model
/// asked for them.
fn drop_unanswered(
    agent: Entity,
    id: &AgentId,
    conversation: &mut Conversation,
    log: &SessionLog,
    notices: &mut MessageWriter<Notice>,
) {
    let unanswered = conversation.messages().last().is_some_and(|message| {
        matches!(message, Message::User { content }
            if !content.iter().any(|item| matches!(item, UserContent::ToolResult(_))))
    });
    if unanswered {
        log.retract(id, conversation);
        notices.write(Notice::info(
            agent,
            "Your last message was taken out of the conversation; send it again.",
        ));
    }
}

/// The request for the agent's next model call, checked against the
/// model's spec, or what the user must fix first. It asks for the
/// provider's prompt cache where the model has one; the preamble and tools
/// come first and do not change between calls, so each call reads the
/// prefix the last one wrote.
fn prepare(
    mut messages: Vec<Message>,
    connection: &Connection,
    effort: &Effort,
    cache_key: &str,
    preamble: String,
    tools: Vec<rig_core::completion::ToolDefinition>,
) -> Result<CompletionRequest, String> {
    let options = connection
        .spec
        .default_options(effort.0)
        .cache_key(cache_key);
    connection
        .spec
        .validate(&options)
        .map_err(|refusal| refusal.to_string())?;
    let prompt_message = messages.pop().ok_or("The conversation is empty.")?;
    Ok(CompletionRequest::new(prompt_message)
        .messages(messages)
        .preamble(preamble)
        .tools(tools)
        .options(options))
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
    mut turns: Query<(&TurnOf, &mut TurnSpending, &mut Recovery)>,
    mut agents: Query<(
        (&AgentId, &mut Conversation, &mut Inbox),
        &Compacted,
        Option<&Connection>,
        &mut Spending,
    )>,
    starter: ToolStarter,
    policy: Res<CompactionPolicy>,
    log: Res<SessionLog>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let call = done.entity;
    let Ok((&CallOf(turn), model_call, Done(reply))) = calls.get(call) else {
        return;
    };
    commands.entity(call).despawn();
    let Ok((&TurnOf(agent), mut turn_spent, mut recovery)) = turns.get_mut(turn) else {
        return;
    };
    let Ok(((id, mut conversation, mut inbox), compacted, connection, mut spent)) =
        agents.get_mut(agent)
    else {
        return;
    };
    let spec = connection.map(|connection| &*connection.spec);
    let response = match reply {
        Ok(response) => {
            // A reply the turn-failure rule rejects was still billed.
            spent.record(&response.usage);
            turn_spent.0.record(&response.usage);
            recovery.retries = 0;
            response
        }
        Err(report) => {
            let failed = Failed {
                agent,
                id,
                turn,
                report,
                spec,
                policy: &policy,
                log: &log,
            };
            failed.recover(
                &mut recovery,
                &mut conversation,
                compacted,
                &wake,
                &mut commands,
                &mut notices,
            );
            return;
        }
    };
    if let Some(message) = response.message() {
        log.commit(id, &mut conversation, message, None);
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
            let results = tool_calls
                .iter()
                .map(|tool_call| failed(tool_call, format!("not run: {failure}")))
                .collect();
            log.commit(id, &mut conversation, Message::tool_results(results), None);
        }
        notices.write(Notice::error(agent, format!("The turn failed: {failure}.")));
        drop_unanswered(agent, id, &mut conversation, &log, &mut notices);
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
        let to = Delivery {
            agent,
            id,
            spec,
            log: &log,
        };
        let carried = deliver_steering(&to, &mut inbox, &mut conversation, &mut notices)
            || deliver_queued(&to, &mut inbox, &mut conversation, &mut notices);
        if carried {
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
    // Every call exists before any starts: an open call may end at once.
    let mut earlier: Vec<Footprint> = Vec::with_capacity(tool_calls.len());
    let mut ready = Vec::new();
    for call in tool_calls {
        let run = starter.run(call, Some(model_call.effect));
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
        starter.start(&mut commands, entity, agent, &run);
    }
}

/// A failed model call of a turn.
struct Failed<'a> {
    agent: Entity,
    id: &'a AgentId,
    turn: Entity,
    report: &'a ErrorReport,
    /// The model that failed.
    spec: Option<&'a ModelSpec>,
    /// How a compaction after an overflow keeps the newest messages.
    policy: &'a CompactionPolicy,
    log: &'a SessionLog,
}

impl Failed<'_> {
    /// Carries the turn on after the failure, as the [`RETRY`] policy
    /// decides: waits and calls again; clears old tool outputs, or else
    /// compacts, and calls again; or ends the turn, keeping the user's message when nothing is
    /// wrong with it.
    fn recover(
        &self,
        recovery: &mut Recovery,
        conversation: &mut Conversation,
        compacted: &Compacted,
        wake: &Wake,
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) {
        let (agent, report) = (self.agent, self.report);
        match RETRY.verdict(report, recovery.retries, now()) {
            Verdict::Retry(delay) => {
                recovery.retries += 1;
                let backoff = Backoff {
                    attempt: recovery.retries,
                    until: Instant::now() + delay,
                    why: report.to_string(),
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
                commands.spawn((
                    Name::new("retry wait"),
                    backoff,
                    Running::spawn(model_pool(), wake, recovery::wait(delay)),
                    CallOf(self.turn),
                ));
            }
            Verdict::Overflow => {
                // Clear old outputs first, which costs no call; then
                // summarize; then fail.
                if !recovery.cleared {
                    recovery.cleared = true;
                    if self.clear(conversation, compacted, notices) {
                        commands.trigger(CallModel { entity: self.turn });
                        return;
                    }
                }
                if recovery.compactions < MAX_COMPACTIONS
                    && self.spec.is_some_and(|spec| {
                        compacted
                            .cut(
                                conversation.messages(),
                                self.policy,
                                spec,
                                &CompactReason::Overflow,
                            )
                            .is_some()
                    })
                {
                    recovery.compactions += 1;
                    notices.write(Notice::info(
                        agent,
                        "The conversation outgrew the model's context window: summarizing its \
                         older messages and sending it again.",
                    ));
                    commands.trigger(Summarize {
                        entity: self.turn,
                        reason: CompactReason::Overflow,
                    });
                    return;
                }
                let why = format!(
                    "The conversation does not fit the model's context window, even \
                     compacted and with old tool outputs cleared: {report}"
                );
                notices.write(Notice::error(agent, why.clone()));
                self.fail(why, conversation, commands, notices);
            }
            Verdict::GaveUp(why) => {
                notices.write(Notice::error(
                    agent,
                    format!(
                        "The model call failed: {report}. Not retrying: {why}. Your message is \
                         kept for a retry."
                    ),
                ));
                end_turn(commands, self.turn, TurnOutcome::Failed(report.to_string()));
            }
            Verdict::Final => {
                let why = format!("The model call failed: {report}");
                notices.write(Notice::error(agent, why.clone()));
                self.fail(why, conversation, commands, notices);
            }
        }
    }

    /// Clears the older tool outputs of the live conversation. Whether it
    /// cleared any.
    fn clear(
        &self,
        conversation: &mut Conversation,
        compacted: &Compacted,
        notices: &mut MessageWriter<Notice>,
    ) -> bool {
        let cleared = recovery::clearing().clear(compacted.live_mut(conversation.messages_mut()));
        if cleared.results > 0 {
            notices.write(Notice::info(
                self.agent,
                format!(
                    "The conversation outgrew the model's context window: cleared {} older \
                     tool outputs (about {} tokens) and sending it again.",
                    cleared.results,
                    usage::tokens(cleared.tokens as u64)
                ),
            ));
        }
        cleared.results > 0
    }

    /// Ends the turn as failed for `why`, taking out the user's message:
    /// the same request would fail again.
    fn fail(
        &self,
        why: String,
        conversation: &mut Conversation,
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) {
        drop_unanswered(self.agent, self.id, conversation, self.log, notices);
        end_turn(commands, self.turn, TurnOutcome::Failed(why));
    }
}

/// Calls the model again once a retry's wait is over.
pub(crate) fn on_retry_due(
    done: On<Add<Done<RetryDue>>>,
    of: Query<&CallOf>,
    mut commands: Commands,
) {
    let Ok(&CallOf(turn)) = of.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    commands.trigger(CallModel { entity: turn });
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
    mut agents: Query<(&AgentId, &mut Conversation)>,
    runs: Query<(&ToolCallRun, Option<&ToolOutput>, Has<Queued>)>,
    starter: ToolStarter,
    log: Res<SessionLog>,
    mut commands: Commands,
) {
    if let Ok((ToolOutput(result), Some(mut open))) = ended.get_mut(done.entity) {
        open.0.settle(Ok(outcome_of(result)));
    }
    commands
        .entity(done.entity)
        .try_remove::<(OpenCall, Running<ToolResult>)>();
    let Ok(&CallOf(turn)) = of.get(done.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), calls)) = turns.get(turn) else {
        return;
    };
    let Ok((id, mut conversation)) = agents.get_mut(agent) else {
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
            results.push(result.clone());
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
    log.commit(id, &mut conversation, Message::tool_results(results), None);
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
            Option<&'static ToolHandler>,
            &'static Footprint,
        ),
    >,
    agents: Query<'w, 's, (&'static AgentId, &'static ToolAccess)>,
    effects: Res<'w, Effects>,
    log: Res<'w, SessionLog>,
    wake: Res<'w, Wake>,
}

impl ToolStarter<'_, '_> {
    /// The footprint of the tool `name`; a tool that is not registered
    /// runs on its own.
    pub fn footprint(&self, name: &str) -> Footprint {
        self.tools
            .iter()
            .find(|(_, def, ..)| def.0.name.as_str() == name)
            .map_or_else(Footprint::default, |(.., &footprint)| footprint)
    }

    /// The run of `call`, asked for by the effect `parent`.
    pub fn run(&self, call: ToolCall, parent: Option<EffectId>) -> ToolCallRun {
        ToolCallRun {
            footprint: self.footprint(call.function.name.as_str()),
            call,
            parent,
        }
    }

    /// Whether a call of the tool `name` left without a result by a restart
    /// starts again: an ordinary read-only tool. Any other such call is
    /// answered as interrupted.
    pub(crate) fn reruns(&self, name: &str) -> bool {
        self.tools.iter().any(|(_, def, handler, footprint)| {
            def.0.name.as_str() == name && handler.is_some() && *footprint == Footprint::ReadOnly
        })
    }

    /// Starts `run`, the call entity `call` of `agent`. An ordinary tool's
    /// call runs on the one dispatch path, and its [`ToolOutput`] is
    /// inserted when it finishes. An open tool's call is recorded as
    /// started and its tool's observer gets [`ToolCalled`]. A call to a
    /// tool that is not registered, that the agent may not use, or with
    /// arguments that do not fit, is dispatched and recorded like any other
    /// and answered with an error. Before a call that may change something, the
    /// session log is written, so the reply that asked for it is on disk
    /// first.
    pub fn start(&self, commands: &mut Commands, call: Entity, agent: Entity, run: &ToolCallRun) {
        let Ok((id, access)) = self.agents.get(agent) else {
            return;
        };
        let name = run.call.function.name.as_str();
        let tool = self
            .tools
            .iter()
            .find(|(_, def, ..)| def.0.name.as_str() == name && access.allows(name));
        if !tool.is_some_and(|(.., footprint)| *footprint == Footprint::ReadOnly) {
            self.log.flush();
        }
        let refused = |kind: ToolErrorKind, why: String| {
            ErasedHandler::new(Refused {
                name: name.to_owned(),
                kind,
                why,
            })
        };
        let why = tool.and_then(|(_, def, ..)| arguments_refusal(&def.0.parameters, &run.call));
        let handler = match (tool, why) {
            (None, _) => refused(
                ToolErrorKind::NotFound,
                format!("no tool named `{name}` is available"),
            ),
            (Some(_), Some(why)) => refused(ToolErrorKind::InvalidArgs, why),
            (Some((tool, _, None, ..)), None) => {
                let args = recorded_args(&run.call);
                let effect = self.effects.open(&id.0, run.parent, name, args);
                commands.entity(call).insert(OpenCall(effect));
                commands.trigger(ToolCalled {
                    entity: tool,
                    call,
                    agent,
                });
                return;
            }
            (Some((_, _, Some(handler), ..)), None) => handler.0.erased(),
        };
        let (_, work) = run_tool_call(&self.effects, &id.0, run.parent, handler, run.call.clone());
        let span = info_span!("tool_call", agent = %id.0, tool = name, parent = ?run.parent);
        commands.entity(call).insert(Running::spawn(
            tool_pool(),
            &self.wake,
            work.instrument(span),
        ));
    }
}

/// The [`Name`] of a tool call's entity.
pub fn tool_name(run: &ToolCallRun) -> Name {
    Name::new(format!("tool call {}", run.call.function.name.as_str()))
}

/// Compacts an idle agent's conversation on the user's request, in a turn
/// of its own that ends with the summary.
pub(crate) fn on_compact(
    compact: On<Compact>,
    agents: Query<
        (
            &Conversation,
            &Compacted,
            Option<&Connection>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    policy: Res<CompactionPolicy>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = compact.entity;
    let Ok((conversation, compacted, connection, busy)) = agents.get(agent) else {
        return;
    };
    if refused_mid_turn(agent, busy, &mut notices) {
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::info(agent, NO_MODEL));
        return;
    };
    let reason = CompactReason::Asked {
        focus: compact.focus.clone(),
    };
    if compacted
        .cut(conversation.messages(), &policy, &connection.spec, &reason)
        .is_none()
    {
        notices.write(Notice::info(agent, "Nothing to compact yet."));
        return;
    }
    let turn = commands
        .spawn((Name::new("compaction"), TurnOf(agent)))
        .id();
    commands.trigger(Summarize {
        entity: turn,
        reason,
    });
}

/// Starts the summary call of a compaction, on the one dispatch path, with
/// the agent's model. When there is nothing to summarize, or no model, the
/// turn carries on, or ends when the user asked.
pub(crate) fn on_summarize(
    summarize: On<Summarize>,
    turns: Query<&TurnOf>,
    agents: Query<(
        &AgentId,
        &Conversation,
        &Compacted,
        Option<&Connection>,
        Option<&EffectParent>,
    )>,
    policy: Res<CompactionPolicy>,
    effects: Res<Effects>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let turn = summarize.entity;
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Ok((id, conversation, compacted, connection, effect_parent)) = agents.get(agent) else {
        return;
    };
    let reason = &summarize.reason;
    let planned = connection
        .ok_or_else(|| "no model is connected".to_owned())
        .and_then(|connection| {
            let upto = compacted
                .cut(conversation.messages(), &policy, &connection.spec, reason)
                .ok_or_else(|| "nothing to summarize yet".to_owned())?;
            compaction::plan(
                &policy,
                compacted,
                conversation.messages(),
                upto,
                &connection.spec,
                reason.clone(),
            )
            .map(|(summarizing, request)| (connection.handler.erased(), summarizing, request))
        });
    let (handler, summarizing, request) = match planned {
        Ok(planned) => planned,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
            carry_on(turn, reason, &mut commands);
            return;
        }
    };
    let (effect, reply) = effects.dispatch(
        &id.0,
        effect_parent.map(|parent| parent.0),
        handler,
        EffectKind::Completion {
            request,
            stream: true,
        },
    );
    let span = info_span!("summary_call", agent = %id.0, effect = %effect);
    let work = effects
        .caught(effect, rig_memory::completion_of(reply))
        .map(Summary)
        .instrument(span);
    commands.spawn((
        Name::new("summary call"),
        summarizing,
        Running::spawn(model_pool(), &wake, work),
        CallOf(turn),
    ));
}

/// Takes a finished summary: the agent's [`Compacted`] now replaces the
/// summarized messages with it, and its log records the compaction. A
/// failed summary replaces nothing. Either way the turn carries on with its
/// model call, or ends when the user asked for the compaction.
pub(crate) fn on_summary_done(
    done: On<Add<Done<Summary>>>,
    calls: Query<(&CallOf, &Summarizing, &Done<Summary>)>,
    mut turns: Query<(&TurnOf, &mut TurnSpending)>,
    mut agents: Query<(&AgentId, &Conversation, &mut Compacted, &mut Spending)>,
    log: Res<SessionLog>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let call = done.entity;
    let Ok((&CallOf(turn), summarizing, Done(Summary(reply)))) = calls.get(call) else {
        return;
    };
    commands.entity(call).despawn();
    let Ok((&TurnOf(agent), mut turn_spent)) = turns.get_mut(turn) else {
        return;
    };
    let Ok((id, conversation, mut compacted, mut spent)) = agents.get_mut(agent) else {
        return;
    };
    let summary = reply
        .as_ref()
        .map_err(ToString::to_string)
        .and_then(|response| {
            spent.record_aside(&response.usage);
            turn_spent.0.record_aside(&response.usage);
            Summarizer::summary_text(response).map_err(|why| why.to_string())
        });
    match summary {
        Ok(summary) => {
            *compacted = Compacted(SummaryState {
                upto: summarizing.upto,
                summary,
                tracked: summarizing.tracked.clone(),
            });
            log.compaction(id, &compacted);
            let left = compacted.estimate(conversation.messages());
            spent.context = Some(left);
            notices.write(Notice::info(
                agent,
                format!(
                    "Compacted {} messages (about {} tokens) into a summary and kept {} \
                     (about {} tokens) as it was; the model now gets about {} tokens of \
                     conversation.",
                    summarizing.messages,
                    usage::tokens(summarizing.tokens),
                    match summarizing.kept {
                        1 => "the newest message".to_owned(),
                        kept => format!("the newest {kept} messages"),
                    },
                    usage::tokens(summarizing.kept_tokens),
                    usage::tokens(left)
                ),
            ));
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Compaction failed: {why}.")));
        }
    }
    carry_on(turn, &summarizing.reason, &mut commands);
}

/// After a compaction: the turn calls the model, or ends when the user
/// asked for the compaction.
fn carry_on(turn: Entity, reason: &CompactReason, commands: &mut Commands) {
    match reason {
        CompactReason::Asked { .. } => {
            commands.entity(turn).despawn();
        }
        CompactReason::Threshold | CompactReason::Overflow => {
            commands.trigger(CallModel { entity: turn });
        }
    }
}

/// The wall clock as std's `SystemTime`, which rig-core's retry policy
/// reads a `Retry-After` date against; read through `web_time` so it also
/// works on the web.
fn now() -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(super::journal::now_ms())
}
