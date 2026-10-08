//! The turn loop. A user message spawns a turn entity, [`TurnOf`] its
//! agent; the turn's model call and tool calls are entities [`CallOf`] the
//! turn, and observers of their [`Done`] outputs carry the turn on. A
//! reply's tool calls run one at a time, in order; rig-core's turn-failure
//! rule decides when a reply ends the turn instead. A failed model call is
//! retried or recovered from as [`recovery`] decides. A conversation near
//! the model's window is [`compaction`]-ed before the next call. Despawning
//! the turn ends it, and the agent's [`ActiveTurn`] going away reports it
//! finished.

use std::pin::Pin;
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemId;
use bevy_log::info_span;
use bevy_log::tracing::Instrument;
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
use rig_core::message::{ToolCall, ToolResult, UserContent};
use rig_core::serve::{Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};

use super::agent::{
    ActiveTurn, Agent, AgentId, CallOf, Calls, Compact, Connection, Conversation, Effort,
    Interrupt, ModelChoice, Notice, Partial, Queued, Retry, SetEffort, SetModel, Submit,
    SystemPrompt, ToolAccess, ToolCallRun, TurnFinished, TurnOf,
};
use super::calls::{Done, Running, Wake};
use super::commands::{CommandArgs, SlashCommand};
use super::compaction::{
    self, CompactReason, Compacted, MAX_COMPACTIONS, Summarize, Summarizing, Summary,
};
use super::effects::Effects;
use super::models;
use super::prompt::{PromptSection, ToolRules, system_prompt};
use super::recovery::{
    self, Backoff, KEEP_RECENT_OUTPUTS, MAX_CLEARINGS, MAX_RETRIES, Recovery, RetryDue, Verdict,
};
use super::tools::{ToolDef, ToolHandler, failed, run_tool_call};
use super::usage::{self, Spending, TurnSpending};

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
/// threads of its own (see [`blocking`](super::blocking::blocking)).
fn tool_pool() -> &'static AsyncComputeTaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// Runs a slash command or starts a turn.
pub(crate) fn on_submit(
    submit: On<Submit>,
    mut agents: Query<(&mut Conversation, Has<ActiveTurn>), With<Agent>>,
    slash: Query<(Entity, &SlashCommand)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = submit.entity;
    let text = submit.text.trim();
    if text.is_empty() {
        return;
    }
    if let Some(line) = text.strip_prefix('/') {
        let (name, args) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match slash.iter().find(|(_, command)| command.name == name) {
            // The command sits on its system's own entity.
            Some((system, _)) => commands.run_system_with(
                SystemId::<In<CommandArgs>>::from_entity(system),
                CommandArgs {
                    agent,
                    args: args.trim().to_owned(),
                },
            ),
            None => {
                // /help comes from a plugin, so point at it only when loaded.
                let hint = if slash.iter().any(|(_, command)| command.name == "help") {
                    " /help lists the commands."
                } else {
                    ""
                };
                notices.write(Notice::error(
                    agent,
                    format!("Unknown command /{name}.{hint}"),
                ));
            }
        }
        return;
    }
    let Ok((mut conversation, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::info(
            agent,
            "The agent is busy. Press Esc to stop the turn.",
        ));
        return;
    }
    conversation.0.push(Message::user(text));
    let turn = commands.spawn((Name::new("turn"), TurnOf(agent))).id();
    commands.trigger(CallModel { entity: turn });
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
    if !matches!(conversation.0.last(), Some(Message::User { .. })) {
        notices.write(Notice::info(
            agent,
            "Nothing to retry: the model answered the last message.",
        ));
        return;
    }
    let turn = commands.spawn((Name::new("turn"), TurnOf(agent))).id();
    commands.trigger(CallModel { entity: turn });
}

/// Reports a finished turn, however its entity went away.
pub(crate) fn on_turn_end(end: On<Remove<ActiveTurn>>, mut finished: MessageWriter<TurnFinished>) {
    finished.write(TurnFinished { agent: end.entity });
}

/// Stops a running turn. Every tool call of the last reply gets a result,
/// real or "interrupted", and despawning the turn cancels its calls.
pub(crate) fn on_interrupt(
    interrupt: On<Interrupt>,
    mut agents: Query<(&mut Conversation, &ActiveTurn)>,
    turns: Query<&Calls>,
    runs: Query<(&ToolCallRun, Option<&Done<ToolResult>>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = interrupt.entity;
    let Ok((mut conversation, active)) = agents.get_mut(agent) else {
        return;
    };
    let turn = active.turn();
    let runs = turns
        .get(turn)
        .into_iter()
        .flat_map(|calls| runs.iter_many(calls.iter()).flatten());
    conversation
        .0
        .extend(stopped_results(runs, "interrupted by the user"));
    commands.entity(turn).despawn();
    notices.write(Notice::info(agent, "Interrupted."));
}

/// The results of a stopped reply's tool calls, in call order: each
/// finished call's result, and an error saying `why` for the others. `None`
/// when the reply made no calls.
pub(crate) fn stopped_results<'a>(
    runs: impl Iterator<Item = (&'a ToolCallRun, Option<&'a Done<ToolResult>>)>,
    why: &str,
) -> Option<Message> {
    let results: Vec<ToolResult> = runs
        .map(|(run, done)| match done {
            Some(Done(result)) => result.clone(),
            None => failed(&run.call, why.to_owned()),
        })
        .collect();
    (!results.is_empty()).then(|| Message::tool_results(results))
}

/// The result of a tool call the session stopped before it finished.
pub(crate) const STOPPED: &str = "the session stopped before this call finished";

/// How long exit waits for running calls to be cancelled. A cancellation
/// waits for a pool thread to drop the call's future, which records its
/// effect as cancelled before the last flush; a plugin tool that blocks in
/// `poll` must not stall the exit, so whatever misses this is dropped.
const EXIT_GRACE: Duration = Duration::from_secs(1);

type Cancelling = Vec<Pin<Box<dyn Future<Output = ()>>>>;

/// On exit, stops every running turn before the session is saved, so the
/// saved conversation never ends in unanswered tool calls. Running calls
/// are cancelled and waited for, within a second for all of them;
/// every unfinished tool call is answered as stopped.
pub(crate) fn stop_turns_on_exit(world: &mut World) {
    let mut cancelling = Cancelling::new();
    take_running::<ModelReply>(world, &mut cancelling);
    take_running::<ToolResult>(world, &mut cancelling);
    take_running::<Summary>(world, &mut cancelling);
    let deadline = Instant::now() + EXIT_GRACE;
    while !cancelling.is_empty() && Instant::now() < deadline {
        cancelling.retain_mut(|cancel| check_ready(cancel).is_none());
        std::thread::sleep(Duration::from_millis(2));
    }
    let turns: Vec<(Entity, Entity)> = world
        .query::<(Entity, &TurnOf)>()
        .iter(world)
        .map(|(turn, of)| (turn, of.0))
        .collect();
    let mut runs = world.query::<(&ToolCallRun, Option<&Done<ToolResult>>)>();
    for (turn, agent) in turns {
        let calls: Vec<Entity> = world
            .get::<Calls>(turn)
            .map(|calls| calls.iter().collect())
            .unwrap_or_default();
        let results = stopped_results(runs.iter_many(world, calls).flatten(), STOPPED);
        if let Some(mut conversation) = world.get_mut::<Conversation>(agent) {
            conversation.0.extend(results);
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
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(busy) = agents.get(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, busy, "model", &mut notices) {
        return;
    }
    match models::resolve(&set.model) {
        Some(spec) => {
            commands
                .entity(set.entity)
                .insert(ModelChoice(models::reference(spec)));
        }
        None => {
            notices.write(Notice::error(
                set.entity,
                format!("No catalog model `{}`. Use vendor/model.", set.model),
            ));
        }
    }
}

/// Connects an agent whose [`ModelChoice`] was inserted, by `/model` or by
/// restoring a session, so requests never re-resolve the provider. A
/// reasoning setting the new model does not take is reset.
pub(crate) fn on_model_chosen(
    chosen: On<Insert<ModelChoice>>,
    mut agents: Query<(&ModelChoice, &mut Effort)>,
    mut effects: ResMut<Effects>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = chosen.entity;
    let Ok((choice, mut effort)) = agents.get_mut(agent) else {
        return;
    };
    let connection = models::resolve(&choice.0)
        .ok_or_else(|| format!("the catalog has no model `{}`", choice.0))
        .and_then(|spec| {
            effects
                .model_handler(spec)
                .map(|handler| Connection { spec, handler })
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
    let spec = connection.spec;
    notices.write(Notice::info(
        agent,
        format!("Model: {} ({}).", spec.display_name, choice.0),
    ));
    if let Err(refusal) = models::check_effort(spec, effort.0) {
        effort.0 = None;
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
    mut agents: Query<(Option<&Connection>, &mut Effort, Has<ActiveTurn>), With<Agent>>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((connection, mut effort, busy)) = agents.get_mut(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, busy, "effort", &mut notices) {
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::info(set.entity, "Pick a model with /model first."));
        return;
    };
    match models::check_effort(connection.spec, set.effort.0) {
        Ok(()) => {
            *effort = set.effort;
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

/// Refuses a model or reasoning change while the agent's turn runs, with a
/// notice naming `/command`: the rest of the turn would go to a model, or
/// use a setting, it did not start with. Every sender of [`SetModel`] and
/// [`SetEffort`] gets the same refusal.
fn refused_mid_turn(
    agent: Entity,
    busy: bool,
    command: &str,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    if busy {
        notices.write(Notice::info(
            agent,
            format!("A turn is running. Press Esc to stop it, then /{command}."),
        ));
    }
    busy
}

/// Sends the conversation of the turn's agent to its model. When that
/// cannot be done the turn ends.
pub(crate) fn on_call_model(
    call: On<CallModel>,
    mut turns: Query<(&TurnOf, &mut Recovery)>,
    mut agents: Query<(
        &AgentId,
        &mut Conversation,
        &Compacted,
        &mut Spending,
        Option<&Connection>,
        &Effort,
        &SystemPrompt,
        &ToolAccess,
    )>,
    tools: Query<(&ToolDef, &ToolRules)>,
    sections: Query<&PromptSection>,
    effects: Res<Effects>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let turn = call.entity;
    let Ok((&TurnOf(agent), mut recovery)) = turns.get_mut(turn) else {
        return;
    };
    let Ok((id, mut conversation, compacted, mut spent, connection, effort, prompt, access)) =
        agents.get_mut(agent)
    else {
        return;
    };
    if let Some(connection) = connection
        && recovery.compactions < MAX_COMPACTIONS
        && must_summarize(
            agent,
            &mut conversation,
            compacted,
            &mut spent,
            connection.spec,
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
    let request = connection
        .ok_or_else(|| "No model is connected. Pick one with /model.".to_owned())
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
            let messages = compacted.request(&conversation.0);
            prepare(messages, connection, effort, preamble, definitions)
                .map(|request| models::with_cache_key(connection.spec, request, &id.0))
                .map(|request| (connection.handler.clone(), connection.spec, request))
        });
    let (handler, spec, request) = match request {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            drop_unanswered(agent, &mut conversation, &mut notices);
            commands.entity(turn).despawn();
            return;
        }
    };
    let (effect, reply) = effects.dispatch(
        &id.0,
        None,
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
        model = %models::reference(spec)
    );
    let work = effects
        .caught(effect, stream_reply(reply, sender, wake.clone()))
        .instrument(span);
    commands.spawn((
        Name::new("model call"),
        ModelCall { effect, feed },
        Running::spawn(model_pool(), &wake, work),
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
    compacted: &Compacted,
    spent: &mut Mut<Spending>,
    spec: &ModelSpec,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    let used = spent
        .context
        .unwrap_or(0)
        .max(compacted.estimate(&conversation.0));
    if !compaction::over_threshold(used, spec) {
        return false;
    }
    let cleared =
        recovery::clear_tool_outputs(compacted.live_mut(&mut conversation.0), KEEP_RECENT_OUTPUTS);
    let left = used.saturating_sub(cleared.tokens);
    if cleared.results > 0 {
        spent.context = Some(left);
        notices.write(Notice::info(
            agent,
            format!(
                "The conversation nears the model's context window: cleared {} older tool \
                 outputs (about {} tokens).",
                cleared.results,
                usage::tokens(cleared.tokens)
            ),
        ));
    }
    compaction::over_threshold(left, spec) && compacted.cut(&conversation.0, spec, false).is_some()
}

/// Removes the user's last message when no model answered it, so the next
/// message does not follow an unanswered one. Tool results stay: the model
/// asked for them.
fn drop_unanswered(
    agent: Entity,
    conversation: &mut Conversation,
    notices: &mut MessageWriter<Notice>,
) {
    let unanswered = conversation.0.last().is_some_and(|message| {
        matches!(message, Message::User { content }
            if content.iter().all(|item| matches!(item, UserContent::Text(_))))
    });
    if unanswered {
        conversation.0.pop();
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
    preamble: String,
    tools: Vec<rig_core::completion::ToolDefinition>,
) -> Result<CompletionRequest, String> {
    let options = models::request_options(connection.spec, effort.0);
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
/// turn; a reply without tool calls ends it too; otherwise its first tool
/// call starts and the rest are [`Queued`] in order.
pub(crate) fn on_model_done(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &ModelCall, &Done<ModelReply>)>,
    mut turns: Query<(&TurnOf, &mut TurnSpending, &mut Recovery)>,
    mut agents: Query<(
        &AgentId,
        &ToolAccess,
        &mut Conversation,
        &Compacted,
        Option<&Connection>,
        &mut Spending,
    )>,
    tools: Query<(&ToolDef, &ToolHandler)>,
    effects: Res<Effects>,
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
    let Ok((id, access, mut conversation, compacted, connection, mut spent)) =
        agents.get_mut(agent)
    else {
        return;
    };
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
                turn,
                report,
                spec: connection.map(|connection| connection.spec),
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
    conversation.0.extend(response.message());
    let tool_calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
    let failure = turn_failure(
        &response.choice,
        Some(&response.stop()),
        response.finish_reason().as_ref(),
    );
    if let Some(failure) = failure {
        // Every call in the history gets a result, though none ran.
        if !tool_calls.is_empty() {
            conversation.0.push(Message::tool_results(
                tool_calls
                    .iter()
                    .map(|tool_call| failed(tool_call, format!("not run: {failure}")))
                    .collect(),
            ));
        }
        notices.write(Notice::error(agent, format!("The turn failed: {failure}.")));
        drop_unanswered(agent, &mut conversation, &mut notices);
        commands.entity(turn).despawn();
        return;
    }
    let mut runs = tool_calls.into_iter().map(|call| ToolCallRun {
        call,
        parent: model_call.effect,
    });
    let Some(first) = runs.next() else {
        commands.entity(turn).despawn();
        return;
    };
    let running = start_tool(&first, &id.0, access, &tools, &effects, &wake);
    commands.spawn((tool_name(&first), first, running, CallOf(turn)));
    for run in runs {
        commands.spawn((tool_name(&run), run, Queued, CallOf(turn)));
    }
}

/// A failed model call of a turn.
struct Failed<'a> {
    agent: Entity,
    turn: Entity,
    report: &'a ErrorReport,
    /// The model that failed.
    spec: Option<&'static ModelSpec>,
}

impl Failed<'_> {
    /// Carries the turn on after the failure, as [`recovery::verdict`]
    /// decides: waits and calls again; clears old tool outputs and calls
    /// again; or ends the turn, keeping the user's message when nothing is
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
        match recovery::verdict(report, recovery.retries) {
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
                        "The model call failed: {report}. Retrying in {}s ({}/{MAX_RETRIES}).",
                        backoff.seconds_left(),
                        backoff.attempt
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
                // Clear first, which costs no call; then summarize; then
                // clear every old output.
                if recovery.clearings == 0 && self.clear(recovery, conversation, compacted, notices)
                {
                    commands.trigger(CallModel { entity: self.turn });
                    return;
                }
                if recovery.compactions < MAX_COMPACTIONS
                    && self
                        .spec
                        .is_some_and(|spec| compacted.cut(&conversation.0, spec, true).is_some())
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
                while recovery.clearings < MAX_CLEARINGS {
                    if self.clear(recovery, conversation, compacted, notices) {
                        commands.trigger(CallModel { entity: self.turn });
                        return;
                    }
                }
                notices.write(Notice::error(
                    agent,
                    format!(
                        "The conversation does not fit the model's context window, even \
                         compacted and with old tool outputs cleared: {report}"
                    ),
                ));
                self.fail(conversation, commands, notices);
            }
            Verdict::GaveUp(why) => {
                notices.write(Notice::error(
                    agent,
                    format!(
                        "The model call failed: {report}. Not retrying: {why}. Your message is \
                         kept; /retry sends it again."
                    ),
                ));
                commands.entity(self.turn).despawn();
            }
            Verdict::Final => {
                notices.write(Notice::error(
                    agent,
                    format!("The model call failed: {report}"),
                ));
                self.fail(conversation, commands, notices);
            }
        }
    }

    /// Clears the tool outputs of the live conversation once, keeping fewer
    /// with each clearing of the turn. Whether it cleared any.
    fn clear(
        &self,
        recovery: &mut Recovery,
        conversation: &mut Conversation,
        compacted: &Compacted,
        notices: &mut MessageWriter<Notice>,
    ) -> bool {
        let keep = recovery::keep_for(recovery.clearings);
        recovery.clearings += 1;
        let cleared = recovery::clear_tool_outputs(compacted.live_mut(&mut conversation.0), keep);
        if cleared.results > 0 {
            notices.write(Notice::info(
                self.agent,
                format!(
                    "The conversation outgrew the model's context window: cleared {} older \
                     tool outputs (about {} tokens) and sending it again.",
                    cleared.results,
                    usage::tokens(cleared.tokens)
                ),
            ));
        }
        cleared.results > 0
    }

    /// Ends the turn, taking out the user's message: the same request
    /// would fail again.
    fn fail(
        &self,
        conversation: &mut Conversation,
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) {
        drop_unanswered(self.agent, conversation, notices);
        commands.entity(self.turn).despawn();
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

/// Takes a finished tool call: starts the next queued call of the reply,
/// or, once every call finished, appends their results in call order and
/// calls the model again.
pub(crate) fn on_tool_done(
    done: On<Add<Done<ToolResult>>>,
    of: Query<&CallOf>,
    turns: Query<(&TurnOf, &Calls)>,
    mut agents: Query<(&AgentId, &ToolAccess, &mut Conversation)>,
    runs: Query<(&ToolCallRun, Option<&Done<ToolResult>>, Has<Queued>)>,
    tools: Query<(&ToolDef, &ToolHandler)>,
    effects: Res<Effects>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(&CallOf(turn)) = of.get(done.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), calls)) = turns.get(turn) else {
        return;
    };
    let Ok((id, access, mut conversation)) = agents.get_mut(agent) else {
        return;
    };
    let mut results = Vec::new();
    for call in calls.iter() {
        let Ok((run, output, queued)) = runs.get(call) else {
            continue;
        };
        match output {
            Some(Done(result)) => results.push(result.clone()),
            None if queued => {
                let running = start_tool(run, &id.0, access, &tools, &effects, &wake);
                commands.entity(call).remove::<Queued>().insert(running);
                return;
            }
            None => return,
        }
    }
    conversation.0.push(Message::tool_results(results));
    commands.entity(turn).despawn_related::<Calls>();
    commands.trigger(CallModel { entity: turn });
}

/// Starts `run` on the one dispatch path. A call to a tool that is not
/// registered, or that the agent may not use, is dispatched and recorded
/// like any other and answered with an error.
fn start_tool(
    run: &ToolCallRun,
    scope: &str,
    access: &ToolAccess,
    tools: &Query<(&ToolDef, &ToolHandler)>,
    effects: &Effects,
    wake: &Wake,
) -> Running<ToolResult> {
    let name = run.call.function.name.as_str();
    let handler = tools
        .iter()
        .find(|(def, _)| def.0.name.as_str() == name && access.allows(name))
        .map(|(_, handler)| handler.0.clone());
    let work = run_tool_call(effects, scope, run.parent, handler, run.call.clone());
    let span = info_span!("tool_call", agent = %scope, tool = name, parent = %run.parent);
    Running::spawn(tool_pool(), wake, work.instrument(span))
}

fn tool_name(run: &ToolCallRun) -> Name {
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
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = compact.entity;
    let Ok((conversation, compacted, connection, busy)) = agents.get(agent) else {
        return;
    };
    if refused_mid_turn(agent, busy, "compact", &mut notices) {
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::info(agent, "Pick a model with /model first."));
        return;
    };
    if compacted
        .cut(&conversation.0, connection.spec, true)
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
        reason: CompactReason::Asked {
            focus: compact.focus.clone(),
        },
    });
}

/// Starts the summary call of a compaction, on the one dispatch path, with
/// the agent's model. When there is nothing to summarize, or no model, the
/// turn carries on, or ends when the user asked.
pub(crate) fn on_summarize(
    summarize: On<Summarize>,
    turns: Query<&TurnOf>,
    agents: Query<(&AgentId, &Conversation, &Compacted, Option<&Connection>)>,
    effects: Res<Effects>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let turn = summarize.entity;
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Ok((id, conversation, compacted, connection)) = agents.get(agent) else {
        return;
    };
    let reason = &summarize.reason;
    let planned = connection
        .ok_or_else(|| "no model is connected".to_owned())
        .and_then(|connection| {
            let upto = compacted
                .cut(&conversation.0, connection.spec, true)
                .ok_or_else(|| "nothing to summarize yet".to_owned())?;
            compaction::plan(
                compacted,
                &conversation.0,
                upto,
                connection.spec,
                reason.clone(),
            )
            .map(|(summarizing, request)| (connection.handler.clone(), summarizing, request))
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
        None,
        handler,
        EffectKind::Completion {
            request,
            stream: true,
        },
    );
    let span = info_span!("summary_call", agent = %id.0, effect = %effect);
    let work = effects
        .caught(effect, compaction::summarize(reply))
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
/// summarized messages with it. A failed summary replaces nothing. Either
/// way the turn carries on with its model call, or ends when the user
/// asked for the compaction.
pub(crate) fn on_summary_done(
    done: On<Add<Done<Summary>>>,
    calls: Query<(&CallOf, &Summarizing, &Done<Summary>)>,
    mut turns: Query<(&TurnOf, &mut TurnSpending)>,
    mut agents: Query<(&Conversation, &mut Compacted, &mut Spending)>,
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
    let Ok((conversation, mut compacted, mut spent)) = agents.get_mut(agent) else {
        return;
    };
    let summary = reply
        .as_ref()
        .map_err(ToString::to_string)
        .and_then(|response| {
            spent.record_aside(&response.usage);
            turn_spent.0.record_aside(&response.usage);
            compaction::summary_text(response)
        });
    match summary {
        Ok(summary) => {
            *compacted = Compacted {
                upto: summarizing.upto,
                summary,
                read: summarizing.read.clone(),
                modified: summarizing.modified.clone(),
            };
            let left = compacted.estimate(&conversation.0);
            spent.context = Some(left);
            notices.write(Notice::info(
                agent,
                format!(
                    "Compacted {} messages (about {} tokens) into a summary; the model now \
                     gets about {} tokens of conversation.",
                    summarizing.messages,
                    usage::tokens(summarizing.tokens),
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
