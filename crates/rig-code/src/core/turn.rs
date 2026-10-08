//! The turn loop: requests from views, then model calls and tool calls as
//! entities polled every frame in [`AgentSystems`]. A reply's tool calls
//! run one at a time, in order; rig-core's turn-failure rule decides when a
//! reply ends the turn instead.

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemId;
use bevy_log::info_span;
use bevy_log::tracing::Instrument;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{AsyncComputeTaskPool, IoTaskPool, Task, TaskPool, block_on};
use crossbeam_channel::{Receiver, Sender};
use futures::StreamExt;
use rig_core::completion::message::turn_failure;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::effect::{EffectId, EffectKind};
use rig_core::error::ErrorReport;
use rig_core::message::{ToolCall, UserContent};
use rig_core::serve::{Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};

use super::agent::{
    Agent, AgentId, AgentStatus, CallOf, Calls, Connection, Conversation, Effort, Interrupt,
    ModelChoice, NeedsCompletion, Notice, Partial, SetEffort, SetModel, Submit, SystemPrompt,
    ToolAccess, ToolCallRun, ToolState, TurnFinished,
};
use super::commands::{CommandArgs, SlashCommand};
use super::effects::Effects;
use super::models;
use super::tools::{ToolDef, ToolHandler, failed, run_tool_call};

/// The turn loop's phases, chained in `Update`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentSystems {
    /// Agents that need a model call send their conversation.
    Start,
    /// Running model and tool calls are polled.
    Poll,
    /// Agents running tools start their next queued call, or get their
    /// results once every call finished.
    Settle,
}

/// A streaming model call of the agent it is a [`CallOf`].
#[derive(Component)]
pub struct ModelCall {
    effect: EffectId,
    task: Task<Result<CompletionResponse, ErrorReport>>,
    feed: Receiver<Delta>,
}

/// A streamed fragment for [`Partial`].
enum Delta {
    Text(String),
    Reasoning(String),
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
pub fn on_submit(
    submit: On<Submit>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus), With<Agent>>,
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
    let Ok((mut conversation, mut status)) = agents.get_mut(agent) else {
        return;
    };
    if *status != AgentStatus::Idle {
        notices.write(Notice::info(
            agent,
            "The agent is busy. Press Esc to stop the turn.".to_owned(),
        ));
        return;
    }
    conversation.0.push(Message::user(text));
    *status = AgentStatus::Thinking;
    commands.entity(agent).insert(NeedsCompletion);
}

/// Stops a running turn. Every tool call of the last reply gets a result,
/// real or "interrupted", and dropping the call entities cancels their
/// tasks.
pub fn on_interrupt(
    interrupt: On<Interrupt>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus, Option<&Calls>), With<Agent>>,
    runs: Query<&ToolCallRun>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut finished: MessageWriter<TurnFinished>,
) {
    let agent = interrupt.entity;
    let Ok((mut conversation, mut status, calls)) = agents.get_mut(agent) else {
        return;
    };
    if *status == AgentStatus::Idle {
        return;
    }
    let runs = calls
        .into_iter()
        .flat_map(|calls| runs.iter_many(calls.iter()).flatten());
    if let Some(results) = stopped_results(runs, "interrupted by the user") {
        conversation.0.push(results);
    }
    commands
        .entity(agent)
        .remove::<NeedsCompletion>()
        .despawn_related::<Calls>();
    *status = AgentStatus::Idle;
    notices.write(Notice::info(agent, "Interrupted.".to_owned()));
    finished.write(TurnFinished { agent });
}

/// The results of a stopped reply's tool calls, in call order: each
/// finished call's result, and an error saying `why` for the others. `None`
/// when the reply made no calls.
pub(crate) fn stopped_results<'a>(
    runs: impl Iterator<Item = &'a ToolCallRun>,
    why: &str,
) -> Option<Message> {
    let mut runs: Vec<&ToolCallRun> = runs.collect();
    runs.sort_by_key(|run| run.index);
    (!runs.is_empty()).then(|| {
        Message::tool_results(
            runs.into_iter()
                .map(|run| {
                    run.result()
                        .cloned()
                        .unwrap_or_else(|| failed(&run.call, why.to_owned()))
                })
                .collect(),
        )
    })
}

/// On exit, stops every running turn before the session is saved, so the
/// saved conversation never ends in unanswered tool calls. Each running
/// call is cancelled and waited for first: dropping a task only schedules
/// its cancellation on a pool thread, which could record the effect as
/// cancelled after the last flush.
pub fn stop_turns_on_exit(world: &mut World) {
    let calls: Vec<Entity> = world
        .query_filtered::<Entity, With<ModelCall>>()
        .iter(world)
        .collect();
    for call in calls {
        if let Ok(mut call) = world.get_entity_mut(call)
            && let Some(call) = call.take::<ModelCall>()
        {
            block_on(call.task.cancel());
        }
    }
    let mut runs = world.query::<&mut ToolCallRun>();
    for mut run in runs.iter_mut(world) {
        if let ToolState::Running(task) = std::mem::replace(&mut run.state, ToolState::Queued) {
            block_on(task.cancel());
        }
    }
    let busy: Vec<Entity> = world
        .query_filtered::<(Entity, &AgentStatus), With<Agent>>()
        .iter(world)
        .filter(|(_, status)| **status != AgentStatus::Idle)
        .map(|(entity, _)| entity)
        .collect();
    for entity in busy {
        world.trigger(Interrupt { entity });
    }
}

/// Chooses the agent's model: a known catalog model replaces the agent's
/// [`ModelChoice`], and [`on_model_chosen`] connects it.
pub fn on_set_model(
    set: On<SetModel>,
    agents: Query<&AgentStatus, With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(status) = agents.get(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, *status, "model", &mut notices) {
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
pub fn on_model_chosen(
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
pub fn on_set_effort(
    set: On<SetEffort>,
    mut agents: Query<(Option<&Connection>, &mut Effort, &AgentStatus), With<Agent>>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((connection, mut effort, status)) = agents.get_mut(set.entity) else {
        return;
    };
    if refused_mid_turn(set.entity, *status, "effort", &mut notices) {
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::info(
            set.entity,
            "Pick a model with /model first.".to_owned(),
        ));
        return;
    };
    match models::check_effort(connection.spec, set.effort) {
        Ok(()) => {
            effort.0 = set.effort;
            notices.write(Notice::info(
                set.entity,
                format!("Reasoning: {}.", models::effort_label(set.effort)),
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
    status: AgentStatus,
    command: &str,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    let running = status != AgentStatus::Idle;
    if running {
        notices.write(Notice::info(
            agent,
            format!("A turn is running. Press Esc to stop it, then /{command}."),
        ));
    }
    running
}

/// Sends the conversation of every agent that needs a model call.
pub fn start_completions(
    mut agents: Query<
        (
            Entity,
            &AgentId,
            &mut Conversation,
            Option<&Connection>,
            &Effort,
            &SystemPrompt,
            &ToolAccess,
            &mut AgentStatus,
        ),
        (With<Agent>, With<NeedsCompletion>),
    >,
    tools: Query<&ToolDef>,
    effects: Res<Effects>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut finished: MessageWriter<TurnFinished>,
) {
    for (agent, id, mut conversation, connection, effort, prompt, access, mut status) in &mut agents
    {
        commands.entity(agent).remove::<NeedsCompletion>();
        let request = connection
            .ok_or_else(|| "No model is connected. Pick one with /model.".to_owned())
            .and_then(|connection| {
                let definitions = tools
                    .iter()
                    .filter(|tool| connection.spec.tools && access.allows(tool.0.name.as_str()))
                    .map(|tool| tool.0.clone())
                    .collect();
                prepare(&conversation, connection, effort, prompt, definitions)
                    .map(|request| (connection.handler.clone(), connection.spec, request))
            });
        match request {
            Ok((handler, spec, request)) => {
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
                let task = model_pool().spawn(
                    effects
                        .caught(effect, stream_reply(reply, sender))
                        .instrument(span),
                );
                commands.spawn((
                    Name::new("model call"),
                    ModelCall { effect, task, feed },
                    Partial::default(),
                    CallOf(agent),
                ));
            }
            Err(why) => {
                notices.write(Notice::error(agent, why));
                drop_unanswered(agent, &mut conversation, &mut notices);
                *status = AgentStatus::Idle;
                finished.write(TurnFinished { agent });
            }
        }
    }
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
/// model's spec, or what the user must fix first.
fn prepare(
    conversation: &Conversation,
    connection: &Connection,
    effort: &Effort,
    prompt: &SystemPrompt,
    tools: Vec<rig_core::completion::ToolDefinition>,
) -> Result<CompletionRequest, String> {
    let options = models::generation_options(effort.0);
    connection
        .spec
        .validate(&options)
        .map_err(|refusal| refusal.to_string())?;
    let (prompt_message, earlier) = conversation
        .0
        .split_last()
        .ok_or("The conversation is empty.")?;
    Ok(CompletionRequest::new(prompt_message.clone())
        .messages(earlier.to_vec())
        .preamble(prompt.0.clone())
        .tools(tools)
        .options(options))
}

/// Streams the reply, feeding text and reasoning to the view, and returns
/// the response the provider ended it with.
async fn stream_reply(
    reply: impl Future<Output = Reply>,
    feed: Sender<Delta>,
) -> Result<CompletionResponse, ErrorReport> {
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
    }
    Err(stream_truncated())
}

/// Polls model calls: streams deltas into [`Partial`], and on a finished
/// reply appends it. rig-core's turn-failure rule then decides: a failed
/// turn runs none of its tool calls and ends; otherwise its tool calls are
/// queued in order, or the turn ends when it made none.
pub fn poll_model_calls(
    mut calls: Query<(Entity, &CallOf, &mut ModelCall, &mut Partial)>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus), With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut finished: MessageWriter<TurnFinished>,
) {
    for (entity, call_of, mut call, mut partial) in &mut calls {
        for delta in call.feed.try_iter() {
            match delta {
                Delta::Text(text) => partial.text.push_str(&text),
                Delta::Reasoning(text) => partial.reasoning.push_str(&text),
            }
        }
        let Some(result) = check_ready(&mut call.task) else {
            continue;
        };
        commands.entity(entity).despawn();
        let agent = call_of.0;
        let Ok((mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        let response = match result {
            Ok(response) => response,
            Err(report) => {
                notices.write(Notice::error(
                    agent,
                    format!("The model call failed: {report}"),
                ));
                drop_unanswered(agent, &mut conversation, &mut notices);
                *status = AgentStatus::Idle;
                finished.write(TurnFinished { agent });
                continue;
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
            *status = AgentStatus::Idle;
            finished.write(TurnFinished { agent });
            continue;
        }
        if tool_calls.is_empty() {
            *status = AgentStatus::Idle;
            finished.write(TurnFinished { agent });
            continue;
        }
        *status = AgentStatus::RunningTools;
        for (index, tool_call) in tool_calls.into_iter().enumerate() {
            commands.spawn((
                Name::new(format!("tool call {}", tool_call.function.name.as_str())),
                ToolCallRun {
                    index,
                    call: tool_call,
                    parent: call.effect,
                    state: ToolState::Queued,
                },
                CallOf(agent),
            ));
        }
    }
}

/// Polls running tool calls and keeps each result on its call entity.
pub fn poll_tool_calls(mut runs: Query<&mut ToolCallRun>) {
    for mut run in &mut runs {
        let ready = match &mut run.bypass_change_detection().state {
            ToolState::Running(task) => check_ready(task),
            ToolState::Queued | ToolState::Done(_) => None,
        };
        if let Some(result) = ready {
            run.state = ToolState::Done(result);
        }
    }
}

/// For each agent running tools: once its earlier calls finished, starts
/// the next queued call; once every call finished, appends their results
/// in call order and asks for the next model call. A call to a tool that
/// is not registered, or that the agent may not use, is dispatched and
/// recorded like any other and answered with an error.
pub fn settle_tools(
    mut agents: Query<
        (
            Entity,
            &AgentId,
            &ToolAccess,
            &Calls,
            &mut Conversation,
            &mut AgentStatus,
        ),
        With<Agent>,
    >,
    mut runs: Query<&mut ToolCallRun>,
    tools: Query<(&ToolDef, &ToolHandler)>,
    effects: Res<Effects>,
    mut commands: Commands,
) {
    for (agent, id, access, calls, mut conversation, mut status) in &mut agents {
        if *status != AgentStatus::RunningTools {
            continue;
        }
        let mut order: Vec<(usize, Entity)> = calls
            .iter()
            .filter_map(|entity| runs.get(entity).ok().map(|run| (run.index, entity)))
            .collect();
        order.sort_unstable();
        let mut results = Vec::with_capacity(order.len());
        let mut waiting = false;
        for (_, entity) in &order {
            let Ok(mut run) = runs.get_mut(*entity) else {
                continue;
            };
            match &run.state {
                ToolState::Done(result) => results.push(result.clone()),
                ToolState::Running(_) => waiting = true,
                ToolState::Queued => {
                    let name = run.call.function.name.as_str();
                    let handler = tools
                        .iter()
                        .find(|(def, _)| def.0.name.as_str() == name && access.allows(name))
                        .map(|(_, handler)| handler.0.clone());
                    let work =
                        run_tool_call(&effects, &id.0, run.parent, handler, run.call.clone());
                    let span =
                        info_span!("tool_call", agent = %id.0, tool = name, parent = %run.parent);
                    run.state = ToolState::Running(tool_pool().spawn(work.instrument(span)));
                    waiting = true;
                }
            }
            if waiting {
                break;
            }
        }
        if waiting || results.is_empty() {
            continue;
        }
        conversation.0.push(Message::tool_results(results));
        commands
            .entity(agent)
            .despawn_related::<Calls>()
            .insert(NeedsCompletion);
        *status = AgentStatus::Thinking;
    }
}
