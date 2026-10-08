//! The turn loop. A user message spawns a turn entity, [`TurnOf`] its
//! agent; the turn's model call and tool calls are entities [`CallOf`] the
//! turn, and observers of their [`Done`] outputs carry the turn on. A
//! reply's tool calls run one at a time, in order; rig-core's turn-failure
//! rule decides when a reply ends the turn instead. Despawning the turn
//! ends it, and the agent's [`ActiveTurn`] going away reports it finished.

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
use futures::StreamExt;
use rig_core::completion::message::turn_failure;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::effect::{EffectId, EffectKind};
use rig_core::error::ErrorReport;
use rig_core::message::{ToolCall, ToolResult, UserContent};
use rig_core::serve::{Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};

use super::agent::{
    ActiveTurn, Agent, AgentId, CallOf, Calls, Connection, Conversation, Effort, Interrupt,
    ModelChoice, Notice, Partial, Queued, SetEffort, SetModel, Submit, SystemPrompt, ToolAccess,
    ToolCallRun, TurnFinished, TurnOf,
};
use super::calls::{Done, Running, Wake};
use super::commands::{CommandArgs, SlashCommand};
use super::effects::Effects;
use super::models;
use super::prompt::{PromptSection, ToolRules, system_prompt};
use super::tools::{ToolDef, ToolHandler, failed, run_tool_call};
use super::usage::{Spending, TurnSpending};

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
    turns: Query<&TurnOf>,
    mut agents: Query<(
        &AgentId,
        &mut Conversation,
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
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Ok((id, mut conversation, connection, effort, prompt, access)) = agents.get_mut(agent)
    else {
        return;
    };
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
            prepare(&conversation, connection, effort, preamble, definitions)
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
    preamble: String,
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
    mut turns: Query<(&TurnOf, &mut TurnSpending)>,
    mut agents: Query<(&AgentId, &ToolAccess, &mut Conversation, &mut Spending)>,
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
    let Ok((&TurnOf(agent), mut turn_spent)) = turns.get_mut(turn) else {
        return;
    };
    let Ok((id, access, mut conversation, mut spent)) = agents.get_mut(agent) else {
        return;
    };
    let response = match reply {
        Ok(response) => {
            // A reply the turn-failure rule rejects was still billed.
            spent.record(&response.usage);
            turn_spent.0.record(&response.usage);
            response
        }
        Err(report) => {
            notices.write(Notice::error(
                agent,
                format!("The model call failed: {report}"),
            ));
            drop_unanswered(agent, &mut conversation, &mut notices);
            commands.entity(turn).despawn();
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
