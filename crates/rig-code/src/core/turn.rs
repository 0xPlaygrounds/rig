//! The turn loop: requests from views, then model calls and tool calls as
//! entities polled every frame in [`AgentSystems`].

use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{AsyncComputeTaskPool, Task, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use futures::StreamExt;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::effect::{EffectId, EffectKind};
use rig_core::error::ErrorReport;
use rig_core::message::{ToolCall, ToolResult};
use rig_core::operation::Completion;
use rig_core::providers::registry::ModelSelector;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::serve::{ErasedHandler, Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};

use super::agent::{
    Agent, AgentId, AgentStatus, CallOf, Calls, Conversation, Effort, Interrupt, ModelChoice,
    NeedsCompletion, Notice, Partial, SetEffort, SetModel, Submit, SystemPrompt, ToolAccess,
    ToolCallRun, TurnFinished,
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
    /// Agents whose tool calls all finished get their results.
    Settle,
}

/// A streaming model call of the agent it is a [`CallOf`].
#[derive(Component)]
pub struct ModelCall {
    effect: EffectId,
    task: Task<Result<CompletionResponse, ErrorReport>>,
    feed: Receiver<Delta>,
}

/// The task running one tool call.
#[derive(Component)]
pub struct ToolTask(Task<ToolResult>);

/// A streamed fragment for [`Partial`].
enum Delta {
    Text(String),
    Reasoning(String),
}

fn pool() -> &'static AsyncComputeTaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// Runs a slash command or starts a turn.
pub fn on_submit(
    submit: On<Submit>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus), With<Agent>>,
    slash: Query<&SlashCommand>,
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
        match slash.iter().find(|command| command.name == name) {
            Some(command) => commands.run_system_with(
                command.system,
                CommandArgs {
                    agent,
                    args: args.trim().to_owned(),
                },
            ),
            None => {
                notices.write(Notice(format!(
                    "Unknown command /{name}. /help lists the commands."
                )));
            }
        }
        return;
    }
    let Ok((mut conversation, mut status)) = agents.get_mut(agent) else {
        return;
    };
    if *status != AgentStatus::Idle {
        notices.write(Notice(
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
    let mut stopped: Vec<&ToolCallRun> = calls
        .into_iter()
        .flat_map(|calls| runs.iter_many(calls.iter()).flatten())
        .collect();
    stopped.sort_by_key(|run| run.index);
    if !stopped.is_empty() {
        conversation.0.push(Message::tool_results(
            stopped
                .into_iter()
                .map(|run| {
                    run.result
                        .clone()
                        .unwrap_or_else(|| failed(&run.call, "interrupted by the user".to_owned()))
                })
                .collect(),
        ));
    }
    commands
        .entity(agent)
        .remove::<NeedsCompletion>()
        .despawn_related::<Calls>();
    *status = AgentStatus::Idle;
    notices.write(Notice("Interrupted.".to_owned()));
    finished.write(TurnFinished { agent });
}

/// Sets the agent's model, resetting a reasoning setting the new model does
/// not take.
pub fn on_set_model(
    set: On<SetModel>,
    mut agents: Query<(&mut ModelChoice, &mut Effort), With<Agent>>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((mut choice, mut effort)) = agents.get_mut(set.entity) else {
        return;
    };
    let Some(spec) = models::resolve(&set.model) else {
        notices.write(Notice(format!(
            "No catalog model `{}`. Use vendor/model.",
            set.model
        )));
        return;
    };
    choice.0 = Some(models::reference(spec));
    notices.write(Notice(format!(
        "Model: {} ({}).",
        spec.display_name,
        models::reference(spec)
    )));
    if let Err(refusal) = models::check_effort(spec, effort.0) {
        effort.0 = None;
        notices.write(Notice(format!("Reasoning reset to default: {refusal}.")));
    }
}

/// Sets the agent's reasoning setting after checking it against the model.
pub fn on_set_effort(
    set: On<SetEffort>,
    mut agents: Query<(&ModelChoice, &mut Effort), With<Agent>>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((choice, mut effort)) = agents.get_mut(set.entity) else {
        return;
    };
    let Some(spec) = choice.0.as_deref().and_then(models::resolve) else {
        notices.write(Notice("Pick a model with /model first.".to_owned()));
        return;
    };
    match models::check_effort(spec, set.effort) {
        Ok(()) => {
            effort.0 = set.effort;
            notices.write(Notice(format!(
                "Reasoning: {}.",
                models::effort_label(set.effort)
            )));
        }
        Err(refusal) => {
            notices.write(Notice(format!("{refusal}.")));
        }
    }
}

/// Sends the conversation of every agent that needs a model call.
pub fn start_completions(
    mut agents: Query<
        (
            Entity,
            &AgentId,
            &Conversation,
            &ModelChoice,
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
    for (agent, id, conversation, choice, effort, prompt, access, mut status) in &mut agents {
        commands.entity(agent).remove::<NeedsCompletion>();
        let definitions = tools
            .iter()
            .filter(|tool| access.allows(tool.0.name.as_str()))
            .map(|tool| tool.0.clone())
            .collect();
        match prepare(conversation, choice, effort, prompt, definitions) {
            Ok((handler, request)) => {
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
                let task = pool().spawn(stream_reply(reply, sender));
                commands.spawn((
                    Name::new("model call"),
                    ModelCall { effect, task, feed },
                    Partial::default(),
                    CallOf(agent),
                ));
            }
            Err(why) => {
                notices.write(Notice(why));
                *status = AgentStatus::Idle;
                finished.write(TurnFinished { agent });
            }
        }
    }
}

/// The handler and request for the agent's next model call, or what the
/// user must fix first.
fn prepare(
    conversation: &Conversation,
    choice: &ModelChoice,
    effort: &Effort,
    prompt: &SystemPrompt,
    tools: Vec<rig_core::completion::ToolDefinition>,
) -> Result<(ErasedHandler, CompletionRequest), String> {
    let reference = choice
        .0
        .as_deref()
        .ok_or("No model is chosen. Pick one with /model.")?;
    let spec = models::resolve(reference)
        .ok_or_else(|| format!("The catalog has no model `{reference}`."))?;
    let options = models::generation_options(effort.0);
    spec.validate(&options)
        .map_err(|refusal| refusal.to_string())?;
    let (prompt_message, earlier) = conversation
        .0
        .split_last()
        .ok_or("The conversation is empty.")?;
    let request = CompletionRequest::new(prompt_message.clone())
        .messages(earlier.to_vec())
        .preamble(prompt.0.clone())
        .tools(tools)
        .options(options);
    let model = ModelSelector::Spec(spec)
        .provider_ref()
        .map_err(|error| error.to_string())?
        .completion_model()
        .map_err(|error| error.to_string())?;
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(reference.to_owned(), model));
    Ok((handler, request))
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
/// reply appends it and starts its tool calls, or ends the turn.
pub fn poll_model_calls(
    mut calls: Query<(Entity, &CallOf, &mut ModelCall, &mut Partial)>,
    mut agents: Query<(&AgentId, &ToolAccess, &mut Conversation, &mut AgentStatus), With<Agent>>,
    tools: Query<(&ToolDef, &ToolHandler)>,
    effects: Res<Effects>,
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
        let Ok((id, access, mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        let response = match result {
            Ok(response) => response,
            Err(report) => {
                notices.write(Notice(format!("The model call failed: {report}")));
                *status = AgentStatus::Idle;
                finished.write(TurnFinished { agent });
                continue;
            }
        };
        conversation.0.extend(response.message());
        let tool_calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
        if tool_calls.is_empty() {
            *status = AgentStatus::Idle;
            finished.write(TurnFinished { agent });
            continue;
        }
        *status = AgentStatus::RunningTools;
        for (index, tool_call) in tool_calls.into_iter().enumerate() {
            let name = tool_call.function.name.as_str();
            let handler = tools
                .iter()
                .find(|(def, _)| def.0.name.as_str() == name && access.allows(name))
                .map(|(_, handler)| handler.0.clone());
            let run = run_tool_call(&effects, &id.0, call.effect, handler, tool_call.clone());
            commands.spawn((
                Name::new(format!("tool call {name}")),
                ToolCallRun {
                    index,
                    call: tool_call,
                    result: None,
                },
                ToolTask(pool().spawn(run)),
                CallOf(agent),
            ));
        }
    }
}

/// Polls tool calls and keeps each result on its call entity.
pub fn poll_tool_calls(
    mut runs: Query<(Entity, &mut ToolCallRun, &mut ToolTask)>,
    mut commands: Commands,
) {
    for (entity, mut run, mut task) in &mut runs {
        if let Some(result) = check_ready(&mut task.0) {
            run.result = Some(result);
            commands.entity(entity).remove::<ToolTask>();
        }
    }
}

/// Appends the results of an agent whose tool calls all finished, in call
/// order, and asks for the next model call.
pub fn settle_tools(
    mut agents: Query<(Entity, &Calls, &mut Conversation, &mut AgentStatus), With<Agent>>,
    runs: Query<&ToolCallRun>,
    mut commands: Commands,
) {
    for (agent, calls, mut conversation, mut status) in &mut agents {
        if *status != AgentStatus::RunningTools {
            continue;
        }
        let mut done: Vec<&ToolCallRun> = runs.iter_many(calls.iter()).flatten().collect();
        if done.is_empty() || done.iter().any(|run| run.result.is_none()) {
            continue;
        }
        done.sort_by_key(|run| run.index);
        conversation.0.push(Message::tool_results(
            done.into_iter()
                .filter_map(|run| run.result.clone())
                .collect(),
        ));
        commands
            .entity(agent)
            .despawn_related::<Calls>()
            .insert(NeedsCompletion);
        *status = AgentStatus::Thinking;
    }
}
