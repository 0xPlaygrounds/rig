//! The agent loop as systems over every agent: start a model call for each
//! agent that needs one, poll the calls, run the tool calls a reply asks
//! for, and feed their results back until a reply calls no tool.

use std::panic::AssertUnwindSafe;

use bevy::prelude::*;
use bevy::tasks::futures::check_ready;
use bevy::tasks::futures_lite::{FutureExt, StreamExt};
use bevy::tasks::{AsyncComputeTaskPool, IoTaskPool, Task, block_on};
use rig_core::ErrorReport;
use rig_core::completion::{
    CompletionRequest, CompletionResponse, GenerationOptions, Message as ChatMessage,
    ToolDefinition,
};
use rig_core::effect::{
    EffectId, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome, family,
};
use rig_core::message::{StopReason, ToolCall, ToolResult, ToolResultContent};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};
use rig_core::tool::ToolExecutionError;

use super::agent::{
    AgentId, AgentStatus, Conversation, EffortChoice, SystemPrompt, ToolAccess, Work, WorkOf,
};
use super::dispatch::Effects;
use super::models::ModelEndpoint;
use super::registry::{NeedsModelCall, Notice, ToolEntry, TurnFinished};

/// A model call in flight, a work entity of its agent.
#[derive(Component)]
pub struct ModelCall {
    /// The call's effect, the parent of the tool calls it asks for.
    pub effect: EffectId,
}

/// The task running a call. Dropping it cancels the call.
#[derive(Component)]
pub struct CallTask<T: Send + 'static>(pub Task<T>);

/// Fragments of a streaming reply on their way from the task.
#[derive(Component)]
pub struct StreamRx(async_channel::Receiver<Delta>);

/// One fragment of a streaming reply.
pub enum Delta {
    /// Answer text.
    Text(String),
    /// Reasoning text.
    Reasoning(String),
}

/// What a model call has streamed so far.
#[derive(Component, Default)]
pub struct StreamingText {
    /// Answer text.
    pub text: String,
    /// Reasoning text.
    pub reasoning: String,
}

/// A tool call in flight: its position in the reply and the call.
#[derive(Component)]
pub struct ToolCallRun {
    /// Position of the call in the reply that asked for it.
    pub index: usize,
    /// The call.
    pub call: ToolCall,
}

/// The result of a finished tool call.
#[derive(Component)]
pub struct ToolCallDone(pub ToolResult);

type ModelTask = CallTask<Result<CompletionResponse, ErrorReport>>;

/// Sends the conversation of every agent that needs a reply.
pub(crate) fn start_model_calls(
    mut commands: Commands,
    effects: Res<Effects>,
    mut agents: Query<
        (
            Entity,
            &AgentId,
            &Conversation,
            &SystemPrompt,
            &ToolAccess,
            &EffortChoice,
            Option<&ModelEndpoint>,
            &mut AgentStatus,
        ),
        With<NeedsModelCall>,
    >,
    tools: Query<&ToolEntry>,
    mut finished: MessageWriter<TurnFinished>,
) {
    for (agent, id, conversation, prompt, access, effort, endpoint, mut status) in &mut agents {
        commands.entity(agent).remove::<NeedsModelCall>();
        let definitions = tools
            .iter()
            .filter(|tool| access.allows(tool.definition.name.as_str()))
            .map(|tool| tool.definition.clone())
            .collect();
        let prepared = endpoint
            .ok_or_else(|| "no model is picked; pick one with /model".to_owned())
            .and_then(|endpoint| {
                build_request(endpoint, conversation, prompt, definitions, effort)
                    .map(|request| (endpoint, request))
            });
        let (endpoint, request) = match prepared {
            Ok(prepared) => prepared,
            Err(reason) => {
                *status = AgentStatus::Failed(reason);
                finished.write(TurnFinished { agent });
                continue;
            }
        };
        let (sender, receiver) = async_channel::unbounded();
        let (effect, reply) = effects.dispatch(
            id,
            None,
            endpoint.handler.clone(),
            EffectKind::Completion {
                request,
                stream: true,
            },
        );
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut stream = reply.await.into_stream();
            while let Some(item) = stream.next().await {
                let delta = match item {
                    Ok(Relayed::Done(response)) => return Ok(*response),
                    Err(report) => return Err(report),
                    Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => {
                        Delta::Text(text)
                    }
                    Ok(Relayed::Item(Item::Event(StreamEvent::Reasoning { text, .. }))) => {
                        Delta::Reasoning(text)
                    }
                    Ok(_) => continue,
                };
                // The receiver is gone only when the call was cancelled.
                let _ = sender.try_send(delta);
            }
            Err(stream_truncated())
        });
        commands.spawn((
            ModelCall { effect },
            WorkOf(agent),
            CallTask(task),
            StreamRx(receiver),
            StreamingText::default(),
        ));
        *status = AgentStatus::Streaming;
    }
}

/// The request for an agent's next reply, checked against the model's
/// catalog entry, or why it cannot be sent.
fn build_request(
    endpoint: &ModelEndpoint,
    conversation: &Conversation,
    prompt: &SystemPrompt,
    tools: Vec<ToolDefinition>,
    effort: &EffortChoice,
) -> Result<CompletionRequest, String> {
    let spec = endpoint.spec;
    if !tools.is_empty() && !spec.tools {
        return Err(format!(
            "{} does not call tools; pick another model",
            spec.id
        ));
    }
    let mut history = conversation.0.clone();
    let last = history.pop().ok_or("the conversation is empty")?;
    let mut options = GenerationOptions::default();
    if let Some(reasoning) = effort.0 {
        options = options.reasoning(reasoning);
    }
    spec.validate(&options).map_err(|error| error.to_string())?;
    let mut request = CompletionRequest::new(last)
        .messages(history)
        .preamble(prompt.0.clone())
        .tools(tools)
        .options(options);
    if let Some(rig_core::completion::Reasoning::Budget { tokens }) = effort.0 {
        // A budget must stay below the reply's token limit.
        let limit = spec
            .max_output_tokens
            .map_or(u64::from(tokens) + 8192, u64::from);
        request = request.max_tokens(limit);
    }
    Ok(request)
}

/// Drains streamed text and finishes model calls: appends the reply and
/// starts the tool calls it asks for.
pub(crate) fn poll_model_calls(
    mut commands: Commands,
    effects: Res<Effects>,
    mut calls: Query<(
        Entity,
        &ModelCall,
        &WorkOf,
        &mut ModelTask,
        &StreamRx,
        &mut StreamingText,
    )>,
    mut agents: Query<(&AgentId, &ToolAccess, &mut Conversation, &mut AgentStatus)>,
    tools: Query<&ToolEntry>,
    mut finished: MessageWriter<TurnFinished>,
) {
    for (call, model_call, work_of, mut task, deltas, mut streamed) in &mut calls {
        while let Ok(delta) = deltas.0.try_recv() {
            match delta {
                Delta::Text(text) => streamed.text.push_str(&text),
                Delta::Reasoning(text) => streamed.reasoning.push_str(&text),
            }
        }
        let Some(result) = check_ready(&mut task.0) else {
            continue;
        };
        commands.entity(call).despawn();
        let agent = work_of.0;
        let Ok((id, access, mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        let response = match result {
            Ok(response) => response,
            Err(report) => {
                *status = AgentStatus::Failed(report.to_string());
                finished.write(TurnFinished { agent });
                continue;
            }
        };
        conversation.0.extend(response.message());
        let stop = response.stop();
        let tool_calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
        if tool_calls.is_empty() {
            *status = match stop {
                StopReason::Error(reason) => AgentStatus::Failed(reason),
                _ => AgentStatus::Idle,
            };
            finished.write(TurnFinished { agent });
            continue;
        }
        *status = AgentStatus::RunningTools;
        let cut_off = matches!(stop, StopReason::Length);
        for (index, call) in tool_calls.into_iter().enumerate() {
            let name = call.function.name.as_str().to_owned();
            let refusal = if cut_off {
                Some("the reply was cut off at the token limit, so this call was not run".into())
            } else if let Some(raw) = &call.function.invalid_arguments {
                Some(rig_core::transcript::invalid_arguments_feedback(&name, raw))
            } else if !access.allows(&name) {
                Some(format!("the tool `{name}` is not allowed for this agent"))
            } else {
                None
            };
            let entry = tools
                .iter()
                .find(|tool| tool.definition.name.as_str() == name);
            let handler = match (refusal, entry) {
                (None, Some(entry)) => entry.handler.clone(),
                (Some(reason), _) => ErasedHandler::new(Refusal {
                    name: name.clone(),
                    reason,
                }),
                (None, None) => ErasedHandler::new(Refusal {
                    reason: format!("there is no tool named `{name}`"),
                    name: name.clone(),
                }),
            };
            let args = call.function.arguments_value().to_string();
            let (_, reply) = effects.dispatch(
                id,
                Some(model_call.effect),
                handler,
                EffectKind::ToolCall { name, args },
            );
            let answered = call.clone();
            let task = IoTaskPool::get().spawn(async move {
                // A panicking plugin tool becomes an error result.
                let outcome = AssertUnwindSafe(async move { reply.await.into_outcome().await })
                    .catch_unwind()
                    .await;
                tool_result(&answered, outcome)
            });
            commands.spawn((ToolCallRun { index, call }, WorkOf(agent), CallTask(task)));
        }
    }
}

/// The history entry answering `call`.
fn tool_result(
    call: &ToolCall,
    outcome: std::thread::Result<Result<Outcome, ErrorReport>>,
) -> ToolResult {
    let text = |text: String| vec![ToolResultContent::text(text)];
    match outcome {
        Ok(Ok(Outcome::ToolResult { result })) if result.is_success() => {
            call.result(result.output().clone().into_content())
        }
        Ok(Ok(Outcome::ToolResult { result })) => {
            call.error_result(result.output().clone().into_content())
        }
        Ok(Ok(_)) => call.error_result(text("the tool answered with something else".into())),
        Ok(Err(report)) => call.error_result(text(report.to_string())),
        Err(_) => call.error_result(text("the tool panicked".into())),
    }
}

/// Stores the result of each finished tool call.
pub(crate) fn poll_tool_calls(
    mut commands: Commands,
    mut calls: Query<(Entity, &mut CallTask<ToolResult>), With<ToolCallRun>>,
) {
    for (call, mut task) in &mut calls {
        if let Some(result) = check_ready(&mut task.0) {
            commands
                .entity(call)
                .try_remove::<CallTask<ToolResult>>()
                .try_insert(ToolCallDone(result));
        }
    }
}

/// Appends the tool results of every agent whose tool calls all finished,
/// and asks for the model's next reply.
pub(crate) fn collect_tool_results(
    mut commands: Commands,
    mut agents: Query<(Entity, &Work, &mut Conversation, &mut AgentStatus)>,
    calls: Query<(&ToolCallRun, Option<&ToolCallDone>)>,
) {
    for (agent, work, mut conversation, mut status) in &mut agents {
        if *status != AgentStatus::RunningTools {
            continue;
        }
        let mut results = Vec::new();
        for entity in work.iter() {
            match calls.get(entity) {
                Ok((run, Some(done))) => results.push((run.index, done.0.clone())),
                Ok((_, None)) => {
                    results.clear();
                    break;
                }
                Err(_) => {}
            }
        }
        if results.is_empty() {
            continue;
        }
        results.sort_by_key(|(index, _)| *index);
        conversation.0.push(ChatMessage::tool_results(
            results.into_iter().map(|(_, result)| result).collect(),
        ));
        commands
            .entity(agent)
            .despawn_related::<Work>()
            .insert(NeedsModelCall);
        *status = AgentStatus::Streaming;
    }
}

/// On exit, cancels every call in flight and waits until each has dropped
/// its future, so the cancellations are recorded before the last effect
/// flush. Dropping a task alone would leave that to a pool thread later.
pub(crate) fn cancel_work_on_exit(world: &mut World) {
    let work: Vec<Entity> = world
        .query_filtered::<Entity, With<WorkOf>>()
        .iter(world)
        .collect();
    for entity in work {
        let Ok(mut entity) = world.get_entity_mut(entity) else {
            continue;
        };
        if let Some(task) = entity.take::<ModelTask>() {
            block_on(task.0.cancel());
        }
        if let Some(task) = entity.take::<CallTask<ToolResult>>() {
            block_on(task.0.cancel());
        }
        entity.despawn();
    }
}

/// Logs each finished turn's outcome.
pub(crate) fn report_turns(
    mut finished: MessageReader<TurnFinished>,
    agents: Query<(&AgentId, &AgentStatus)>,
    mut notices: MessageWriter<Notice>,
) {
    for turn in finished.read() {
        let Ok((id, status)) = agents.get(turn.agent) else {
            continue;
        };
        match status {
            AgentStatus::Failed(reason) => {
                warn!(agent = id.0, "turn failed: {reason}");
                notices.write(Notice::error(turn.agent, reason.clone()));
            }
            _ => info!(agent = id.0, "turn finished"),
        }
    }
}

/// Answers a tool call that must not run with a refusal the model reads.
/// It goes through the same dispatch as a real tool, so the effect log
/// shows it.
struct Refusal {
    name: String,
    reason: String,
}

impl Serve for Refusal {
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            // Its own key, so the effect-log header keeps the real tool's
            // descriptor under `tool:<name>`.
            key: HandlerKey::from(format!("refused:{}", self.name)),
            family: FamilyDescriptor::Tool {
                name: self.name.clone(),
                description: "a call rig-code refused to run".to_owned(),
                parameters: serde_json::json!({ "type": "object" }),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Ok(Outcome::ToolResult {
            result: rig_core::tool::ToolResult::failed(ToolExecutionError::refused(
                self.reason.clone(),
            )),
        }))
    }
}
