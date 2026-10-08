//! The agent loop. A submitted prompt queues a model call; the reply's tool
//! calls run one at a time; their results queue the next model call, until
//! the model stops calling tools. Each call in flight is an entity related
//! to its agent by [`CallOf`], so despawning it cancels the work.

use async_channel::{Receiver, Sender};
use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{AsyncComputeTaskPool, IoTaskPool, Task, TaskPool};
use futures::StreamExt;
use rig_core::ErrorReport;
use rig_core::completion::options::Reasoning;
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::effect::{
    EffectKind, FamilyDescriptor, HandlerDescriptor, Outcome, family, tool_key,
};
use rig_core::message::{
    self, AssistantContent, AssistantMessage, Message, StopReason, ToolResultContent,
};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent, delivered};
use rig_core::tool::{ToolExecutionError, ToolResult};

use super::agent::{
    AgentId, CallOf, Calls, Connection, Conversation, Effort, Status, SystemPrompt, ToolAccess,
};
use super::dispatch::{Effects, caught, dispatch};
use super::registry::{Notice, ToolSpec};

/// Sends a user message to an agent. An agent that is not idle refuses it
/// with a notice.
#[derive(EntityEvent, Clone, Debug)]
pub struct Submit {
    /// The agent.
    pub entity: Entity,
    /// The message.
    pub text: String,
}

/// Stops the agent's running turn: its calls in flight are cancelled, and
/// the conversation records what arrived and which tool calls never ran.
#[derive(EntityEvent, Clone, Debug)]
pub struct Stop {
    /// The agent.
    pub entity: Entity,
}

/// The agent finished a turn, whether the model answered, failed or was
/// stopped. The agent is idle again.
#[derive(EntityEvent, Clone, Debug)]
pub struct TurnEnded {
    /// The agent.
    pub entity: Entity,
}

/// A model call in flight: the streaming reply and what arrived so far.
#[derive(Component)]
pub struct ModelCall {
    task: Task<Result<CompletionResponse, ErrorReport>>,
    events: Receiver<Item<StreamEvent>>,
    /// The reply's events so far, in order.
    pub items: Vec<Item<StreamEvent>>,
}

impl ModelCall {
    /// The assistant content the events so far add up to.
    pub fn delivered(&self) -> Vec<AssistantContent> {
        delivered(&self.items)
    }
}

/// Where a tool call is.
pub enum ToolState {
    /// Waiting for the agent's earlier tool calls.
    Queued,
    /// Running.
    Running(Task<ToolResult>),
    /// Finished.
    Done(ToolResult),
}

/// One tool call of the last reply.
#[derive(Component)]
pub struct ToolCall {
    /// The call's position in the reply.
    pub index: usize,
    /// The call as the model made it.
    pub call: message::ToolCall,
    /// Where it is.
    pub state: ToolState,
}

/// Appends a submitted message and queues a model call.
pub(crate) fn submit(
    submit: On<Submit>,
    mut agents: Query<(&mut Conversation, &mut Status)>,
    mut commands: Commands,
) -> Result {
    let (mut conversation, mut status) = agents.get_mut(submit.entity)?;
    let text = submit.text.trim();
    if text.is_empty() {
        return Ok(());
    }
    if *status != Status::Idle {
        commands.trigger(Notice::error(
            submit.entity,
            "The agent is busy. Press Esc to stop it.",
        ));
        return Ok(());
    }
    conversation.0.push(Message::user(text));
    *status = Status::Queued;
    Ok(())
}

/// Cancels the agent's calls in flight and makes the conversation whole:
/// the partial reply is kept as an aborted turn, and every tool call that
/// did not finish gets a cancelled result.
pub(crate) fn stop(
    stop: On<Stop>,
    mut agents: Query<(&mut Conversation, &mut Status, Option<&Calls>)>,
    model_calls: Query<&ModelCall>,
    tool_calls: Query<&ToolCall>,
    mut commands: Commands,
) -> Result {
    let agent = stop.entity;
    let (mut conversation, mut status, calls) = agents.get_mut(agent)?;
    if *status == Status::Idle {
        return Ok(());
    }
    let mut results = Vec::new();
    for entity in calls.into_iter().flat_map(|calls| calls.iter()) {
        if let Ok(call) = model_calls.get(entity) {
            let content = call.delivered();
            if !content.is_empty() {
                conversation
                    .0
                    .push(Message::Assistant(AssistantMessage::aborted(
                        None,
                        content,
                        "stopped by the user",
                    )));
            }
        }
        if let Ok(tool) = tool_calls.get(entity) {
            let result = match &tool.state {
                ToolState::Done(result) => tool_result(&tool.call, result),
                ToolState::Queued | ToolState::Running(_) => tool
                    .call
                    .error_result(vec![ToolResultContent::text("Cancelled by the user.")]),
            };
            results.push((tool.index, result));
        }
        commands.entity(entity).despawn();
    }
    if !results.is_empty() {
        results.sort_by_key(|(index, _)| *index);
        conversation.0.push(Message::tool_results(
            results.into_iter().map(|(_, result)| result).collect(),
        ));
    }
    commands.trigger(Notice::info(agent, "Stopped."));
    end_turn(&mut commands, agent, &mut status);
    Ok(())
}

/// Starts a model call for every queued agent.
pub(crate) fn start_model_calls(
    mut agents: Query<(
        Entity,
        &AgentId,
        &Conversation,
        &SystemPrompt,
        &ToolAccess,
        &Effort,
        Option<&Connection>,
        &mut Status,
    )>,
    tools: Query<&ToolSpec>,
    effects: Res<Effects>,
    mut commands: Commands,
) {
    for (agent, id, conversation, prompt, access, effort, connection, mut status) in &mut agents {
        if *status != Status::Queued {
            continue;
        }
        let Some(connection) = connection else {
            commands.trigger(Notice::error(agent, "Pick a model with /model first."));
            end_turn(&mut commands, agent, &mut status);
            continue;
        };
        let request = request(prompt, conversation, access, effort, connection, &tools);
        if let Err(refusal) = connection.spec.validate(&request.options) {
            commands.trigger(Notice::error(agent, refusal.to_string()));
            end_turn(&mut commands, agent, &mut status);
            continue;
        }
        let reply = dispatch(
            &effects,
            id,
            connection.key.clone(),
            connection.handler.clone(),
            EffectKind::Completion {
                request,
                stream: true,
            },
        );
        let (sender, events) = async_channel::unbounded();
        let task = IoTaskPool::get_or_init(TaskPool::new)
            .spawn(async move { caught(read_reply(reply, sender)).await.flatten() });
        commands.spawn((
            CallOf(agent),
            ModelCall {
                task,
                events,
                items: Vec::new(),
            },
        ));
        *status = Status::Streaming;
    }
}

/// The request for the agent's next reply: the system prompt and the
/// conversation, the tools it may call when the model takes tools, and its
/// effort. A token budget lets the reply use the model's whole output limit.
fn request(
    prompt: &SystemPrompt,
    conversation: &Conversation,
    access: &ToolAccess,
    effort: &Effort,
    connection: &Connection,
    tools: &Query<&ToolSpec>,
) -> CompletionRequest {
    let mut history = Vec::with_capacity(conversation.0.len() + 1);
    if !prompt.0.is_empty() {
        history.push(Message::system(prompt.0.clone()));
    }
    history.extend(conversation.0.iter().cloned());
    let mut request = CompletionRequest::from(history);
    if connection.spec.tools {
        request.tools = tools
            .iter()
            .filter(|tool| access.allows(tool.definition.name.as_str()))
            .map(|tool| tool.definition.clone())
            .collect();
    }
    request.options.reasoning = effort.0;
    if let Some(Reasoning::Budget { .. }) = effort.0 {
        request.max_tokens = connection.spec.max_output_tokens.map(u64::from);
    }
    request
}

/// Reads a streamed reply, forwarding its events, until the response.
async fn read_reply(
    reply: impl Future<Output = Result<Reply, ErrorReport>>,
    events: Sender<Item<StreamEvent>>,
) -> Result<CompletionResponse, ErrorReport> {
    let mut stream = reply.await?.into_stream();
    while let Some(item) = stream.next().await {
        match item? {
            Relayed::Origin(_) => {}
            // A closed channel means the call was despawned; the task is
            // dropped with it.
            Relayed::Item(item) => {
                let _ = events.try_send(item);
            }
            Relayed::Done(response) => return Ok(*response),
        }
    }
    Err(stream_truncated())
}

/// Collects streamed events and applies finished replies: the reply joins
/// the conversation, and its tool calls are queued, or the turn ends.
pub(crate) fn poll_model_calls(
    mut calls: Query<(Entity, &CallOf, &mut ModelCall)>,
    mut agents: Query<(&mut Conversation, &mut Status)>,
    mut commands: Commands,
) {
    for (entity, call_of, mut call) in &mut calls {
        if !call.events.is_empty() {
            let call = &mut *call;
            while let Ok(item) = call.events.try_recv() {
                call.items.push(item);
            }
        }
        let Some(result) = check_ready(&mut call.bypass_change_detection().task) else {
            continue;
        };
        commands.entity(entity).despawn();
        let agent = call_of.0;
        let Ok((mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                let content = call.delivered();
                if !content.is_empty() {
                    conversation
                        .0
                        .push(Message::Assistant(AssistantMessage::aborted(
                            None,
                            content,
                            error.message.clone(),
                        )));
                }
                commands.trigger(Notice::error(agent, error.to_string()));
                end_turn(&mut commands, agent, &mut status);
                continue;
            }
        };
        if let Some(message) = response.message() {
            conversation.0.push(message);
        }
        let tool_calls: Vec<message::ToolCall> = response.tool_calls().cloned().collect();
        match response.stop() {
            StopReason::ToolUse if !tool_calls.is_empty() => {
                for (index, call) in tool_calls.into_iter().enumerate() {
                    commands.spawn((
                        CallOf(agent),
                        ToolCall {
                            index,
                            call,
                            state: ToolState::Queued,
                        },
                    ));
                }
                *status = Status::Tools;
            }
            // Arguments cut off by the output limit are not run. The model
            // is told when the user continues; retrying at once could
            // repeat paid calls that hit the limit every time.
            StopReason::Length if !tool_calls.is_empty() => {
                conversation.0.push(Message::tool_results(
                    tool_calls
                        .iter()
                        .map(|call| {
                            call.error_result(vec![ToolResultContent::text(
                                "The reply hit the output token limit, so this call was cut \
                                 off and not run.",
                            )])
                        })
                        .collect(),
                ));
                commands.trigger(Notice::error(
                    agent,
                    "The reply hit the output token limit, so its tool calls were not run.",
                ));
                end_turn(&mut commands, agent, &mut status);
            }
            stop => {
                if let StopReason::Error(reason) = stop {
                    commands.trigger(Notice::error(agent, reason));
                }
                end_turn(&mut commands, agent, &mut status);
            }
        }
    }
}

/// Starts the next queued tool call of every agent running tools, once its
/// earlier calls have finished.
pub(crate) fn start_tool_calls(
    agents: Query<(&AgentId, &ToolAccess, &Status, &Calls)>,
    mut tool_calls: Query<&mut ToolCall>,
    tools: Query<&ToolSpec>,
    effects: Res<Effects>,
) {
    for (id, access, status, calls) in &agents {
        if *status != Status::Tools {
            continue;
        }
        let mut next: Option<(usize, Entity)> = None;
        let mut running = false;
        for entity in calls.iter() {
            let Ok(tool) = tool_calls.get(entity) else {
                continue;
            };
            match tool.state {
                ToolState::Running(_) => running = true,
                ToolState::Queued if next.is_none_or(|(index, _)| tool.index < index) => {
                    next = Some((tool.index, entity));
                }
                ToolState::Queued | ToolState::Done(_) => {}
            }
        }
        if running {
            continue;
        }
        if let Some((_, entity)) = next
            && let Ok(mut tool) = tool_calls.get_mut(entity)
        {
            tool.state = start_tool(&effects, id, access, &tools, &tool.call);
        }
    }
}

/// Dispatches one tool call to its tool. A call to a tool the agent may not
/// call, or that does not exist, is dispatched to [`missing_tool`], so it
/// is recorded like any other. Tools run on the `AsyncComputeTaskPool`, as
/// they may block, so they never hold up the model streams on the
/// `IoTaskPool`.
fn start_tool(
    effects: &Effects,
    agent: &AgentId,
    access: &ToolAccess,
    tools: &Query<&ToolSpec>,
    call: &message::ToolCall,
) -> ToolState {
    let name = call.function.name.as_str();
    let handler = tools
        .iter()
        .find(|tool| tool.definition.name.as_str() == name && access.allows(name))
        .map_or_else(|| missing_tool(name), |tool| tool.handler.clone());
    let args = call
        .function
        .invalid_arguments
        .clone()
        .unwrap_or_else(|| call.function.arguments_value().to_string());
    let reply = dispatch(
        effects,
        agent,
        tool_key(name),
        handler,
        EffectKind::ToolCall {
            name: name.to_owned(),
            args,
        },
    );
    ToolState::Running(
        AsyncComputeTaskPool::get_or_init(TaskPool::new).spawn(async move {
            let outcome = match reply.await {
                Ok(reply) => reply.into_outcome().await,
                Err(panic) => Err(panic),
            };
            match outcome {
                Ok(Outcome::ToolResult { result }) => result,
                Ok(other) => ToolResult::failed(ToolExecutionError::other(format!(
                    "The tool answered with a {} outcome.",
                    other.family()
                ))),
                Err(error) => ToolResult::failed(ToolExecutionError::other(error.message)),
            }
        }),
    )
}

/// The handler of a call to a tool that is not registered, or that the
/// agent may not call: it answers with a not-found error.
fn missing_tool(name: &str) -> ErasedHandler {
    ErasedHandler::new(MissingTool(name.to_owned()))
}

/// See [`missing_tool`].
struct MissingTool(String);

impl Serve for MissingTool {
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: tool_key(&self.0),
            family: FamilyDescriptor::Tool {
                name: self.0.clone(),
                description: "A tool that is not registered.".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Ok(Outcome::ToolResult {
            result: ToolResult::failed(ToolExecutionError::not_found(format!(
                "There is no tool named `{}`.",
                self.0
            ))),
        }))
    }
}

/// Collects finished tool calls. When every tool call of an agent has
/// finished, their results join the conversation in call order and the next
/// model call is queued.
pub(crate) fn poll_tool_calls(
    mut agents: Query<(&mut Conversation, &mut Status, &Calls)>,
    mut tool_calls: Query<&mut ToolCall>,
    mut commands: Commands,
) {
    for mut tool in &mut tool_calls {
        let ready = match &mut tool.bypass_change_detection().state {
            ToolState::Running(task) => check_ready(task),
            ToolState::Queued | ToolState::Done(_) => None,
        };
        if let Some(result) = ready {
            tool.state = ToolState::Done(result);
        }
    }
    'agents: for (mut conversation, mut status, calls) in &mut agents {
        if *status != Status::Tools {
            continue;
        }
        let mut results = Vec::new();
        for entity in calls.iter() {
            let Ok(tool) = tool_calls.get(entity) else {
                continue;
            };
            let ToolState::Done(result) = &tool.state else {
                continue 'agents;
            };
            results.push((tool.index, tool_result(&tool.call, result)));
        }
        if results.is_empty() {
            continue;
        }
        results.sort_by_key(|(index, _)| *index);
        conversation.0.push(Message::tool_results(
            results.into_iter().map(|(_, result)| result).collect(),
        ));
        for entity in calls.iter() {
            commands.entity(entity).despawn();
        }
        *status = Status::Queued;
    }
}

/// The conversation's record of a finished tool call.
fn tool_result(call: &message::ToolCall, result: &ToolResult) -> message::ToolResult {
    let mut content = result.output().as_content().to_vec();
    if content.is_empty() {
        content.push(ToolResultContent::text("(no output)"));
    }
    if result.is_error() {
        call.error_result(content)
    } else {
        call.result(content)
    }
}

/// Makes the agent idle and announces the end of its turn.
fn end_turn(commands: &mut Commands, agent: Entity, status: &mut Status) {
    *status = Status::Idle;
    commands.trigger(TurnEnded { entity: agent });
}
