//! The agent loop: send the conversation with the tool definitions, stream
//! the reply, run its tool calls, append the results, and repeat until the
//! model stops calling tools. Every system works on all agents at once.

use std::{collections::HashMap, sync::Arc};

use bevy::{
    prelude::*,
    tasks::{
        AsyncComputeTaskPool, IoTaskPool, Task, futures::check_ready, futures_lite::StreamExt,
    },
};
use crossbeam_channel::{Receiver, Sender};
use rig_core::{
    completion::{CompletionRequest, CompletionResponse},
    effect::{EffectKind, Outcome},
    error::ProviderError,
    message::{Message, StopReason, ToolCall, ToolResult, ToolResultContent},
    operation::Completion,
    providers::registry::ModelSelector,
    serve::{ErasedHandler, Reply, adapters::ModelAdapter},
    streaming::{Item, StreamEvent, Streamed},
};

use super::{
    Interrupt, Notice, Submit, TurnEnded,
    agent::{
        AgentId, AgentStatus, CallOf, Calls, Conversation, Draft, EffortChoice, ModelChoice,
        NeedsReply, SystemPrompt, ToolAccess, Workdir,
    },
    catalog,
    command::{CommandId, CommandInput, SlashCommand},
    dispatch::Effects,
    tools::RegisteredTool,
};

/// A model call in flight: the task streaming the reply, and the events it
/// forwards for the agent's [`Draft`].
#[derive(Component)]
pub(super) struct ModelCall {
    task: Task<Result<CompletionResponse, ProviderError>>,
    events: Receiver<StreamEvent>,
}

/// One tool call of a reply: its position, its task until it is done, and
/// then its result.
#[derive(Component)]
pub(super) struct ToolCallRun {
    index: usize,
    task: Option<Task<ToolResult>>,
    done: Option<ToolResult>,
}

/// Route submitted text: a slash command runs its system, other text
/// becomes a user message when the agent is idle.
pub(super) fn read_submits(
    mut submits: MessageReader<Submit>,
    slash_commands: Query<(Entity, &SlashCommand)>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for submit in submits.read() {
        let agent = submit.agent;
        let text = submit.text.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(line) = text.strip_prefix('/') {
            let (name, args) = line
                .split_once(char::is_whitespace)
                .map_or((line, ""), |(name, args)| (name, args.trim()));
            match slash_commands
                .iter()
                .find(|(_, command)| command.name == name)
            {
                Some((system, _)) => commands.run_system_with(
                    CommandId::from_entity(system),
                    CommandInput {
                        agent,
                        args: args.to_owned(),
                    },
                ),
                None => {
                    notices.write(Notice::error(
                        agent,
                        format!("Unknown command /{name}; /help lists the commands"),
                    ));
                }
            }
            continue;
        }
        let Ok((mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        if *status != AgentStatus::Idle {
            notices.write(Notice::error(
                agent,
                "A turn is running. Stop it first, then send again.",
            ));
            continue;
        }
        conversation.0.push(Message::user(text));
        *status = AgentStatus::Thinking;
        commands.entity(agent).insert(NeedsReply);
    }
}

/// Stop a running turn: despawning the agent's calls drops their tasks,
/// which cancels them. Calls that will never be answered get error results,
/// so the conversation stays valid.
pub(super) fn read_interrupts(
    mut interrupts: MessageReader<Interrupt>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for interrupt in interrupts.read() {
        let agent = interrupt.agent;
        let Ok((mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        if *status == AgentStatus::Idle {
            continue;
        }
        if *status == AgentStatus::Tools {
            answer_all(&mut conversation, "interrupted by the user");
        }
        *status = AgentStatus::Idle;
        commands
            .entity(agent)
            .despawn_related::<Calls>()
            .remove::<(NeedsReply, Draft)>()
            .trigger(|entity| TurnEnded { entity });
        notices.write(Notice::info(agent, "Stopped."));
    }
}

/// Answer every tool call of the last assistant message with an error.
fn answer_all(conversation: &mut Conversation, reason: &str) {
    let Some(Message::Assistant(message)) = conversation.0.last() else {
        return;
    };
    let results = message
        .content
        .iter()
        .filter_map(|content| match content {
            rig_core::message::AssistantContent::ToolCall(call) => {
                Some(call.error_result(vec![ToolResultContent::text(reason)]))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if !results.is_empty() {
        conversation.0.push(Message::tool_results(results));
    }
}

/// Send the conversation of every agent that needs a reply.
pub(super) fn start_model_calls(
    mut agents: Query<
        (
            Entity,
            &AgentId,
            &Conversation,
            &ModelChoice,
            &EffortChoice,
            &SystemPrompt,
            &Workdir,
            &ToolAccess,
            &mut AgentStatus,
        ),
        With<NeedsReply>,
    >,
    tools: Query<&RegisteredTool>,
    mut effects: ResMut<Effects>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (agent, id, conversation, model, effort, prompt, workdir, access, mut status) in &mut agents
    {
        commands.entity(agent).remove::<NeedsReply>();
        let prepared = prepare(
            model,
            *effort,
            conversation,
            prompt,
            workdir,
            access,
            &tools,
        );
        let (label, handler, request) = match prepared {
            Ok(prepared) => prepared,
            Err(reason) => {
                notices.write(Notice::error(agent, reason));
                *status = AgentStatus::Idle;
                commands.trigger(TurnEnded { entity: agent });
                continue;
            }
        };
        *status = AgentStatus::Thinking;
        let reply = effects.dispatch(
            id,
            handler,
            EffectKind::Completion {
                request,
                stream: true,
            },
            Vec::new(),
        );
        let (sender, events) = crossbeam_channel::unbounded();
        let task = AsyncComputeTaskPool::get().spawn(stream_reply(label, reply, sender));
        commands.entity(agent).insert(Draft::default());
        commands.spawn((CallOf(agent), ModelCall { task, events }));
    }
}

/// The model label, its handler and the request for one agent, or why the
/// conversation cannot be sent.
fn prepare(
    model: &ModelChoice,
    effort: EffortChoice,
    conversation: &Conversation,
    prompt: &SystemPrompt,
    workdir: &Workdir,
    access: &ToolAccess,
    tools: &Query<&RegisteredTool>,
) -> Result<(String, ErasedHandler, CompletionRequest), String> {
    let reference = model
        .0
        .as_deref()
        .ok_or("No model is selected. Pick one with /model.")?;
    let spec = catalog::resolve(reference)
        .ok_or_else(|| format!("`{reference}` is not in the model catalog"))?;
    let (options, max_tokens) = catalog::request_options(spec, effort);
    spec.validate(&options).map_err(|error| error.to_string())?;
    let model = ModelSelector::from(spec)
        .provider_ref()
        .map_err(|error| error.to_string())?
        .completion_model()
        .map_err(|error| error.to_string())?;
    let definitions = tools
        .iter()
        .filter(|tool| access.allows(tool.definition.name.as_str()))
        .map(|tool| tool.definition.clone())
        .collect();
    let preamble = format!("{}\n\nWorking directory: {}", prompt.0, workdir.0.display());
    let request = CompletionRequest::from(conversation.0.clone())
        .preamble(preamble)
        .tools(definitions)
        .options(options)
        .max_tokens(max_tokens);
    let handler = ErasedHandler::new(ModelAdapter::<Completion>::new(reference, model));
    Ok((reference.to_owned(), handler, request))
}

/// Read the reply, forwarding each event, and fold it into the response.
async fn stream_reply(
    label: String,
    reply: impl Future<Output = Reply>,
    events: Sender<StreamEvent>,
) -> Result<CompletionResponse, ProviderError> {
    let mut stream = Streamed::relay(label, reply.await.into_stream());
    while let Some(item) = stream.next().await {
        // The receiver is gone only when the call was despawned, and then
        // this task is dropped too.
        if let Ok(Item::Event(event)) = item {
            let _ = events.send(event);
        }
    }
    stream.finish().await
}

/// Move streamed text into each agent's [`Draft`].
pub(super) fn drain_streams(calls: Query<(&CallOf, &ModelCall)>, mut drafts: Query<&mut Draft>) {
    for (call_of, call) in &calls {
        let Ok(mut draft) = drafts.get_mut(call_of.0) else {
            continue;
        };
        for event in call.events.try_iter() {
            match event {
                StreamEvent::Text { text, .. } => draft.text.push_str(&text),
                StreamEvent::Reasoning { text, .. } => draft.reasoning.push_str(&text),
                _ => {}
            }
        }
    }
}

/// Append each finished reply, then start its tool calls or end the turn.
pub(super) fn finish_model_calls(
    mut calls: Query<(Entity, &CallOf, &mut ModelCall)>,
    mut agents: Query<(
        &AgentId,
        &Workdir,
        &ToolAccess,
        &mut Conversation,
        &mut AgentStatus,
    )>,
    tools: Query<&RegisteredTool>,
    mut effects: ResMut<Effects>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (call_entity, call_of, mut call) in &mut calls {
        let Some(result) = check_ready(&mut call.task) else {
            continue;
        };
        let agent = call_of.0;
        commands.entity(call_entity).despawn();
        commands.entity(agent).remove::<Draft>();
        let Ok((id, workdir, access, mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                notices.write(Notice::error(agent, error.to_string()));
                *status = AgentStatus::Idle;
                commands.trigger(TurnEnded { entity: agent });
                continue;
            }
        };
        conversation.0.extend(response.message());
        let tool_calls = response.tool_calls().cloned().collect::<Vec<_>>();
        match response.stop() {
            StopReason::ToolUse if !tool_calls.is_empty() => {
                *status = AgentStatus::Tools;
                for (index, call) in tool_calls.into_iter().enumerate() {
                    let run =
                        start_tool_call(index, call, id, workdir, access, &tools, &mut effects);
                    commands.spawn((CallOf(agent), run));
                }
            }
            StopReason::Length if !tool_calls.is_empty() => {
                // A reply cut off by the length limit may hold truncated
                // arguments, so its calls are answered instead of run.
                answer_all(
                    &mut conversation,
                    "not run: the reply hit the output limit; retry with a shorter call",
                );
                commands.entity(agent).insert(NeedsReply);
            }
            stop => {
                if let StopReason::Error(reason) | StopReason::Aborted(reason) = stop {
                    notices.write(Notice::error(agent, reason));
                }
                *status = AgentStatus::Idle;
                commands.trigger(TurnEnded { entity: agent });
            }
        }
    }
}

/// Dispatch one tool call on the IO pool, or answer it at once when the
/// agent has no such tool.
fn start_tool_call(
    index: usize,
    call: ToolCall,
    agent: &AgentId,
    workdir: &Workdir,
    access: &ToolAccess,
    tools: &Query<&RegisteredTool>,
    effects: &mut Effects,
) -> ToolCallRun {
    let name = call.function.name.as_str();
    let tool = tools
        .iter()
        .find(|tool| tool.definition.name.as_str() == name && access.allows(name));
    let Some(tool) = tool else {
        let result = call.error_result(vec![ToolResultContent::text(format!(
            "there is no tool named `{name}`"
        ))]);
        return ToolCallRun {
            index,
            task: None,
            done: Some(result),
        };
    };
    let args =
        call.function.invalid_arguments.clone().unwrap_or_else(|| {
            serde_json::Value::Object(call.function.arguments.clone()).to_string()
        });
    let reply = effects.dispatch(
        agent,
        tool.handler.clone(),
        EffectKind::ToolCall {
            name: name.to_owned(),
            args,
        },
        vec![Arc::new(workdir.clone())],
    );
    let answered = call;
    let task = IoTaskPool::get().spawn(async move {
        match reply.await.into_outcome().await {
            Ok(Outcome::ToolResult { result }) => {
                let content = result.output().as_content().to_vec();
                if result.is_error() {
                    answered.error_result(content)
                } else {
                    answered.result(content)
                }
            }
            Ok(_) => answered.error_result(vec![ToolResultContent::text(
                "the tool answered with something other than a tool result",
            )]),
            Err(report) => answered.error_result(vec![ToolResultContent::text(report.to_string())]),
        }
    });
    ToolCallRun {
        index,
        task: Some(task),
        done: None,
    }
}

/// Collect finished tool calls. Once every call of an agent is done, the
/// results are appended in call order and the conversation goes back to the
/// model.
pub(super) fn finish_tool_calls(
    mut runs: Query<(Entity, &CallOf, &mut ToolCallRun)>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus)>,
    mut commands: Commands,
) {
    let mut by_agent: HashMap<Entity, Vec<(usize, Entity, Option<ToolResult>)>> = HashMap::new();
    for (entity, call_of, mut run) in &mut runs {
        if run.done.is_none()
            && let Some(task) = &mut run.task
            && let Some(result) = check_ready(task)
        {
            run.done = Some(result);
        }
        by_agent
            .entry(call_of.0)
            .or_default()
            .push((run.index, entity, run.done.clone()));
    }
    for (agent, mut runs) in by_agent {
        if runs.iter().any(|(_, _, done)| done.is_none()) {
            continue;
        }
        runs.sort_by_key(|(index, _, _)| *index);
        let mut results = Vec::with_capacity(runs.len());
        for (_, entity, done) in runs {
            commands.entity(entity).despawn();
            results.extend(done);
        }
        let Ok((mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        conversation.0.push(Message::tool_results(results));
        *status = AgentStatus::Thinking;
        commands.entity(agent).insert(NeedsReply);
    }
}
