//! The agent loop: send the conversation with the tool definitions, stream
//! the reply, run its tool calls, append the results, and repeat until the
//! model stops calling tools. Every system works on all agents at once.

use std::{collections::HashMap, panic::AssertUnwindSafe, sync::Arc};

use bevy::{
    prelude::*,
    tasks::{
        AsyncComputeTaskPool, IoTaskPool, Task,
        futures::check_ready,
        futures_lite::{FutureExt, StreamExt},
    },
};
use crossbeam_channel::{Receiver, Sender};
use rig_core::{
    completion::{CompletionRequest, CompletionResponse},
    effect::{EffectKind, Outcome},
    error::ProviderError,
    message::{Message, StopReason, ToolCall, ToolResult, ToolResultContent, turn_failure},
    operation::Completion,
    providers::registry::ModelSelector,
    serve::{
        ErasedHandler, Reply,
        adapters::{ModelAdapter, ToolFn},
    },
    streaming::{Item, StreamEvent, Streamed},
    tool::{ToolContext, ToolExecutionError, ToolOutput},
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

/// One tool call of a reply: its position, the call while it waits for the
/// calls before it, its task while it runs, and its result once done.
///
/// The calls of one reply run one at a time, in order: two edits of one
/// file in a reply would otherwise both read the original text, and one
/// would be lost.
#[derive(Component)]
pub(super) struct ToolCallRun {
    index: usize,
    waiting: Option<(ToolCall, ErasedHandler)>,
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
/// which cancels them. Finished tool calls keep their results; calls that
/// will never finish get error results, so the conversation stays valid.
pub(super) fn read_interrupts(
    mut interrupts: MessageReader<Interrupt>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus, Option<&Calls>)>,
    runs: Query<&ToolCallRun>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for interrupt in interrupts.read() {
        let agent = interrupt.agent;
        let Ok((mut conversation, mut status, calls)) = agents.get_mut(agent) else {
            continue;
        };
        if *status == AgentStatus::Idle {
            continue;
        }
        if *status == AgentStatus::Tools {
            let finished = runs
                .iter_many(calls.into_iter().flat_map(|calls| calls.iter()))
                .filter_map(|run| {
                    let run = run.ok()?;
                    Some((run.index, run.done.clone()?))
                })
                .collect();
            answer_interrupted(&mut conversation, finished);
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

/// Answer every tool call of the last assistant message: with its result
/// when `finished` holds one by call position, else as interrupted.
fn answer_interrupted(conversation: &mut Conversation, mut finished: HashMap<usize, ToolResult>) {
    let Some(Message::Assistant(message)) = conversation.0.last() else {
        return;
    };
    let results = message
        .tool_calls()
        .enumerate()
        .map(|(index, call)| {
            finished.remove(&index).unwrap_or_else(|| {
                call.error_result(vec![ToolResultContent::text("interrupted by the user")])
            })
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
        // A panicking provider must end the call, not poison its task.
        let task = AsyncComputeTaskPool::get().spawn(async move {
            AssertUnwindSafe(stream_reply(label, reply, sender))
                .catch_unwind()
                .await
                .unwrap_or_else(|_| Err(ProviderError::Response("the model call panicked".into())))
        });
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
    mut agents: Query<(&ToolAccess, &mut Conversation, &mut AgentStatus)>,
    tools: Query<&RegisteredTool>,
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
        let Ok((access, mut conversation, mut status)) = agents.get_mut(agent) else {
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
        let stop = response.stop();
        let failure = turn_failure(
            &response.choice,
            Some(&stop),
            response.finish_reason().as_ref(),
        );
        conversation.0.extend(response.message());
        let tool_calls = response.tool_calls().cloned().collect::<Vec<_>>();
        if failure.is_none() && !stop.is_failure() && !tool_calls.is_empty() {
            *status = AgentStatus::Tools;
            for (index, call) in tool_calls.into_iter().enumerate() {
                commands.spawn((CallOf(agent), queue_tool_call(index, call, access, &tools)));
            }
            continue;
        }
        if let Some(reason) = failure.or(match stop {
            StopReason::Aborted(reason) => Some(reason),
            _ => None,
        }) {
            notices.write(Notice::error(agent, reason));
        }
        *status = AgentStatus::Idle;
        commands.trigger(TurnEnded { entity: agent });
    }
}

/// A tool call waiting for its turn, with the handler that will serve it.
/// A call to a tool the agent does not have goes to a handler that refuses
/// it, so the effect log shows it too.
fn queue_tool_call(
    index: usize,
    call: ToolCall,
    access: &ToolAccess,
    tools: &Query<&RegisteredTool>,
) -> ToolCallRun {
    let name = call.function.name.as_str();
    let handler = tools
        .iter()
        .find(|tool| tool.definition.name.as_str() == name && access.allows(name))
        .map_or_else(|| missing_tool(name), |tool| tool.handler.clone());
    ToolCallRun {
        index,
        waiting: Some((call, handler)),
        task: None,
        done: None,
    }
}

/// Dispatch one tool call on the IO pool.
fn start_tool_call(
    call: ToolCall,
    handler: ErasedHandler,
    agent: &AgentId,
    workdir: &Workdir,
    effects: &mut Effects,
) -> Task<ToolResult> {
    let name = call.function.name.as_str();
    let args =
        call.function.invalid_arguments.clone().unwrap_or_else(|| {
            serde_json::Value::Object(call.function.arguments.clone()).to_string()
        });
    let reply = effects.dispatch(
        agent,
        handler,
        EffectKind::ToolCall {
            name: name.to_owned(),
            args,
        },
        vec![Arc::new(workdir.clone())],
    );
    let answered = call;
    IoTaskPool::get().spawn(async move {
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
    })
}

/// A handler for a call to `name`, which the agent has no tool for.
fn missing_tool(name: &str) -> ErasedHandler {
    fn refuse<'a>(
        _context: &'a mut ToolContext,
        _args: serde_json::Value,
    ) -> rig_core::wasm_compat::WasmBoxedFuture<'a, Result<ToolOutput, ToolExecutionError>> {
        Box::pin(async {
            Err(ToolExecutionError::not_found(
                "there is no tool by this name",
            ))
        })
    }
    ErasedHandler::new(ToolFn::new(
        name,
        "A tool the agent does not have.",
        serde_json::json!({"type": "object"}),
        refuse,
    ))
}

/// Collect finished tool calls and start each agent's next waiting one.
/// Once every call of an agent is done, the results are appended in call
/// order and the conversation goes back to the model.
pub(super) fn finish_tool_calls(
    mut runs: Query<(Entity, &CallOf, &mut ToolCallRun)>,
    mut agents: Query<(&AgentId, &Workdir, &mut Conversation, &mut AgentStatus)>,
    mut effects: ResMut<Effects>,
    mut commands: Commands,
) {
    let mut by_agent: HashMap<Entity, Vec<(usize, Entity)>> = HashMap::new();
    for (entity, call_of, mut run) in &mut runs {
        if let Some(task) = &mut run.task
            && let Some(result) = check_ready(task)
        {
            run.task = None;
            run.done = Some(result);
        }
        by_agent
            .entry(call_of.0)
            .or_default()
            .push((run.index, entity));
    }
    for (agent, mut entities) in by_agent {
        let Ok((id, workdir, mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        entities.sort_by_key(|(index, _)| *index);
        let mut results = Vec::with_capacity(entities.len());
        let mut running = false;
        for (_, entity) in &entities {
            let Ok((_, _, mut run)) = runs.get_mut(*entity) else {
                continue;
            };
            if let Some(done) = &run.done {
                results.push(done.clone());
                continue;
            }
            if !running && let Some((call, handler)) = run.waiting.take() {
                run.task = Some(start_tool_call(call, handler, id, workdir, &mut effects));
            }
            running = true;
        }
        if running {
            continue;
        }
        for (_, entity) in entities {
            commands.entity(entity).despawn();
        }
        conversation.0.push(Message::tool_results(results));
        *status = AgentStatus::Thinking;
        commands.entity(agent).insert(NeedsReply);
    }
}
