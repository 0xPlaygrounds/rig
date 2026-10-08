//! The agent loop, as systems over agent and call entities.
//!
//! Each frame, in [`RigSet`] order: submitted text becomes a prompt or a
//! command run; an agent owing a reply starts one model call; finished calls
//! are collected, a reply's tool calls start as call entities; and once all
//! of a turn's tool calls have results, they are appended and the agent owes
//! the model a reply again. The loop ends when a reply calls no tools.

use std::sync::mpsc;

use bevy_app::{App, AppExit, Last, Plugin, Startup, Update};
use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use rig_core::{
    ErrorReport,
    completion::AssistantContent,
    effect::{EffectKind, Outcome, model_key, tool_key},
    message::{self, AssistantMessage, Message, ToolResultContent, turn_failure},
};

use crate::{
    agent::{
        Agent, AgentCalls, AgentId, AgentStatus, CallOf, Choose, Conversation, EffectTask, Effort,
        Interrupt, ModelCall, ModelChoice, NeedsReply, Notice, Quit, RigSet, Submit, SystemPrompt,
        ToolAccess, ToolCallSlot, TurnEnded, turn_running,
    },
    commands::route_submit,
    effects::{EffectHub, describe_tools, flush_effects},
    model::{self, Credentials},
    session,
    tools::ToolDef,
};

/// The agent loop: messages, the [`RigSet`] stages, the effect hub and
/// model credentials, and the agents restored (or one new agent) at startup.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Submit>()
            .add_message::<Interrupt>()
            .add_message::<Quit>()
            .add_message::<Notice>()
            .add_message::<Choose>()
            .init_resource::<EffectHub>()
            .init_resource::<Credentials>()
            .configure_sets(
                Update,
                (RigSet::Input, RigSet::Start, RigSet::Poll, RigSet::Finish).chain(),
            )
            .add_systems(Startup, (session::restore, describe_tools))
            .add_systems(
                Update,
                (
                    (route_submit, interrupt, quit).in_set(RigSet::Input),
                    start_model_calls.in_set(RigSet::Start),
                    (drain_feeds, poll_calls).chain().in_set(RigSet::Poll),
                    collect_tool_results.in_set(RigSet::Finish),
                ),
            )
            .add_systems(Last, flush_effects);
    }
}

/// Save the session and leave.
fn quit(mut quits: MessageReader<Quit>, mut commands: Commands, mut exit: MessageWriter<AppExit>) {
    if quits.read().last().is_some() {
        commands.queue(session::save);
        exit.write(AppExit::Success);
    }
}

/// Set the agent idle and announce the end of its turn.
fn end_turn(commands: &mut Commands, agent: Entity, status: &mut AgentStatus) {
    *status = AgentStatus::Idle;
    commands.trigger(TurnEnded { entity: agent });
}

/// Stop each interrupted agent's turn. What a cancelled model call
/// streamed is kept as an aborted turn, and every unanswered tool call gets
/// an "Operation aborted" result.
fn interrupt(
    mut interrupts: MessageReader<Interrupt>,
    mut agents: Query<
        (
            &mut Conversation,
            &mut AgentStatus,
            Has<NeedsReply>,
            Option<&AgentCalls>,
        ),
        With<Agent>,
    >,
    calls: Query<(Option<&ModelCall>, Option<&ToolCallSlot>)>,
    mut commands: Commands,
) {
    for Interrupt { agent } in interrupts.read() {
        let Ok((mut conversation, mut status, needs_reply, agent_calls)) = agents.get_mut(*agent)
        else {
            continue;
        };
        // An idle agent has no turn to end.
        if !turn_running(*status, needs_reply, agent_calls.is_some()) {
            continue;
        }
        let mut results = Vec::new();
        for (model_call, slot) in agent_calls
            .into_iter()
            .flat_map(|calls| calls.iter())
            .filter_map(|call| calls.get(call).ok())
        {
            if let Some(call) = model_call
                && !call.text.is_empty()
            {
                conversation
                    .0
                    .push(Message::Assistant(AssistantMessage::aborted(
                        call.origin.clone(),
                        vec![AssistantContent::text(call.text.clone())],
                        "stopped by the user",
                    )));
            }
            if let Some(slot) = slot {
                let result = slot.result.clone().unwrap_or_else(|| {
                    slot.call
                        .error_result(vec![ToolResultContent::text("Operation aborted")])
                });
                results.push((slot.index, result));
            }
        }
        if !results.is_empty() {
            results.sort_by_key(|(index, _)| *index);
            conversation.0.push(Message::tool_results(
                results.into_iter().map(|(_, result)| result).collect(),
            ));
        }
        commands
            .entity(*agent)
            .remove::<NeedsReply>()
            .despawn_related::<AgentCalls>();
        end_turn(&mut commands, *agent, &mut status);
    }
}

/// Start a model call for each agent that owes the model a reply and has
/// nothing in flight.
fn start_model_calls(
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
        (With<Agent>, With<NeedsReply>, Without<AgentCalls>),
    >,
    tools: Query<&ToolDef>,
    mut hub: ResMut<EffectHub>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (agent, id, conversation, choice, effort, system, access, mut status) in &mut agents {
        commands.entity(agent).remove::<NeedsReply>();
        let definitions = tools
            .iter()
            .filter(|tool| allowed(access, tool.definition.name.as_str()))
            .map(|tool| tool.definition.clone())
            .collect();
        let started = choice
            .0
            .as_deref()
            .ok_or_else(|| "Pick a model with /model first.".to_owned())
            .and_then(|reference| {
                let spec = model::resolve(reference)
                    .ok_or_else(|| format!("{reference} is not in the model catalog."))?;
                let request =
                    model::build_request(spec, &system.0, &conversation.0, effort.0, definitions)?;
                let handler = hub.model(reference, spec)?;
                Ok((reference, request, handler))
            });
        let (reference, request, handler) = match started {
            Ok(started) => started,
            Err(text) => {
                notices.write(Notice { agent, text });
                end_turn(&mut commands, agent, &mut status);
                continue;
            }
        };
        let (feed, receiver) = mpsc::channel();
        let task = hub.dispatch(
            id,
            model_key(reference),
            Some(handler),
            EffectKind::Completion {
                request,
                stream: true,
            },
            Some(feed),
        );
        commands.spawn((CallOf(agent), EffectTask(task), ModelCall::new(receiver)));
        *status = AgentStatus::Thinking;
    }
}

fn allowed(access: &ToolAccess, name: &str) -> bool {
    access
        .0
        .as_ref()
        .is_none_or(|names| names.iter().any(|allowed| allowed == name))
}

/// Move streamed text into each model call, for views to read.
fn drain_feeds(mut calls: Query<&mut ModelCall>) {
    for mut call in &mut calls {
        if call.bypass_change_detection().drain() {
            call.set_changed();
        }
    }
}

/// Collect finished calls. A finished reply is appended and its tool calls
/// start; a finished tool call stores its result in its slot.
fn poll_calls(
    mut calls: Query<(Entity, &CallOf, &mut EffectTask, Option<&mut ToolCallSlot>)>,
    mut agents: Query<(&AgentId, &ToolAccess, &mut Conversation, &mut AgentStatus), With<Agent>>,
    tools: Query<&ToolDef>,
    mut hub: ResMut<EffectHub>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, &CallOf(agent), mut task, slot) in &mut calls {
        let Some(outcome) = check_ready(&mut task.0) else {
            continue;
        };
        commands.entity(call).remove::<EffectTask>();
        if let Some(mut slot) = slot {
            slot.result = Some(tool_result(&slot.call, outcome));
            // Count down the calls still running; the batch ends at the last.
            if let Ok((.., mut status)) = agents.get_mut(agent)
                && let AgentStatus::Tools(running) = *status
            {
                *status = AgentStatus::Tools(running.saturating_sub(1).max(1));
            }
            continue;
        }
        commands.entity(call).despawn();
        let Ok((id, access, mut conversation, mut status)) = agents.get_mut(agent) else {
            continue;
        };
        let response = match outcome {
            Ok(Outcome::Completion(response)) => response,
            Ok(other) => {
                let text = format!("The model answered with a {} outcome.", other.family());
                notices.write(Notice { agent, text });
                end_turn(&mut commands, agent, &mut status);
                continue;
            }
            Err(error) => {
                let text = format!("The model call failed: {}", error.message);
                notices.write(Notice { agent, text });
                end_turn(&mut commands, agent, &mut status);
                continue;
            }
        };
        if let Some(message) = response.message() {
            conversation.0.push(message);
        }
        let stop = response.stop();
        if let Some(text) = turn_failure(
            &response.choice,
            Some(&stop),
            response.finish_reason().as_ref(),
        ) {
            notices.write(Notice { agent, text });
            end_turn(&mut commands, agent, &mut status);
            continue;
        }
        let tool_calls: Vec<message::ToolCall> = response.tool_calls().cloned().collect();
        if tool_calls.is_empty() {
            end_turn(&mut commands, agent, &mut status);
            continue;
        }
        *status = AgentStatus::Tools(tool_calls.len());
        for (index, tool_call) in tool_calls.into_iter().enumerate() {
            let name = tool_call.function.name.as_str().to_owned();
            let handler = tools
                .iter()
                .find(|tool| tool.definition.name.as_str() == name && allowed(access, &name))
                .map(|tool| tool.handler.clone());
            let args = tool_call
                .function
                .invalid_arguments
                .clone()
                .unwrap_or_else(|| tool_call.function.arguments_value().to_string());
            let task = hub.dispatch(
                id,
                tool_key(&name),
                handler,
                EffectKind::ToolCall { name, args },
                None,
            );
            commands.spawn((
                CallOf(agent),
                EffectTask(task),
                ToolCallSlot {
                    index,
                    call: tool_call,
                    result: None,
                },
            ));
        }
    }
}

/// The result the model reads for a finished tool call. A failure of any
/// kind, including a missing tool or a panic, is an error result.
fn tool_result(
    call: &message::ToolCall,
    outcome: Result<Outcome, ErrorReport>,
) -> message::ToolResult {
    let content = match outcome {
        Ok(Outcome::ToolResult { result }) => {
            let content = result.output().clone().into_content();
            return if result.is_success() {
                call.result(content)
            } else {
                call.error_result(content)
            };
        }
        Ok(other) => format!("the tool answered with a {} outcome", other.family()),
        Err(error) => error.message,
    };
    call.error_result(vec![ToolResultContent::text(content)])
}

/// Answer each agent whose tool calls all finished: append their results in
/// call order and owe the model a reply.
fn collect_tool_results(
    mut agents: Query<(Entity, &AgentCalls, &mut Conversation), With<Agent>>,
    slots: Query<&ToolCallSlot>,
    mut commands: Commands,
) {
    for (agent, calls, mut conversation) in &mut agents {
        let mut results = Vec::new();
        for call in calls.iter() {
            match slots
                .get(call)
                .ok()
                .and_then(|slot| slot.result.clone().map(|result| (slot.index, result)))
            {
                Some(result) => results.push(result),
                None => {
                    results.clear();
                    break;
                }
            }
        }
        if results.is_empty() {
            continue;
        }
        results.sort_by_key(|(index, _)| *index);
        conversation.0.push(Message::tool_results(
            results.into_iter().map(|(_, result)| result).collect(),
        ));
        commands
            .entity(agent)
            .despawn_related::<AgentCalls>()
            .insert(NeedsReply);
    }
}
