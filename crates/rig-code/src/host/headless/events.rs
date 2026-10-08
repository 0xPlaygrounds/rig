//! The JSON event stream of `--print --json` and `--rpc`: one object per
//! line on stdout, each with a `type`. Every event about an agent names it
//! by its stable [`AgentId`] as `agent`; requests may name it so too.
//!
//! - `session`: `id`, `directory`, once at the start.
//! - `agent`: `agent`, `entity` (its `Entity` as bits, for reflected
//!   requests), `parent` and `task` for a subagent, `forked_from`.
//! - `model`: `agent`, `model`, whenever its model is chosen.
//! - `turn_start`, `turn_end` (with the agent's `spending`).
//! - `text_delta`, `reasoning_delta`: `agent`, `delta`, as the model
//!   streams.
//! - `message`: `agent`, `message`, each message added to a conversation
//!   (the user's, the model's, tool results), as rig-core serializes it.
//! - `tool_call`: `agent`, `call` (entity bits), `id`, `name`,
//!   `arguments`; `tool_result`: `agent`, `call`, `id`, `name`, `result`,
//!   `error`.
//! - `approval`: `agent`, `call`, `tool`, `subject`: answer it with the
//!   reflected `Approve` request.
//! - `notice`: `agent` (or null), `level`, `text`.
//! - `pick`: `agent`, `kind`: a command asked a view to let the user pick.

use std::collections::HashMap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::message::ToolResult;
use serde_json::{Value, json};

use super::emit;
use crate::core::agent::{
    Agent, AgentId, CallOf, Conversation, ModelChoice, Notice, NoticeLevel, Partial, PickRequest,
    ToolCallRun, TurnFinished, TurnOf,
};
use crate::core::approval::AwaitingApproval;
use crate::core::calls::Done;
use crate::core::rewind::Forked;
use crate::core::save::SessionPaths;
use crate::core::subagents::{Delegated, SubagentOf};
use crate::core::turn::PollCalls;
use crate::core::usage::Spending;

/// Writes the event stream.
pub struct EventStreamPlugin;

impl Plugin for EventStreamPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, emit_session)
            .add_systems(
                Update,
                (emit_agents, emit_deltas, emit_messages, emit_notices)
                    .chain()
                    .after(PollCalls),
            )
            .add_systems(Last, emit_turn_ends)
            .add_observer(emit_turn_start)
            .add_observer(emit_tool_call)
            .add_observer(emit_tool_result)
            .add_observer(emit_approval);
    }
}

/// The [`AgentId`] of the agent a call entity's turn belongs to.
fn agent_of_call(
    call: Entity,
    calls: &Query<&CallOf>,
    turns: &Query<&TurnOf>,
    ids: &Query<&AgentId>,
) -> Option<String> {
    let &CallOf(turn) = calls.get(call).ok()?;
    let &TurnOf(agent) = turns.get(turn).ok()?;
    ids.get(agent).ok().map(|id| id.0.clone())
}

fn emit_session(paths: Option<Res<SessionPaths>>) {
    let id = paths
        .as_ref()
        .and_then(|paths| paths.path().file_name())
        .map(|name| name.to_string_lossy().into_owned());
    let directory = std::env::current_dir().ok();
    emit(&json!({
        "type": "session",
        "id": id,
        "directory": directory,
    }));
}

/// Each new agent, and each model chosen.
fn emit_agents(
    added: Query<
        (
            Entity,
            &AgentId,
            Option<&SubagentOf>,
            Option<&Delegated>,
            Option<&Forked>,
        ),
        Added<Agent>,
    >,
    chosen: Query<(&AgentId, &ModelChoice), Changed<ModelChoice>>,
    ids: Query<&AgentId>,
) {
    for (entity, id, parent, delegated, forked) in &added {
        emit(&json!({
            "type": "agent",
            "agent": id.0,
            "entity": entity.to_bits(),
            "parent": parent.and_then(|of| ids.get(of.0).ok()).map(|id| &id.0),
            "task": delegated.map(|delegated| &delegated.task),
            "forked_from": forked.map(|forked| &forked.from),
        }));
    }
    for (id, model) in &chosen {
        emit(&json!({"type": "model", "agent": id.0, "model": model.0}));
    }
}

fn emit_turn_start(started: On<Add<TurnOf>>, turns: Query<&TurnOf>, ids: Query<&AgentId>) {
    let Some(id) = turns
        .get(started.entity)
        .ok()
        .and_then(|&TurnOf(agent)| ids.get(agent).ok())
    else {
        return;
    };
    emit(&json!({"type": "turn_start", "agent": id.0}));
}

fn emit_turn_ends(mut finished: MessageReader<TurnFinished>, agents: Query<(&AgentId, &Spending)>) {
    for turn in finished.read() {
        if let Ok((id, spending)) = agents.get(turn.agent) {
            emit(&json!({"type": "turn_end", "agent": id.0, "spending": to_json(spending)}));
        }
    }
}

/// What each streaming call streamed since the last frame. A call's last
/// fragments may arrive with its end; its `message` holds them.
fn emit_deltas(
    calls: Query<(Entity, &CallOf, &Partial), Changed<Partial>>,
    live: Query<(), With<Partial>>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
    mut sent: Local<HashMap<Entity, (usize, usize)>>,
) {
    sent.retain(|call, _| live.contains(*call));
    for (call, &CallOf(turn), partial) in &calls {
        let Some(id) = turns
            .get(turn)
            .ok()
            .and_then(|&TurnOf(agent)| ids.get(agent).ok())
        else {
            continue;
        };
        let (text, reasoning) = sent.entry(call).or_default();
        for (kind, streamed, sent) in [
            ("reasoning_delta", &partial.reasoning, reasoning),
            ("text_delta", &partial.text, text),
        ] {
            if let Some(delta) = streamed.get(*sent..).filter(|delta| !delta.is_empty()) {
                emit(&json!({"type": kind, "agent": id.0, "delta": delta}));
            }
            *sent = streamed.len();
        }
    }
}

/// Each message added to a conversation since the last frame. A
/// conversation that got shorter, by a rewind, is followed from its new
/// end.
fn emit_messages(
    conversations: Query<(Entity, &AgentId, &Conversation), Changed<Conversation>>,
    mut sent: Local<HashMap<Entity, usize>>,
) {
    for (agent, id, conversation) in &conversations {
        let sent = sent.entry(agent).or_insert(conversation.0.len());
        let from = if *sent > conversation.0.len() {
            conversation.0.len()
        } else {
            *sent
        };
        for message in conversation.0.iter().skip(from) {
            emit(&json!({"type": "message", "agent": id.0, "message": to_json(message)}));
        }
        *sent = conversation.0.len();
    }
}

fn emit_tool_call(
    started: On<Add<ToolCallRun>>,
    runs: Query<&ToolCallRun>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
) {
    let Ok(run) = runs.get(started.entity) else {
        return;
    };
    emit(&json!({
        "type": "tool_call",
        "agent": agent_of_call(started.entity, &calls, &turns, &ids),
        "call": started.entity.to_bits(),
        "id": to_json(&run.call.id),
        "name": run.call.function.name.as_str(),
        "arguments": run.call.function.arguments,
    }));
}

fn emit_tool_result(
    done: On<Add<Done<ToolResult>>>,
    results: Query<&Done<ToolResult>>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
) {
    let Ok(Done(result)) = results.get(done.entity) else {
        return;
    };
    emit(&json!({
        "type": "tool_result",
        "agent": agent_of_call(done.entity, &calls, &turns, &ids),
        "call": done.entity.to_bits(),
        "id": to_json(&result.call),
        "name": result.name.as_str(),
        "error": result.is_error,
        "result": to_json(result),
    }));
}

fn emit_approval(
    waiting: On<Add<AwaitingApproval>>,
    asks: Query<&AwaitingApproval>,
    calls: Query<&CallOf>,
    turns: Query<&TurnOf>,
    ids: Query<&AgentId>,
) {
    let Ok(ask) = asks.get(waiting.entity) else {
        return;
    };
    emit(&json!({
        "type": "approval",
        "agent": agent_of_call(waiting.entity, &calls, &turns, &ids),
        "call": waiting.entity.to_bits(),
        "tool": ask.tool,
        "subject": ask.subject,
    }));
}

fn emit_notices(
    mut notices: MessageReader<Notice>,
    mut picks: MessageReader<PickRequest>,
    ids: Query<&AgentId>,
) {
    for notice in notices.read() {
        let agent = notice.agent.and_then(|agent| ids.get(agent).ok());
        emit(&json!({
            "type": "notice",
            "agent": agent.map(|id| &id.0),
            "level": match notice.level {
                NoticeLevel::Info => "info",
                NoticeLevel::Error => "error",
            },
            "text": notice.text,
        }));
    }
    for pick in picks.read() {
        let agent = ids.get(pick.agent).ok();
        emit(&json!({
            "type": "pick",
            "agent": agent.map(|id| &id.0),
            "kind": format!("{:?}", pick.kind),
        }));
    }
}

/// `value` with JSON's `null` for what does not serialize.
pub(crate) fn to_json(value: &impl serde::Serialize) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}
