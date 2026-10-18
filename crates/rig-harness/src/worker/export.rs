//! The session's durable events, derived from what its logs hold: each
//! record a log write stored becomes the events it means, keyed by its
//! place in its log ([`SourceKey`]), so a record exported twice (after a
//! crash between the save and the send) is journaled once.

use std::collections::HashMap;

use bevy_reflect::TypePath;
use rig_core::conversation::{Change, ChangeRecord, Conversation, TurnEnd, TurnRecord};
use rig_core::message::{AssistantContent, Message, UserContent};
use rig_memory::LogRecord;
use rig_protocol::event::{self, Event, TurnStatus};
use rig_protocol::methods::WorkerEvent;
use rig_protocol::{AgentId, CommandSeq, SourceKey};

use super::Through;

/// What the exporter knows of each log, folded from its records.
#[derive(Default)]
pub(crate) struct Exporter {
    logs: HashMap<String, Shadow>,
}

/// One log as its records left it.
#[derive(Default)]
struct Shadow {
    conversation: Conversation,
    /// The last command the agent's turn answers ([`Through`]).
    through: u64,
    /// The model the agent chose last, as saved.
    model: Option<String>,
}

impl Exporter {
    /// The events of `record`, the record `seq` of `log`; `emit` false
    /// only folds it in, for a record exported before.
    pub(crate) fn record(
        &mut self,
        log: &str,
        seq: u64,
        record: &LogRecord,
        emit: bool,
        out: &mut Vec<WorkerEvent>,
    ) {
        let shadow = self.logs.entry(log.to_owned()).or_default();
        let source = SourceKey {
            log: log.to_owned(),
            index: seq,
        };
        let agent = AgentId(log.to_owned());
        let mut events: Vec<Event> = Vec::new();
        match record {
            LogRecord::Saved { key, value } => {
                if key == Through::type_path() {
                    shadow.through = value.as_ref().and_then(seq_of).unwrap_or(0);
                } else if key == rig_memory::MODEL {
                    shadow.model = value.as_ref().and_then(|v| v.as_str()).map(str::to_owned);
                } else if key == rig_ecs::agent::LastUsage::type_path()
                    && let Some(usage) = value.as_ref().and_then(|v| v.get("usage"))
                    && emit
                {
                    out.push(WorkerEvent {
                        kind: "usage".into(),
                        data: usage_data(&agent, shadow.model.as_deref(), usage),
                        source: Some(source.clone()),
                    });
                }
            }
            LogRecord::Retired => {}
            LogRecord::Change(change) => {
                events.extend(change_events(&agent, shadow, change));
                shadow.conversation.apply(change).ok();
                if let Some(record) = serde_json::to_value(change).ok().filter(|_| emit) {
                    out.push(durable(
                        event::ConversationChange {
                            agent_id: agent.clone(),
                            log: log.to_owned(),
                            index: seq,
                            record,
                        }
                        .into(),
                        &source,
                    ));
                }
            }
        }
        if emit {
            out.extend(events.iter().map(|e| durable(e.clone(), &source)));
        }
    }
}

/// A model call's `usage` event from what its reply reported, as the agent's
/// saved [`LastUsage`](rig_ecs::agent::LastUsage) holds it: `tokens` its
/// input and output (or the provider's total when only that came), and
/// `costUsd` the cost the kernel priced it at, `null` when unpriced. `tree`
/// is `rigd`'s to sum.
fn usage_data(agent: &AgentId, model: Option<&str>, usage: &serde_json::Value) -> serde_json::Value {
    let count = |name: &str| usage.get(name).and_then(serde_json::Value::as_u64);
    let (input, output) = (count("input_tokens"), count("output_tokens"));
    let tokens = match (input, output) {
        (None, None) => count("total_tokens").unwrap_or(0),
        _ => input.unwrap_or(0).saturating_add(output.unwrap_or(0)),
    };
    let cost = usage
        .get("cost")
        .and_then(|cost| cost.get("total"))
        .and_then(serde_json::Value::as_f64);
    serde_json::json!({
        "agentId": agent.0,
        "model": model,
        "tokens": tokens,
        "inputTokens": input,
        "outputTokens": output,
        "cachedInputTokens": count("cached_input_tokens"),
        "costUsd": cost,
        "tree": {"tokens": 0},
    })
}

fn durable(event: Event, source: &SourceKey) -> WorkerEvent {
    WorkerEvent {
        kind: event.kind().to_owned(),
        data: event.encode(),
        source: Some(source.clone()),
    }
}

/// The events a change means, read against the conversation before it.
fn change_events(agent: &AgentId, shadow: &Shadow, change: &ChangeRecord) -> Vec<Event> {
    match &change.change {
        Change::Push { entries } => entries
            .iter()
            .flat_map(|entry| {
                let message = text_of(&entry.message).map(|(role, text)| {
                    event::Message {
                        agent_id: agent.clone(),
                        message_id: entry.id.map(id_text).unwrap_or_default(),
                        role: role.to_owned(),
                        text,
                    }
                    .into()
                });
                message.into_iter().chain(call_events(agent, &entry.message))
            })
            .collect(),
        Change::Aside {
            aside: Some(aside), ..
        } => {
            let Some(turn) = TurnRecord::of(aside) else {
                return Vec::new();
            };
            let turn_id = turn.index.to_string();
            let Some(end) = turn.end else {
                return vec![
                    event::TurnStarted {
                        agent_id: agent.clone(),
                        turn_id,
                    }
                    .into(),
                ];
            };
            let (status, answer, error) = match end {
                TurnEnd::Answered { answer } => {
                    let text = answer
                        .and_then(|id| shadow.conversation.index_of(id))
                        .and_then(|at| shadow.conversation.messages().get(at).cloned())
                        .and_then(|message| message.final_answer());
                    (TurnStatus::Answered, text, None)
                }
                TurnEnd::Failed { why } => (TurnStatus::Failed, None, Some(why)),
                TurnEnd::Stopped { why } => (TurnStatus::Stopped, None, why),
            };
            vec![
                event::TurnEnded {
                    agent_id: agent.clone(),
                    turn_id,
                    status,
                    answer,
                    error,
                    through: CommandSeq(shadow.through),
                }
                .into(),
            ]
        }
        _ => Vec::new(),
    }
}

/// The calls a reply asks for, or the results a message answers them with.
fn call_events(agent: &AgentId, message: &Message) -> Vec<Event> {
    // A call's id is its `CallId` as serde JSON text, which a mirror reads back.
    let call_id = |id: &rig_core::message::CallId| serde_json::to_string(id).unwrap_or_default();
    match message {
        Message::Assistant(reply) => reply
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(
                    event::CallStarted {
                        agent_id: agent.clone(),
                        call_id: call_id(&call.id),
                        tool: call.function.name.to_string(),
                        args: serde_json::Value::Object(call.function.arguments.clone()),
                    }
                    .into(),
                ),
                _ => None,
            })
            .collect(),
        Message::User { content } => content
            .iter()
            .filter_map(|content| match content {
                UserContent::ToolResult(result) => {
                    let text: Vec<String> = result
                        .content
                        .iter()
                        .map(|part| match part {
                            rig_core::message::ToolResultContent::Text(text) => text.text.clone(),
                            rig_core::message::ToolResultContent::Json { value } => {
                                value.to_string()
                            }
                            rig_core::message::ToolResultContent::Image(_) => "[image]".into(),
                        })
                        .collect();
                    let text = text.join("\n");
                    Some(
                        event::CallEnded {
                            call_id: call_id(&result.call),
                            ok: !result.is_error,
                            result: serde_json::Value::String(text.clone()),
                            error: result.is_error.then_some(text),
                        }
                        .into(),
                    )
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn id_text(id: rig_core::conversation::MessageId) -> String {
    serde_json::to_value(id)
        .map(|v| v.to_string())
        .unwrap_or_default()
}

/// A message's role and its text, when it has text.
fn text_of(message: &Message) -> Option<(&'static str, String)> {
    let (role, parts): (&str, Vec<&str>) = match message {
        Message::User { content } => (
            "user",
            content
                .iter()
                .filter_map(|c| match c {
                    UserContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect(),
        ),
        Message::Assistant(reply) => (
            "assistant",
            reply
                .content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect(),
        ),
        _ => return None,
    };
    let text = parts.join("\n\n");
    (!text.trim().is_empty()).then_some((role, text))
}

/// The number a saved newtype such as [`Through`] holds.
pub(crate) fn seq_of(value: &serde_json::Value) -> Option<u64> {
    value.as_u64().or_else(|| value.get(0)?.as_u64())
}

/// The most a `worker/events` batch carries, well under the wire's frame
/// limit ([`rig_protocol::MAX_FRAME`]), leaving room for the request around
/// it and for JSON escapes.
const BATCH: usize = rig_protocol::MAX_FRAME / 4;

/// The longest string an event keeps when the event alone would not fit a
/// batch; the rest is cut, with a note saying so. The session's log keeps
/// it whole.
const KEPT: usize = BATCH / 4;

/// `events` in batches that each fit a frame, in order, the events of one
/// saved record together. An event too big for a batch of its own has its
/// long strings cut ([`cut`]). A record's events together stay under the
/// frame: each fits a quarter of it, and a record makes at most a few.
pub(crate) fn batches(events: Vec<WorkerEvent>) -> Vec<Vec<WorkerEvent>> {
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut size = 0;
    for mut event in events {
        let mut len = serde_json::to_vec(&event).map_or(0, |bytes| bytes.len());
        if len > BATCH {
            cut(&mut event.data);
            len = serde_json::to_vec(&event).map_or(0, |bytes| bytes.len());
        }
        // The events of one record go in one batch: `rigd` drops an event
        // whose record a batch before it journaled.
        let same_record = event.source.is_some()
            && batch.last().is_some_and(|last: &WorkerEvent| last.source == event.source);
        if !batch.is_empty() && size + len > BATCH && !same_record {
            batches.push(std::mem::take(&mut batch));
            size = 0;
        }
        size += len;
        batch.push(event);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

/// Cuts every string of `data` longer than [`KEPT`] bytes to that, ending
/// it with a note of how much was left out.
fn cut(data: &mut serde_json::Value) {
    match data {
        serde_json::Value::String(text) if text.len() > KEPT => {
            let at = (0..=KEPT)
                .rev()
                .find(|at| text.is_char_boundary(*at))
                .unwrap_or(0);
            let left = text.len() - at;
            text.truncate(at);
            text.push_str(&format!(
                "\n[cut: {left} more bytes; the session's log holds the whole text]"
            ));
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(cut),
        serde_json::Value::Object(fields) => fields.values_mut().for_each(cut),
        _ => {}
    }
}
