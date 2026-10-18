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
            .filter_map(|entry| {
                let (role, text) = text_of(&entry.message)?;
                Some(
                    event::Message {
                        agent_id: agent.clone(),
                        message_id: entry.id.map(id_text).unwrap_or_default(),
                        role: role.to_owned(),
                        text,
                    }
                    .into(),
                )
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
