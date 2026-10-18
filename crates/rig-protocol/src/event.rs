//! The events a session's journal holds: [`EventRecord`] on the wire and the
//! typed [`Event`] of each built-in kind. A kind this build doesn't know (a
//! plugin's `x/<plugin>/<name>`, or a newer server's) stays an
//! [`EventRecord`]; it is never an error.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    AgentId, Answerers, AskId, BuildId, ClientId, CommandSeq, Grant, Key, Seq, SessionId, SourceKey,
};

/// One journaled event on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub session: SessionId,
    pub seq: Seq,
    pub at_ms: u64,
    pub kind: String,
    pub data: serde_json::Value,
    /// Where a durable worker event came from in the worker's saved log:
    /// a restarted `rigd` rebuilds its re-export dedupe from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceKey>,
}

/// A session's lifecycle state, as `rig/session/list` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Idle,
    Running,
    NeedsApproval,
    NeedsAnswer,
    Stopping,
    Stopped,
    Faulted,
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Answered,
    Failed,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AskKind {
    Approval,
    Question,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStarted {
    pub cwd: PathBuf,
    pub name: Option<String>,
    pub host: Option<String>,
    pub parent: Option<SessionId>,
    pub build: BuildId,
    pub grant: Option<Grant>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandAccepted {
    pub command_seq: CommandSeq,
    pub key: Key,
    pub method: String,
    pub by: ClientId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// The accepted command itself, so a restarted `rigd` can replay it
    /// and answer a repeated key from the journal alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<crate::methods::Command>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerReady {
    pub pid: u32,
    pub build: BuildId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerExited {
    pub code: Option<i32>,
    pub signal: Option<i32>,
    pub restarting: bool,
    pub log: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnStarted {
    pub agent_id: AgentId,
    pub turn_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageDelta {
    pub agent_id: AgentId,
    pub message_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// A message's authoritative final text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub agent_id: AgentId,
    pub message_id: String,
    pub role: String,
    pub text: String,
}

/// One saved conversation change; `record` is a
/// `rig_core::conversation::ChangeRecord` as JSON, for mirrors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationChange {
    pub agent_id: AgentId,
    pub log: String,
    pub index: u64,
    pub record: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallStarted {
    pub agent_id: AgentId,
    pub call_id: String,
    pub tool: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallProgress {
    pub call_id: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallEnded {
    pub call_id: String,
    pub ok: bool,
    pub result: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AskOption {
    pub option_id: String,
    pub label: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AskOpened {
    pub ask_id: AskId,
    pub agent_id: AgentId,
    pub agent_name: String,
    pub kind: AskKind,
    pub title: String,
    pub body: String,
    pub options: Vec<AskOption>,
    pub deadline_ms: u64,
    pub answerers: Answerers,
}

/// `by` is `"client:<id>"`, `"rule"`, `"timeout"` or `"parent"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AskResolved {
    pub ask_id: AskId,
    pub by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub option_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// `through` is the last command the turn answered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnEnded {
    pub agent_id: AgentId,
    pub turn_id: String,
    pub status: TurnStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub through: CommandSeq,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageTotals {
    pub tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// A session's usage, with its tree's running total.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    pub tree: UsageTotals,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Notice {
    pub level: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusItem {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub items: Vec<StatusItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildStarted {
    pub session_id: SessionId,
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildEnded {
    pub session_id: SessionId,
    pub status: State,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DriverChanged {
    pub driver: Option<ClientId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BudgetExhausted {
    pub tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReloadStarted {
    pub build: BuildId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReloadDone {
    pub build: BuildId,
}

/// `kept` is the build the session went on running.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReloadFailed {
    pub build: BuildId,
    pub error: String,
    pub kept: BuildId,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SessionIdle {}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SessionStopped {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionFaulted {
    pub reason: String,
    pub log: Option<PathBuf>,
}

macro_rules! events {
    ($($kind:literal => $variant:ident,)*) => {
        /// The typed view of a built-in kind.
        #[derive(Debug, Clone, PartialEq)]
        pub enum Event {
            $($variant($variant),)*
        }

        impl Event {
            /// Every built-in kind.
            pub const KINDS: &'static [&'static str] = &[$($kind),*];

            /// The typed event of `r`, or `None` for an unknown kind or a
            /// `data` that doesn't fit its kind.
            pub fn decode(r: &EventRecord) -> Option<Event> {
                match r.kind.as_str() {
                    $($kind => serde_json::from_value(r.data.clone()).ok().map(Event::$variant),)*
                    _ => None,
                }
            }

            pub fn kind(&self) -> &'static str {
                match self {
                    $(Event::$variant(_) => $kind,)*
                }
            }

            /// The record's `data`.
            pub fn encode(&self) -> serde_json::Value {
                match self {
                    $(Event::$variant(data) => serde_json::to_value(data).unwrap_or_default(),)*
                }
            }
        }

        $(impl From<$variant> for Event {
            fn from(data: $variant) -> Event {
                Event::$variant(data)
            }
        })*
    };
}

events! {
    "session.started" => SessionStarted,
    "command.accepted" => CommandAccepted,
    "worker.ready" => WorkerReady,
    "worker.exited" => WorkerExited,
    "turn.started" => TurnStarted,
    "message.delta" => MessageDelta,
    "message" => Message,
    "conversation.change" => ConversationChange,
    "call.started" => CallStarted,
    "call.progress" => CallProgress,
    "call.ended" => CallEnded,
    "ask.opened" => AskOpened,
    "ask.resolved" => AskResolved,
    "turn.ended" => TurnEnded,
    "usage" => Usage,
    "notice" => Notice,
    "status" => Status,
    "child.started" => ChildStarted,
    "child.ended" => ChildEnded,
    "driver.changed" => DriverChanged,
    "budget.exhausted" => BudgetExhausted,
    "reload.started" => ReloadStarted,
    "reload.done" => ReloadDone,
    "reload.failed" => ReloadFailed,
    "session.idle" => SessionIdle,
    "session.stopped" => SessionStopped,
    "session.faulted" => SessionFaulted,
}
