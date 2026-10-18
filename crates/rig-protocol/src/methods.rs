//! Every method of the wire, one [`Method`] each. A method's type is its
//! params; its `Result` type is what it answers. Mutating methods carry a
//! [`Key`]. Field names are camelCase on the wire.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::event::{AskOpened, State, UsageTotals};
use crate::{
    AskId, BuildId, ClientId, CommandSeq, EventRecord, Grant, Key, Method, Outcome, Policy, Seq,
    SessionId, SourceKey,
};

macro_rules! method {
    ($name:literal, $params:ident => $result:ty) => {
        impl Method for $params {
            const NAME: &'static str = $name;
            type Params = $params;
            type Result = $result;
        }
    };
}

/// An empty result, `{}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Empty {}

/// The notification carrying one [`EventRecord`].
pub const EVENT: &str = "rig/event";
/// The notification of a subscription cancelled for exceeding its queue.
pub const LAGGED: &str = "rig/lagged";
/// The notification carrying a [`WorkerCommand`] from `rigd` to a worker.
pub const WORKER_COMMAND: &str = "worker/command";

/// The params of the [`EVENT`] notification: the subscription it belongs
/// to and the record. A connection may hold several subscriptions on one
/// session, each with its own replay, so a client routes by `subscription`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventNotice {
    pub subscription: SubscriptionId,
    #[serde(flatten)]
    pub record: EventRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lagged {
    pub session_id: SessionId,
    pub next_seq: Seq,
}

// rig/hello

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hello {
    pub token: String,
    pub client: ClientInfo,
    pub rig: VersionRange,
}
method!("rig/hello", Hello => HelloResult);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VersionRange {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HelloResult {
    pub rig: u32,
    pub server: ServerInfo,
    pub client_id: ClientId,
    pub methods: Vec<MethodInfo>,
    pub events: Vec<EventInfo>,
    pub limits: Limits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub version: String,
    pub pid: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodInfo {
    pub name: String,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventInfo {
    pub kind: String,
    pub version: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Limits {
    pub frame: usize,
}

// rig/session/start

/// `cwd` is required. A worker caller's session is the implicit parent;
/// `build` is honoured for owner callers only.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Start {
    pub cwd: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<BuildId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant: Option<Grant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<Policy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<Budget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Context>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unload: Option<Unload>,
    #[serde(default)]
    pub view: bool,
    pub key: Key,
}
method!("rig/session/start", Start => Started);

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Budget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// What a parent hands a child: its environment facts and instructions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Context {
    pub facts: Vec<String>,
}

/// When the worker unloads: after `session.idle_unload` or after each turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unload {
    Idle,
    Turn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Started {
    pub session_id: SessionId,
    pub seq: Seq,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_seq: Option<CommandSeq>,
}

// rig/session/send

/// `text` or `content` (ACP content blocks). A `/cmd` line runs a slash command.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Send {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<SendMode>,
    pub key: Key,
}
method!("rig/session/send", Send => Sent);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SendMode {
    Steer,
    Queue,
    Interrupt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Started,
    Steered,
    Queued,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sent {
    pub command_seq: CommandSeq,
    pub disposition: Disposition,
}

// rig/session/wait

/// `afterCommand` defaults to the latest command accepted when the call arrives.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Wait {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_command: Option<CommandSeq>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}
method!("rig/session/wait", Wait => WaitResult);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitStatus {
    Answered,
    NeedsApproval,
    NeedsAnswer,
    Failed,
    Stopped,
    TimedOut,
}

impl WaitStatus {
    /// The exit code of a command that waited for this.
    pub fn exit(self) -> crate::Exit {
        match self {
            WaitStatus::Answered => crate::Exit::Answered,
            WaitStatus::NeedsApproval => crate::Exit::NeedsApproval,
            WaitStatus::NeedsAnswer => crate::Exit::NeedsAnswer,
            WaitStatus::Failed => crate::Exit::Failed,
            WaitStatus::Stopped => crate::Exit::Stopped,
            WaitStatus::TimedOut => crate::Exit::TimedOut,
        }
    }
}

/// An open ask, with the session it belongs to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AskInfo {
    pub session_id: SessionId,
    #[serde(flatten)]
    pub ask: AskOpened,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WaitResult {
    pub status: WaitStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask: Option<AskInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub seq: Seq,
    pub command_seq: CommandSeq,
}

// rig/session/answer

/// `optionId` or `text`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Answer {
    pub session_id: SessionId,
    pub ask_id: AskId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub option_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub key: Key,
}
method!("rig/session/answer", Answer => Resolved);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolved {
    pub resolved: bool,
}

// rig/session/stop

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Stop {
    pub session_id: SessionId,
    pub scope: StopScope,
    pub key: Key,
}
method!("rig/session/stop", Stop => Stopping);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopScope {
    Turn,
    Session,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stopping {
    pub stopping: Vec<SessionId>,
}

// rig/session/list, rig/session/read

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct List {
    #[serde(default)]
    pub tree: bool,
    #[serde(default)]
    pub all: bool,
}
method!("rig/session/list", List => Vec<SessionInfo>);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub session_id: SessionId,
    pub name: Option<String>,
    pub cwd: PathBuf,
    pub host: Option<String>,
    pub state: State,
    pub parent: Option<SessionId>,
    pub children: Vec<SessionId>,
    pub last_seq: Seq,
    pub usage: UsageTotals,
    pub worker: bool,
}

/// Answered from `rigd`'s state; never wakes a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Read {
    pub session_id: SessionId,
}
method!("rig/session/read", Read => ReadResult);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadResult {
    pub info: SessionInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_answer: Option<String>,
    pub open_asks: Vec<AskInfo>,
    pub last_seq: Seq,
}

// rig/events/*

/// A subscription's id, per connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubscriptionId(pub u64);

/// [`EVENT`] notifications follow the response: the journal after
/// `afterSeq`, then live when `follow`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Subscribe {
    pub session_id: SessionId,
    pub after_seq: Seq,
    pub follow: bool,
}
method!("rig/events/subscribe", Subscribe => Subscribed);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Subscribed {
    pub subscription: SubscriptionId,
    pub next_seq: Seq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unsubscribe {
    pub subscription: SubscriptionId,
}
method!("rig/events/unsubscribe", Unsubscribe => Empty);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadEvents {
    pub session_id: SessionId,
    pub after_seq: Seq,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}
method!("rig/events/read", ReadEvents => EventsRead);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventsRead {
    pub events: Vec<EventRecord>,
    pub next_seq: Seq,
}

// rig/session/attach, detach, reload

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attach {
    pub session_id: SessionId,
    pub role: AttachRole,
    #[serde(default)]
    pub takeover: bool,
}
method!("rig/session/attach", Attach => Attached);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachRole {
    Viewer,
    Driver,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attached {
    pub driver: Option<ClientId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Detach {
    pub session_id: SessionId,
}
method!("rig/session/detach", Detach => Empty);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reload {
    pub session_id: SessionId,
    pub build: BuildId,
    pub key: Key,
}
method!("rig/session/reload", Reload => Reloading);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reloading {
    pub command_seq: CommandSeq,
}

// The worker channel: callable only by a worker's own connection.

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerHello {
    pub rig: u32,
    pub session_id: SessionId,
    pub build: BuildId,
    pub applied: CommandSeq,
    pub logs: Vec<LogLen>,
}
method!("worker/hello", WorkerHello => WorkerWelcome);

/// A saved log of the worker's session and how many records it holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogLen {
    pub log: String,
    pub len: u64,
}

/// `knownThrough`: the last journaled record index per log, so the worker
/// re-exports everything after it. `replay`: commands after `applied`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerWelcome {
    pub known_through: BTreeMap<String, u64>,
    pub replay: Vec<WorkerCommand>,
}

/// One batch of events, coalesced per worker frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEvents {
    pub events: Vec<WorkerEvent>,
}
method!("worker/events", WorkerEvents => Empty);

/// An event before `rigd` journals it; `source` is set on durable kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEvent {
    pub kind: String,
    pub data: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerApplied {
    pub through: CommandSeq,
}
method!("worker/applied", WorkerApplied => Empty);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerQuiescent {
    pub quiescent: bool,
}
method!("worker/quiescent", WorkerQuiescent => Empty);

/// The worker's answer to a [`Command::Plugin`] call: `result` or `error`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerReply {
    pub reply: u64,
    #[serde(flatten)]
    pub outcome: Outcome,
}
method!("worker/reply", WorkerReply => Empty);

/// The params of the [`WORKER_COMMAND`] notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerCommand {
    pub command_seq: CommandSeq,
    pub command: Command,
}

/// What `rigd` forwards to a worker. `Plugin` is an `x/<plugin>/<verb>` call
/// answered with `worker/reply` carrying `reply`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Send(Send),
    Stop(Stop),
    Answer(Answer),
    Plugin {
        method: String,
        params: serde_json::Value,
        reply: u64,
    },
    Reload(Reload),
}
