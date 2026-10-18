#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! The wire every rig front speaks with the rig server (`rigd`): JSON-RPC 2.0
//! frames, one per line (NDJSON), each carrying `"rig": 1`. It holds the ids,
//! the frames and their reading and writing, the typed [`methods`], the
//! journaled [`event`]s, least-privilege [`Grant`]s, the exit codes every
//! waiting command shares, and the local transport. It has no Bevy and no
//! async runtime.
//!
//! ```
//! use rig_protocol::{Frame, Request, RequestId, read_frame, write_frame};
//!
//! let mut wire = Vec::new();
//! let request = Request::new(RequestId::Num(1), "rig/session/list", serde_json::json!({}));
//! write_frame(&mut wire, &Frame::Request(request))?;
//! let mut line = Vec::new();
//! let frame = read_frame(&mut wire.as_slice(), &mut line)?;
//! assert!(matches!(frame, Some(Frame::Request(_))));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod event;
mod frame;
mod grant;
pub mod methods;
mod transport;
pub mod upgrade;

pub use event::{Event, EventRecord};
pub use frame::{
    ErrorData, Frame, FrameError, JsonRpc, Notification, Outcome, Request, RequestId, Response,
    RpcError, Side, read_frame, write_frame,
};
pub use grant::{Answerers, DirGrant, Grant, Policy, Rule, TimeoutOutcome, Wider};
#[cfg(unix)]
pub use transport::{Conn, Listener, Peer};
pub use transport::{Endpoint, read_token};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The rig wire major this build speaks.
pub const PROTOCOL: u32 = 1;
/// Every wire major this build accepts.
pub const SUPPORTED: core::ops::RangeInclusive<u32> = 1..=1;
/// The longest frame, in bytes, without its line ending.
pub const MAX_FRAME: usize = 1 << 20;

/// A session's id, in the session id format of `rig-harness-protocol`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SessionId(pub String);
/// A connection's id, given by `rig/hello`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClientId(pub String);
/// An open approval or question.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AskId(pub String);
/// An agent inside a session.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);
/// A client's idempotency key, required on every mutating method.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Key(pub String);
/// An immutable agent build, `$RIG_HOME/bin/builds/<BuildId>/rig-agent`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BuildId(pub String);
/// A per-session event position: starts at 1, never reused, below 2^53.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct Seq(pub u64);
/// A per-session accepted-command position.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct CommandSeq(pub u64);
/// The dedupe key of a durable event a worker exports from its saved log:
/// the log's name and the record's index in it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceKey {
    pub log: String,
    pub index: u64,
}

/// The error codes of the wire, beside JSON-RPC's own.
pub mod codes {
    pub const PARSE: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL: i32 = -32603;
    /// `data.update` names the side to update, `data.supported` lists versions.
    pub const VERSION: i32 = -32010;
    /// A bad or missing token, a wrong uid, or a principal that may not call this.
    pub const UNAUTHORIZED: i32 = -32011;
    /// An unknown session, ask or agent id.
    pub const NOT_FOUND: i32 = -32012;
    /// A key reused with a different command.
    pub const KEY_CONFLICT: i32 = -32013;
    /// An ask answered or expired already (`data.by`, `data.choice`).
    pub const ALREADY_RESOLVED: i32 = -32014;
    /// A child asked for more than its parent has.
    pub const GRANT_WIDER: i32 = -32015;
    /// The inbound queue is full; retry with the same key.
    pub const OVERLOADED: i32 = -32016;
    /// A narrower grant this platform can't enforce.
    pub const SANDBOX_UNAVAILABLE: i32 = -32017;
    /// A worker could not start or crashed repeatedly (`data.log`).
    pub const WORKER_FAILED: i32 = -32018;
    /// The children or depth limit; `data.next` says what to stop.
    pub const LIMIT: i32 = -32019;
    pub const FRAME_TOO_LARGE: i32 = -32020;
}

/// A typed method: lanes register and call through these, never through
/// string literals.
pub trait Method {
    const NAME: &'static str;
    type Params: Serialize + DeserializeOwned + Send + 'static;
    type Result: Serialize + DeserializeOwned + Send + 'static;
}

/// The exit codes every waiting command uses (the CLI verbs, `rig -p`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    Answered = 0,
    Failed = 1,
    Usage = 2,
    NeedsApproval = 3,
    NeedsAnswer = 4,
    Stopped = 5,
    NotAllowed = 6,
    Unavailable = 7,
    TimedOut = 124,
    Interrupted = 130,
}

impl Exit {
    /// The process exit code.
    pub fn code(self) -> u8 {
        self as u8
    }
}

#[cfg(test)]
mod tests;
