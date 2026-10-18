//! The Rust client of the rig agent server (`rigd`). [`Client::connect`]
//! reaches the local server, starting it when none answers, or a remote one
//! through its `[remote.hosts]` transport; [`Client::call`] calls a typed
//! [`Method`]; [`Client::subscribe`] follows a session's events, resuming
//! across reconnects without gaps or repeats. It is sync, on std threads,
//! with no Bevy and no async runtime.

use std::path::PathBuf;

use rig_protocol::{EventRecord, Exit, Method, RpcError, Seq, SessionId, Side};

/// Where the server is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// The local `rigd` of this rig home, started when none answers.
    Local { rig_home: PathBuf },
    /// The `rigd` of a `[remote.hosts]` entry, reached through its transport.
    Remote { host: String },
}

/// A connection to `rigd` that has said hello.
pub struct Client {
    _connection: (),
}

impl Client {
    /// Connects to `target` as client `name`, says hello and checks versions.
    pub fn connect(target: &Target, name: &str) -> Result<Client, ClientError> {
        let _ = (target, name);
        Err(not_built())
    }

    /// Calls `M` and waits for its answer.
    pub fn call<M: Method>(&self, params: &M::Params) -> Result<M::Result, ClientError> {
        let _ = params;
        Err(not_built())
    }

    /// `session`'s events after `after`, then live ones.
    pub fn subscribe(&self, session: &SessionId, after: Seq) -> Result<Events, ClientError> {
        let _ = (session, after);
        Err(not_built())
    }
}

fn not_built() -> ClientError {
    ClientError::Unavailable {
        next: "not built yet".into(),
    }
}

/// A session's events in `seq` order, each once.
pub struct Events {
    last: Seq,
}

impl Events {
    /// The `seq` of the last event this iterator yielded.
    pub fn last_seq(&self) -> Seq {
        self.last
    }
}

impl Iterator for Events {
    type Item = Result<EventRecord, ClientError>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// No server could be reached or started; `next` says what to run.
    #[error("the rig server is unavailable")]
    Unavailable { next: String },
    /// The two sides share no wire version; `update` must be updated.
    #[error("no common rig wire version: update the {update:?} (it speaks {supported:?})")]
    Version { update: Side, supported: Vec<u32> },
    #[error(transparent)]
    Rpc(RpcError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl ClientError {
    /// The exit code a command ends with on this error.
    pub fn exit(&self) -> Exit {
        match self {
            ClientError::Unavailable { .. } | ClientError::Version { .. } | ClientError::Io(_) => {
                Exit::Unavailable
            }
            ClientError::Rpc(e) => match e.code {
                rig_protocol::codes::UNAUTHORIZED
                | rig_protocol::codes::ALREADY_RESOLVED
                | rig_protocol::codes::GRANT_WIDER
                | rig_protocol::codes::SANDBOX_UNAVAILABLE => Exit::NotAllowed,
                rig_protocol::codes::INVALID_PARAMS
                | rig_protocol::codes::METHOD_NOT_FOUND
                | rig_protocol::codes::NOT_FOUND => Exit::Usage,
                _ => Exit::Failed,
            },
        }
    }

    /// The command that fixes it, when one is known.
    pub fn next(&self) -> Option<&str> {
        match self {
            ClientError::Unavailable { next } => Some(next),
            ClientError::Rpc(e) => e.data.as_ref().and_then(|d| d.next.as_deref()),
            ClientError::Version { .. } | ClientError::Io(_) => None,
        }
    }
}
