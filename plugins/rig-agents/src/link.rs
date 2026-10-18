//! The connection the helper tools drive children through: a
//! [`rig_client::Client`] to `rigd` authenticated as this worker's session,
//! the same channel and methods an outside driver uses. Its principal limits
//! it to this session and its descendants.

use std::sync::Arc;

use rig_client::{Client, ClientError, Events};
use rig_harness::prelude::*;
use rig_protocol::{EventRecord, Grant, Method, RpcError, Seq, SessionId, codes};

/// What the tools need of a channel to `rigd`: calls, and a session's
/// events from a `seq` on, read without blocking.
pub trait Channel: Send + Sync + 'static {
    /// Calls `method` and waits for its answer. Blocks: run it off the
    /// main thread.
    fn call_raw(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, RpcError>;

    /// `session`'s events after `after`, then live ones, each once.
    fn subscribe(&self, session: &SessionId, after: Seq) -> Result<Box<dyn Feed>, RpcError>;
}

/// A session's events, read once a frame.
pub trait Feed: Send + Sync + 'static {
    /// The next event if one is here, without blocking.
    fn try_next(&mut self) -> Option<Result<EventRecord, RpcError>>;
}

/// [`Channel::call_raw`] for a typed method.
pub fn call<M: Method>(channel: &dyn Channel, params: &M::Params) -> Result<M::Result, RpcError> {
    let params = serde_json::to_value(params)
        .map_err(|e| RpcError::new(codes::INVALID_PARAMS, e.to_string()))?;
    let value = channel.call_raw(M::NAME, params)?;
    serde_json::from_value(value).map_err(|e| {
        RpcError::new(
            codes::INTERNAL,
            format!("the answer to {} does not parse: {e}", M::NAME),
        )
    })
}

fn rpc(error: ClientError) -> RpcError {
    match error {
        ClientError::Rpc(error) => error,
        other => {
            let next = other.next().map(str::to_owned);
            let error = RpcError::new(codes::INTERNAL, other.to_string());
            match next {
                Some(next) => error.with_next(next),
                None => error,
            }
        }
    }
}

impl Channel for Client {
    fn call_raw(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, RpcError> {
        Client::call_raw(self, method, params).map_err(rpc)
    }

    fn subscribe(&self, session: &SessionId, after: Seq) -> Result<Box<dyn Feed>, RpcError> {
        Ok(Box::new(
            Client::subscribe(self, session, after).map_err(rpc)?,
        ))
    }
}

impl Feed for Events {
    fn try_next(&mut self) -> Option<Result<EventRecord, RpcError>> {
        Events::try_next(self).map(|r| r.map_err(rpc))
    }
}

/// This session's link to `rigd`, and who it is there.
#[derive(Resource, Clone)]
pub struct AgentsLink {
    pub channel: Arc<dyn Channel>,
    /// This session.
    pub me: SessionId,
    /// Where this session works: a helper's default `cwd`.
    pub cwd: std::path::PathBuf,
    /// The remote host this session runs on, if any.
    pub host: Option<String>,
    /// What this session may use (`None`: everything, an owner session).
    pub grant: Option<Grant>,
}

impl AgentsLink {
    /// The link of session `me` over `channel`, reading where it works and
    /// its grant from its `session.started` event.
    pub fn new(channel: Arc<dyn Channel>, me: SessionId) -> Result<AgentsLink, RpcError> {
        use rig_protocol::event::{Event, SessionStarted};
        use rig_protocol::methods::ReadEvents;
        let first = call::<ReadEvents>(
            channel.as_ref(),
            &ReadEvents {
                session_id: me.clone(),
                after_seq: Seq(0),
                limit: Some(1),
            },
        )?;
        let started = first.events.iter().find_map(|r| match Event::decode(r) {
            Some(Event::SessionStarted(started)) => Some(started),
            _ => None,
        });
        let SessionStarted {
            cwd, host, grant, ..
        } = started.ok_or_else(|| {
            RpcError::new(codes::NOT_FOUND, "this session's journal has no start")
        })?;
        Ok(AgentsLink {
            channel,
            me,
            cwd,
            host,
            grant,
        })
    }

    /// Whether this session is unrestricted (an owner session).
    pub fn unrestricted(&self) -> bool {
        self.grant.as_ref().is_none_or(|g| g.dirs.is_none())
    }
}
