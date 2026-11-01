//! What the router and each session's process share: the answer to
//! `initialize`, and a response owed to the client.

use std::sync::{Mutex, PoisonError};

use agent_client_protocol::schema::{ProtocolVersion, v1 as acp};
use agent_client_protocol::{Error, JsonRpcResponse, Responder};
use tokio::sync::watch;

/// A response the app owes the client, answered once: with what it is sent,
/// else, once dropped, with what it was made to answer then, so no request
/// is left unanswered, however the session or the router ended.
pub(crate) struct Reply<T: JsonRpcResponse> {
    responder: Mutex<Option<Responder<T>>>,
    /// What it answers when dropped unanswered; behind a lock only so a
    /// reply shared between tasks stays `Sync`.
    unanswered: Mutex<Option<Result<T, Error>>>,
    _held: Option<Held>,
}

impl<T: JsonRpcResponse> Reply<T> {
    /// Owed until sent; dropped unanswered, an error says the session ended.
    pub(crate) fn new(responder: Responder<T>) -> Self {
        Self::owed(responder, Err(ended()), None)
    }

    /// Owed until sent, counted in `held` meanwhile; dropped unanswered, it
    /// answers `unanswered`.
    pub(crate) fn owed(
        responder: Responder<T>,
        unanswered: Result<T, Error>,
        held: Option<Held>,
    ) -> Self {
        Self {
            responder: Mutex::new(Some(responder)),
            unanswered: Mutex::new(Some(unanswered)),
            _held: held,
        }
    }

    /// Answers with `result`, unless answered already.
    pub(crate) fn send(&self, result: Result<T, Error>) {
        let mut slot = self
            .responder
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(responder) = slot.take() {
            responder.respond_with_result(result).ok();
        }
    }
}

impl<T: JsonRpcResponse> Drop for Reply<T> {
    fn drop(&mut self) {
        let unanswered = self
            .unanswered
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        let unanswered = unanswered.take().unwrap_or_else(|| Err(ended()));
        self.send(unanswered);
    }
}

/// Why a request was answered as its session or the router ended.
pub(crate) fn ended() -> Error {
    error(
        Error::internal_error(),
        "the session ended before it answered",
    )
}

/// How many requests are owed, each counted by a [`Held`] while it is.
#[derive(Clone)]
pub(crate) struct Pending(watch::Sender<usize>);

impl Pending {
    pub(crate) fn new() -> Self {
        Self(watch::Sender::new(0))
    }

    /// Counts one more until the [`Held`] is dropped.
    pub(crate) fn hold(&self) -> Held {
        self.0.send_modify(|count| *count += 1);
        Held(self.0.clone())
    }

    /// Whether none is owed.
    pub(crate) fn is_none(&self) -> bool {
        *self.0.borrow() == 0
    }

    /// Done once none is owed.
    pub(crate) async fn none(&self) {
        self.0.subscribe().wait_for(|count| *count == 0).await.ok();
    }
}

/// One owed request of a [`Pending`].
pub(crate) struct Held(watch::Sender<usize>);

impl Drop for Held {
    fn drop(&mut self) {
        self.0.send_modify(|count| *count = count.saturating_sub(1));
    }
}

/// The answer to `initialize`, of the router and of each session's
/// process alike: ACP v1, whatever version the client asked for.
pub(crate) fn initialized() -> acp::InitializeResponse {
    let prompts = acp::PromptCapabilities::new()
        .image(true)
        .embedded_context(true);
    let sessions = acp::SessionCapabilities::new()
        .list(acp::SessionListCapabilities::new())
        .resume(acp::SessionResumeCapabilities::new())
        .close(acp::SessionCloseCapabilities::new());
    let capabilities = acp::AgentCapabilities::new()
        .load_session(true)
        .prompt_capabilities(prompts)
        .mcp_capabilities(acp::McpCapabilities::new().http(true))
        .session_capabilities(sessions);
    acp::InitializeResponse::new(ProtocolVersion::V1)
        .agent_capabilities(capabilities)
        .agent_info(acp::Implementation::new("rig", env!("CARGO_PKG_VERSION")))
}

/// An error of `kind` that says `why` as its message, which a client shows
/// as it is.
pub(crate) fn error(mut kind: Error, why: impl ToString) -> Error {
    kind.message = why.to_string();
    kind
}
