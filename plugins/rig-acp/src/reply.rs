//! What the router and each session's process share: the answer to
//! `initialize`, and a response owed to the client.

use std::sync::Mutex;

use agent_client_protocol::schema::{ProtocolVersion, v1 as acp};
use agent_client_protocol::{Error, JsonRpcResponse, Responder};

/// A response the app owes the client, answered once.
pub(crate) struct Reply<T: JsonRpcResponse>(Mutex<Option<Responder<T>>>);

impl<T: JsonRpcResponse> Reply<T> {
    pub(crate) fn new(responder: Responder<T>) -> Self {
        Self(Mutex::new(Some(responder)))
    }

    /// Answers with `result`, unless answered already.
    pub(crate) fn send(&self, result: Result<T, Error>) {
        let taken = self.0.lock().ok().and_then(|mut slot| slot.take());
        if let Some(responder) = taken {
            responder.respond_with_result(result).ok();
        }
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
