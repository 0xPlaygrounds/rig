//! Error types for `rig-a2a` operations.

use rig_core::completion::CompletionError;

/// Errors that occur while resolving or validating a remote
/// [`AgentCard`](a2a::AgentCard).
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum AgentCardError {
    /// The builder was finalized without setting a source (URL or card).
    #[error("A2AClientBuilder requires either .url(...) or .card(...) before .build()")]
    MissingSource,

    /// The well-known agent-card endpoint returned a non-2xx status. The
    /// `body` snippet (truncated to 512 chars) often carries a JSON
    /// `{"error":"..."}` payload with the underlying reason; surfacing it
    /// is essential for diagnosing tenant-auth and configuration failures
    /// that would otherwise show up as bare HTTP status codes.
    #[error("failed to fetch agent card: HTTP {status}{}", body.as_deref().map(|b| format!(" — {b}")).unwrap_or_default())]
    FetchFailed {
        status: reqwest::StatusCode,
        body: Option<String>,
    },

    /// The agent-card response exceeded the client-side safety limit.
    #[error("agent card response exceeded {limit} bytes")]
    ResponseTooLarge { limit: usize },

    /// The card does not advertise a JSON-RPC or HTTP+JSON interface for the
    /// A2A protocol version implemented by this client.
    #[error("agent card does not advertise a supported A2A interface")]
    NoSupportedInterface,

    /// A selected HTTP interface is not an absolute HTTP(S) URL.
    #[error("agent card interface URL {interface_url} must be an absolute HTTP(S) URL")]
    InvalidInterfaceUrl { interface_url: String },

    /// A fetched card advertised an interface outside the card origin.
    #[error(
        "agent card interface URL {interface_url} is outside fetched card origin {base_origin}"
    )]
    CrossOriginInterface {
        base_origin: String,
        interface_url: String,
    },
}

/// Errors returned by `rig-a2a` operations.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum A2AError {
    /// An error returned by the underlying A2A protocol stack.
    #[error("A2A protocol error: {0}")]
    Protocol(#[from] a2a::A2AError),

    /// An HTTP-level error while fetching an `AgentCard` or talking to the
    /// transport layer.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// An error resolving or validating a remote
    /// [`AgentCard`](a2a::AgentCard).
    #[error("AgentCard error: {0}")]
    AgentCard(#[from] AgentCardError),

    /// A caller supplied an invalid A2A conversation context id.
    #[error("A2A contextId must be a non-empty string")]
    InvalidContextId,

    /// A caller supplied an invalid A2A task id.
    #[error("A2A taskId must be a non-empty string")]
    InvalidTaskId,

    /// A caller supplied a malformed `a2a` block in `additional_params`.
    #[error("`additional_params.a2a` {detail}")]
    InvalidThreadParams { detail: &'static str },

    /// A payload exceeded a client-side projection limit.
    #[error("A2A payload exceeded {limit} bytes while projecting {what}")]
    PayloadTooLarge { what: &'static str, limit: usize },

    /// A completion request used a feature the A2A protocol cannot express.
    ///
    /// Returned by [`A2AModel`](crate::A2AModel) rather than silently dropping
    /// the field, because dropping it would change the result without saying
    /// so. See the [`model`](crate::model) module docs for the full table.
    #[error("A2A cannot honor `{what}`: {detail}")]
    Unsupported {
        what: &'static str,
        detail: &'static str,
    },

    /// A completion request projected to a message with no content to send.
    #[error("completion request produced no text to send to the remote A2A agent")]
    EmptyRequest,

    /// A serde / JSON error converting between Rig and A2A representations.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A URL could not be parsed.
    #[error("invalid URL: {0}")]
    Url(#[from] url::ParseError),
}

impl From<A2AError> for CompletionError {
    fn from(err: A2AError) -> Self {
        match err {
            // A request Rig built but A2A cannot express is a request-side
            // fault, not an upstream failure — keep the two distinguishable.
            err @ (A2AError::Unsupported { .. }
            | A2AError::EmptyRequest
            | A2AError::InvalidThreadParams { .. }
            | A2AError::InvalidContextId
            | A2AError::InvalidTaskId) => CompletionError::RequestError(Box::new(err)),
            err => CompletionError::ProviderError(err.to_string()),
        }
    }
}
