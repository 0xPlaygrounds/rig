//! Success-or-error envelope classification for OpenAI-style JSON responses.
//!
//! Several providers wrap 2xx bodies in an untagged `Ok(payload) | Err(error)`
//! enum and only use the decoded error for logging — the raw body is what gets
//! preserved on the returned error. [`ProviderEnvelope`] abstracts over each
//! provider's private envelope type so the shared request drivers in this
//! module tree can classify responses without changing how any provider
//! deserializes its own error shape.

/// Error envelope returned by OpenAI-style providers alongside 2xx statuses.
///
/// Providers spell the message field differently (`message`, `error`, nested
/// objects such as `{"error": {"message": ...}}`), so anything that isn't a
/// valid success payload is treated as an error envelope and the raw body is
/// preserved for the caller; `message` is only used for logging.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct ApiErrorResponse {
    #[serde(
        default,
        alias = "error",
        deserialize_with = "crate::providers::internal::envelope::error_message_or_value"
    )]
    pub(crate) message: String,
}

/// Accept either a plain string message or any other JSON shape (nested error
/// objects, arrays), stringifying the latter so the envelope never fails to
/// decode on an unexpected error spelling.
pub(crate) fn error_message_or_value<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::String(message) => message,
        other => other.to_string(),
    })
}

/// A decoded provider response envelope: either the success payload or the
/// provider's error message.
///
/// The error message is used only for logging; callers preserve the raw
/// response body via `from_http_response` when the envelope is an error.
pub(crate) trait ProviderEnvelope {
    /// The success payload carried by the envelope.
    type Payload;

    /// Split the envelope into its payload or the provider's error message.
    fn into_payload(self) -> Result<Self::Payload, String>;
}

impl<T> ProviderEnvelope for crate::providers::openai::client::ApiResponse<T> {
    type Payload = T;

    fn into_payload(self) -> Result<T, String> {
        match self {
            Self::Ok(value) => Ok(value),
            Self::Err(error) => Err(error.message),
        }
    }
}
