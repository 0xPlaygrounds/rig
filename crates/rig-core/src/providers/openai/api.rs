//! Shared OpenAI-compatible API envelope types.
//!
//! Every OpenAI-compatible provider's `functions` module deserializes success
//! bodies through `ApiResponse`, which tolerates the error envelopes some
//! providers return alongside a 2xx status. These types used to live in the
//! deleted `openai::client` module; they are pure wire data and belong with the
//! rest of the wire layer.

use serde::Deserialize;

/// Error envelope returned by OpenAI-compatible providers alongside 2xx
/// statuses. Providers spell the message field differently (`message`,
/// `error`, nested objects), so anything that isn't a valid success payload
/// is treated as an error envelope and the raw body is preserved for the
/// caller; `message` is only used for logging.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    #[serde(default, alias = "error", deserialize_with = "error_message_or_value")]
    pub(crate) message: String,
}

fn error_message_or_value<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::String(message) => message,
        other => other.to_string(),
    })
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
