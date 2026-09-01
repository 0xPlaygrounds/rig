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
#[derive(Debug)]
pub(crate) struct ApiErrorResponse {
    pub(crate) message: String,
}

impl<'de> serde::Deserialize<'de> for ApiErrorResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            message: error_message(deserializer)?,
        })
    }
}

/// Extract a loggable error message from an error-envelope object.
///
/// Accepts `{"message": ...}`, `{"error": ...}`, and bodies carrying BOTH
/// keys (a field-level `alias = "error"` would reject those as a duplicate
/// field); the non-null `error` key wins since it is the canonical provider
/// error object. String values pass through, any other JSON shape (nested
/// error objects, arrays) is stringified, and a body with neither key still
/// classifies as an error envelope with an empty message — the raw body is
/// what callers preserve.
pub(crate) fn error_message<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    let serde_json::Value::Object(body) = value else {
        return Err(serde::de::Error::custom(
            "error envelope must be a JSON object",
        ));
    };
    Ok(body
        .get("error")
        .filter(|value| !value.is_null())
        .or_else(|| body.get("message"))
        .map(|value| match value {
            serde_json::Value::String(message) => message.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default())
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

/// Identity envelope for providers whose 2xx body IS the success payload
/// (no error envelope can arrive with a success status).
#[derive(serde::Deserialize)]
#[serde(transparent)]
pub(crate) struct DirectPayload<T>(T);

impl<T> ProviderEnvelope for DirectPayload<T> {
    type Payload = T;

    fn into_payload(self) -> Result<T, String> {
        Ok(self.0)
    }
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

#[cfg(test)]
mod tests {
    use crate::providers::openai::client::ApiResponse;

    #[derive(Debug, serde::Deserialize)]
    struct Success {
        #[allow(dead_code)]
        text: String,
    }

    fn classify(body: &str) -> String {
        match serde_json::from_str::<ApiResponse<Success>>(body).expect("body must decode") {
            ApiResponse::Err(error) => error.message,
            ApiResponse::Ok(_) => panic!("error body must classify as the error envelope"),
        }
    }

    /// A body carrying BOTH `message` and `error` must still classify as the
    /// error envelope (a field-level `alias = "error"` rejected it as a
    /// duplicate field), with the canonical `error` object winning.
    #[test]
    fn dual_message_and_error_keys_classify_as_the_error_envelope() {
        assert_eq!(
            classify(r#"{"message":"quota exceeded","error":{"code":"429"}}"#),
            r#"{"code":"429"}"#
        );
    }

    #[test]
    fn null_error_key_falls_back_to_message() {
        assert_eq!(
            classify(r#"{"error":null,"message":"over capacity"}"#),
            "over capacity"
        );
    }
}
