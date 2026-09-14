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
#[cfg(any(
    test,
    feature = "anthropic",
    feature = "azure",
    feature = "chatgpt",
    feature = "copilot",
    feature = "deepseek",
    feature = "doubleword",
    feature = "groq",
    feature = "huggingface",
    feature = "hyperbolic",
    feature = "llamacpp",
    feature = "minimax",
    feature = "mira",
    feature = "mistral",
    feature = "moonshot",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "together",
    feature = "venice",
    feature = "xai",
    feature = "xiaomimimo",
    feature = "zai"
))]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[cfg(any(
    test,
    feature = "anthropic",
    feature = "azure",
    feature = "chatgpt",
    feature = "copilot",
    feature = "deepseek",
    feature = "doubleword",
    feature = "groq",
    feature = "huggingface",
    feature = "hyperbolic",
    feature = "llamacpp",
    feature = "minimax",
    feature = "mira",
    feature = "mistral",
    feature = "moonshot",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "together",
    feature = "venice",
    feature = "xai",
    feature = "xiaomimimo",
    feature = "zai"
))]
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
#[cfg(any(
    test,
    feature = "anthropic",
    feature = "azure",
    feature = "chatgpt",
    feature = "cohere",
    feature = "copilot",
    feature = "deepseek",
    feature = "doubleword",
    feature = "groq",
    feature = "huggingface",
    feature = "hyperbolic",
    feature = "llamacpp",
    feature = "minimax",
    feature = "mira",
    feature = "mistral",
    feature = "moonshot",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "together",
    feature = "venice",
    feature = "voyageai",
    feature = "xai",
    feature = "xiaomimimo",
    feature = "zai"
))]
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
#[cfg(any(
    feature = "anthropic",
    feature = "azure",
    feature = "chatgpt",
    feature = "cohere",
    feature = "copilot",
    feature = "deepseek",
    feature = "doubleword",
    feature = "gemini",
    feature = "groq",
    feature = "huggingface",
    feature = "hyperbolic",
    feature = "llamacpp",
    feature = "minimax",
    feature = "mira",
    feature = "mistral",
    feature = "moonshot",
    feature = "ollama",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "together",
    feature = "venice",
    feature = "xai",
    feature = "xiaomimimo",
    feature = "zai"
))]
pub(crate) trait ProviderEnvelope {
    /// The success payload carried by the envelope.
    type Payload;

    /// Split the envelope into its payload or the provider's error message.
    fn into_payload(self) -> Result<Self::Payload, String>;
}

/// Success-or-error envelope shared by OpenAI-compatible HTTP APIs.
#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
#[cfg(any(
    test,
    feature = "azure",
    feature = "chatgpt",
    feature = "copilot",
    feature = "deepseek",
    feature = "doubleword",
    feature = "groq",
    feature = "huggingface",
    feature = "hyperbolic",
    feature = "llamacpp",
    feature = "minimax",
    feature = "mira",
    feature = "mistral",
    feature = "moonshot",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "together",
    feature = "venice",
    feature = "xai",
    feature = "xiaomimimo",
    feature = "zai"
))]
pub(crate) enum OpenAiApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

/// Identity envelope for providers whose 2xx body IS the success payload
/// (no error envelope can arrive with a success status).
#[derive(serde::Deserialize)]
#[serde(transparent)]
#[cfg(any(
    all(
        test,
        any(
            feature = "anthropic",
            feature = "azure",
            feature = "chatgpt",
            feature = "cohere",
            feature = "copilot",
            feature = "deepseek",
            feature = "doubleword",
            feature = "gemini",
            feature = "groq",
            feature = "huggingface",
            feature = "hyperbolic",
            feature = "llamacpp",
            feature = "minimax",
            feature = "mira",
            feature = "mistral",
            feature = "moonshot",
            feature = "ollama",
            feature = "openai",
            feature = "openrouter",
            feature = "perplexity",
            feature = "together",
            feature = "venice",
            feature = "xai",
            feature = "xiaomimimo",
            feature = "zai"
        )
    ),
    feature = "chatgpt",
    feature = "cohere",
    feature = "copilot",
    feature = "gemini",
    feature = "ollama",
    feature = "openai",
    feature = "xai",
))]
pub(crate) struct DirectPayload<T>(T);

#[cfg(any(
    all(
        test,
        any(
            feature = "anthropic",
            feature = "azure",
            feature = "chatgpt",
            feature = "cohere",
            feature = "copilot",
            feature = "deepseek",
            feature = "doubleword",
            feature = "gemini",
            feature = "groq",
            feature = "huggingface",
            feature = "hyperbolic",
            feature = "llamacpp",
            feature = "minimax",
            feature = "mira",
            feature = "mistral",
            feature = "moonshot",
            feature = "ollama",
            feature = "openai",
            feature = "openrouter",
            feature = "perplexity",
            feature = "together",
            feature = "venice",
            feature = "xai",
            feature = "xiaomimimo",
            feature = "zai"
        )
    ),
    feature = "chatgpt",
    feature = "cohere",
    feature = "copilot",
    feature = "gemini",
    feature = "ollama",
    feature = "openai",
    feature = "xai",
))]
impl<T> ProviderEnvelope for DirectPayload<T> {
    type Payload = T;

    fn into_payload(self) -> Result<T, String> {
        Ok(self.0)
    }
}

#[cfg(any(
    all(
        test,
        any(
            feature = "anthropic",
            feature = "azure",
            feature = "chatgpt",
            feature = "cohere",
            feature = "copilot",
            feature = "deepseek",
            feature = "doubleword",
            feature = "gemini",
            feature = "groq",
            feature = "huggingface",
            feature = "hyperbolic",
            feature = "llamacpp",
            feature = "minimax",
            feature = "mira",
            feature = "mistral",
            feature = "moonshot",
            feature = "ollama",
            feature = "openai",
            feature = "openrouter",
            feature = "perplexity",
            feature = "together",
            feature = "venice",
            feature = "xai",
            feature = "xiaomimimo",
            feature = "zai"
        )
    ),
    feature = "azure",
    feature = "chatgpt",
    feature = "copilot",
    feature = "deepseek",
    feature = "doubleword",
    feature = "groq",
    feature = "huggingface",
    feature = "hyperbolic",
    feature = "llamacpp",
    feature = "minimax",
    feature = "mira",
    feature = "mistral",
    feature = "moonshot",
    feature = "openai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "together",
    feature = "venice",
    feature = "xai",
    feature = "xiaomimimo",
    feature = "zai"
))]
impl<T> ProviderEnvelope for OpenAiApiResponse<T> {
    type Payload = T;

    fn into_payload(self) -> Result<T, String> {
        match self {
            Self::Ok(value) => Ok(value),
            Self::Err(error) => Err(error.message),
        }
    }
}

#[cfg(test)]
mod tests;
