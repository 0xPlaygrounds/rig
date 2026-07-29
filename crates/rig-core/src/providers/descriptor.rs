//! Provider capability descriptors and provider configuration primitives.
//!
//! A [`ProviderDescriptor`] is a provider's compile-time capability sheet as
//! plain data — the replacement for capability `const`s scattered across
//! provider traits. Fulfilment code consults it at request-build time, which
//! is how capability mismatches fail fast without a provider round-trip.

use serde::{Deserialize, Serialize};

/// A provider's capability sheet, as one `const` value per provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProviderDescriptor {
    /// Canonical provider name (telemetry `gen_ai.provider.name`, the
    /// `provider` field on normalized responses).
    pub name: &'static str,
    /// Whether the provider supports tool calling. When false, tools and
    /// tool_choice are dropped with a warning during request conversion.
    pub supports_tools: bool,
    /// Whether `output_schema` maps to a native structured-output request
    /// parameter.
    pub supports_response_format: bool,
    /// Whether streaming requests ask for usage on the final chunk
    /// (OpenAI-style `stream_options.include_usage`).
    pub stream_include_usage: bool,
    /// Whether the backend can emit a whole tool call in a single streaming
    /// chunk (llama.cpp-style servers).
    pub emits_complete_single_chunk_tool_calls: bool,
    /// Whether native structured output composes with tool calls in the same
    /// request (see issue #1928).
    pub composes_native_output_with_tools: bool,
    /// Maximum documents per embedding request, for providers with an
    /// embeddings API.
    pub max_embedding_documents: Option<usize>,
}

impl ProviderDescriptor {
    /// A descriptor with every capability off — a starting point for
    /// provider modules to override with their real capabilities.
    pub const fn named(name: &'static str) -> Self {
        Self {
            name,
            supports_tools: false,
            supports_response_format: false,
            stream_include_usage: false,
            emits_complete_single_chunk_tool_calls: false,
            composes_native_output_with_tools: false,
            max_embedding_documents: None,
        }
    }

    /// Set tool-calling support (const builder for out-of-crate descriptors).
    pub const fn with_tools(mut self, value: bool) -> Self {
        self.supports_tools = value;
        self
    }

    /// Set native structured-output support.
    pub const fn with_response_format(mut self, value: bool) -> Self {
        self.supports_response_format = value;
        self
    }

    /// Set OpenAI-style streaming usage-chunk requests.
    pub const fn with_stream_include_usage(mut self, value: bool) -> Self {
        self.stream_include_usage = value;
        self
    }

    /// Set whole-tool-call-per-chunk streaming behavior.
    pub const fn with_single_chunk_tool_calls(mut self, value: bool) -> Self {
        self.emits_complete_single_chunk_tool_calls = value;
        self
    }

    /// Set whether native structured output composes with tools (#1928).
    pub const fn with_composes_native_output_with_tools(mut self, value: bool) -> Self {
        self.composes_native_output_with_tools = value;
        self
    }

    /// Set the embedding batch limit.
    pub const fn with_max_embedding_documents(mut self, value: usize) -> Self {
        self.max_embedding_documents = Some(value);
        self
    }
}

/// Where a provider credential comes from.
///
/// A serialized provider config can reference the environment instead of
/// embedding secrets; `Inline` supports the explicit-key path (serializing
/// an inline key is the caller's deliberate choice).
///
/// `Debug` redacts the `Inline` key so credentials never leak through
/// `{:?}` of a config or agent; `Serialize` stays faithful (resuming a
/// serialized config requires the real key), so treat serialized configs
/// as secrets.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiKeyLocation {
    /// Read the key from this environment variable at use time.
    Env(String),
    /// The key itself, carried inline.
    Inline(String),
    /// No credential (local or unauthenticated endpoints).
    None,
}

impl std::fmt::Debug for ApiKeyLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Env(var) => f.debug_tuple("Env").field(var).finish(),
            // Never print the key itself — configs and agents holding one
            // are routinely logged via `{:?}`.
            Self::Inline(_) => f.debug_tuple("Inline").field(&"******").finish(),
            Self::None => f.write_str("None"),
        }
    }
}

impl ApiKeyLocation {
    /// Resolve the credential to a key string, if one is configured.
    ///
    /// # Errors
    /// [`ApiKeyError::MissingEnv`] when an `Env` variable is unset or empty.
    pub fn resolve(&self) -> Result<Option<String>, ApiKeyError> {
        match self {
            Self::Env(var) => match std::env::var(var) {
                Ok(value) if !value.is_empty() => Ok(Some(value)),
                _ => Err(ApiKeyError::MissingEnv(var.clone())),
            },
            Self::Inline(key) => Ok(Some(key.clone())),
            Self::None => Ok(None),
        }
    }
}

/// Credential resolution failure.
#[derive(Debug, thiserror::Error)]
pub enum ApiKeyError {
    /// The configured environment variable is unset or empty.
    #[error("environment variable `{0}` is unset or empty")]
    MissingEnv(String),
}

#[cfg(test)]
mod tests {
    use super::ApiKeyLocation;

    #[test]
    fn debug_redacts_inline_key() {
        let rendered = format!("{:?}", ApiKeyLocation::Inline("sk-secret".to_string()));
        assert!(!rendered.contains("sk-secret"), "leaked key: {rendered}");
        assert_eq!(rendered, r#"Inline("******")"#);
    }

    #[test]
    fn debug_keeps_env_and_none_readable() {
        assert_eq!(
            format!("{:?}", ApiKeyLocation::Env("OPENAI_API_KEY".to_string())),
            r#"Env("OPENAI_API_KEY")"#
        );
        assert_eq!(format!("{:?}", ApiKeyLocation::None), "None");
    }
}
