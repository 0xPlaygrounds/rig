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

/// Which wire `usage` payload a Chat Completions dialect sends.
///
/// The variants exist because the providers' usage accounting differs (Mistral's
/// cached-token fallbacks, DeepSeek's cache hit/miss counters, OpenRouter's
/// cost fields); each is parsed into its own concrete type and converted at the
/// parse site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatCompletionsUsageDialect {
    /// OpenAI's `Usage` — the shape almost every compatible provider sends.
    OpenAi,
    /// DeepSeek's cache hit/miss accounting.
    DeepSeek,
    /// Mistral's usage with cached-token fallbacks.
    Mistral,
    /// OpenRouter's usage.
    OpenRouter,
}

/// A Chat Completions streaming dialect, as plain data.
///
/// Everything the shared state machine needs to know about a provider: the name
/// stamped on the terminal record, which wire usage payload to parse, and the
/// two behavioral knobs (both already
/// [`ProviderDescriptor`] fields or a single provider's quirk).
#[derive(Debug, Clone, Copy)]
pub struct ChatCompletionsDialect {
    /// Provider name stamped on the normalized terminal record.
    pub provider: &'static str,
    /// Which wire usage payload the provider sends.
    pub usage: ChatCompletionsUsageDialect,
    /// Whether a whole tool call can arrive in one chunk (llama.cpp-style
    /// servers); mirrors the descriptor field of the same name.
    pub emits_complete_single_chunk_tool_calls: bool,
    /// Whether `reasoning_details` payloads decorate accumulated tool calls
    /// (OpenRouter's encrypted-reasoning signatures).
    pub decorates_reasoning_details: bool,
}

impl ChatCompletionsDialect {
    /// The dialect implied by a provider's capability sheet: OpenAI-shaped
    /// usage, no reasoning-detail decoration.
    pub const fn from_descriptor(descriptor: &ProviderDescriptor) -> Self {
        Self {
            provider: descriptor.name,
            usage: ChatCompletionsUsageDialect::OpenAi,
            emits_complete_single_chunk_tool_calls: descriptor
                .emits_complete_single_chunk_tool_calls,
            decorates_reasoning_details: false,
        }
    }

    /// Parse a provider-specific wire usage payload instead of OpenAI's.
    pub const fn with_usage(mut self, usage: ChatCompletionsUsageDialect) -> Self {
        self.usage = usage;
        self
    }

    /// Decorate accumulated tool calls from `reasoning_details` payloads.
    pub const fn with_reasoning_detail_decoration(mut self) -> Self {
        self.decorates_reasoning_details = true;
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

/// Failure while building a provider configuration from the environment.
///
/// Returned by every provider's `functions::Config::from_env` (and the
/// modality configs' `from_env`). These are the problems detectable before any
/// request is sent: a missing credential variable, a variable holding invalid
/// Unicode, or a combination of variables that cannot describe a usable
/// endpoint.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A required or optional environment variable could not be read.
    ///
    /// For required variables this is also the "variable not present" case.
    #[error("environment variable `{name}` is not set or is invalid")]
    EnvironmentVariable {
        /// The environment variable name.
        name: &'static str,
        /// The underlying environment lookup error.
        #[source]
        source: std::env::VarError,
    },
    /// The variables that were present do not describe a usable configuration.
    #[error("{0}")]
    InvalidConfiguration(&'static str),
}

/// Read a required environment variable for provider configuration.
///
/// # Errors
/// [`ConfigError::EnvironmentVariable`] when the variable is missing or holds
/// invalid Unicode.
pub fn required_env_var(name: &'static str) -> Result<String, ConfigError> {
    std::env::var(name).map_err(|source| ConfigError::EnvironmentVariable { name, source })
}

/// Read an optional environment variable for provider configuration.
///
/// Missing variables return `Ok(None)`.
///
/// # Errors
/// [`ConfigError::EnvironmentVariable`] when the variable holds invalid
/// Unicode.
pub fn optional_env_var(name: &'static str) -> Result<Option<String>, ConfigError> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(source) => Err(ConfigError::EnvironmentVariable { name, source }),
    }
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
