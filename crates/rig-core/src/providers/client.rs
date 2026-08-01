//! Concrete, data-backed provider client building blocks.
//!
//! Provider modules expose their own monomorphic `Client` and `ClientBuilder`
//! wrappers. This module owns the shared connection record and construction
//! mechanics so those wrappers remain small and cannot drift in credential,
//! header, or transport handling.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::http_runtime::HttpRuntime;

use super::descriptor::ApiKeyLocation;

/// Reusable HTTP connection data shared by provider clients and their configs.
///
/// Serialization preserves credentials so a config can be resumed elsewhere.
/// Treat serialized values as secrets. `Debug` redacts inline credentials and
/// every extra-header value. Header values are redacted by default because
/// callers may use arbitrary vendor-specific credential names.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HttpConnectionConfig {
    /// Provider API base URL.
    pub base_url: String,
    /// Where the provider credential is resolved from.
    pub api_key: ApiKeyLocation,
    /// Headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl HttpConnectionConfig {
    /// Construct connection data with no extra headers.
    pub fn new(base_url: impl Into<String>, api_key: ApiKeyLocation) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
            extra_headers: Vec::new(),
        }
    }
}

macro_rules! impl_http_connection_config {
    ($config:ty) => {
        impl std::ops::Deref for $config {
            type Target = $crate::providers::client::HttpConnectionConfig;

            fn deref(&self) -> &Self::Target {
                &self.connection
            }
        }

        impl std::ops::DerefMut for $config {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.connection
            }
        }
    };
}

pub(crate) use impl_http_connection_config;

impl fmt::Debug for HttpConnectionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let headers = redacted_headers(&self.extra_headers);

        f.debug_struct("HttpConnectionConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key)
            .field("extra_headers", &headers)
            .finish()
    }
}

pub(crate) fn redacted_headers(headers: &[(String, String)]) -> Vec<(&str, &str)> {
    headers
        .iter()
        .map(|(name, _)| (name.as_str(), "******"))
        .collect()
}

/// Failure while building a concrete provider client.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientBuildError {
    /// A provider requiring authentication was built without a credential.
    #[error("an API key is required; call `api_key`, `api_key_location`, or `from_env`")]
    MissingApiKey,
    /// Azure-style clients require a resource endpoint.
    #[error("a provider endpoint is required")]
    MissingEndpoint,
}

#[derive(Clone, Debug)]
pub(crate) struct HttpClientState {
    pub(crate) connection: HttpConnectionConfig,
    pub(crate) http: HttpRuntime,
}

#[derive(Clone)]
pub(crate) struct HttpClientBuilderState {
    base_url: String,
    api_key: Option<ApiKeyLocation>,
    extra_headers: Vec<(String, String)>,
    http: HttpRuntime,
    api_key_required: bool,
}

impl fmt::Debug for HttpClientBuilderState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpClientBuilderState")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key)
            .field("extra_headers", &redacted_headers(&self.extra_headers))
            .field("http", &self.http)
            .field("api_key_required", &self.api_key_required)
            .finish()
    }
}

impl HttpClientBuilderState {
    pub(crate) fn new(base_url: impl Into<String>, api_key_required: bool) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: (!api_key_required).then_some(ApiKeyLocation::None),
            extra_headers: Vec::new(),
            http: HttpRuntime::new(),
            api_key_required,
        }
    }

    pub(crate) fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(ApiKeyLocation::Inline(key.into()));
        self
    }

    pub(crate) fn api_key_location(mut self, api_key: ApiKeyLocation) -> Self {
        self.api_key = Some(api_key);
        self
    }

    pub(crate) fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub(crate) fn extra_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    pub(crate) fn extra_headers(
        mut self,
        headers: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        self.extra_headers.extend(headers);
        self
    }

    pub(crate) fn http_runtime(mut self, http: HttpRuntime) -> Self {
        self.http = http;
        self
    }

    pub(crate) fn build(self) -> Result<HttpClientState, ClientBuildError> {
        let api_key = match self.api_key {
            Some(api_key) => api_key,
            None if self.api_key_required => return Err(ClientBuildError::MissingApiKey),
            None => ApiKeyLocation::None,
        };
        Ok(HttpClientState {
            connection: HttpConnectionConfig {
                base_url: self.base_url,
                api_key,
                extra_headers: self.extra_headers,
            },
            http: self.http,
        })
    }
}

macro_rules! define_http_client {
    (
        config = $config:path,
        default_base_url = $default_base_url:expr,
        api_key_required = $api_key_required:expr $(,)?
    ) => {
        /// Concrete reusable connection client for this provider.
        #[derive(Clone, Debug)]
        pub struct Client {
            inner: $crate::providers::client::HttpClientState,
        }

        /// Monomorphic builder for [`Client`].
        #[derive(Clone, Debug)]
        pub struct ClientBuilder {
            inner: $crate::providers::client::HttpClientBuilderState,
        }

        impl Client {
            /// Build a client from this provider's conventional environment.
            pub fn from_env() -> Result<Self, $crate::providers::ConfigError> {
                // Only the connection projection is retained. `config(model)`
                // constructs a fresh config, so model-derived defaults never
                // come from this model-free environment probe.
                let config = <$config>::from_env(String::new())?;
                Ok(Self::from_connection(
                    config.connection,
                    $crate::http_runtime::HttpRuntime::new(),
                ))
            }

            /// Start a concrete client builder.
            pub fn builder() -> ClientBuilder {
                ClientBuilder {
                    inner: $crate::providers::client::HttpClientBuilderState::new(
                        $default_base_url,
                        $api_key_required,
                    ),
                }
            }

            /// Build a client with an inline API key and provider defaults.
            pub fn new(api_key: impl Into<String>) -> Self {
                Self::from_connection(
                    $crate::providers::client::HttpConnectionConfig::new(
                        $default_base_url,
                        $crate::providers::ApiKeyLocation::Inline(api_key.into()),
                    ),
                    $crate::http_runtime::HttpRuntime::new(),
                )
            }

            /// Materialize plain provider configuration for `model`.
            pub fn config(&self, model: impl Into<String>) -> $config {
                let mut config = <$config>::new(model);
                config.connection = self.inner.connection.clone();
                config
            }

            /// The canonical connection data used by every materialized config.
            pub fn connection_config(&self) -> &$crate::providers::client::HttpConnectionConfig {
                &self.inner.connection
            }

            /// The shared concrete HTTP runtime.
            pub fn http_runtime(&self) -> $crate::http_runtime::HttpRuntime {
                self.inner.http.clone()
            }

            /// Compatibility alias for [`Self::http_runtime`].
            pub fn http(&self) -> $crate::http_runtime::HttpRuntime {
                self.http_runtime()
            }

            pub(crate) fn from_connection(
                connection: $crate::providers::client::HttpConnectionConfig,
                http: $crate::http_runtime::HttpRuntime,
            ) -> Self {
                Self {
                    inner: $crate::providers::client::HttpClientState { connection, http },
                }
            }
        }

        impl ClientBuilder {
            /// Set an inline API key.
            pub fn api_key(self, key: impl Into<String>) -> Self {
                Self {
                    inner: self.inner.api_key(key),
                }
            }

            /// Set a deferred or inline credential location.
            pub fn api_key_location(self, api_key: $crate::providers::ApiKeyLocation) -> Self {
                Self {
                    inner: self.inner.api_key_location(api_key),
                }
            }

            /// Override the provider API base URL.
            pub fn base_url(self, base_url: impl Into<String>) -> Self {
                Self {
                    inner: self.inner.base_url(base_url),
                }
            }

            /// Append one connection-wide header.
            pub fn extra_header(self, name: impl Into<String>, value: impl Into<String>) -> Self {
                Self {
                    inner: self.inner.extra_header(name, value),
                }
            }

            /// Append connection-wide headers.
            pub fn extra_headers(
                self,
                headers: impl IntoIterator<Item = (String, String)>,
            ) -> Self {
                Self {
                    inner: self.inner.extra_headers(headers),
                }
            }

            /// Reuse an existing concrete HTTP runtime.
            pub fn http_runtime(self, http: $crate::http_runtime::HttpRuntime) -> Self {
                Self {
                    inner: self.inner.http_runtime(http),
                }
            }

            /// Reuse an existing reqwest client.
            pub fn reqwest_client(self, client: reqwest::Client) -> Self {
                self.http_runtime($crate::http_runtime::HttpRuntime::from_reqwest(client))
            }

            /// Validate the required values and build the client.
            pub fn build(self) -> Result<Client, $crate::providers::client::ClientBuildError> {
                Ok(Client {
                    inner: self.inner.build()?,
                })
            }
        }
    };
}

pub(crate) use define_http_client;

macro_rules! impl_http_embedding_config_factory {
    ($client:ident, $config:path) => {
        impl $client {
            /// Materialize plain embedding configuration sharing this client's connection.
            pub fn embedding_config(&self, model: impl Into<String>) -> $config {
                let mut config = <$config>::new(model);
                config.connection = self.connection_config().clone();
                config
            }
        }
    };
}

pub(crate) use impl_http_embedding_config_factory;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{anthropic, ollama, openai};

    #[test]
    fn debug_redacts_credentials_in_key_locations_and_headers() {
        let config = HttpConnectionConfig {
            base_url: "https://example.com".to_string(),
            api_key: ApiKeyLocation::Inline("inline-secret".to_string()),
            extra_headers: vec![
                (
                    "Authorization".to_string(),
                    "Bearer header-secret".to_string(),
                ),
                ("x-api-key".to_string(), "other-secret".to_string()),
                (
                    "cf-access-client-secret".to_string(),
                    "vendor-secret".to_string(),
                ),
                ("x-request-id".to_string(), "request-id-value".to_string()),
            ],
        };

        let debug = format!("{config:?}");
        assert!(!debug.contains("inline-secret"));
        assert!(!debug.contains("header-secret"));
        assert!(!debug.contains("other-secret"));
        assert!(!debug.contains("vendor-secret"));
        assert!(!debug.contains("request-id-value"));
        assert!(debug.contains("cf-access-client-secret"));
        assert!(debug.contains("x-request-id"));
    }

    #[test]
    fn authenticated_client_builder_requires_a_key() {
        assert!(matches!(
            openai::Client::builder().build(),
            Err(ClientBuildError::MissingApiKey)
        ));
    }

    #[test]
    fn client_materializes_models_from_one_connection_record() {
        let client = openai::Client::builder()
            .api_key("secret")
            .base_url("https://proxy.example/v1")
            .extra_header("x-tenant", "acme")
            .build()
            .expect("complete builder should succeed");

        let responses = client.responses_config("response-model");
        let completions = client.completions_config("chat-model");
        let embeddings = client.embedding_config("embedding-model");
        let transcription = client.transcription_config("transcription-model");

        assert_eq!(responses.model, "response-model");
        assert_eq!(completions.model, "chat-model");
        assert_eq!(embeddings.model, "embedding-model");
        assert_eq!(transcription.model, "transcription-model");
        assert_eq!(responses.connection, *client.connection_config());
        assert_eq!(completions.connection, *client.connection_config());
        assert_eq!(embeddings.connection, *client.connection_config());
        assert_eq!(transcription.connection, *client.connection_config());

        #[cfg(feature = "image")]
        assert_eq!(
            client.image_generation_config("image-model").connection,
            *client.connection_config()
        );
        #[cfg(feature = "audio")]
        assert_eq!(
            client.audio_generation_config("audio-model").connection,
            *client.connection_config()
        );
    }

    #[test]
    fn model_derived_defaults_are_recomputed() {
        let client = anthropic::Client::new("secret");
        let known = client.config(anthropic::completion::CLAUDE_SONNET_4_6);
        let unknown = client.config("future-model");

        assert_eq!(
            known.default_max_tokens,
            anthropic::functions::Config::new(anthropic::completion::CLAUDE_SONNET_4_6)
                .default_max_tokens
        );
        assert_eq!(
            unknown.default_max_tokens,
            anthropic::functions::Config::new("future-model").default_max_tokens
        );
    }

    #[test]
    fn client_reuses_the_supplied_runtime() {
        let runtime = HttpRuntime::recording(crate::test_utils::RecordingHttpClient::new("{}"));
        let client = openai::Client::builder()
            .api_key("secret")
            .http_runtime(runtime)
            .build()
            .expect("complete builder should succeed");

        assert!(format!("{:?}", client.http_runtime()).contains("recording"));
        assert!(format!("{:?}", client.completions_api().http_runtime()).contains("recording"));
    }

    #[test]
    fn client_debug_redacts_builder_credentials() {
        let client = openai::Client::builder()
            .api_key("inline-secret")
            .extra_header("authorization", "Bearer header-secret")
            .extra_header("x-request-id", "request-id-value")
            .build()
            .expect("complete builder should succeed");

        let debug = format!("{client:?}");
        assert!(!debug.contains("inline-secret"));
        assert!(!debug.contains("header-secret"));
        assert!(!debug.contains("request-id-value"));
        assert!(debug.contains("x-request-id"));
    }

    #[test]
    fn client_builder_debug_redacts_credentials_before_build() {
        let builder = openai::Client::builder()
            .api_key("inline-secret")
            .extra_header("Authorization", "Bearer header-secret")
            .extra_header("proxy-authorization", "proxy-secret")
            .extra_header("x-api-key", "x-secret")
            .extra_header("api-key", "api-secret")
            .extra_header("x-goog-api-key", "google-secret")
            .extra_header("cf-access-client-secret", "vendor-secret")
            .extra_header("x-request-id", "request-id-value");

        let debug = format!("{builder:?}");
        for secret in [
            "inline-secret",
            "header-secret",
            "proxy-secret",
            "x-secret",
            "api-secret",
            "google-secret",
            "vendor-secret",
            "request-id-value",
        ] {
            assert!(!debug.contains(secret));
        }
        assert!(debug.contains("cf-access-client-secret"));
        assert!(debug.contains("x-request-id"));
    }

    #[test]
    fn unauthenticated_client_can_be_constructed_from_environment() {
        let client = ollama::Client::from_env().expect("Ollama does not require a credential");
        let config = client.config("local-model");
        assert_eq!(config.model, "local-model");
        assert_eq!(config.connection, *client.connection_config());
    }
}
