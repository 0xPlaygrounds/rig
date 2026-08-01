//! Google Gemini API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::gemini;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = gemini::functions::EmbeddingConfig::from_env(gemini::EMBEDDING_001)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//!
//! let response = gemini::functions::embed(&cfg, &rt, vec!["Hello world!".to_string()]).await?;
//! # Ok(())
//! # }
//! ```

use serde::Deserialize;

pub mod completion;
pub mod embedding;
pub mod functions;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod interactions_api;
pub mod model_listing;
pub mod streaming;
pub mod transcription;

mod generate_content_client {
    crate::providers::client::define_http_client! {
        config = super::functions::Config,
        default_base_url = super::functions::DEFAULT_BASE_URL,
        api_key_required = true,
    }
}

mod interactions_client {
    crate::providers::client::define_http_client! {
        config = super::interactions_api::functions::Config,
        default_base_url = super::interactions_api::functions::DEFAULT_BASE_URL,
        api_key_required = true,
    }
}

pub use generate_content_client::{Client, ClientBuilder};
pub use interactions_client::{
    Client as InteractionsClient, ClientBuilder as InteractionsClientBuilder,
};

impl Client {
    /// Materialize embedding configuration sharing this connection.
    ///
    /// Known Gemini models retain their documented default dimensionality so
    /// requests built through the concrete client match the former fluent
    /// embedding model API on the wire.
    pub fn embedding_config(&self, model: impl Into<String>) -> functions::EmbeddingConfig {
        let model = model.into();
        let dimensions = embedding::model_default_ndims(&model);
        let mut config = functions::EmbeddingConfig::new(model);
        config.connection = self.connection_config().clone();
        config.dimensions = dimensions;
        config
    }

    /// Select the Interactions API while preserving this connection and runtime.
    pub fn interactions_api(&self) -> InteractionsClient {
        InteractionsClient::from_connection(self.connection_config().clone(), self.http_runtime())
    }

    /// Keep the `generateContent` API selected.
    pub fn generate_content_api(&self) -> Self {
        self.clone()
    }

    /// Materialize transcription configuration sharing this connection.
    pub fn transcription_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }

    /// Materialize image-generation configuration sharing this connection.
    #[cfg(feature = "image")]
    pub fn image_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }
}

impl InteractionsClient {
    /// Materialize generateContent embedding configuration sharing this connection.
    pub fn embedding_config(&self, model: impl Into<String>) -> functions::EmbeddingConfig {
        self.generate_content_api().embedding_config(model)
    }

    /// Select `generateContent` while preserving this connection and runtime.
    pub fn generate_content_api(&self) -> Client {
        Client::from_connection(self.connection_config().clone(), self.http_runtime())
    }

    /// Keep the Interactions API selected.
    pub fn interactions_api(&self) -> Self {
        self.clone()
    }

    /// Materialize generateContent transcription configuration using this connection.
    pub fn transcription_config(&self, model: impl Into<String>) -> functions::Config {
        self.generate_content_api().config(model)
    }

    /// Materialize generateContent image configuration using this connection.
    #[cfg(feature = "image")]
    pub fn image_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.generate_content_api().config(model)
    }
}

pub use embedding::{EMBEDDING_001, EMBEDDING_004};
#[cfg(feature = "image")]
pub use image_generation::GEMINI_2_5_FLASH_IMAGE;

// ================================================================
// Shared Gemini response envelope
// ================================================================
// Moved here from the deleted `gemini::client` module: these are wire types,
// not client plumbing, and `embedding`/`image_generation` parse through them.

/// Error response payload returned by Gemini.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    /// Structured error details.
    pub error: ApiError,
}

/// Error details returned in a Gemini API error response.
#[derive(Debug, Deserialize)]
pub struct ApiError {
    /// Human-readable description of the error.
    pub message: String,
}

/// Wrapper for successful or error Gemini API responses.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    // Untagged variants are tried in order, and some Gemini success response
    // types contain only defaulted or optional fields that accept error objects.
    Err(ApiErrorResponse),
    Ok(T),
}

pub mod gemini_api_types {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum ExecutionLanguage {
        /// Unspecified language. This value should not be used.
        LanguageUnspecified,
        /// Python >= 3.10, with numpy and simply available.
        Python,
    }

    /// Code generated by the model that is meant to be executed, and the result returned to the model.
    /// Only generated when using the CodeExecution tool, in which the code will be automatically executed,
    /// and a corresponding CodeExecutionResult will also be generated.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct ExecutableCode {
        /// Programming language of the code.
        pub language: ExecutionLanguage,
        /// The code to be executed.
        pub code: String,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct CodeExecutionResult {
        /// Outcome of the code execution.
        pub outcome: CodeExecutionOutcome,
        /// Contains stdout when code execution is successful, stderr or other description otherwise.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub output: Option<String>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub enum CodeExecutionOutcome {
        /// Unspecified status. This value should not be used.
        #[serde(rename = "OUTCOME_UNSPECIFIED")]
        Unspecified,
        /// Code execution completed successfully.
        #[serde(rename = "OUTCOME_OK")]
        Ok,
        /// Code execution finished but with a failure. stderr should contain the reason.
        #[serde(rename = "OUTCOME_FAILED")]
        Failed,
        /// Code execution ran for too long, and was cancelled. There may or may not be a partial output present.
        #[serde(rename = "OUTCOME_DEADLINE_EXCEEDED")]
        DeadlineExceeded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_response_detects_nested_error_before_permissive_success() {
        #[derive(Debug, Deserialize)]
        struct PermissiveResponse {
            #[serde(default)]
            candidates: Vec<serde_json::Value>,
        }

        let response: ApiResponse<PermissiveResponse> = serde_json::from_str(
            r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#,
        )
        .expect("nested Gemini error should deserialize");

        match response {
            ApiResponse::Err(err) => assert_eq!(err.error.message, "boom"),
            ApiResponse::Ok(response) => panic!(
                "expected nested error, got success with {} candidates",
                response.candidates.len()
            ),
        }
    }

    #[test]
    fn api_response_allows_top_level_message_in_success() {
        #[derive(Debug, Deserialize)]
        struct MessageResponse {
            message: String,
        }

        let response: ApiResponse<MessageResponse> =
            serde_json::from_str(r#"{"message":"success"}"#)
                .expect("success response should deserialize");

        match response {
            ApiResponse::Ok(response) => assert_eq!(response.message, "success"),
            ApiResponse::Err(err) => panic!("expected success, got error: {err:?}"),
        }
    }

    #[test]
    fn concrete_clients_preserve_embedding_model_defaults() {
        let client = Client::new("test-key");
        let config = client.embedding_config(EMBEDDING_001);
        assert_eq!(config.dimensions, Some(3072));
        assert_eq!(config.connection, *client.connection_config());

        let interactions_config = client.interactions_api().embedding_config(EMBEDDING_004);
        assert_eq!(interactions_config.dimensions, Some(768));
        assert_eq!(interactions_config.connection, *client.connection_config());

        let unknown = client.embedding_config("custom-embedding-model");
        assert_eq!(unknown.dimensions, None);
    }
}
