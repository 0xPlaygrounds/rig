//! Unary Vertex AI completions and model identifiers. Streaming is unsupported.
//!
//! ```no_run
//! use rig_vertexai::{Client, completion::{CompletionModel, GEMINI_2_5_FLASH}};
//!
//! # async fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
//! let model = CompletionModel::new(Client::from_env()?, GEMINI_2_5_FLASH);
//! # Ok(())
//! # }
//! ```

use super::Client;
use crate::types::completion_request::VertexCompletionRequest;
pub use crate::types::completion_response::VertexGenerateContentOutput;
use rig_core::completion::{
    CompletionModel as CompletionModelTrait, CompletionRequest, CompletionResponse,
};
use rig_core::error::ProviderError;
use rig_core::streaming::StreamingCompletionResponse;

/// `gemini-1.5-pro`
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-flash`
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro-latest`
pub const GEMINI_1_5_PRO_LATEST: &str = "gemini-1.5-pro-latest";
/// `gemini-1.5-flash-latest`
pub const GEMINI_1_5_FLASH_LATEST: &str = "gemini-1.5-flash-latest";
/// `gemini-2.0-flash-exp`
pub const GEMINI_2_0_FLASH_EXP: &str = "gemini-2.0-flash-exp";
/// `gemini-2.5-flash-lite`
pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
/// `gemini-2.5-flash`
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.5-pro`
pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: crate::client::Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub fn with_model(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    fn model_path(&self) -> String {
        let project = self.client.project();
        let location = self.client.location();
        format!(
            "projects/{project}/locations/{location}/publishers/google/models/{}",
            self.model
        )
    }
}

impl CompletionModel {
    /// Executes one completion RPC and returns provider-native output.
    /// Returns request-conversion, client-initialization, or RPC errors.
    /// The output type also deserializes from [`CompletionResponse::raw`].
    pub async fn raw_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<VertexGenerateContentOutput, ProviderError> {
        tracing::debug!(
            target: "rig_core::vertexai",
            "Vertex AI completion request: {request:?}"
        );

        let vertex_request = VertexCompletionRequest(request);

        let generation_config = vertex_request.generation_config()?;
        let system_instruction = vertex_request.system_instruction();
        let tools = vertex_request.tools();
        let tool_config = vertex_request.tool_config();
        let contents = vertex_request.contents()?;
        let model_path = self.model_path();

        let mut request_builder = self
            .client
            .inner()
            .await
            .map_err(|error| ProviderError::Provider(error.to_string()))?
            .generate_content()
            .set_model(&model_path)
            .set_contents(contents);

        if let Some(config) = generation_config {
            request_builder = request_builder.set_generation_config(config);
        }

        if let Some(system_instruction) = system_instruction {
            request_builder = request_builder.set_system_instruction(system_instruction);
        }

        if let Some(tools) = tools {
            request_builder = request_builder.set_tools([tools]);
        }

        if let Some(tool_config) = tool_config {
            request_builder = request_builder.set_tool_config(tool_config);
        }

        let response = request_builder
            .send()
            .await
            .map_err(|error| rpc_error(&error))?;

        tracing::debug!(
            target: "rig_core::vertexai",
            "Vertex AI completion response: {response:?}"
        );

        Ok(VertexGenerateContentOutput(response))
    }
}

fn streaming_unsupported() -> ProviderError {
    ProviderError::Provider(
        "Streaming is not supported for Vertex AI in this integration".to_string(),
    )
}

impl CompletionModelTrait for CompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        self.raw_completion(request).await?.try_into()
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, ProviderError> {
        Err(streaming_unsupported())
    }
}

/// Preserves SDK error display text, HTTP status, RPC code, and retry hints.
/// Transport errors also use the provider-body representation.
fn rpc_error(error: &google_cloud_aiplatform_v1::Error) -> ProviderError {
    let status = error
        .http_status_code()
        .and_then(|code| rig_core::http_client::StatusCode::from_u16(code).ok());
    let code = error.status().map(|status| status.code.name());
    // SDK transport classifications supply retry hints when no RPC code exists.
    let transient = code.map(transient_rpc_code).or_else(|| {
        (error.is_transport() || error.is_io() || error.is_timeout() || error.is_connect())
            .then_some(true)
    });
    ProviderError::from_provider_body(error.to_string())
        .with_provider_status(status)
        .with_provider_code(code.map(str::to_owned))
        .with_transient(transient)
}

/// Recognizes transient RPC codes; unlisted codes are non-transient.
fn transient_rpc_code(code: &str) -> bool {
    matches!(
        code,
        "UNAVAILABLE" | "RESOURCE_EXHAUSTED" | "DEADLINE_EXCEEDED" | "ABORTED"
    )
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
