//! All supported models: <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini>

use super::Client;
use crate::types::{
    completion_request::VertexCompletionRequest, completion_response::VertexGenerateContentOutput,
};
use rig_core::completion::{
    CompletionError, CompletionModel as CompletionModelTrait, CompletionRequest,
    CompletionResponse, GetTokenUsage,
};
use rig_core::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Serialize, Deserialize)]
pub struct PlaceholderStreamingResponse;

impl GetTokenUsage for PlaceholderStreamingResponse {
    fn token_usage(&self) -> rig_core::completion::Usage {
        rig_core::completion::Usage::new()
    }
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

    fn model_path(&self) -> Result<String, CompletionError> {
        let project = self.client.project();
        let location = self.client.location();
        Ok(format!(
            "projects/{project}/locations/{location}/publishers/google/models/{}",
            self.model
        ))
    }
}

impl CompletionModelTrait for CompletionModel {
    type Response = VertexGenerateContentOutput;
    type StreamingResponse = PlaceholderStreamingResponse;

    type Client = Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into())
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        tracing::debug!(
            target: "rig_core::vertexai",
            "Vertex AI completion request: {request:?}"
        );

        let vertex_request = VertexCompletionRequest(request);

        let contents = vertex_request.contents()?;
        let generation_config = vertex_request.generation_config()?;
        let system_instruction = vertex_request.system_instruction();
        let tools = vertex_request.tools();
        let tool_config = vertex_request.tool_config();
        let model_path = self.model_path()?;

        let mut request_builder = self
            .client
            .get_inner()
            .await
            .map_err(|error| CompletionError::ProviderError(error.to_string()))?
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

        let response = request_builder.send().await.map_err(rpc_error)?;

        tracing::debug!(
            target: "rig_core::vertexai",
            "Vertex AI completion response: {response:?}"
        );

        let vertex_output = VertexGenerateContentOutput(response);
        let completion_response = vertex_output.try_into()?;

        Ok(completion_response)
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Err(CompletionError::ProviderError(
            "Streaming is not supported for Vertex AI in this integration".to_string(),
        ))
    }
}

/// Map a failed `send()` RPC into a [`CompletionError`] that preserves the
/// provider's gRPC error text verbatim.
///
/// Vertex AI uses a non-HTTP (gRPC/SDK) transport, so there is no
/// [`http::StatusCode`] to attach; the error body is preserved via
/// [`CompletionError::from_provider_body`] (`status: None`) rather than a
/// Rig-prefixed [`CompletionError::ProviderError`] diagnostic. (The
/// `get_inner()` client-init failure stays a `ProviderError` because it is a
/// Rig-side setup failure, not a provider response.)
///
/// Note: the SDK does not distinguish a server-returned gRPC error from a
/// transport/connection failure, so a pure connection error is also preserved
/// here (`status: None`) rather than gated out as a Rig diagnostic the way
/// Bedrock's typed service errors are.
fn rpc_error(error: impl std::fmt::Display) -> CompletionError {
    CompletionError::from_provider_body(error.to_string())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    // The `send()` RPC error type comes from the `google-cloud-aiplatform-v1`
    // SDK and is not trivially constructible, so `rpc_error` is generic over
    // `impl Display` and we pin it here with a representative error string of
    // its parameter type. This guards against a revert to `ProviderError`,
    // which would surface the body as `None`.
    #[test]
    fn rpc_error_preserves_raw_text_without_http_status() {
        let raw = "status: Unavailable, message: \"the service is currently unavailable\"";

        let err = rpc_error(raw);

        assert_eq!(err.provider_response_body(), Some(raw));
        assert_eq!(err.provider_response_status(), None);
    }
}
