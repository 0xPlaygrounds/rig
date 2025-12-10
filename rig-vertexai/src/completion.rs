//! All supported models: <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini>

use super::Client;
use crate::types::{
    completion_request::VertexCompletionRequest, completion_response::VertexGenerateContentOutput,
};
use rig::completion::{
    CompletionError, CompletionModel as CompletionModelTrait, CompletionRequest,
    CompletionResponse, GetTokenUsage,
};
use rig::streaming::StreamingCompletionResponse;
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
    fn token_usage(&self) -> Option<rig::completion::Usage> {
        None
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
            target: "rig::vertexai",
            "Vertex AI completion request: {request:?}"
        );

        let vertex_request = VertexCompletionRequest(request);

        let contents = vertex_request.contents()?;
        let generation_config = vertex_request.generation_config();
        let system_instruction = vertex_request.system_instruction();
        let tools = vertex_request.tools();
        let tool_config = vertex_request.tool_config();
        let model_path = self.model_path()?;

        let mut request_builder = self
            .client
            .get_inner()
            .await
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
            .map_err(|e| CompletionError::ProviderError(format!("Vertex AI API error: {e}")))?;

        tracing::debug!(
            target: "rig::vertexai",
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
