use crate::types::{
    completion_request::VertexCompletionRequest, completion_response::VertexGenerateContentOutput,
};
use rig::completion::{
    CompletionError, CompletionModel as CompletionModelTrait, CompletionRequest,
    CompletionResponse, GetTokenUsage,
};
use rig::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use std::env;

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
    pub fn new(client: crate::client::Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    fn model_path(&self) -> Result<String, CompletionError> {
        let project_id = env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
            CompletionError::ProviderError(
                "GOOGLE_CLOUD_PROJECT environment variable is not set".to_string(),
            )
        })?;

        // Extract model name from full path if provided, otherwise use as-is
        let model_name = if self.model.contains('/') {
            self.model.clone()
        } else {
            format!(
                "projects/{project_id}/locations/global/publishers/google/models/{}",
                self.model
            )
        };

        Ok(model_name)
    }
}

impl CompletionModelTrait for CompletionModel {
    type Response = VertexGenerateContentOutput;
    type StreamingResponse = PlaceholderStreamingResponse;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let vertex_request = VertexCompletionRequest(request);

        let contents = vertex_request.contents()?;
        let generation_config = vertex_request.generation_config();
        let system_instruction = vertex_request.system_instruction();
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

        let response = request_builder
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("Vertex AI API error: {e}")))?;

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
