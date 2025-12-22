//! Streaming support for Vertex AI Gemini 3 models
//!
//! This module provides streaming support for Vertex AI Gemini 3 models (pro and flash).
//! It uses the Rig framework's generic HTTP client abstraction and SSE parsing.
//!
//! Streaming is currently only supported for Gemini 3 models due to differences in the
//! streaming response format between Gemini 2.5 and 3.0.

use crate::client::Client;
use crate::types::completion_request::VertexCompletionRequest;
use http::Request;
use rig::completion::{
    CompletionError, CompletionModel as CompletionModelTrait, CompletionRequest,
    CompletionResponse, GetTokenUsage,
};
use rig::http_client::sse::{Event, GenericEventSource};
use rig::http_client::HttpClientExt;
use rig::streaming::{RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

/// Usage metadata from Vertex AI streaming response
#[derive(Debug, Deserialize, Serialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PartialUsage {
    pub total_token_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<i32>,
    pub prompt_token_count: Option<i32>,
}

impl GetTokenUsage for PartialUsage {
    fn token_usage(&self) -> Option<rig::completion::Usage> {
        let mut usage = rig::completion::Usage::new();

        #[allow(clippy::cast_sign_loss)]
        {
            usage.input_tokens = self.prompt_token_count.unwrap_or_default().max(0) as u64;
            usage.output_tokens = (self.cached_content_token_count.unwrap_or_default().max(0)
                + self.candidates_token_count.unwrap_or_default().max(0)
                + self.thoughts_token_count.unwrap_or_default().max(0))
                as u64;
            usage.total_tokens = self.total_token_count.unwrap_or_default().max(0) as u64;
        }

        Some(usage)
    }
}

/// Final streaming response containing usage metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexAIStreamingResponse {
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: PartialUsage,
}

impl GetTokenUsage for VertexAIStreamingResponse {
    fn token_usage(&self) -> Option<rig::completion::Usage> {
        self.usage_metadata.token_usage()
    }
}

/// Function call in a streaming response
#[derive(Debug, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

/// Part of a content response (can be text or function call)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallPart {
    #[serde(rename = "functionCall")]
    pub function_call: Option<FunctionCall>,
    #[serde(rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
    #[serde(rename = "thought_signature")]
    pub thought_signature_snake: Option<String>,
    pub text: Option<String>,
}

/// Content in a candidate response
#[derive(Debug, Deserialize)]
pub struct ContentPart {
    pub parts: Vec<FunctionCallPart>,
}

/// Candidate response from the model
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: Option<ContentPart>,
    pub finish_reason: Option<String>,
}

/// Streaming response chunk from Vertex AI
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamGenerateContentResponse {
    pub candidates: Vec<Candidate>,
    pub usage_metadata: Option<PartialUsage>,
}

/// Vertex AI streaming `CompletionModel` for Gemini 3 models
///
/// This model provides streaming support for Vertex AI Gemini 3 models (pro and flash).
/// It uses the Rig framework's generic HTTP client abstraction, allowing users to provide
/// their own HTTP client with custom configuration (e.g., HTTP/3, connection pooling, etc).
#[derive(Clone)]
pub struct StreamingCompletionModel<HttpClient: HttpClientExt> {
    client: Client,
    model_name: String,
    http_client: HttpClient,
}

impl<HttpClient: HttpClientExt + Clone + 'static> StreamingCompletionModel<HttpClient> {
    /// Create a new streaming completion model for Vertex AI
    ///
    /// # Arguments
    ///
    /// * `client` - Vertex AI client with project and location configuration
    /// * `model_name` - Name of the model (must be Gemini 3 - e.g., "gemini-3-pro" or "gemini-3-flash")
    /// * `http_client` - HTTP client implementing `HttpClientExt` for making streaming requests
    pub fn new(client: Client, model_name: impl Into<String>, http_client: HttpClient) -> Self {
        Self {
            client,
            model_name: model_name.into(),
            http_client,
        }
    }

    /// Get the streaming endpoint URL for this model
    fn streaming_endpoint(&self) -> String {
        let project = self.client.project();
        let location = self.client.location();

        // Gemini 3 models use 'global' location with non-regional endpoint
        if self.model_name.contains("gemini-3") {
            format!(
                "https://aiplatform.googleapis.com/v1/projects/{}/locations/global/publishers/google/models/{}:streamGenerateContent",
                project, self.model_name
            )
        } else {
            format!(
                "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent",
                location, project, location, self.model_name
            )
        }
    }

}

impl<HttpClient: HttpClientExt + Clone + 'static> CompletionModelTrait
    for StreamingCompletionModel<HttpClient>
{
    type Response = VertexAIStreamingResponse;
    type StreamingResponse = VertexAIStreamingResponse;
    type Client = HttpClient;

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        unimplemented!("Use StreamingCompletionModel::new() instead")
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        Err(CompletionError::ProviderError(
            "Non-streaming completion not implemented. Use stream() instead.".to_string(),
        ))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        // Streaming is currently only supported for Gemini 3 models
        if !self.model_name.contains("gemini-3") {
            return Err(CompletionError::ProviderError(
                format!(
                    "Streaming not yet supported for {}. Currently supported: gemini-3-pro, gemini-3-flash",
                    self.model_name
                ),
            ));
        }

        debug!("Starting VertexAI streaming request for model: {}", self.model_name);

        // Build request using existing VertexCompletionRequest
        let vertex_request = VertexCompletionRequest(request);

        let contents = vertex_request
            .contents()
            .map_err(|e| CompletionError::ProviderError(format!("Request build error: {e}")))?;
        let generation_config = vertex_request.generation_config();
        let system_instruction = vertex_request.system_instruction();
        let tools = vertex_request.tools();

        // Build request body
        let mut request_body = serde_json::json!({
            "contents": contents,
        });

        if let Some(config) = generation_config {
            request_body["generationConfig"] = serde_json::to_value(config)
                .map_err(|e| CompletionError::ProviderError(format!("Config error: {e}")))?;
        }

        if let Some(system_instruction) = system_instruction {
            request_body["systemInstruction"] = serde_json::to_value(system_instruction)
                .map_err(|e| CompletionError::ProviderError(format!("System instruction error: {e}")))?;
        }

        if let Some(tools) = tools {
            request_body["tools"] = serde_json::json!([tools]);
        }

        let body = serde_json::to_vec(&request_body)
            .map_err(|e| CompletionError::ProviderError(format!("Serialization error: {e}")))?;

        let url = self.streaming_endpoint();
        debug!("Streaming to: {}", url);

        // Create SSE request
        // AUTHENTICATION: The provided HttpClient must handle GCP authentication
        // (e.g., via interceptors, middleware, or pre-configured auth headers)
        // Callers should pass an authenticated HTTP client that includes Bearer tokens
        // in Authorization headers for Vertex AI API requests.
        let req = Request::post(url)
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::ProviderError(format!("Request building error: {e}")))?;

        // Create event source for SSE parsing
        let mut event_source = GenericEventSource::new(self.http_client.clone(), req);

        // Create streaming response using async_stream
        use futures::StreamExt;
        let stream = async_stream::stream! {
            let mut final_usage = None;

            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        debug!("SSE connection opened");
                        continue;
                    }
                    Ok(Event::Message(message)) => {
                        // Skip empty messages
                        if message.data.trim().is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<StreamGenerateContentResponse>(&message.data) {
                            Ok(data) => {
                                // Process candidates
                                for candidate in data.candidates {
                                    // Get finish reason if present
                                    if let Some(finish_reason) = candidate.finish_reason {
                                        debug!("Stream finished: {}", finish_reason);
                                        final_usage = data.usage_metadata;
                                        break;
                                    }

                                    // Process content
                                    if let Some(content) = candidate.content {
                                        for part in content.parts {
                                            // Handle function calls
                                            if let Some(function_call) = part.function_call {
                                                debug!("Tool call: {}", function_call.name);

                                                let thought_sig = part
                                                    .thought_signature
                                                    .or(part.thought_signature_snake);

                                                let tool_call = RawStreamingToolCall::new(
                                                    function_call.name.clone(),
                                                    function_call.name,
                                                    function_call.args,
                                                )
                                                .with_signature(thought_sig);

                                                yield Ok(RawStreamingChoice::ToolCall(tool_call));
                                            }

                                            // Handle text content
                                            if let Some(text) = part.text {
                                                if !text.is_empty() {
                                                    debug!("Text chunk: {} bytes", text.len());
                                                    yield Ok(RawStreamingChoice::Message(text));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse stream chunk: {} - data: {}", e, message.data);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Stream error: {}", e);
                        yield Err(CompletionError::ProviderError(format!("Stream error: {e}")));
                        break;
                    }
                }
            }

            yield Ok(RawStreamingChoice::FinalResponse(
                VertexAIStreamingResponse {
                    usage_metadata: final_usage.unwrap_or_default(),
                },
            ));
        };

        Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_stream_response_with_text() {
        let json_data = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello, world!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        assert_eq!(response.candidates.len(), 1);
        let content = response.candidates[0].content.as_ref().expect("should have content");
        assert_eq!(content.parts.len(), 1);
        assert_eq!(content.parts[0].text, Some("Hello, world!".to_string()));
    }

    #[test]
    fn test_deserialize_stream_response_with_function_call() {
        let json_data = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "San Francisco"}
                        },
                        "thoughtSignature": "thinking..."
                    }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 20,
                "totalTokenCount": 70
            }
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        let content = response.candidates[0]
            .content
            .as_ref()
            .expect("should have content");
        let part = &content.parts[0];
        assert!(part.function_call.is_some());
        assert_eq!(
            part.function_call.as_ref().unwrap().name,
            "get_weather"
        );
        assert_eq!(part.thought_signature, Some("thinking...".to_string()));
    }

    #[test]
    fn test_token_usage() {
        let usage = PartialUsage {
            total_token_count: Some(100),
            cached_content_token_count: Some(20),
            candidates_token_count: Some(30),
            thoughts_token_count: Some(10),
            prompt_token_count: Some(40),
        };

        let token_usage = usage.token_usage().unwrap();
        assert_eq!(token_usage.input_tokens, 40);
        assert_eq!(token_usage.output_tokens, 60); // 20 + 30 + 10
        assert_eq!(token_usage.total_tokens, 100);
    }

    #[test]
    fn test_streaming_response_token_usage() {
        let response = VertexAIStreamingResponse {
            usage_metadata: PartialUsage {
                total_token_count: Some(100),
                cached_content_token_count: None,
                candidates_token_count: Some(30),
                thoughts_token_count: None,
                prompt_token_count: Some(70),
            },
        };

        let usage = response.token_usage().unwrap();
        assert_eq!(usage.input_tokens, 70);
        assert_eq!(usage.output_tokens, 30);
        assert_eq!(usage.total_tokens, 100);
    }
}
