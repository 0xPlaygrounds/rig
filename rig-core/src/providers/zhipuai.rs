//! ZhipuAI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::zhipuai;
//!
//! let client = zhipuai::Client::new("YOUR_API_KEY");
//!
//! let zhipu_model = client.completion_model(zhipuai::ZHIPU_CHAT);
//! ```

use crate::json_utils::merge;
use crate::message;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::{StreamingCompletionModel, StreamingCompletionResponse};
use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    extractor::ExtractorBuilder,
    json_utils,
    providers::openai,
    OneOrMany,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;

// ================================================================
// Main ZhipuAI Client
// ================================================================
const ZHIPU_API_BASE_URL: &str = "https://open.bigmodel.cn/api/paas/v4";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new ZhipuAI client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, ZHIPU_API_BASE_URL)
    }

    /// Create a new ZhipuAI client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {api_key}")
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("ZhipuAI reqwest client should build"),
        }
    }

    /// Create a new ZhipuAI client from the `ZHIPUAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("ZHIPUAI_API_KEY").expect("ZHIPUAI_API_KEY not set");
        Self::new(&api_key)
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an agent builder with the given completion model.
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ZhipuAIError,
}

#[derive(Debug, Deserialize)]
struct ZhipuAIError {
    message: String,
}

#[derive(Debug, Deserialize)]
pub struct ZhipuAICompletionResponse {
    #[allow(dead_code)]
    model: Option<String>,
    #[allow(dead_code)]
    created: Option<i64>,
    choices: Vec<ZhipuAICompletionChoice>,
    #[allow(dead_code)]
    request_id: Option<String>,
    #[allow(dead_code)]
    id: Option<String>,
    usage: Option<ZhipuAIUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ZhipuAICompletionChoice {
    #[allow(dead_code)]
    index: i64,
    #[allow(dead_code)]
    finish_reason: String,
    message: ZhipuAIMessage,
}

#[derive(Debug, Deserialize)]
pub struct ZhipuAIMessage {
    content: Option<String>,
    #[allow(dead_code)]
    role: String,
    tool_calls: Option<Vec<ZhipuAIToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct ZhipuAIToolCall {
    id: String,
    function: ZhipuAIFunction,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    type_: String,
}

#[derive(Debug, Deserialize)]
pub struct ZhipuAIFunction {
    arguments: String,
    name: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ZhipuAIUsage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

impl fmt::Display for ZhipuAIUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prompt_tokens: {}, completion_tokens: {}, total_tokens: {}",
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// ZhipuAI Completion API
// ================================================================
pub const ZHIPU_CHAT: &str = "glm-4";

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<openai::Message> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| {
                vec![openai::Message::system(&preamble)]
            });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(openai::ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = ZhipuAICompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<ZhipuAICompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "ZhipuAI completion error: {}", t);

            match serde_json::from_str::<ApiResponse<ZhipuAICompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "ZhipuAI completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );

                    let message = &response.choices[0].message;
                    let mut content = Vec::new();
                    // Add text content if present
                    if let Some(text) = &message.content {
                        if !text.trim().is_empty() {
                            content.push(completion::message::AssistantContent::text(text.clone()));
                        }
                    }
                    // Add tool calls if present
                    if let Some(tool_calls) = &message.tool_calls {
                        for tool_call in tool_calls {
                            let arguments = serde_json::from_str(&tool_call.function.arguments)
                                .unwrap_or(serde_json::Value::String(
                                    tool_call.function.arguments.clone(),
                                ));
                            content.push(completion::message::AssistantContent::tool_call(
                                &tool_call.id,
                                &tool_call.function.name,
                                arguments,
                            ));
                        }
                    }
                    // Ensure we have at least some content
                    if content.is_empty() {
                        content.push(completion::message::AssistantContent::text(String::new()));
                    }
                    let choice = OneOrMany::many(content).map_err(|_| {
                        CompletionError::ResponseError(
                            "Response contained no valid content".to_owned(),
                        )
                    })?;

                    Ok(completion::CompletionResponse {
                        choice,
                        raw_response: response,
                    })
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.error.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}

impl StreamingCompletionModel for CompletionModel {
    type StreamingResponse = openai::StreamingCompletionResponse;

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let mut request = self.create_completion_request(request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self.client.post("/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}
