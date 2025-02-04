//! DeepSeek API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::deepseek;
//!
//! let client = deepseek::Client::new("DEEPSEEK_API_KEY");
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_CHAT);
//! ```
use crate::{
    completion::{CompletionModel, CompletionRequest, CompletionResponse},
    extractor::ExtractorBuilder,
    json_utils,
};
use reqwest::Client as HttpClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Clone)]
pub struct Client {
    pub base_url: String,
    pub api_key: String,
    http_client: HttpClient,
}

impl Client {
    // Create a new DeepSeek client from an API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            base_url: DEEPSEEK_API_BASE_URL.to_string(),
            api_key: api_key.to_string(),
            http_client: HttpClient::new(),
        }
    }

    // If you prefer the environment variable approach:
    pub fn from_env() -> Self {
        let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set");
        Self::new(&api_key)
    }

    // Handy for advanced usage, e.g. letting user override base_url or set timeouts:
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        // Possibly configure a custom HTTP client here if needed.
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            http_client: HttpClient::new(),
        }
    }

    /// Creates a DeepSeek completion model with the given `model_name`.
    pub fn completion_model(&self, model_name: &str) -> DeepSeekCompletionModel {
        DeepSeekCompletionModel {
            client: self.clone(),
            model: model_name.to_string(),
        }
    }

    /// Optionally add an agent() convenience:
    pub fn agent(&self, model_name: &str) -> crate::agent::AgentBuilder<DeepSeekCompletionModel> {
        crate::agent::AgentBuilder::new(self.completion_model(model_name))
    }

    /// Create an extractor builder with the given completion model.
    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, DeepSeekCompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

/// The response shape from the DeepSeek API
#[derive(Debug, Deserialize)]
pub struct DeepSeekResponse {
    // We'll match the JSON:
    pub choices: Vec<Choice>,
    // you may want usage or other fields
}

impl TryFrom<DeepSeekResponse> for CompletionResponse<DeepSeekResponse> {
    type Error = crate::completion::CompletionError;

    fn try_from(value: DeepSeekResponse) -> Result<Self, Self::Error> {
        match value.choices.as_slice() {
            [Choice {
                message:
                    Some(DeepSeekMessage {
                        tool_calls: Some(calls),
                        ..
                    }),
                ..
            }, ..]
                if !calls.is_empty() =>
            {
                let call = calls.first().unwrap();

                Ok(crate::completion::CompletionResponse {
                    choice: crate::completion::ModelChoice::ToolCall(
                        call.function.name.clone(),
                        "".to_owned(),
                        serde_json::from_str(&call.function.arguments)?,
                    ),
                    raw_response: value,
                })
            }
            [Choice {
                message:
                    Some(DeepSeekMessage {
                        content: Some(content),
                        ..
                    }),
                ..
            }, ..] => Ok(crate::completion::CompletionResponse {
                choice: crate::completion::ModelChoice::Message(content.to_string()),
                raw_response: value,
            }),
            _ => Err(crate::completion::CompletionError::ResponseError(
                "Response did not contain a message or tool call".into(),
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Option<DeepSeekMessage>,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekMessage {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<DeepSeekToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekToolCall {
    pub id: String,
    pub r#type: String,
    pub function: DeepSeekFunction,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DeepSeekToolDefinition {
    pub r#type: String,
    pub function: crate::completion::ToolDefinition,
}

impl From<crate::completion::ToolDefinition> for DeepSeekToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

/// The struct implementing the `CompletionModel` trait
#[derive(Clone)]
pub struct DeepSeekCompletionModel {
    pub client: Client,
    pub model: String,
}

impl CompletionModel for DeepSeekCompletionModel {
    type Response = DeepSeekResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<DeepSeekResponse>, crate::completion::CompletionError> {
        // 1. Build the array of messages from request.chat_history + user prompt
        // if request.preamble is set, it becomes "system" or the first message.
        // So let's gather them in the style "system" + "user" + chat_history => JSON messages.

        let mut messages_json = vec![];

        // If preamble is present, push a system message
        if let Some(preamble) = &request.preamble {
            messages_json.push(json!({
                "role": "system",
                "content": preamble,
            }));
        }

        // If chat_history is present, we can push them.
        // Typically, a "user" role is "USER" and an "assistant" role is "system" or "assistant"
        for msg in &request.chat_history {
            let role = match msg.role.as_str() {
                "system" => "system",
                "assistant" => "assistant",
                _ => "user",
            };
            messages_json.push(json!({
                "role": role,
                "content": msg.content,
            }));
        }

        // Add user’s prompt as well
        messages_json.push(json!({
            "role": "user",
            "content": request.prompt_with_context(),
        }));

        // 2. Prepare the body as DeepSeek expects
        let body = json!({
            "model": self.model,
            "messages": messages_json,
            "frequency_penalty": 0,
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "presence_penalty": 0,
            "temperature": request.temperature.unwrap_or(1.0),
            "top_p": 1,
            "logprobs": false,
            "stream": false,
        });

        // prepare tools
        let tools = if request.tools.is_empty() {
            json!({
                "tool_choice": "none",
            })
        } else {
            json!({
                "tools": request.tools.into_iter().map(DeepSeekToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let body = json_utils::merge(body, tools);

        // if user set additional_params, merge them:
        let final_body = if let Some(params) = request.additional_params {
            json_utils::merge(body, params)
        } else {
            body
        };

        // 3. Execute the HTTP call
        let url = format!("{}/chat/completions", self.client.base_url);
        let resp = self
            .client
            .http_client
            .post(url)
            .bearer_auth(&self.client.api_key)
            .json(&final_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(crate::completion::CompletionError::ProviderError(format!(
                "DeepSeek call failed: {status} - {text}"
            )));
        }

        let deep_seek_response: DeepSeekResponse = resp.json().await?;

        // 4. Convert DeepSeekResponse -> rig’s `CompletionResponse<DeepSeekResponse>`
        deep_seek_response.try_into()
    }
}

// ================================================================
// DeepSeek Completion API
// ================================================================
/// `deepseek-chat` completion model
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
/// `deepseek-reasoner` completion model
pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";
