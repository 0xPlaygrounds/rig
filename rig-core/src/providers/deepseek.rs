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
    json_utils,
};
use reqwest::Client as HttpClient;
use serde::Deserialize;
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
}

/// The response shape from the DeepSeek API
#[derive(Debug, Deserialize)]
pub struct DeepSeekResponse {
    // We'll match the JSON:
    pub choices: Option<Vec<Choice>>,
    // you may want usage or other fields
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Option<DeepSeekMessage>,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

/// The struct implementing the `CompletionModel` trait
#[derive(Clone)]
pub struct DeepSeekCompletionModel {
    pub client: Client,
    pub model: String,
}

impl CompletionModel for DeepSeekCompletionModel {
    type Response = DeepSeekResponse;

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
            "tool_choice": "none",
            "logprobs": false,
            "stream": false,
        });

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

        let json_resp: DeepSeekResponse = resp.json().await?;
        // 4. Convert DeepSeekResponse -> rig’s `CompletionResponse<DeepSeekResponse>`

        // If no choices or content, return an empty message
        let content = if let Some(choices) = &json_resp.choices {
            if let Some(choice) = choices.first() {
                if let Some(msg) = &choice.message {
                    msg.content.clone().unwrap_or_default()
                } else {
                    "".to_string()
                }
            } else {
                "".to_string()
            }
        } else {
            "".to_string()
        };

        // For now, we just treat it as a normal text message
        let model_choice = crate::completion::ModelChoice::Message(content);

        Ok(CompletionResponse {
            choice: model_choice,
            raw_response: json_resp,
        })
    }
}

// ================================================================
// DeepSeek Completion API
// ================================================================
/// `deepseek-chat` completion model
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
