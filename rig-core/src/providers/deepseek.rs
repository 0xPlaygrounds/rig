//! DeepSeek API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::deepseek;
//!
//! let client = deepseek::Client::new("YOUR_API_KEY");
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_CHAT);
//! ```

use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    extractor::ExtractorBuilder,
    json_utils,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com/v1";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new DeepSeek client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, DEEPSEEK_API_BASE_URL)
    }

    /// Create a new DeepSeek client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .timeout(Duration::from_secs(120))
                .build()
                .expect("DeepSeek reqwest client should build"),
        }
    }

    /// Create a new DeepSeek client from the `DEEPSEEK_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set");
        Self::new(&api_key)
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::deepseek::{Client, self};
    ///
    /// // Initialize the DeepSeek client
    /// let deepseek = Client::new("your-deepseek-api-key");
    ///
    /// let completion_model = deepseek.completion_model(deepseek::DEEPSEEK_CHAT);
    /// ```
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
    error: DeepSeekError,
}

#[derive(Debug, Deserialize)]
struct DeepSeekError {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// DeepSeek Completion API
// ================================================================
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.error.message)
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(value: CompletionResponse) -> std::prelude::v1::Result<Self, Self::Error> {
        match value.choices.as_slice() {
            [Choice {
                message:
                    Message {
                        content: Some(content),
                        ..
                    },
                ..
            }, ..] => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(content.to_string()),
                raw_response: value,
            }),
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a message".into(),
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

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
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    async fn completion(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let mut full_history = if let Some(preamble) = &completion_request.preamble {
            vec![completion::Message {
                role: "system".into(),
                content: preamble.clone(),
            }]
        } else {
            vec![]
        };

        full_history.append(&mut completion_request.chat_history);

        full_history.push(completion::Message {
            role: "user".into(),
            content: completion_request.prompt_with_context(),
        });

        let request = json!({
            "model": self.model,
            "messages": full_history,
            "temperature": completion_request.temperature,
        });

        let response = self
            .client
            .post("/chat/completions")
            .json(
                &if let Some(params) = completion_request.additional_params {
                    json_utils::merge(request, params)
                } else {
                    request
                },
            )
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "DeepSeek completion token usage: {}",
                        response.usage
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.error.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}
