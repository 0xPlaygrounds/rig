//! Moonshot API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::moonshot;
//!
//! let client = moonshot::Client::new("YOUR_API_KEY");
//!
//! let moonshot_model = client.completion_model(moonshot::MOONSHOT_CHAT);
//! ```

use crate::json_utils::merge;
use crate::message;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::{StreamingCompletionModel, StreamingResult};
use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    extractor::ExtractorBuilder,
    json_utils,
    providers::openai,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ================================================================
// Main Moonshot Client
// ================================================================
const MOONSHOT_API_BASE_URL: &str = "https://api.moonshot.cn/v1";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new Moonshot client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, MOONSHOT_API_BASE_URL)
    }

    /// Create a new Moonshot client with the given API key and base API URL.
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
                .build()
                .expect("Moonshot reqwest client should build"),
        }
    }

    /// Create a new Moonshot client from the `MOONSHOT_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("MOONSHOT_API_KEY").expect("MOONSHOT_API_KEY not set");
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
    /// use rig::providers::moonshot::{Client, self};
    ///
    /// // Initialize the Moonshot client
    /// let moonshot = Client::new("your-moonshot-api-key");
    ///
    /// let completion_model = moonshot.completion_model(moonshot::MOONSHOT_CHAT);
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
    error: MoonshotError,
}

#[derive(Debug, Deserialize)]
struct MoonshotError {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Moonshot Completion API
// ================================================================
pub const MOONSHOT_CHAT: &str = "moonshot-v1-128k";

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
    type Response = openai::CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "MoonShot completion error: {}", t);

            match serde_json::from_str::<ApiResponse<openai::CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "MoonShot completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
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

impl StreamingCompletionModel for CompletionModel {
    async fn stream(&self, request: CompletionRequest) -> Result<StreamingResult, CompletionError> {
        let mut request = self.create_completion_request(request)?;

        request = merge(request, json!({"stream": true}));

        let builder = self.client.post("/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}
