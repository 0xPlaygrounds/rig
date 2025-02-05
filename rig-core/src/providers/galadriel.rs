//! Galadriel API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::galadriel;
//!
//! let client = galadriel::Client::new("YOUR_API_KEY", None);
//! // to use a fine-tuned model
//! // let client = galadriel::Client::new("YOUR_API_KEY", "FINE_TUNE_API_KEY");
//!
//! let gpt4o = client.completion_model(galadriel::GPT_4O);
//! ```
use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    extractor::ExtractorBuilder,
    json_utils, message, OneOrMany,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::openai;

// ================================================================
// Main Galadriel Client
// ================================================================
const GALADRIEL_API_BASE_URL: &str = "https://api.galadriel.com/v1/verified";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new Galadriel client with the given API key and optional fine-tune API key.
    pub fn new(api_key: &str, fine_tune_api_key: Option<&str>) -> Self {
        Self::from_url_with_optional_key(api_key, GALADRIEL_API_BASE_URL, fine_tune_api_key)
    }

    /// Create a new Galadriel client with the given API key, base API URL, and optional fine-tune API key.
    pub fn from_url(api_key: &str, base_url: &str, fine_tune_api_key: Option<&str>) -> Self {
        Self::from_url_with_optional_key(api_key, base_url, fine_tune_api_key)
    }

    pub fn from_url_with_optional_key(
        api_key: &str,
        base_url: &str,
        fine_tune_api_key: Option<&str>,
    ) -> Self {
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
                    if let Some(key) = fine_tune_api_key {
                        headers.insert(
                            "Fine-Tune-Authorization",
                            format!("Bearer {}", key)
                                .parse()
                                .expect("Bearer token should parse"),
                        );
                    }
                    headers
                })
                .build()
                .expect("Galadriel reqwest client should build"),
        }
    }

    /// Create a new Galadriel client from the `GALADRIEL_API_KEY` environment variable,
    /// and optionally from the `GALADRIEL_FINE_TUNE_API_KEY` environment variable.
    /// Panics if the `GALADRIEL_API_KEY` environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("GALADRIEL_API_KEY").expect("GALADRIEL_API_KEY not set");
        let fine_tune_api_key = std::env::var("GALADRIEL_FINE_TUNE_API_KEY").ok();
        Self::new(&api_key, fine_tune_api_key.as_deref())
    }
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::galadriel::{Client, self};
    ///
    /// // Initialize the Galadriel client
    /// let galadriel = Client::new("your-galadriel-api-key", None);
    ///
    /// let gpt4 = galadriel.completion_model(galadriel::GPT_4);
    /// ```
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an agent builder with the given completion model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::galadriel::{Client, self};
    ///
    /// // Initialize the Galadriel client
    /// let galadriel = Client::new("your-galadriel-api-key", None);
    ///
    /// let agent = galadriel.agent(galadriel::GPT_4)
    ///    .preamble("You are comedian AI with a mission to make people laugh.")
    ///    .temperature(0.0)
    ///    .build();
    /// ```
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
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
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

// ================================================================
// Galadriel Completion API
// ================================================================
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini completion model
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-0125` completion model
pub const GPT_35_TURBO_0125: &str = "gpt-3.5-turbo-0125";
/// `gpt-3.5-turbo-1106` completion model
pub const GPT_35_TURBO_1106: &str = "gpt-3.5-turbo-1106";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let Choice { message, .. } = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let mut content = message
            .content
            .as_ref()
            .map(|c| vec![completion::AssistantContent::text(c)])
            .unwrap_or_default();

        content.extend(message.tool_calls.iter().map(|call| {
            completion::AssistantContent::tool_call(
                &call.function.name,
                &call.function.name,
                call.function.arguments.clone(),
            )
        }));

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    pub tool_calls: Vec<openai::ToolCall>,
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        let tool_calls: Vec<message::ToolCall> = message
            .tool_calls
            .into_iter()
            .map(|tool_call| tool_call.into())
            .collect();

        match message.role.as_str() {
            "user" => Ok(Self::User {
                content: OneOrMany::one(
                    message
                        .content
                        .map(|content| message::UserContent::text(&content))
                        .ok_or_else(|| {
                            message::MessageError::ConversionError("Empty user message".to_string())
                        })?,
                ),
            }),
            "assistant" => Ok(Self::Assistant {
                content: OneOrMany::many(
                    tool_calls
                        .into_iter()
                        .map(message::AssistantContent::ToolCall)
                        .chain(
                            message
                                .content
                                .map(|content| message::AssistantContent::text(&content))
                                .into_iter(),
                        ),
                )
                .map_err(|_| {
                    message::MessageError::ConversionError("Empty assistant message".to_string())
                })?,
            }),
            _ => Err(message::MessageError::ConversionError(format!(
                "Unknown role: {}",
                message.role
            ))),
        }
    }
}

impl TryFrom<message::Message> for Message {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => Ok(Self {
                role: "user".to_string(),
                content: content.iter().find_map(|c| match c {
                    message::UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                }),
                tool_calls: vec![],
            }),
            message::Message::Assistant { content } => {
                let mut text_content: Option<String> = None;
                let mut tool_calls = vec![];

                for c in content.iter() {
                    match c {
                        message::AssistantContent::Text(text) => {
                            text_content = Some(
                                text_content
                                    .map(|mut existing| {
                                        existing.push('\n');
                                        existing.push_str(&text.text);
                                        existing
                                    })
                                    .unwrap_or_else(|| text.text.clone()),
                            );
                        }
                        message::AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(tool_call.clone().into());
                        }
                    }
                }

                Ok(Self {
                    role: "assistant".to_string(),
                    content: text_content,
                    tool_calls,
                })
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
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

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Add preamble to chat history (if available)
        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message {
                role: "system".to_string(),
                content: Some(preamble.to_string()),
                tool_calls: vec![],
            }],
            None => vec![],
        };

        // Convert prompt to user message
        let prompt: Message = completion_request.prompt_with_context().try_into()?;

        // Convert existing chat history
        let chat_history: Vec<Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Message>, _>>()?;

        // Combine all messages into a single history
        full_history.extend(chat_history);
        full_history.push(prompt);

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
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

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
            let t = response.text().await?;
            tracing::debug!(target: "rig", "Galadriel completion error: {}", t);

            match serde_json::from_str::<ApiResponse<CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Galadriel completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}
