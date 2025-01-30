//! Anthropic completion api implementation

use std::iter;

use crate::{
    completion::{self, CompletionError},
    json_utils,
};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Client;

// ================================================================
// Anthropic Completion API
// ================================================================
/// `claude-3-5-sonnet-latest` completion model
pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-latest";

/// `claude-3-5-haiku-latest` completion model
pub const CLAUDE_3_5_HAIKU: &str = "claude-3-5-haiku-latest";

/// `claude-3-5-haiku-latest` completion model
pub const CLAUDE_3_OPUS: &str = "claude-3-opus-latest";

/// `claude-3-sonnet-20240229` completion model
pub const CLAUDE_3_SONNET: &str = "claude-3-sonnet-20240229";

/// `claude-3-haiku-20240307` completion model
pub const CLAUDE_3_HAIKU: &str = "claude-3-haiku-20240307";

pub const ANTHROPIC_VERSION_2023_01_01: &str = "2023-01-01";
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
pub const ANTHROPIC_VERSION_LATEST: &str = ANTHROPIC_VERSION_2023_06_01;

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    String(String),
    Text {
        r#type: String,
        text: String,
    },
    ToolUse {
        r#type: String,
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub output_tokens: u64,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nCache read input tokens: {}\nCache creation input tokens: {}\nOutput tokens: {}",
            self.input_tokens,
            match self.cache_read_input_tokens {
                Some(token) => token.to_string(),
                None => "n/a".to_string(),
            },
            match self.cache_creation_input_tokens {
                Some(token) => token.to_string(),
                None => "n/a".to_string(),
            },
            self.output_tokens
        )
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    Ephemeral,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        if let Some(tool_use) = response.content.iter().find_map(|content| match content {
            Content::ToolUse {
                name, input, id, ..
            } => Some((name.clone(), id.clone(), input.clone())),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::ToolCall(tool_use.0, tool_use.1, tool_use.2),
                raw_response: response,
            });
        }

        if let Some(text_content) = response.content.iter().find_map(|content| match content {
            Content::String(text) => Some(text.clone()),
            Content::Text { text, .. } => Some(text.clone()),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(text_content),
                raw_response: response,
            });
        }

        Err(CompletionError::ResponseError(
            "Response did not contain a message or tool call".into(),
        ))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl From<completion::Message> for Message {
    fn from(message: completion::Message) -> Self {
        Self {
            role: message.role,
            content: message.content,
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
    pub default_max_tokens: Option<u64>,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            default_max_tokens: calculate_max_tokens(model),
        }
    }
}

/// Anthropic requires a `max_tokens` parameter to be set, which is dependent on the model. If not
/// set or if set too high, the request will fail. The following values are based on the models
/// available at the time of writing.
///
/// Dev Note: This is really bad design, I'm not sure why they did it like this..
fn calculate_max_tokens(model: &str) -> Option<u64> {
    if model.starts_with("claude-3-5-sonnet") || model.starts_with("claude-3-5-haiku") {
        Some(8192)
    } else if model.starts_with("claude-3-opus")
        || model.starts_with("claude-3-sonnet")
        || model.starts_with("claude-3-haiku")
    {
        Some(4096)
    } else {
        None
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Metadata {
    user_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Note: Ideally we'd introduce provider-specific Request models to handle the
        // specific requirements of each provider. For now, we just manually check while
        // building the request as a raw JSON document.

        let prompt_with_context = completion_request.prompt_with_context();

        // Check if max_tokens is set, required for Anthropic
        let max_tokens = if let Some(tokens) = completion_request.max_tokens {
            tokens
        } else if let Some(tokens) = self.default_max_tokens {
            tokens
        } else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let mut request = json!({
            "model": self.model,
            "messages": completion_request
                .chat_history
                .into_iter()
                .map(Message::from)
                .chain(iter::once(Message {
                    role: "user".to_owned(),
                    content: prompt_with_context,
                }))
                .collect::<Vec<_>>(),
            "max_tokens": max_tokens,
            "system": completion_request.preamble.unwrap_or("".to_string()),
        });

        if let Some(temperature) = completion_request.temperature {
            json_utils::merge_inplace(&mut request, json!({ "temperature": temperature }));
        }

        if !completion_request.tools.is_empty() {
            json_utils::merge_inplace(
                &mut request,
                json!({
                    "tools": completion_request
                        .tools
                        .into_iter()
                        .map(|tool| ToolDefinition {
                            name: tool.name,
                            description: Some(tool.description),
                            input_schema: tool.parameters,
                        })
                        .collect::<Vec<_>>(),
                    "tool_choice": ToolChoice::Auto,
                }),
            );
        }

        if let Some(ref params) = completion_request.additional_params {
            json_utils::merge_inplace(&mut request, params.clone())
        }

        let response = self
            .client
            .post("/v1/messages")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Message(completion) => {
                    tracing::info!(target: "rig",
                        "Anthropic completion token usage: {}",
                        completion.usage
                    );
                    completion.try_into()
                }
                ApiResponse::Error(error) => Err(CompletionError::ProviderError(error.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiResponse<T> {
    Message(T),
    Error(ApiErrorResponse),
}
