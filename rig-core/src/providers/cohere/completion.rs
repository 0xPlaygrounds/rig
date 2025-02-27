use std::collections::HashMap;

use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    extractor::ExtractorBuilder,
    json_utils, message, Embed, OneOrMany,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Cohere Completion API
// ================================================================
/// `command-r-plus` completion model
pub const COMMAND_R_PLUS: &str = "comman-r-plus";
/// `command-r` completion model
pub const COMMAND_R: &str = "command-r";
/// `command` completion model
pub const COMMAND: &str = "command";
/// `command-nightly` completion model
pub const COMMAND_NIGHTLY: &str = "command-nightly";
/// `command-light` completion model
pub const COMMAND_LIGHT: &str = "command-light";
/// `command-light-nightly` completion model
pub const COMMAND_LIGHT_NIGHTLY: &str = "command-light-nightly";

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub finish_reason: FinishReason,
    pub message: Message,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    MaxTokens,
    StopSequence,
    Complete,
    Error,
    ToolCall,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    billed_units: Option<BilledUnits>,
    tokens: Option<Tokens>
}

#[derive(Debug, Deserialize)]
pub struct BilledUnits {
    #[serde(default)]
    output_tokens: Option<f64>,
    #[serde(default)]
    classifications: Option<f64>,
    #[serde(default)]
    search_units: Option<f64>,
    #[serde(default)]
    input_tokens: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Tokens {
    #[serde(default)]
    input_tokens: Option<f64>,
    #[serde(default)]
    output_tokens: Option<f64>,
}

impl From<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    fn from(response: CompletionResponse) -> Self {
        let CompletionResponse {
            message: text,
            tool_calls,
            ..
        } = &response;

        let model_response = if !tool_calls.is_empty() {
            tool_calls
                .iter()
                .map(|tool_call| {
                    completion::AssistantContent::tool_call(
                        tool_call.name.clone(),
                        tool_call.name.clone(),
                        tool_call.parameters.clone(),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            vec![completion::AssistantContent::text(text.clone())]
        };

        completion::CompletionResponse {
            choice: OneOrMany::many(model_response).expect("There is atleast one content"),
            raw_response: response,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Document {
    pub id: String,
    pub data: HashMap<String, serde_json::Value>,
}

impl From<completion::Document> for Document {
    fn from(document: completion::Document) -> Self {
        let mut data: HashMap<String, serde_json::Value> = HashMap::new();

        // We use `.into()` here explictely since the `document.additional_props` type will likely
        //  evolve into `serde_json::Value` in the future.
        document
            .additional_props
            .into_iter()
            .for_each(|(key, value)| {
                data.insert(key, value.into());
            });
        data.insert("text".to_string(), document.text.into());

        Self {
            id: document.id,
            data,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub generation_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub search_query: SearchQuery,
    pub connector: Connector,
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub error_message: Option<String>,
    #[serde(default)]
    pub continue_on_failure: bool,
}

#[derive(Debug, Deserialize)]
pub struct Connector {
    pub id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatHistory {
    pub role: String,
    pub message: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Parameter {
    pub description: String,
    pub r#type: String,
    pub required: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameter_definitions: HashMap<String, Parameter>,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        fn convert_type(r#type: &serde_json::Value) -> String {
            fn convert_type_str(r#type: &str) -> String {
                match r#type {
                    "string" => "string".to_owned(),
                    "number" => "number".to_owned(),
                    "integer" => "integer".to_owned(),
                    "boolean" => "boolean".to_owned(),
                    "array" => "array".to_owned(),
                    "object" => "object".to_owned(),
                    _ => "string".to_owned(),
                }
            }
            match r#type {
                serde_json::Value::String(r#type) => convert_type_str(r#type.as_str()),
                serde_json::Value::Array(types) => convert_type_str(
                    types
                        .iter()
                        .find(|t| t.as_str() != Some("null"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("string"),
                ),
                _ => "string".to_owned(),
            }
        }

        let maybe_required = tool
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .map(|required| {
                required
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Self {
            name: tool.name,
            description: tool.description,
            parameter_definitions: tool
                .parameters
                .get("properties")
                .expect("Tool properties should exist")
                .as_object()
                .expect("Tool properties should be an object")
                .iter()
                .map(|(argname, argdef)| {
                    (
                        argname.clone(),
                        Parameter {
                            description: argdef
                                .get("description")
                                .expect("Argument description should exist")
                                .as_str()
                                .expect("Argument description should be a string")
                                .to_string(),
                            r#type: convert_type(
                                argdef.get("type").expect("Argument type should exist"),
                            ),
                            required: maybe_required.contains(&argname.as_str()),
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "UPPERCASE")]
pub enum Message {
    User {
        content: OneOrMany<Content>,
    },

    Assistant {
        content: OneOrMany<Content>,
        #[serde(default)]
        citations: Vec<Citation>,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
    },

    Tool {
        tool_results: Vec<ToolResult>,
    },

    System {
        content: String,
        tool_calls: Vec<ToolCall>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    Text { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    #[serde(default)]
    pub start: Option<u32>,
    #[serde(default)]
    pub end: Option<u32>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(rename = "type")]
    pub citation_type: Option<CitationType>,
    #[serde(default)]
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Source {
    Document {
        id: Option<String>,
        document: Option<serde_json::Map<String, serde_json::Value>>,
    },
    Tool {
        id: Option<String>,
        tool_output: Option<serde_json::Map<String, serde_json::Value>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CitationType {
    TextContent,
    Plan,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolResult {
    pub call: ToolCall,
    pub outputs: Vec<serde_json::Value>,
}

impl TryFrom<message::Message> for Message {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => message::Message::User {
                content: content.try_map(|content| match content {
                    message::UserContent::Text(message::Text { text }) => {
                        Ok(Content::Text { text })
                    }
                    _ => Err(message::MessageError::ConversionError(
                        "Only text content is supported by Cohere".to_owned(),
                    )),
                })?,
            },
            _ => Err(message::MessageError::ConversionError(
                "Only user messages are supported by Cohere".to_owned(),
            )),
        })
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

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let messages = completion_request
            .chat_history
            .into_iter()
            .map(Message::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let request = json!({
            "model": self.model,
            "preamble": completion_request.preamble,
            "messages": messages,
            "documents": completion_request.documents,
            "temperature": completion_request.temperature,
            "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
        });

        let response = self
            .client
            .post("/v1/chat")
            .json(
                &if let Some(ref params) = completion_request.additional_params {
                    json_utils::merge(request.clone(), params.clone())
                } else {
                    request.clone()
                },
            )
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(completion) => Ok(completion.into()),
                ApiResponse::Err(error) => Err(CompletionError::ProviderError(error.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}
