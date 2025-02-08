//! Ollama API client and Rig integration
//!
//! # Example
//! ```rust
//! use rig::providers::ollama;
//!
//! // Create a new Ollama client (defaults to http://localhost:11434)
//! let client = ollama::Client::new();
//!
//! // Create a completion model interface using, for example, the "llama3.2" model
//! let comp_model = client.completion_model("llama3.2");
//!
//! let req = rig::completion::CompletionRequest {
//!     preamble: Some("You are now a humorous AI assistant.".to_owned()),
//!     chat_history: vec![],  // internal messages, if any
//!     prompt: "Please tell me why the sky is blue.".to_owned(),
//!     temperature: 0.7,
//!     additional_params: None,
//!     tools: vec![],
//! };
//!
//! let response = comp_model.completion(req).await.unwrap();
//! println!("Ollama completion response: {:?}", response.choice);
//!
//! // Create an embedding interface using the "all-minilm" model
//! let emb_model = ollama::Client::new().embedding_model("all-minilm");
//! let docs = vec![
//!     "Why is the sky blue?".to_owned(),
//!     "Why is the grass green?".to_owned()
//! ];
//! let embeddings = emb_model.embed_texts(docs).await.unwrap();
//! println!("Embedding response: {:?}", embeddings);
//!
//! // Also create an agent and extractor if needed
//! let agent = client.agent("llama3.2");
//! let extractor = client.extractor::<serde_json::Value>("llama3.2");
//! ```


// =================================================================
// Imports
// =================================================================

use std::{convert::Infallible, str::FromStr, convert::TryFrom};

use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    extractor::ExtractorBuilder,
    json_utils,
    message::{AudioMediaType, ImageDetail},
    one_or_many::string_or_one_or_many,
    Embed, OneOrMany,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use reqwest;

// =================================================================
// FromStr implementations for provider types (used by deserializers)
// =================================================================

impl FromStr for SystemContent {
    type Err = Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent {
            r#type: SystemContentType::Text,
            text: s.to_owned(),
        })
    }
}

impl FromStr for UserContent {
    type Err = Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text { text: s.to_owned() })
    }
}

impl FromStr for AssistantContent {
    type Err = Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent::Text { text: s.to_owned() })
    }
}

// =================================================================
// Main Ollama Client
// =================================================================

/// Default Ollama API base URL.
const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new Ollama client using the default API URL.
    pub fn new() -> Self {
        Self::from_url(OLLAMA_API_BASE_URL)
    }

    /// Create a new Ollama client using the specified API URL.
    pub fn from_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_owned(),
            http_client: reqwest::Client::builder()
                .build()
                .expect("Ollama reqwest client should build"),
        }
    }

    /// Create a new HTTP POST request builder.
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path);
        self.http_client.post(url)
    }

    /// Create an embedding model interface.
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, 0)
    }

    /// Create an embedding model interface with a specified number of dimensions.
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embeddings builder.
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }

    /// Create a completion model interface.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an agent builder.
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder.
    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

// =================================================================
// Generic API Error and Response Structures
// =================================================================

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

// =================================================================
// Embedding API
// =================================================================

/// Example constant for an Ollama embedding model.
pub const ALL_MINILM: &str = "all-minilm";

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_owned(),
            ndims,
        }
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;
    fn ndims(&self) -> usize {
        self.ndims
    }
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let docs: Vec<String> = documents.into_iter().collect();
        let payload = json!({
            "model": self.model,
            "input": docs,
        });
        let response = self.client.post("api/embed")
            .json(&payload)
            .send()
            .await
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
        if response.status().is_success() {
            let api_resp: EmbeddingResponse = response.json()
                .await
                .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
            if api_resp.embeddings.len() != docs.len() {
                return Err(EmbeddingError::ResponseError("Number of returned embeddings does not match input".into()));
            }
            Ok(api_resp.embeddings.into_iter()
                .zip(docs.into_iter())
                .map(|(vec, document)| embeddings::Embedding { document, vec })
                .collect())
        } else {
            Err(EmbeddingError::ProviderError(response.text().await?))
        }
    }
}

// =================================================================
// Completion API
// =================================================================

/// Example constants for Ollama completion models.
pub const LLAMA3_2: &str = "llama3.2";
pub const LLAVA: &str = "llava";
pub const MISTRAL: &str = "mistral";

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub context: Option<serde_json::Value>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

/// For single-turn generation, we convert a CompletionResponse into our unified CompletionResponse type
impl TryFrom<CompletionResponse> for completion::CompletionResponse<serde_json::Value> {
    type Error = CompletionError;
    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let assistant = completion::AssistantContent::text(&response.response);
        // Instead of using map_err, we check manually:
        let choice = OneOrMany::one(assistant);
        if choice.is_empty() {
            return Err(CompletionError::ResponseError("Empty response".into()));
        }
        let raw = serde_json::to_value(&response)
            .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
        Ok(completion::CompletionResponse { choice, raw_response: raw })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl TryFrom<ChatResponse> for completion::CompletionResponse<serde_json::Value> {
    type Error = CompletionError;
    fn try_from(resp: ChatResponse) -> Result<Self, Self::Error> {
        match resp.message {
            Message::Assistant { ref content, .. } => {
                let texts: Vec<completion::AssistantContent> = content.into_iter().map(|c| {
                    match c {
                        AssistantContent::Text { text } => completion::AssistantContent::text(text),
                        AssistantContent::Refusal { refusal } => completion::AssistantContent::text(refusal),
                    }
                }).collect();
                let choice = OneOrMany::many(texts)
                    .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
                if choice.is_empty() {
                    return Err(CompletionError::ResponseError("Empty chat response".into()));
                }
                let raw = serde_json::to_value(&resp)
                    .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
                Ok(completion::CompletionResponse { choice, raw_response: raw })
            },
            _ => Err(CompletionError::ResponseError("Chat response does not include an assistant message".into())),
        }
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// The model name (e.g. "llama3.2").
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self { client, model: model.to_owned() }
    }
}

/// We set our associated type Response to be a JSON value.
impl completion::CompletionModel for CompletionModel {
    type Response = serde_json::Value;
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        if !completion_request.chat_history.is_empty() {
            // Chat mode:
            // Assume prompt_with_context() returns a single internal message.
            let prompt_internal: crate::message::Message = completion_request.prompt_with_context();
            let prompt_msg: Message = prompt_internal.try_into()?;
            let chat_history: Vec<Message> = completion_request
                .chat_history
                .into_iter()
                .map(|m| m.try_into())
                .collect::<Result<_, _>>()?;
            let mut full_history = Vec::new();
            if let Some(preamble) = &completion_request.preamble {
                full_history.push(Message::system(preamble));
            }
            full_history.extend(chat_history);
            full_history.push(prompt_msg);
            let mut request_payload = json!({
                "model": self.model,
                "messages": full_history,
                "stream": false,
            });
            if let Some(params) = &completion_request.additional_params {
                request_payload = json_utils::merge(request_payload, params.clone());
            }
            let response = self.client.post("api/chat")
                .json(&request_payload)
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            if response.status().is_success() {
                let text = response.text().await.map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                tracing::debug!(target: "rig", "Ollama chat response: {}", text);
                let chat_resp: ChatResponse = serde_json::from_str(&text)
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                let conv: completion::CompletionResponse<serde_json::Value> = chat_resp.try_into()?;
                Ok(conv)
            } else {
                let err_text = response.text().await.map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                Err(CompletionError::ProviderError(err_text))
            }
        } else {
            // Single-turn mode:
            let full_prompt = completion_request.prompt.clone();
            let mut request_payload = json!({
                "model": self.model,
                "prompt": full_prompt,
                "stream": false,
            });
            if let Some(params) = &completion_request.additional_params {
                request_payload = json_utils::merge(request_payload, params.clone());
            }
            let response = self.client.post("api/generate")
                .json(&request_payload)
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            if response.status().is_success() {
                let text = response.text().await.map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                tracing::debug!(target: "rig", "Ollama generate response: {}", text);
                let gen_resp: CompletionResponse = serde_json::from_str(&text)
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                let conv: completion::CompletionResponse<serde_json::Value> = gen_resp.try_into()?;
                Ok(conv)
            } else {
                let err_text = response.text().await.map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                Err(CompletionError::ProviderError(err_text))
            }
        }
    }
}

// =================================================================
// Provider Message Definitions and Conversions (following openai.rs)
// =================================================================

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, deserialize_with = "crate::json_utils::string_or_vec")]
        content: Vec<AssistantContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<AudioAssistant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, deserialize_with = "crate::json_utils::null_or_vec")]
        tool_calls: Vec<ToolCall>,
    },
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "Tool")]
    ToolResult {
        tool_call_id: String,
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<ToolResultContent>,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::many(vec![SystemContent {
                r#type: SystemContentType::Text,
                text: content.to_owned(),
            }]).expect("Non-empty system content"),
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AudioAssistant {
    pub id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    pub r#type: SystemContentType,
    pub text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
    #[default]
    Text,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
    Refusal { refusal: String },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text { text: String },
    Image { image_url: ImageUrl },
    Audio { input_audio: InputAudio },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: ImageDetail,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: AudioMediaType,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolResultContent {
    pub text: String,
}

impl FromStr for ToolResultContent {
    type Err = Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ToolResultContent { text: s.to_owned() })
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent { text: s }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "crate::json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

// =================================================================
// Conversion from internal Rig message (crate::message::Message)
// to provider Message. (Only User and Assistant variants are supported.)
// =================================================================

impl TryFrom<crate::message::Message> for Message {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        use crate::message::Message as InternalMessage;
        match internal_msg {
            InternalMessage::User { content } => {
                let converted: Result<Vec<UserContent>, _> = content.into_iter().map(|uc| {
                    match uc {
                        crate::message::UserContent::Text(t) => Ok(UserContent::Text { text: t.text }),
                        crate::message::UserContent::Image(img) => Ok(UserContent::Image {
                            image_url: ImageUrl {
                                url: img.data,
                                detail: img.detail.unwrap_or_default(),
                            },
                        }),
                        crate::message::UserContent::Audio(audio) => Ok(UserContent::Audio {
                            input_audio: InputAudio {
                                data: audio.data,
                                format: audio.media_type.unwrap_or(AudioMediaType::MP3),
                            },
                        }),
                        other => Err(crate::message::MessageError::ConversionError(format!("Unsupported user content: {:?}", other))),
                    }
                }).collect();
                let one = OneOrMany::many(converted?).map_err(|e| crate::message::MessageError::ConversionError(e.to_string()))?;
                Ok(Message::User { content: one, name: None })
            }
            InternalMessage::Assistant { content } => {
                let converted: Result<Vec<AssistantContent>, _> = content.into_iter().map(|ac| {
                    match ac {
                        crate::message::AssistantContent::Text(t) => Ok(AssistantContent::Text { text: t.text }),
                        other => Err(crate::message::MessageError::ConversionError(format!("Unsupported assistant content: {:?}", other))),
                    }
                }).collect();
                Ok(Message::Assistant {
                    content: converted?,
                    refusal: None,
                    audio: None,
                    name: None,
                    tool_calls: vec![],
                })
            }
        }
    }
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_path_to_error::deserialize;

    #[test]
    fn test_deserialize_message() {
        let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;
        let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                }
            ],
            "tool_calls": null
        }
        "#;
        let assistant_message_json3 = r#"
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_h89ipqYUjEpCPI6SxspMnoUU",
                    "type": "function",
                    "function": {
                        "name": "subtract",
                        "arguments": "{\"x\": 2, \"y\": 5}"
                    }
                }
            ],
            "content": null,
            "refusal": null
        }
        "#;
        let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                },
                {
                    "type": "audio",
                    "input_audio": {
                        "data": "...",
                        "format": "mp3"
                    }
                }
            ]
        }
        "#;
        let assistant_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err.inner())
            })
        };
        let assistant_message2: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err.inner())
            })
        };
        let assistant_message3: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json3);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err.inner())
            })
        };
        let user_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(user_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err.inner())
            })
        };
        if let Message::Assistant { content, .. } = assistant_message {
            assert_eq!(content[0],
                       AssistantContent::Text { text: "\n\nHello there, how may I assist you today?".to_owned() }
            );
        } else {
            panic!("Expected assistant message");
        }
        if let Message::Assistant { content, tool_calls, .. } = assistant_message2 {
            assert_eq!(content[0],
                       AssistantContent::Text { text: "\n\nHello there, how may I assist you today?".to_owned() }
            );
            assert_eq!(tool_calls, vec![]);
        } else {
            panic!("Expected assistant message");
        }
        if let Message::Assistant { content, tool_calls, refusal, .. } = assistant_message3 {
            assert!(content.is_empty());
            assert!(refusal.is_none());
            assert_eq!(tool_calls[0],
                       ToolCall {
                           id: "call_h89ipqYUjEpCPI6SxspMnoUU".to_owned(),
                           r#type: ToolType::Function,
                           function: Function {
                               name: "subtract".to_owned(),
                               arguments: serde_json::json!({"x": 2, "y": 5}),
                           },
                       }
            );
        } else {
            panic!("Expected assistant message");
        }
        if let Message::User { content, .. } = user_message {
            let mut iter = content.into_iter();
            let first = iter.next().unwrap();
            let second = iter.next().unwrap();
            assert_eq!(first,
                       UserContent::Text { text: "What's in this image?".to_owned() }
            );
            assert_eq!(second,
                       UserContent::Image {
                           image_url: ImageUrl {
                               url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_owned(),
                               detail: ImageDetail::default()
                           }
                       }
            );
        } else {
            panic!("Expected user message");
        }
    }
    #[test]
    fn test_message_to_message_conversion() {
        let internal_user = crate::message::Message::User {
            content: OneOrMany::many(vec![crate::message::UserContent::text("Hello")]).expect("Non-empty"),
        };
        let internal_assistant = crate::message::Message::Assistant {
            content: OneOrMany::many(vec![crate::message::AssistantContent::text("Hi there!")]).expect("Non-empty"),
        };
        let converted_user: Message = Message::try_from(internal_user).unwrap();
        let converted_assistant: Message = Message::try_from(internal_assistant).unwrap();
        if let Message::User { ref content, .. } = converted_user {
            assert_eq!(content.first(), UserContent::Text { text: "Hello".to_owned() });
        } else {
            panic!("Expected user message");
        }
        if let Message::Assistant { ref content, .. } = converted_assistant {
            assert_eq!(content.first(), Some(&AssistantContent::Text { text: "Hi there!".to_owned() }));
        } else {
            panic!("Expected assistant message");
        }
    }
}