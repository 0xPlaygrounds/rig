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
//!     chat_history: vec![],  // internal messages (if any)
//!     prompt: /* a crate::message::Message value representing the prompt */
//!         rig::message::Message::User {
//!             content: rig::one_or_many::OneOrMany::one(rig::message::UserContent::text("Please tell me why the sky is blue.")),
//!             name: None
//!         },
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

use std::{convert::Infallible, convert::TryFrom, str::FromStr};

use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    extractor::ExtractorBuilder,
    json_utils,
    message::{AudioMediaType, ImageDetail},
    Embed, OneOrMany,
};
use reqwest;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// =================================================================
// FromStr implementations for provider types (for deserialization)
// =================================================================

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

/// Main Ollama Client
const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

#[derive(Clone, Default)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}
impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    pub fn new() -> Self {
        Self::from_url(OLLAMA_API_BASE_URL)
    }
    pub fn from_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_owned(),
            http_client: reqwest::Client::builder()
                .build()
                .expect("Ollama reqwest client should build"),
        }
    }
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path);
        self.http_client.post(url)
    }
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, 0)
    }
    pub fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }
    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

// =================================================================
// API Error and Response Structures
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

/// =================================================================
/// Embedding API
/// =================================================================
pub const ALL_MINILM: &str = "all-minilm";
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";

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
        let response = self
            .client
            .post("api/embed")
            .json(&payload)
            .send()
            .await
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
        if response.status().is_success() {
            let api_resp: EmbeddingResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
            if api_resp.embeddings.len() != docs.len() {
                return Err(EmbeddingError::ResponseError(
                    "Number of returned embeddings does not match input".into(),
                ));
            }
            Ok(api_resp
                .embeddings
                .into_iter()
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

impl TryFrom<CompletionResponse> for completion::CompletionResponse<serde_json::Value> {
    type Error = CompletionError;
    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let assistant = completion::AssistantContent::text(&response.response);
        let choice = OneOrMany::one(assistant);
        if choice.is_empty() {
            return Err(CompletionError::ResponseError("Empty response".into()));
        }
        let raw = serde_json::to_value(&response)
            .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
        Ok(completion::CompletionResponse {
            choice,
            raw_response: raw,
        })
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
                // Since the provider Message's content is now a String,
                // create a single AssistantContent from it.
                let assistant_content = completion::AssistantContent::text(content);
                // Directly construct OneOrMany from the assistant_content.
                let choice = OneOrMany::one(assistant_content);
                if choice.is_empty() {
                    return Err(CompletionError::ResponseError("Empty chat response".into()));
                }
                let raw = serde_json::to_value(&resp)
                    .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
                Ok(completion::CompletionResponse {
                    choice,
                    raw_response: raw,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Chat response does not include an assistant message".into(),
            )),
        }
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
            model: model.to_owned(),
        }
    }
}

/// In our unified API, we set the associated Response type to be a JSON value.
/// -----------------------------
/// Additional conversion implementations
/// -----------------------------
/// This implementation allows converting an internal message (crate::message::Message)
/// into a Vec of provider Message. This is used when combining prompt context.
impl TryFrom<crate::message::Message> for Vec<Message> {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        // For now, simply convert the internal message to a provider Message and wrap it in a Vec.
        Ok(vec![Message::try_from(internal_msg)?])
    }
}

/// This implementation allows the '?' operator to convert an Infallible error into a CompletionError.
impl From<std::convert::Infallible> for CompletionError {
    fn from(_: std::convert::Infallible) -> Self {
        CompletionError::ProviderError("Infallible error".to_string())
    }
}

/// -----------------------------
/// CompletionModel implementation
/// -----------------------------
/// Helper method for provider Message conversion to a plain prompt string.
impl completion::CompletionModel for CompletionModel {
    type Response = serde_json::Value;
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        // Convert internal prompt using prompt_with_context() into Vec<Message>
        let prompt: Vec<Message> = completion_request.prompt_with_context().try_into()?;
        let default_options = json!({
            "temperature": completion_request.temperature,
        });
        // Determine chat mode: if chat history is non-empty OR prompt returns more than one message, use chat mode.
        if !completion_request.chat_history.is_empty() || prompt.len() > 1 {
            // Chat mode: build full conversation history as an array.
            let mut full_history: Vec<Message> = match &completion_request.preamble {
                Some(preamble) => vec![Message::system(preamble)],
                None => vec![],
            };

            // Convert chat history: each internal message may yield multiple provider messages.
            let chat_history: Vec<Message> = completion_request
                .chat_history
                .into_iter()
                .map(|m| m.try_into())
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect();

            full_history.extend(chat_history);
            full_history.extend(prompt);
            let options = if let Some(extra) = completion_request.additional_params {
                json_utils::merge(default_options, extra)
            } else {
                default_options
            };

            let request_payload = json!({
                "model": self.model,
                "messages": full_history,  // Send as an array, per API specification.
                "temperature": options,
                "stream": false,
            });

            tracing::debug!(target: "rig", "Chat mode payload: {}", request_payload);
            let response = self
                .client
                .post("api/chat")
                .json(&request_payload)
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            if response.status().is_success() {
                let text = response
                    .text()
                    .await
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                tracing::debug!(target: "rig", "Ollama chat response: {}", text);
                let chat_resp: ChatResponse = serde_json::from_str(&text)
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                let conv: completion::CompletionResponse<serde_json::Value> =
                    chat_resp.try_into()?;
                Ok(conv)
            } else {
                let err_text = response
                    .text()
                    .await
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                Err(CompletionError::ProviderError(err_text))
            }
        } else {
            // Single-turn mode: if prompt_with_context() returns empty, fallback to converting internal prompt to a plain string.
            let full_prompt = provider_messages_to_string(&prompt);
            let mut request_payload = json!({
                "model": self.model,
                "prompt": full_prompt, // prompt must be a string
                "temperature": completion_request.temperature,
                "stream": false,
            });
            if let Some(params) = completion_request.additional_params {
                request_payload = json_utils::merge(request_payload, params);
            }
            tracing::debug!(target: "rig", "Single-turn payload: {}", request_payload);
            let response = self
                .client
                .post("api/generate")
                .json(&request_payload)
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            if response.status().is_success() {
                let text = response
                    .text()
                    .await
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                tracing::debug!(target: "rig", "Ollama generate response: {}", text);
                let gen_resp: CompletionResponse = serde_json::from_str(&text)
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                let conv: completion::CompletionResponse<serde_json::Value> =
                    gen_resp.try_into()?;
                Ok(conv)
            } else {
                let err_text = response
                    .text()
                    .await
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                Err(CompletionError::ProviderError(err_text))
            }
        }
    }
}

// Helper function: convert a slice of provider Message into a plain string.
// For each message, we extract the text from User, Assistant or System variants.
fn provider_messages_to_string(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|msg| msg.to_prompt())
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<AudioAssistant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "Tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

// Implement a helper method on provider Message to extract text for prompt.
impl Message {
    pub fn to_prompt(&self) -> String {
        match self {
            Message::User { content, .. } => content.clone(),
            Message::Assistant { content, .. } => content.clone(),
            Message::System { content, .. } => content.clone(),
            Message::ToolResult { content, .. } => content.clone(),
        }
    }

    // A convenience method to create a system message from a string.
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            images: None,
            name: None,
        }
    }
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    r#type: SystemContentType,
    text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
    #[default]
    Text,
}

impl From<String> for SystemContent {
    fn from(s: String) -> Self {
        SystemContent {
            r#type: SystemContentType::default(),
            text: s,
        }
    }
}

impl FromStr for SystemContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent {
            r#type: SystemContentType::default(),
            text: s.to_string(),
        })
    }
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AudioAssistant {
    pub id: String,
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
// to provider Message.
// (Only User, Assistant and System variants are supported.)
// =================================================================

impl TryFrom<crate::message::Message> for Message {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        use crate::message::Message as InternalMessage;
        match internal_msg {
            InternalMessage::User { content, .. } => {
                let mut texts = Vec::new();
                let mut images = Vec::new();
                for uc in content.into_iter() {
                    match uc {
                        crate::message::UserContent::Text(t) => texts.push(t.text),
                        crate::message::UserContent::Image(img) => images.push(img.data),
                        crate::message::UserContent::Audio(_audio) => {}
                        _ => {}
                    }
                }
                let content_str = texts.join(" ");
                let images_opt = if images.is_empty() {
                    None
                } else {
                    Some(images)
                };
                Ok(Message::User {
                    content: content_str,
                    images: images_opt,
                    name: None,
                })
            }
            InternalMessage::Assistant { content, .. } => {
                let mut texts = Vec::new();
                let images = Vec::new();
                for ac in content.into_iter() {
                    match ac {
                        crate::message::AssistantContent::Text(t) => texts.push(t.text),
                        _ => {}
                    }
                }
                let content_str = texts.join(" ");
                let images_opt = if images.is_empty() {
                    None
                } else {
                    Some(images)
                };
                Ok(Message::Assistant {
                    content: content_str,
                    images: images_opt,
                    refusal: None,
                    audio: None,
                    name: None,
                })
            }
        }
    }
}
