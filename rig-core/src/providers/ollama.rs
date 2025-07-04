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
//!     prompt: rig::message::Message::User {
//!         content: rig::one_or_many::OneOrMany::one(rig::message::UserContent::text("Please tell me why the sky is blue.")),
//!         name: None
//!     },
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
use crate::client::{CompletionClient, EmbeddingsClient, ProviderClient};
use crate::json_utils::merge_inplace;
use crate::message::MessageError;
use crate::streaming::RawStreamingChoice;
use crate::{
    Embed, OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    impl_conversion_traits, json_utils, message,
    message::{ImageDetail, Text},
    streaming,
};
use async_stream::stream;
use futures::StreamExt;
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::{convert::TryFrom, str::FromStr};
// ---------- Main Client ----------

const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

#[derive(Clone, Debug)]
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

    /// Use your own `reqwest::Client`.
    /// The required headers will be automatically attached upon trying to make a request.
    pub fn with_custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;

        self
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path);
        self.http_client.post(url)
    }
}

impl ProviderClient for Client {
    fn from_env() -> Self
    where
        Self: Sized,
    {
        let api_base = std::env::var("OLLAMA_API_BASE_URL").expect("OLLAMA_API_BASE_URL not set");
        Self::from_url(&api_base)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;
    fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, 0)
    }
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
    fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client
);

// ---------- API Error and Response Structures ----------

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

// ---------- Embedding API ----------

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

// ---------- Embedding Model ----------

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
    #[cfg_attr(feature = "worker", worker::send)]
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

// ---------- Completion API ----------

pub const LLAMA3_2: &str = "llama3.2";
pub const LLAVA: &str = "llava";
pub const MISTRAL: &str = "mistral";

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
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
impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;
    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        match resp.message {
            // Process only if an assistant message is present.
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut assistant_contents = Vec::new();
                // Add the assistant's text content if any.
                if !content.is_empty() {
                    assistant_contents.push(completion::AssistantContent::text(&content));
                }
                // Process tool_calls following Ollama's chat response definition.
                // Each ToolCall has an id, a type, and a function field.
                for tc in tool_calls.iter() {
                    assistant_contents.push(completion::AssistantContent::tool_call(
                        tc.function.name.clone(),
                        tc.function.name.clone(),
                        tc.function.arguments.clone(),
                    ));
                }
                let choice = OneOrMany::many(assistant_contents).map_err(|_| {
                    CompletionError::ResponseError("No content provided".to_owned())
                })?;
                let raw_response = CompletionResponse {
                    model: resp.model,
                    created_at: resp.created_at,
                    done: resp.done,
                    done_reason: resp.done_reason,
                    total_duration: resp.total_duration,
                    load_duration: resp.load_duration,
                    prompt_eval_count: resp.prompt_eval_count,
                    prompt_eval_duration: resp.prompt_eval_duration,
                    eval_count: resp.eval_count,
                    eval_duration: resp.eval_duration,
                    message: Message::Assistant {
                        content,
                        images: None,
                        name: None,
                        tool_calls,
                    },
                };
                Ok(completion::CompletionResponse {
                    choice,
                    raw_response,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Chat response does not include an assistant message".into(),
            )),
        }
    }
}

// ---------- Completion Model ----------

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
        let mut full_history: Vec<Message> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| vec![Message::system(&preamble)]);

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(|msg| msg.try_into())
                .collect::<Result<Vec<Message>, _>>()?,
        );

        // Convert internal prompt into a provider Message
        let options = if let Some(extra) = completion_request.additional_params {
            json_utils::merge(
                json!({ "temperature": completion_request.temperature }),
                extra,
            )
        } else {
            json!({ "temperature": completion_request.temperature })
        };

        let mut request_payload = json!({
            "model": self.model,
            "messages": full_history,
            "options": options,
            "stream": false,
        });
        if !completion_request.tools.is_empty() {
            request_payload["tools"] = json!(
                completion_request
                    .tools
                    .into_iter()
                    .map(|tool| tool.into())
                    .collect::<Vec<ToolDefinition>>()
            );
        }

        tracing::debug!(target: "rig", "Chat mode payload: {}", request_payload);

        Ok(request_payload)
    }
}

// ---------- CompletionModel Implementation ----------

#[derive(Clone)]
pub struct StreamingCompletionResponse {
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}
impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let request_payload = self.create_completion_request(completion_request)?;

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
            let chat_resp: CompletionResponse = serde_json::from_str(&text)
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            let conv: completion::CompletionResponse<CompletionResponse> = chat_resp.try_into()?;
            Ok(conv)
        } else {
            let err_text = response
                .text()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            Err(CompletionError::ProviderError(err_text))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        let mut request_payload = self.create_completion_request(request)?;
        merge_inplace(&mut request_payload, json!({"stream": true}));

        let response = self
            .client
            .post("api/chat")
            .json(&request_payload)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        if !response.status().is_success() {
            let err_text = response
                .text()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            return Err(CompletionError::ProviderError(err_text));
        }

        let stream = Box::pin(stream! {
            let mut stream = response.bytes_stream();
            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(CompletionError::from(e));
                        break;
                    }
                };

                let text = match String::from_utf8(chunk.to_vec()) {
                    Ok(t) => t,
                    Err(e) => {
                        yield Err(CompletionError::ResponseError(e.to_string()));
                        break;
                    }
                };


                for line in text.lines() {
                    let line = line.to_string();

                    let Ok(response) = serde_json::from_str::<CompletionResponse>(&line) else {
                        continue;
                    };

                    match response.message {
                        Message::Assistant{ content, tool_calls, .. } => {
                            if !content.is_empty() {
                                yield Ok(RawStreamingChoice::Message(content))
                            }

                            for tool_call in tool_calls.iter() {
                                let function = tool_call.function.clone();

                                yield Ok(RawStreamingChoice::ToolCall {
                                    id: "".to_string(),
                                    name: function.name,
                                    arguments: function.arguments,
                                    call_id: None
                                });
                            }
                        }
                        _ => {
                            continue;
                        }
                    }

                    if response.done {
                        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                            total_duration: response.total_duration,
                            load_duration: response.load_duration,
                            prompt_eval_count: response.prompt_eval_count,
                            prompt_eval_duration: response.prompt_eval_duration,
                            eval_count: response.eval_count,
                            eval_duration: response.eval_duration,
                            done_reason: response.done_reason,
                        }));
                    }
                }
            }
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}

// ---------- Tool Definition Conversion ----------

/// Ollama-required tool definition format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String, // Fixed as "function"
    pub function: completion::ToolDefinition,
}

/// Convert internal ToolDefinition (from the completion module) into Ollama's tool definition.
impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        ToolDefinition {
            type_field: "function".to_owned(),
            function: completion::ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    // pub id: String,
    #[serde(default, rename = "type")]
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
    pub arguments: Value,
}

// ---------- Provider Message Definition ----------

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
        #[serde(default)]
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        tool_calls: Vec<ToolCall>,
    },
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "tool")]
    ToolResult { name: String, content: String },
}

/// -----------------------------
/// Provider Message Conversions
/// -----------------------------
/// Conversion from an internal Rig message (crate::message::Message) to a provider Message.
/// (Only User and Assistant variants are supported.)
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
                        crate::message::UserContent::ToolResult(result) => {
                            let content = result
                                .content
                                .into_iter()
                                .map(ToolResultContent::try_from)
                                .collect::<Result<Vec<ToolResultContent>, MessageError>>()?;

                            let content = OneOrMany::many(content).map_err(|x| {
                                MessageError::ConversionError(format!(
                                    "Couldn't make a OneOrMany from a list of tool results: {x}"
                                ))
                            })?;

                            return Ok(Message::ToolResult {
                                name: result.id,
                                content: content.first().text,
                            });
                        }
                        _ => {} // Audio variant removed since Ollama API does not support it.
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
                let mut tool_calls = Vec::new();
                for ac in content.into_iter() {
                    match ac {
                        crate::message::AssistantContent::Text(t) => texts.push(t.text),
                        crate::message::AssistantContent::ToolCall(tc) => {
                            tool_calls.push(ToolCall {
                                r#type: ToolType::Function, // Assuming internal tool call provides these fields
                                function: Function {
                                    name: tc.function.name,
                                    arguments: tc.function.arguments,
                                },
                            });
                        }
                    }
                }
                let content_str = texts.join(" ");
                Ok(Message::Assistant {
                    content: content_str,
                    images: None,
                    name: None,
                    tool_calls,
                })
            }
        }
    }
}

/// Conversion from provider Message to a completion message.
/// This is needed so that responses can be converted back into chat history.
impl From<Message> for crate::completion::Message {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User { content, .. } => crate::completion::Message::User {
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
                    text: content,
                })),
            },
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut assistant_contents =
                    vec![crate::completion::message::AssistantContent::Text(Text {
                        text: content,
                    })];
                for tc in tool_calls {
                    assistant_contents.push(
                        crate::completion::message::AssistantContent::tool_call(
                            tc.function.name.clone(),
                            tc.function.name,
                            tc.function.arguments,
                        ),
                    );
                }
                crate::completion::Message::Assistant {
                    id: None,
                    content: OneOrMany::many(assistant_contents).unwrap(),
                }
            }
            // System and ToolResult are converted to User message as needed.
            Message::System { content, .. } => crate::completion::Message::User {
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
                    text: content,
                })),
            },
            Message::ToolResult { name, content } => crate::completion::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    name,
                    OneOrMany::one(message::ToolResultContent::Text(Text { text: content })),
                )),
            },
        }
    }
}

impl Message {
    /// Constructs a system message.
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            images: None,
            name: None,
        }
    }
}

// ---------- Additional Message Types ----------

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolResultContent {
    text: String,
}

impl TryFrom<crate::message::ToolResultContent> for ToolResultContent {
    type Error = MessageError;
    fn try_from(value: crate::message::ToolResultContent) -> Result<Self, Self::Error> {
        let crate::message::ToolResultContent::Text(Text { text }) = value else {
            return Err(MessageError::ConversionError(
                "Non-text tool results not supported".into(),
            ));
        };

        Ok(Self { text })
    }
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent { text: s }
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
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent {
            r#type: SystemContentType::default(),
            text: s.to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AssistantContent {
    pub text: String,
}

impl FromStr for AssistantContent {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent { text: s.to_owned() })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text { text: String },
    Image { image_url: ImageUrl },
    // Audio variant removed as Ollama API does not support audio input.
}

impl FromStr for UserContent {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text { text: s.to_owned() })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: ImageDetail,
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Test deserialization and conversion for the /api/chat endpoint.
    #[tokio::test]
    async fn test_chat_completion() {
        // Sample JSON response from /api/chat (non-streaming) based on Ollama docs.
        let sample_chat_response = json!({
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "The sky is blue because of Rayleigh scattering.",
                "images": null,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {
                                "location": "San Francisco, CA",
                                "format": "celsius"
                            }
                        }
                    }
                ]
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 61u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 468u64,
            "eval_duration": 7700000000u64
        });
        let sample_text = sample_chat_response.to_string();

        let chat_resp: CompletionResponse =
            serde_json::from_str(&sample_text).expect("Invalid JSON structure");
        let conv: completion::CompletionResponse<CompletionResponse> =
            chat_resp.try_into().unwrap();
        assert!(
            !conv.choice.is_empty(),
            "Expected non-empty choice in chat response"
        );
    }

    // Test conversion from provider Message to completion Message.
    #[test]
    fn test_message_conversion() {
        // Construct a provider Message (User variant with String content).
        let provider_msg = Message::User {
            content: "Test message".to_owned(),
            images: None,
            name: None,
        };
        // Convert it into a completion::Message.
        let comp_msg: crate::completion::Message = provider_msg.into();
        match comp_msg {
            crate::completion::Message::User { content } => {
                // Assume OneOrMany<T> has a method first() to access the first element.
                let first_content = content.first();
                // The expected type is crate::completion::message::UserContent::Text wrapping a Text struct.
                match first_content {
                    crate::completion::message::UserContent::Text(text_struct) => {
                        assert_eq!(text_struct.text, "Test message");
                    }
                    _ => panic!("Expected text content in conversion"),
                }
            }
            _ => panic!("Conversion from provider Message to completion Message failed"),
        }
    }

    // Test conversion of internal tool definition to Ollama's ToolDefinition format.
    #[test]
    fn test_tool_definition_conversion() {
        // Internal tool definition from the completion module.
        let internal_tool = crate::completion::ToolDefinition {
            name: "get_current_weather".to_owned(),
            description: "Get the current weather for a location".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for, e.g. San Francisco, CA"
                    },
                    "format": {
                        "type": "string",
                        "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "format"]
            }),
        };
        // Convert internal tool to Ollama's tool definition.
        let ollama_tool: ToolDefinition = internal_tool.into();
        assert_eq!(ollama_tool.type_field, "function");
        assert_eq!(ollama_tool.function.name, "get_current_weather");
        assert_eq!(
            ollama_tool.function.description,
            "Get the current weather for a location"
        );
        // Check JSON fields in parameters.
        let params = &ollama_tool.function.parameters;
        assert_eq!(params["properties"]["location"]["type"], "string");
    }
}
