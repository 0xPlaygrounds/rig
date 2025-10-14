//! This module provides functionality for working with completion models.
//! It provides traits, structs, and enums for generating completion requests,
//! handling completion responses, and defining completion models.
//!
//! The main traits defined in this module are:
//! - [Prompt]: Defines a high-level LLM one-shot prompt interface.
//! - [Chat]: Defines a high-level LLM chat interface with chat history.
//! - [Completion]: Defines a low-level LLM completion interface for generating completion requests.
//! - [CompletionModel]: Defines a completion model that can be used to generate completion
//!   responses from requests.
//!
//! The [Prompt] and [Chat] traits are high level traits that users are expected to use
//! to interact with LLM models. Moreover, it is good practice to implement one of these
//! traits for composite agents that use multiple LLM models to generate responses.
//!
//! The [Completion] trait defines a lower level interface that is useful when the user want
//! to further customize the request before sending it to the completion model provider.
//!
//! The [CompletionModel] trait is meant to act as the interface between providers and
//! the library. It defines the methods that need to be implemented by the user to define
//! a custom base completion model (i.e.: a private or third party LLM provider).
//!
//! The module also provides various structs and enums for representing generic completion requests,
//! responses, and errors.
//!
//! Example Usage:
//! ```rust
//! use rig::providers::openai::{Client, self};
//! use rig::completion::*;
//!
//! // Initialize the OpenAI client and a completion model
//! let openai = Client::new("your-openai-api-key");
//!
//! let gpt_4 = openai.completion_model(openai::GPT_4);
//!
//! // Create the completion request
//! let request = gpt_4.completion_request("Who are you?")
//!     .preamble("\
//!         You are Marvin, an extremely smart but depressed robot who is \
//!         nonetheless helpful towards humanity.\
//!     ")
//!     .temperature(0.5)
//!     .build();
//!
//! // Send the completion request and get the completion response
//! let response = gpt_4.completion(request)
//!     .await
//!     .expect("Failed to get completion response");
//!
//! // Handle the completion response
//! match completion_response.choice {
//!     ModelChoice::Message(message) => {
//!         // Handle the completion response as a message
//!         println!("Received message: {}", message);
//!     }
//!     ModelChoice::ToolCall(tool_name, tool_params) => {
//!         // Handle the completion response as a tool call
//!         println!("Received tool call: {} {:?}", tool_name, tool_params);
//!     }
//! }
//! ```
//!
//! For more information on how to use the completion functionality, refer to the documentation of
//! the individual traits, structs, and enums defined in this module.

use super::message::{AssistantContent, DocumentMediaType};
use crate::client::builder::FinalCompletionResponse;
use crate::client::completion::CompletionModelHandle;
use crate::message::ToolChoice;
use crate::streaming::StreamingCompletionResponse;
use crate::tool::server::ToolServerError;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use crate::{OneOrMany, http_client, streaming};
use crate::{
    json_utils,
    message::{Message, UserContent},
    tool::ToolSetError,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Add, AddAssign};
use std::sync::Arc;
use thiserror::Error;

// Errors
#[derive(Debug, Error)]
pub enum CompletionError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Url error (e.g.: invalid URL)
    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[cfg(not(target_family = "wasm"))]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + 'static>),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the completion model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),
}

/// Prompt errors
#[derive(Debug, Error)]
pub enum PromptError {
    /// Something went wrong with the completion
    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),

    /// There was an error while using a tool
    #[error("ToolCallError: {0}")]
    ToolError(#[from] ToolSetError),

    /// There was an issue while executing a tool on a tool server
    #[error("ToolServerError: {0}")]
    ToolServerError(#[from] ToolServerError),

    /// The LLM tried to call too many tools during a multi-turn conversation.
    /// To fix this, you may either need to lower the amount of tools your model has access to (and then create other agents to share the tool load)
    /// or increase the amount of turns given in `.multi_turn()`.
    #[error("MaxDepthError: (reached limit: {max_depth})")]
    MaxDepthError {
        max_depth: usize,
        chat_history: Box<Vec<Message>>,
        prompt: Message,
    },

    /// A prompting loop was cancelled.
    #[error("PromptCancelled")]
    PromptCancelled { chat_history: Box<Vec<Message>> },
}

impl PromptError {
    pub(crate) fn prompt_cancelled(chat_history: Vec<Message>) -> Self {
        Self::PromptCancelled {
            chat_history: Box::new(chat_history),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(flatten)]
    pub additional_props: HashMap<String, String>,
}

impl std::fmt::Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            concat!("<file id: {}>\n", "{}\n", "</file>\n"),
            self.id,
            if self.additional_props.is_empty() {
                self.text.clone()
            } else {
                let mut sorted_props = self.additional_props.iter().collect::<Vec<_>>();
                sorted_props.sort_by(|a, b| a.0.cmp(b.0));
                let metadata = sorted_props
                    .iter()
                    .map(|(k, v)| format!("{k}: {v:?}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("<metadata {} />\n{}", metadata, self.text)
            }
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ================================================================
// Implementations
// ================================================================
/// Trait defining a high-level LLM simple prompt interface (i.e.: prompt in, response out).
pub trait Prompt: WasmCompatSend + WasmCompatSync {
    /// Send a simple prompt to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and
    /// the result is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: WasmCompatSend>;
}

/// Trait defining a high-level LLM chat interface (i.e.: prompt and chat history in, response out).
pub trait Chat: WasmCompatSend + WasmCompatSync {
    /// Send a prompt with optional chat history to the underlying completion model.
    ///
    /// If the completion model's response is a message, then it is returned as a string.
    ///
    /// If the completion model's response is a tool call, then the tool is called and the result
    /// is returned as a string.
    ///
    /// If the tool does not exist, or the tool call fails, then an error is returned.
    fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: WasmCompatSend>;
}

/// Trait defining a low-level LLM completion interface
pub trait Completion<M: CompletionModel> {
    /// Generates a completion request builder for the given `prompt` and `chat_history`.
    /// This function is meant to be called by the user to further customize the
    /// request at prompt time before sending it.
    ///
    /// ❗IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    ///
    /// For example, the request builder returned by [`Agent::completion`](crate::agent::Agent::completion) will already
    /// contain the `preamble` provided when creating the agent.
    fn completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<CompletionRequestBuilder<M>, CompletionError>>
    + WasmCompatSend;
}

/// General completion response struct that contains the high-level completion choice
/// and the raw response. The completion choice contains one or more assistant content.
#[derive(Debug)]
pub struct CompletionResponse<T> {
    /// The completion choice (represented by one or more assistant message content)
    /// returned by the completion model provider
    pub choice: OneOrMany<AssistantContent>,
    /// Tokens used during prompting and responding
    pub usage: Usage,
    /// The raw response returned by the completion model provider
    pub raw_response: T,
}

/// A trait for grabbing the token usage of a completion response.
///
/// Primarily designed for streamed completion responses in streamed multi-turn, as otherwise it would be impossible to do.
pub trait GetTokenUsage {
    fn token_usage(&self) -> Option<crate::completion::Usage>;
}

impl GetTokenUsage for () {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        None
    }
}

impl<T> GetTokenUsage for Option<T>
where
    T: GetTokenUsage,
{
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        if let Some(usage) = self {
            usage.token_usage()
        } else {
            None
        }
    }
}

/// Struct representing the token usage for a completion request.
/// If tokens used are `0`, then the provider failed to supply token usage metrics.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// The number of input ("prompt") tokens used in a given request.
    pub input_tokens: u64,
    /// The number of output ("completion") tokens used in a given request.
    pub output_tokens: u64,
    /// We store this separately as some providers may only report one number
    pub total_tokens: u64,
}

impl Usage {
    /// Creates a new instance of `Usage`.
    pub fn new() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
        }
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// Trait defining a completion model that can be used to generate completion responses.
/// This trait is meant to be implemented by the user to define a custom completion model,
/// either from a third party provider (e.g.: OpenAI) or a local model.
pub trait CompletionModel: Clone + WasmCompatSend + WasmCompatSync {
    /// The raw response type returned by the underlying completion model.
    type Response: WasmCompatSend + WasmCompatSync + Serialize + DeserializeOwned;
    /// The raw response type returned by the underlying completion model when streaming.
    type StreamingResponse: Clone
        + Unpin
        + WasmCompatSend
        + WasmCompatSync
        + Serialize
        + DeserializeOwned
        + GetTokenUsage;

    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<CompletionResponse<Self::Response>, CompletionError>,
    > + WasmCompatSend;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + WasmCompatSend;

    /// Generates a completion request builder for the given `prompt`.
    fn completion_request(&self, prompt: impl Into<Message>) -> CompletionRequestBuilder<Self> {
        CompletionRequestBuilder::new(self.clone(), prompt)
    }
}
pub trait CompletionModelDyn: WasmCompatSend + WasmCompatSync {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'_, Result<CompletionResponse<()>, CompletionError>>;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<
        '_,
        Result<StreamingCompletionResponse<FinalCompletionResponse>, CompletionError>,
    >;

    fn completion_request(
        &self,
        prompt: Message,
    ) -> CompletionRequestBuilder<CompletionModelHandle<'_>>;
}

impl<T, R> CompletionModelDyn for T
where
    T: CompletionModel<StreamingResponse = R>,
    R: Clone + Unpin + GetTokenUsage + 'static,
{
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'_, Result<CompletionResponse<()>, CompletionError>> {
        Box::pin(async move {
            self.completion(request)
                .await
                .map(|resp| CompletionResponse {
                    choice: resp.choice,
                    usage: resp.usage,
                    raw_response: (),
                })
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<
        '_,
        Result<StreamingCompletionResponse<FinalCompletionResponse>, CompletionError>,
    > {
        Box::pin(async move {
            let resp = self.stream(request).await?;
            let inner = resp.inner;

            let stream = streaming::StreamingResultDyn {
                inner: Box::pin(inner),
            };

            Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
        })
    }

    /// Generates a completion request builder for the given `prompt`.
    fn completion_request(
        &self,
        prompt: Message,
    ) -> CompletionRequestBuilder<CompletionModelHandle<'_>> {
        CompletionRequestBuilder::new(
            CompletionModelHandle {
                inner: Arc::new(self.clone()),
            },
            prompt,
        )
    }
}

/// Struct representing a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// The preamble to be sent to the completion model provider
    pub preamble: Option<String>,
    /// The chat history to be sent to the completion model provider.
    /// The very last message will always be the prompt (hence why there is *always* one)
    pub chat_history: OneOrMany<Message>,
    /// The documents to be sent to the completion model provider
    pub documents: Vec<Document>,
    /// The tools to be sent to the completion model provider
    pub tools: Vec<ToolDefinition>,
    /// The temperature to be sent to the completion model provider
    pub temperature: Option<f64>,
    /// The max tokens to be sent to the completion model provider
    pub max_tokens: Option<u64>,
    /// Whether tools are required to be used by the model provider or not before providing a response.
    pub tool_choice: Option<ToolChoice>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    pub additional_params: Option<serde_json::Value>,
}

impl CompletionRequest {
    /// Returns documents normalized into a message (if any).
    /// Most providers do not accept documents directly as input, so it needs to convert into a
    ///  `Message` so that it can be incorporated into `chat_history` as a
    pub fn normalized_documents(&self) -> Option<Message> {
        if self.documents.is_empty() {
            return None;
        }

        // Most providers will convert documents into a text unless it can handle document messages.
        // We use `UserContent::document` for those who handle it directly!
        let messages = self
            .documents
            .iter()
            .map(|doc| {
                UserContent::document(
                    doc.to_string(),
                    // In the future, we can customize `Document` to pass these extra types through.
                    // Most providers ditch these but they might want to use them.
                    Some(DocumentMediaType::TXT),
                )
            })
            .collect::<Vec<_>>();

        Some(Message::User {
            content: OneOrMany::many(messages).expect("There will be atleast one document"),
        })
    }
}

/// Builder struct for constructing a completion request.
///
/// Example usage:
/// ```rust
/// use rig::{
///     providers::openai::{Client, self},
///     completion::CompletionRequestBuilder,
/// };
///
/// let openai = Client::new("your-openai-api-key");
/// let model = openai.completion_model(openai::GPT_4O).build();
///
/// // Create the completion request and execute it separately
/// let request = CompletionRequestBuilder::new(model, "Who are you?".to_string())
///     .preamble("You are Marvin from the Hitchhiker's Guide to the Galaxy.".to_string())
///     .temperature(0.5)
///     .build();
///
/// let response = model.completion(request)
///     .await
///     .expect("Failed to get completion response");
/// ```
///
/// Alternatively, you can execute the completion request directly from the builder:
/// ```rust
/// use rig::{
///     providers::openai::{Client, self},
///     completion::CompletionRequestBuilder,
/// };
///
/// let openai = Client::new("your-openai-api-key");
/// let model = openai.completion_model(openai::GPT_4O).build();
///
/// // Create the completion request and execute it directly
/// let response = CompletionRequestBuilder::new(model, "Who are you?".to_string())
///     .preamble("You are Marvin from the Hitchhiker's Guide to the Galaxy.".to_string())
///     .temperature(0.5)
///     .send()
///     .await
///     .expect("Failed to get completion response");
/// ```
///
/// Note: It is usually unnecessary to create a completion request builder directly.
/// Instead, use the [CompletionModel::completion_request] method.
pub struct CompletionRequestBuilder<M: CompletionModel> {
    model: M,
    prompt: Message,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    tool_choice: Option<ToolChoice>,
    additional_params: Option<serde_json::Value>,
}

impl<M: CompletionModel> CompletionRequestBuilder<M> {
    pub fn new(model: M, prompt: impl Into<Message>) -> Self {
        Self {
            model,
            prompt: prompt.into(),
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        }
    }

    /// Sets the preamble for the completion request.
    pub fn preamble(mut self, preamble: String) -> Self {
        self.preamble = Some(preamble);
        self
    }

    pub fn without_preamble(mut self) -> Self {
        self.preamble = None;
        self
    }

    /// Adds a message to the chat history for the completion request.
    pub fn message(mut self, message: Message) -> Self {
        self.chat_history.push(message);
        self
    }

    /// Adds a list of messages to the chat history for the completion request.
    pub fn messages(self, messages: Vec<Message>) -> Self {
        messages
            .into_iter()
            .fold(self, |builder, msg| builder.message(msg))
    }

    /// Adds a document to the completion request.
    pub fn document(mut self, document: Document) -> Self {
        self.documents.push(document);
        self
    }

    /// Adds a list of documents to the completion request.
    pub fn documents(self, documents: Vec<Document>) -> Self {
        documents
            .into_iter()
            .fold(self, |builder, doc| builder.document(doc))
    }

    /// Adds a tool to the completion request.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds a list of tools to the completion request.
    pub fn tools(self, tools: Vec<ToolDefinition>) -> Self {
        tools
            .into_iter()
            .fold(self, |builder, tool| builder.tool(tool))
    }

    /// Adds additional parameters to the completion request.
    /// This can be used to set additional provider-specific parameters. For example,
    /// Cohere's completion models accept a `connectors` parameter that can be used to
    /// specify the data connectors used by Cohere when executing the completion
    /// (see `examples/cohere_connectors.rs`).
    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        match self.additional_params {
            Some(params) => {
                self.additional_params = Some(json_utils::merge(params, additional_params));
            }
            None => {
                self.additional_params = Some(additional_params);
            }
        }
        self
    }

    /// Sets the additional parameters for the completion request.
    /// This can be used to set additional provider-specific parameters. For example,
    /// Cohere's completion models accept a `connectors` parameter that can be used to
    /// specify the data connectors used by Cohere when executing the completion
    /// (see `examples/cohere_connectors.rs`).
    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }

    /// Sets the temperature for the completion request.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the temperature for the completion request.
    pub fn temperature_opt(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the max tokens for the completion request.
    /// Note: This is required if using Anthropic
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the max tokens for the completion request.
    /// Note: This is required if using Anthropic
    pub fn max_tokens_opt(mut self, max_tokens: Option<u64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the thing.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Builds the completion request.
    pub fn build(self) -> CompletionRequest {
        let chat_history = OneOrMany::many([self.chat_history, vec![self.prompt]].concat())
            .expect("There will always be atleast the prompt");

        CompletionRequest {
            preamble: self.preamble,
            chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            tool_choice: self.tool_choice,
            additional_params: self.additional_params,
        }
    }

    /// Sends the completion request to the completion model provider and returns the completion response.
    pub async fn send(self) -> Result<CompletionResponse<M::Response>, CompletionError> {
        let model = self.model.clone();
        model.completion(self.build()).await
    }

    /// Stream the completion request
    pub async fn stream<'a>(
        self,
    ) -> Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError>
    where
        <M as CompletionModel>::StreamingResponse: 'a,
        Self: 'a,
    {
        let model = self.model.clone();
        model.stream(self.build()).await
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_document_display_without_metadata() {
        let doc = Document {
            id: "123".to_string(),
            text: "This is a test document.".to_string(),
            additional_props: HashMap::new(),
        };

        let expected = "<file id: 123>\nThis is a test document.\n</file>\n";
        assert_eq!(format!("{doc}"), expected);
    }

    #[test]
    fn test_document_display_with_metadata() {
        let mut additional_props = HashMap::new();
        additional_props.insert("author".to_string(), "John Doe".to_string());
        additional_props.insert("length".to_string(), "42".to_string());

        let doc = Document {
            id: "123".to_string(),
            text: "This is a test document.".to_string(),
            additional_props,
        };

        let expected = concat!(
            "<file id: 123>\n",
            "<metadata author: \"John Doe\" length: \"42\" />\n",
            "This is a test document.\n",
            "</file>\n"
        );
        assert_eq!(format!("{doc}"), expected);
    }

    #[test]
    fn test_normalize_documents_with_documents() {
        let doc1 = Document {
            id: "doc1".to_string(),
            text: "Document 1 text.".to_string(),
            additional_props: HashMap::new(),
        };

        let doc2 = Document {
            id: "doc2".to_string(),
            text: "Document 2 text.".to_string(),
            additional_props: HashMap::new(),
        };

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one("What is the capital of France?".into()),
            documents: vec![doc1, doc2],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let expected = Message::User {
            content: OneOrMany::many(vec![
                UserContent::document(
                    "<file id: doc1>\nDocument 1 text.\n</file>\n".to_string(),
                    Some(DocumentMediaType::TXT),
                ),
                UserContent::document(
                    "<file id: doc2>\nDocument 2 text.\n</file>\n".to_string(),
                    Some(DocumentMediaType::TXT),
                ),
            ])
            .expect("There will be at least one document"),
        };

        assert_eq!(request.normalized_documents(), Some(expected));
    }

    #[test]
    fn test_normalize_documents_without_documents() {
        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one("What is the capital of France?".into()),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        assert_eq!(request.normalized_documents(), None);
    }
}
