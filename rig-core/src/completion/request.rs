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
//! # Examples
//!
//! ## Basic Completion
//!
//! ```ignore
//! use rig::providers::openai::{Client, self};
//! use rig::client::completion::CompletionClient;
//! use rig::completion::Prompt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the OpenAI client and a completion model
//! let openai = Client::new("your-openai-api-key");
//! let gpt_4 = openai.completion_model(openai::GPT_4);
//!
//! // Send a simple prompt
//! let response = gpt_4.prompt("Who are you?").await?;
//! println!("Response: {}", response);
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Completion Request
//!
//! ```ignore
//! use rig::providers::openai::{Client, self};
//! use rig::client::completion::CompletionClient;
//! use rig::completion::Completion;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let openai = Client::new("your-openai-api-key");
//! let gpt_4 = openai.completion_model(openai::GPT_4);
//!
//! // Build a custom completion request
//! let request = gpt_4.completion_request("Who are you?")
//!     .preamble("You are Marvin, a depressed but helpful robot.")
//!     .temperature(0.5)
//!     .build()?;
//!
//! // Send the request
//! let response = gpt_4.completion(request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! For more information on how to use the completion functionality, refer to the documentation of
//! the individual traits, structs, and enums defined in this module.

use super::message::{AssistantContent, DocumentMediaType};
use crate::client::completion::CompletionModelHandle;
use crate::message::ToolChoice;
use crate::streaming::StreamingCompletionResponse;
use crate::{OneOrMany, streaming};
use crate::{
    json_utils,
    message::{Message, UserContent},
    tool::ToolSetError,
};
use futures::future::BoxFuture;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Add, AddAssign};
use std::sync::Arc;
use thiserror::Error;

// ================================================================
// Error types
// ================================================================

/// Errors that can occur during completion operations.
///
/// This enum covers all possible errors that may occur when making completion
/// requests to LLM providers, from network issues to provider-specific errors.
///
/// # Examples
///
/// ## Basic Error Handling
///
/// ```ignore
/// use rig::completion::{CompletionError, Prompt};
/// use rig::providers::openai;
/// use rig::client::completion::CompletionClient;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = openai::Client::new("api-key");
/// let model = client.completion_model(openai::GPT_4);
///
/// match model.prompt("Hello").await {
///     Ok(response) => println!("Success: {}", response),
///     Err(CompletionError::HttpError(e)) => {
///         eprintln!("Network error: {}. Check your internet connection.", e);
///     }
///     Err(CompletionError::ProviderError(msg)) if msg.contains("rate_limit") => {
///         eprintln!("Rate limited. Please wait and try again.");
///     }
///     Err(CompletionError::ProviderError(msg)) if msg.contains("invalid_api_key") => {
///         eprintln!("Invalid API key. Check your credentials.");
///     }
///     Err(CompletionError::ProviderError(msg)) => {
///         eprintln!("Provider error: {}", msg);
///     }
///     Err(e) => eprintln!("Unexpected error: {}", e),
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Retry with Exponential Backoff
///
/// ```ignore
/// use rig::completion::{CompletionError, Prompt};
/// use rig::providers::openai;
/// use rig::client::completion::CompletionClient;
/// use std::time::Duration;
/// use tokio::time::sleep;
///
/// # async fn example() -> Result<String, Box<dyn std::error::Error>> {
/// let client = openai::Client::new("api-key");
/// let model = client.completion_model(openai::GPT_4);
///
/// let mut retries = 0;
/// let max_retries = 3;
///
/// loop {
///     match model.prompt("Hello").await {
///         Ok(response) => return Ok(response),
///         Err(CompletionError::HttpError(_)) if retries < max_retries => {
///             retries += 1;
///             let delay = Duration::from_secs(2_u64.pow(retries));
///             eprintln!("Network error. Retry {}/{} in {:?}", retries, max_retries, delay);
///             sleep(delay).await;
///         }
///         Err(CompletionError::ProviderError(msg)) if msg.contains("rate_limit") && retries < max_retries => {
///             retries += 1;
///             let delay = Duration::from_secs(5 * retries as u64);
///             eprintln!("Rate limited. Waiting {:?} before retry...", delay);
///             sleep(delay).await;
///         }
///         Err(e) => return Err(e.into()),
///     }
/// }
/// # }
/// ```
///
/// ## Fallback to Different Model
///
/// ```ignore
/// use rig::completion::{CompletionError, Prompt};
/// use rig::providers::openai;
/// use rig::client::completion::CompletionClient;
///
/// # async fn example() -> Result<String, Box<dyn std::error::Error>> {
/// let client = openai::Client::new("api-key");
///
/// // Try GPT-4 first
/// let gpt4 = client.completion_model(openai::GPT_4);
/// match gpt4.prompt("Explain quantum computing").await {
///     Ok(response) => return Ok(response),
///     Err(CompletionError::ProviderError(msg)) if msg.contains("rate_limit") => {
///         eprintln!("GPT-4 rate limited, falling back to GPT-3.5...");
///         // Fall back to cheaper model
///         let gpt35 = client.completion_model(openai::GPT_3_5_TURBO);
///         return Ok(gpt35.prompt("Explain quantum computing").await?);
///     }
///     Err(e) => return Err(e.into()),
/// }
/// # }
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CompletionError {
    /// HTTP request failed.
    ///
    /// This occurs when there are network connectivity issues, timeouts,
    /// or other HTTP-level problems communicating with the provider.
    ///
    /// Common causes:
    /// - No internet connection
    /// - Request timeout
    /// - DNS resolution failure
    /// - SSL/TLS errors
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON serialization or deserialization failed.
    ///
    /// This occurs when:
    /// - The provider returns malformed JSON
    /// - Request data cannot be serialized
    /// - Response data doesn't match expected schema
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Invalid URL provided.
    ///
    /// This occurs when attempting to construct an invalid URL for the provider endpoint.
    #[error("Invalid URL: {0}")]
    UrlError(#[from] url::ParseError),

    /// Error building the completion request.
    ///
    /// This occurs when there's an issue constructing the request,
    /// such as invalid parameters or missing required fields.
    #[error("Request building error: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Error parsing the completion response.
    ///
    /// This occurs when the provider returns a valid HTTP response
    /// but the content cannot be parsed or understood.
    #[error("Response parsing error: {0}")]
    ResponseError(String),

    /// Error returned by the LLM provider.
    ///
    /// This represents errors from the provider's API, such as:
    /// - Invalid API key
    /// - Rate limits exceeded
    /// - Model not found
    /// - Content policy violations
    /// - Insufficient credits/quota
    #[error("Provider error: {0}")]
    ProviderError(String),
}

/// Errors that can occur during prompt operations.
///
/// This enum covers errors specific to high-level prompt operations,
/// including multi-turn conversations and tool calling.
///
/// # Examples
///
/// ```ignore
/// use rig::completion::{PromptError, Prompt};
/// use rig::providers::openai;
/// use rig::client::completion::CompletionClient;
///
/// # async fn example() -> Result<(), PromptError> {
/// let client = openai::Client::new("api-key");
/// let model = client.completion_model(openai::GPT_4);
///
/// match model.prompt("Hello").await {
///     Ok(response) => println!("Success: {}", response),
///     Err(PromptError::MaxDepthError { max_depth, .. }) => {
///         eprintln!("Too many tool calls (limit: {})", max_depth);
///     },
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PromptError {
    /// Underlying completion operation failed.
    ///
    /// See [`CompletionError`] for details on specific failure modes.
    #[error("Completion error: {0}")]
    CompletionError(#[from] CompletionError),

    /// Tool execution failed.
    ///
    /// This occurs when the LLM requests a tool call but the tool
    /// execution encounters an error.
    #[error("Tool call error: {0}")]
    ToolError(#[from] ToolSetError),

    /// Maximum conversation depth exceeded.
    ///
    /// This occurs when the LLM attempts too many tool calls in a multi-turn
    /// conversation, exceeding the configured maximum depth.
    ///
    /// # Solutions
    ///
    /// To resolve this:
    /// - Reduce the number of available tools (distribute across multiple agents)
    /// - Increase the maximum depth with `.multi_turn(depth)`
    /// - Simplify the task to require fewer tool calls
    #[error("Maximum conversation depth exceeded (limit: {max_depth})")]
    MaxDepthError {
        /// The maximum depth that was exceeded.
        max_depth: usize,

        /// The conversation history up to the point of failure.
        chat_history: Box<Vec<Message>>,

        /// The prompt that triggered the error.
        prompt: Message,
    },
}

/// A document that can be provided as context to an LLM.
///
/// Documents are structured pieces of text that models can reference during
/// completion. They differ from regular text messages in important ways:
/// - Have unique IDs for reference and citation
/// - Are formatted as files (typically with XML-like tags)
/// - Can include metadata via `additional_props`
/// - Are treated as "reference material" rather than conversation turns
///
/// # When to Use Documents vs. Messages
///
/// **Use Documents when:**
/// - Providing reference material (documentation, knowledge base articles, code files)
/// - The content is factual/static rather than conversational
/// - You want the model to cite sources by ID
/// - Implementing RAG (Retrieval-Augmented Generation) patterns
/// - Building code analysis or document Q&A systems
///
/// **Use Messages when:**
/// - Having a conversation with the model
/// - The text is a question or response
/// - Building chat history
/// - User input or model output
///
/// # Formatting
///
/// Documents are typically formatted as XML-like files when sent to the model:
///
/// ```text
/// <file id="doc-1">
/// [metadata: source=API Docs, version=1.0]
/// Content of the document...
/// </file>
/// ```
///
/// # Examples
///
/// ## Basic Document
///
/// ```
/// use rig::completion::request::Document;
/// use std::collections::HashMap;
///
/// let doc = Document {
///     id: "rust-ownership".to_string(),
///     text: "Rust's ownership system ensures memory safety without garbage collection.".to_string(),
///     additional_props: HashMap::new(),
/// };
/// ```
///
/// ## Document with Metadata
///
/// ```
/// use rig::completion::request::Document;
/// use std::collections::HashMap;
///
/// let mut metadata = HashMap::new();
/// metadata.insert("source".to_string(), "Rust Book".to_string());
/// metadata.insert("chapter".to_string(), "4".to_string());
/// metadata.insert("topic".to_string(), "Ownership".to_string());
/// metadata.insert("last_updated".to_string(), "2024-01-15".to_string());
///
/// let doc = Document {
///     id: "rust-book-ch4".to_string(),
///     text: "Ownership is Rust's most unique feature...".to_string(),
///     additional_props: metadata,
/// };
/// ```
///
/// ## RAG Pattern (Retrieval-Augmented Generation)
///
/// ```ignore
/// # use rig::providers::openai;
/// # use rig::client::completion::CompletionClient;
/// # use rig::completion::{Prompt, request::Document};
/// # use std::collections::HashMap;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = openai::Client::new("your-api-key");
/// let model = client.completion_model(openai::GPT_4);
///
/// // Retrieved from vector database based on user query
/// let relevant_docs = vec![
///     Document {
///         id: "ownership-basics".to_string(),
///         text: "Rust's ownership system manages memory through a set of rules...".to_string(),
///         additional_props: HashMap::new(),
///     },
///     Document {
///         id: "borrowing-rules".to_string(),
///         text: "Borrowing allows you to have references to a value...".to_string(),
///         additional_props: HashMap::new(),
///     },
/// ];
///
/// // Build prompt with context
/// let context = relevant_docs.iter()
///     .map(|doc| format!("<doc id=\"{}\">\n{}\n</doc>", doc.id, doc.text))
///     .collect::<Vec<_>>()
///     .join("\n\n");
///
/// let prompt = format!(
///     "Context:\n{}\n\nQuestion: How does Rust prevent memory leaks?\n\nAnswer based on the context above:",
///     context
/// );
///
/// let response = model.prompt(&prompt).await?;
/// # Ok(())
/// # }
/// ```
///
/// ## Code Documentation Example
///
/// ```
/// use rig::completion::request::Document;
/// use std::collections::HashMap;
///
/// let mut metadata = HashMap::new();
/// metadata.insert("file".to_string(), "src/main.rs".to_string());
/// metadata.insert("language".to_string(), "rust".to_string());
/// metadata.insert("lines".to_string(), "1-50".to_string());
///
/// let code_doc = Document {
///     id: "main-rs".to_string(),
///     text: r#"
/// fn main() {
///     let config = load_config();
///     run_server(config);
/// }
/// "#.to_string(),
///     additional_props: metadata,
/// };
/// ```
///
/// # Performance Tips
///
/// - Keep document size reasonable (typically 1000-5000 words each)
/// - Use concise, relevant excerpts rather than full documents
/// - Limit the number of documents (3-5 most relevant is often optimal)
/// - Include metadata to help the model understand context and provenance
/// - Consider pre-processing documents to remove irrelevant information
///
/// # See also
///
/// - [`Message`] for conversational messages
/// - Vector embeddings for retrieving relevant documents
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Document {
    /// Unique identifier for this document.
    pub id: String,

    /// The text content of the document.
    pub text: String,

    /// Additional properties for this document.
    ///
    /// These are flattened during serialization and can contain
    /// metadata like author, date, source, etc.
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ================================================================
// Implementations
// ================================================================
/// Trait defining a high-level LLM simple prompt interface (i.e.: prompt in, response out).
pub trait Prompt: Send + Sync {
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
        prompt: impl Into<Message> + Send,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: Send>;
}

/// Trait defining a high-level LLM chat interface (i.e.: prompt and chat history in, response out).
pub trait Chat: Send + Sync {
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
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: Send>;
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
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<CompletionRequestBuilder<M>, CompletionError>> + Send;
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
pub trait CompletionModel: Clone + Send + Sync {
    /// The raw response type returned by the underlying completion model.
    type Response: Send + Sync + Serialize + DeserializeOwned;
    /// The raw response type returned by the underlying completion model when streaming.
    type StreamingResponse: Clone
        + Unpin
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + GetTokenUsage;

    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<CompletionResponse<Self::Response>, CompletionError>,
    > + Send;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + Send;

    /// Generates a completion request builder for the given `prompt`.
    fn completion_request(&self, prompt: impl Into<Message>) -> CompletionRequestBuilder<Self> {
        CompletionRequestBuilder::new(self.clone(), prompt)
    }
}
pub trait CompletionModelDyn: Send + Sync {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'_, Result<CompletionResponse<()>, CompletionError>>;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'_, Result<StreamingCompletionResponse<()>, CompletionError>>;

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
    ) -> BoxFuture<'_, Result<CompletionResponse<()>, CompletionError>> {
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
    ) -> BoxFuture<'_, Result<StreamingCompletionResponse<()>, CompletionError>> {
        Box::pin(async move {
            let resp = self.stream(request).await?;
            let inner = resp.inner;

            let stream = Box::pin(streaming::StreamingResultDyn {
                inner: Box::pin(inner),
            });

            Ok(StreamingCompletionResponse::stream(stream))
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
/// ```ignore
/// use rig::{
///     providers::openai::{Client, self},
///     completion::CompletionRequestBuilder,
/// };
/// use rig::client::completion::CompletionClient;
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
/// ```ignore
/// use rig::{
///     providers::openai::{Client, self},
///     completion::CompletionRequestBuilder,
/// };
/// use rig::client::completion::CompletionClient;
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
