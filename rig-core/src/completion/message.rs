//! Message types for LLM conversations.
//!
//! This module provides the core message structures used in conversations with
//! Large Language Models. Messages are the fundamental building blocks of
//! interactions, representing both user inputs and assistant responses.
//!
//! # Main Components
//!
//! - [`Message`]: The primary message type with user and assistant variants
//! - [`UserContent`]: Content types that can appear in user messages (text, images, audio, etc.)
//! - [`AssistantContent`]: Content types in assistant responses (text, tool calls, reasoning)
//! - [`ConvertMessage`]: Trait for converting messages to custom formats
//!
//! # Quick Start
//!
//! ```
//! use rig::completion::Message;
//!
//! // Create a simple user message
//! let user_msg = Message::user("What is the capital of France?");
//!
//! // Create an assistant response
//! let assistant_msg = Message::assistant("The capital of France is Paris.");
//! ```
//!
//! # Multimodal Messages
//!
//! Messages can contain multiple types of content:
//!
//! ```ignore
//! use rig::completion::message::{Message, UserContent, ImageMediaType, ImageDetail};
//! use rig::OneOrMany;
//!
//! let msg = Message::User {
//!     content: OneOrMany::many(vec![
//!         UserContent::text("What's in this image?"),
//!         UserContent::image_url(
//!             "https://example.com/image.png",
//!             Some(ImageMediaType::PNG),
//!             Some(ImageDetail::High)
//!         ),
//!     ])
//! };
//! ```
//!
//! # Provider Compatibility
//!
//! Different LLM providers support different content types and features.
//! Rig handles conversion between its generic message format and provider-specific
//! formats automatically, with some potential loss of information for unsupported features.
//!
//! # Common Patterns
//!
//! ## Building Conversation History
//!
//! ```
//! use rig::completion::Message;
//!
//! let conversation = vec![
//!     Message::user("What's the capital of France?"),
//!     Message::assistant("The capital of France is Paris."),
//!     Message::user("What's its population?"),
//!     Message::assistant("Paris has approximately 2.2 million inhabitants."),
//! ];
//! ```
//!
//! ## Multimodal Messages
//!
//! Combining text with images for vision-capable models:
//!
//! ```ignore
//! use rig::completion::message::{Message, UserContent, ImageMediaType, ImageDetail};
//! use rig::OneOrMany;
//!
//! let msg = Message::User {
//!     content: OneOrMany::many(vec![
//!         UserContent::text("Analyze this architecture diagram and explain the data flow:"),
//!         UserContent::image_url(
//!             "https://example.com/architecture.png",
//!             Some(ImageMediaType::PNG),
//!             Some(ImageDetail::High)
//!         ),
//!     ])
//! };
//! ```
//!
//! ## Working with Tool Results
//!
//! ```
//! use rig::completion::message::{Message, UserContent, ToolResult, ToolResultContent, Text};
//! use rig::OneOrMany;
//!
//! // After the model requests a tool call, provide the result
//! let tool_result = ToolResult {
//!     id: "call_123".to_string(),
//!     call_id: Some("msg_456".to_string()),
//!     content: OneOrMany::one(ToolResultContent::Text(Text {
//!         text: "The current temperature is 72°F".to_string(),
//!     })),
//! };
//!
//! let msg = Message::User {
//!     content: OneOrMany::one(UserContent::ToolResult(tool_result)),
//! };
//! ```
//!
//! # Troubleshooting
//!
//! ## Common Issues
//!
//! ### "Media type required" Error
//!
//! When creating images from base64 data, you must specify the media type:
//!
//! ```
//! # use rig::completion::message::UserContent;
//! // ❌ This may fail during provider conversion
//! let img = UserContent::image_base64("iVBORw0KGgo...", None, None);
//! ```
//!
//! ```
//! # use rig::completion::message::{UserContent, ImageMediaType};
//! // ✅ Correct: always specify media type for base64
//! let img = UserContent::image_base64(
//!     "iVBORw0KGgo...",
//!     Some(ImageMediaType::PNG),
//!     None
//! );
//! ```
//!
//! ### Provider Doesn't Support Content Type
//!
//! Not all providers support all content types:
//!
//! | Content Type | Supported By |
//! |--------------|--------------|
//! | Text | All providers |
//! | Images | GPT-4V, GPT-4o, Claude 3+, Gemini Pro Vision |
//! | Audio | OpenAI Whisper, specific models |
//! | Video | Gemini 1.5+, very limited support |
//! | Tool calls | GPT-4+, Claude 3+, most modern models |
//!
//! **Solution**: Check your provider's documentation before using multimedia content.
//!
//! ### Large Base64 Images Failing
//!
//! Base64-encoded images count heavily toward token limits:
//!
//! ```
//! # use rig::completion::message::{UserContent, ImageMediaType, ImageDetail};
//! // ❌ Large base64 image (may exceed limits)
//! // let huge_image = UserContent::image_base64(very_large_base64, ...);
//!
//! // ✅ Better: use URL for large images
//! let img = UserContent::image_url(
//!     "https://example.com/large-image.png",
//!     Some(ImageMediaType::PNG),
//!     Some(ImageDetail::High)
//! );
//! ```
//!
//! **Tips**:
//! - Resize images before encoding (768x768 is often sufficient)
//! - Use URLs for images >1MB
//! - Use `ImageDetail::Low` for thumbnails or simple images
//!
//! ### Builder Pattern Not Chaining
//!
//! Make sure to capture the return value from builder methods:
//!
//! ```
//! # use rig::completion::message::Reasoning;
//! // ❌ This doesn't work (discards the returned value)
//! let mut reasoning = Reasoning::new("step 1");
//! reasoning.with_id("id-123".to_string());  // Returns new value, not stored!
//! // reasoning.id is still None
//! ```
//!
//! ```
//! # use rig::completion::message::Reasoning;
//! // ✅ Correct: chain the calls or reassign
//! let reasoning = Reasoning::new("step 1")
//!     .with_id("id-123".to_string());  // Proper chaining
//! assert_eq!(reasoning.id, Some("id-123".to_string()));
//! ```
//!
//! # Performance Tips
//!
//! ## Message Size
//! - Keep conversation history manageable (typically last 10-20 messages)
//! - Summarize old context rather than sending full history
//! - Images use 85-765 tokens each depending on size
//!
//! ## Content Type Selection
//! - Prefer URLs over base64 for multimedia (faster, fewer tokens)
//! - Use `ImageDetail::Low` when high detail isn't needed (saves tokens)
//! - Remove tool results from history after they've been used
//!
//! # See also
//!
//! - [`crate::completion::request`] for sending messages to models
//! - [`crate::providers`] for provider-specific implementations

use std::{convert::Infallible, str::FromStr};

use crate::OneOrMany;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CompletionError;

// ================================================================
// Message models
// ================================================================

/// A trait for converting [`Message`] to custom message types.
///
/// This trait provides a clean way to convert Rig's generic message format
/// into your own custom message types without running into Rust's orphan rule.
/// Since `Vec` is a foreign type (owned by stdlib), implementing `TryFrom<Message>`
/// for `Vec<YourType>` would violate the orphan rule. This trait solves that problem.
///
/// # When to implement
///
/// Implement this trait when:
/// - You need to integrate Rig with existing message-based systems
/// - You want to convert between Rig's format and your own message types
/// - You need custom conversion logic beyond simple type mapping
///
/// # Examples
///
/// ```
/// use rig::completion::message::{ConvertMessage, Message, UserContent, AssistantContent};
/// use rig::OneOrMany;
///
/// #[derive(Debug)]
/// struct MyMessage {
///     role: String,
///     content: String,
/// }
///
/// #[derive(Debug)]
/// struct ConversionError(String);
///
/// impl std::fmt::Display for ConversionError {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         write!(f, "Conversion error: {}", self.0)
///     }
/// }
///
/// impl std::error::Error for ConversionError {}
///
/// impl ConvertMessage for MyMessage {
///     type Error = ConversionError;
///
///     fn convert_from_message(message: Message) -> Result<Vec<Self>, Self::Error> {
///         match message {
///             Message::User { content } => {
///                 // Extract text from all text content items
///                 let mut messages = Vec::new();
///
///                 for item in content.iter() {
///                     if let UserContent::Text(text) = item {
///                         messages.push(MyMessage {
///                             role: "user".to_string(),
///                             content: text.text.clone(),
///                         });
///                     }
///                 }
///
///                 if messages.is_empty() {
///                     return Err(ConversionError("No text content found".to_string()));
///                 }
///
///                 Ok(messages)
///             }
///             Message::Assistant { content, .. } => {
///                 // Extract text from assistant content
///                 let mut messages = Vec::new();
///
///                 for item in content.iter() {
///                     if let AssistantContent::Text(text) = item {
///                         messages.push(MyMessage {
///                             role: "assistant".to_string(),
///                             content: text.text.clone(),
///                         });
///                     }
///                 }
///
///                 if messages.is_empty() {
///                     return Err(ConversionError("No text content found".to_string()));
///                 }
///
///                 Ok(messages)
///             }
///         }
///     }
/// }
///
/// // Usage
/// let msg = Message::user("Hello, world!");
/// let converted = MyMessage::convert_from_message(msg).unwrap();
/// assert_eq!(converted[0].role, "user");
/// assert_eq!(converted[0].content, "Hello, world!");
/// ```
///
/// # See also
///
/// - [`Message`] for the source message type
/// - [`From`] and [`TryFrom`] for simpler conversions
pub trait ConvertMessage: Sized + Send + Sync {
    /// The error type returned when conversion fails.
    type Error: std::error::Error + Send;

    /// Converts a Rig message into a vector of custom message types.
    ///
    /// Returns a vector because a single Rig message may map to multiple
    /// messages in your format (e.g., separating user content and tool results).
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion cannot be performed, such as when
    /// the message contains content types unsupported by your format.
    fn convert_from_message(message: Message) -> Result<Vec<Self>, Self::Error>;
}

/// A message in a conversation between a user and an AI assistant.
///
/// Messages form the core communication structure in Rig. Each message has a role
/// (user or assistant) and can contain various types of content such as text, images,
/// audio, documents, or tool-related information.
///
/// While messages can contain multiple content items, most commonly you'll see one
/// content type per message (e.g., an image with a text description, or just text).
///
/// # Provider Compatibility
///
/// Each LLM provider converts these generic messages to their provider-specific format
/// using [`From`] or [`TryFrom`] traits. Since not all providers support all features,
/// conversion may be lossy (e.g., images might be discarded for non-vision models).
///
/// # Conversions
///
/// This type implements several convenient conversions:
///
/// ```
/// use rig::completion::Message;
///
/// // From string types
/// let msg: Message = "Hello".into();
/// let msg: Message = String::from("Hello").into();
/// ```
///
/// # Examples
///
/// Creating a simple text message:
///
/// ```
/// use rig::completion::Message;
///
/// let msg = Message::user("Hello, world!");
/// ```
///
/// Creating a message with an assistant response:
///
/// ```
/// use rig::completion::Message;
///
/// let response = Message::assistant("I'm doing well, thank you!");
/// ```
///
/// # See also
///
/// - [`UserContent`] for user message content types
/// - [`AssistantContent`] for assistant response content types
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// User message containing one or more content types.
    ///
    /// User messages typically contain prompts, questions, or follow-up responses.
    /// They can include text, images, audio, documents, and tool results.
    ///
    /// See [`UserContent`] for all supported content types.
    User { content: OneOrMany<UserContent> },

    /// Assistant message containing one or more content types.
    ///
    /// Assistant messages contain the AI's responses, which can be text,
    /// tool calls, or reasoning steps.
    ///
    /// The optional `id` field identifies specific response turns in multi-turn
    /// conversations.
    ///
    /// See [`AssistantContent`] for all supported content types.
    Assistant {
        id: Option<String>,
        content: OneOrMany<AssistantContent>,
    },
}

/// Content types that can be included in user messages.
///
/// User messages can contain various types of content including text, multimedia
/// (images, audio, video), documents, and tool execution results. Provider support
/// for each content type varies.
///
/// # Content Type Support
///
/// - **Text**: Universally supported by all providers
/// - **Images**: Supported by vision-capable models (GPT-4V, Claude 3, Gemini Pro Vision, etc.)
/// - **Audio**: Supported by audio-capable models (Whisper, etc.)
/// - **Video**: Supported by select multimodal models
/// - **Documents**: Supported by document-aware models
/// - **Tool Results**: Supported by function-calling capable models
///
/// # Multimedia Encoding
///
/// Multimedia content (images, audio, video) can be provided in two formats:
/// - **Base64-encoded**: Data embedded directly in the message
/// - **URL**: Reference to externally hosted content (provider support varies)
///
/// # Choosing the Right Content Type
///
/// - **Text**: Use for all text-based user input. Universal support across all providers.
/// - **Image**: Use for visual analysis tasks. Requires vision-capable models (GPT-4V, Claude 3+, Gemini Pro Vision).
///   - URLs are preferred for large images (faster, less token usage)
///   - Base64 for small images or when URLs aren't available
/// - **Audio**: Use for transcription or audio analysis. Limited provider support (OpenAI Whisper, etc.).
/// - **Video**: Use for video understanding. Very limited provider support (Gemini 1.5+).
/// - **Document**: Use for document analysis (PDFs, etc.). Provider-specific support.
/// - **ToolResult**: Use only for returning tool execution results to the model in multi-turn conversations.
///
/// # Size Limitations
///
/// Be aware of size limits:
/// - **Base64 images**: Typically 20MB max, counts heavily toward token limits (85-765 tokens per image)
/// - **URLs**: Fetched by provider, usually larger limits
/// - **Documents**: Provider-specific, often 10-100 pages
///
/// # Examples
///
/// Creating text content:
///
/// ```
/// use rig::completion::message::UserContent;
///
/// let content = UserContent::text("Hello, world!");
/// ```
///
/// Creating image content from a URL (preferred):
///
/// ```
/// use rig::completion::message::{UserContent, ImageMediaType, ImageDetail};
///
/// let image = UserContent::image_url(
///     "https://example.com/image.png",
///     Some(ImageMediaType::PNG),
///     Some(ImageDetail::High)
/// );
/// ```
///
/// Creating image content from base64 data:
///
/// ```
/// use rig::completion::message::{UserContent, ImageMediaType, ImageDetail};
///
/// let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
/// let image = UserContent::image_base64(
///     base64_data,
///     Some(ImageMediaType::PNG),
///     Some(ImageDetail::Low)  // Use Low for small images
/// );
/// ```
///
/// # Performance Tips
///
/// - Prefer URLs over base64 for images when possible (saves tokens and is faster)
/// - Resize images to appropriate dimensions before sending (768x768 is often sufficient)
/// - Use `ImageDetail::Low` for thumbnails or simple images (saves ~200 tokens per image)
/// - For multi-image scenarios, consider whether all images are needed or if quality can be reduced
///
/// # See also
///
/// - [`Text`] for text content
/// - [`Image`] for image content
/// - [`ToolResult`] for tool execution results
/// - [`Audio`] for audio content
/// - [`Video`] for video content
/// - [`Document`] for document content
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    /// Plain text content.
    Text(Text),

    /// Result from a tool execution.
    ToolResult(ToolResult),

    /// Image content (base64-encoded or URL).
    Image(Image),

    /// Audio content (base64-encoded or URL).
    Audio(Audio),

    /// Video content (base64-encoded or URL).
    Video(Video),

    /// Document content (base64-encoded or URL).
    Document(Document),
}

/// Content types that can be included in assistant messages.
///
/// Assistant responses can contain text, requests to call tools/functions,
/// or reasoning steps (for models that support chain-of-thought reasoning).
///
/// # Examples
///
/// Creating text content:
///
/// ```
/// use rig::completion::message::AssistantContent;
///
/// let content = AssistantContent::text("The answer is 42.");
/// ```
///
/// # See also
///
/// - [`Text`] for text responses
/// - [`ToolCall`] for function/tool calls
/// - [`Reasoning`] for chain-of-thought reasoning
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum AssistantContent {
    /// Plain text response from the assistant.
    Text(Text),

    /// A request to call a tool or function.
    ToolCall(ToolCall),

    /// Chain-of-thought reasoning steps (for reasoning-capable models).
    Reasoning(Reasoning),
}

/// Chain-of-thought reasoning from an AI model.
///
/// Some advanced AI models can provide explicit reasoning steps alongside their
/// responses, allowing you to understand the model's thought process. This is
/// particularly useful for complex problem-solving, mathematical proofs, and
/// transparent decision-making.
///
/// # Model Support
///
/// As of 2024, reasoning is supported by:
/// - **OpenAI o1 models** (o1-preview, o1-mini) - Native reasoning support
/// - **Anthropic Claude 3.5 Sonnet** - With extended thinking mode
/// - **Google Gemini Pro** - With chain-of-thought prompting
///
/// Check your provider's documentation for the latest support and capabilities.
///
/// # Use Cases
///
/// Reasoning is particularly valuable for:
/// - **Complex problem-solving**: Multi-step analytical tasks
/// - **Mathematical proofs**: Step-by-step mathematical reasoning
/// - **Debugging and troubleshooting**: Understanding decision paths
/// - **Transparent AI**: Making AI decisions explainable
/// - **Educational applications**: Showing work and explanations
///
/// # Performance Impact
///
/// Note that enabling reasoning:
/// - **Increases latency**: Models think longer before responding (5-30 seconds typical)
/// - **Increases token usage**: Each reasoning step counts as tokens (can add 500-2000 tokens)
/// - **May improve accuracy**: Particularly for complex, multi-step tasks
/// - **Not always necessary**: Simple tasks don't benefit from reasoning overhead
///
/// # Invariants
///
/// - The `reasoning` vector should not be empty when used in a response
/// - Each reasoning step should be a complete thought or sentence
/// - Steps are ordered chronologically (first thought to last)
///
/// # Examples
///
/// Creating reasoning from a single step:
///
/// ```
/// use rig::completion::message::Reasoning;
///
/// let reasoning = Reasoning::new("First, I'll analyze the input data");
/// assert_eq!(reasoning.reasoning.len(), 1);
/// ```
///
/// Creating reasoning from multiple steps:
///
/// ```
/// use rig::completion::message::Reasoning;
///
/// let steps = vec![
///     "First, analyze the problem structure".to_string(),
///     "Then, identify the key variables".to_string(),
///     "Next, apply the relevant formula".to_string(),
///     "Finally, verify the result".to_string(),
/// ];
/// let reasoning = Reasoning::multi(steps);
/// assert_eq!(reasoning.reasoning.len(), 4);
/// ```
///
/// Using reasoning-capable models:
///
/// ```ignore
/// # use rig::providers::openai;
/// # use rig::client::completion::CompletionClient;
/// # use rig::completion::Prompt;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = openai::Client::new("your-api-key");
/// // Use a reasoning-capable model
/// let model = client.completion_model(openai::O1_PREVIEW);
///
/// // The model will automatically include reasoning in complex tasks
/// let response = model.prompt(
///     "Prove that the square root of 2 is irrational using proof by contradiction"
/// ).await?;
///
/// // Response includes both reasoning steps and final answer
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[non_exhaustive]
pub struct Reasoning {
    /// Optional identifier for this reasoning instance.
    ///
    /// Used to associate reasoning with specific response turns in
    /// multi-turn conversations.
    pub id: Option<String>,

    /// The individual reasoning steps.
    ///
    /// Each string represents one step in the model's reasoning process.
    pub reasoning: Vec<String>,
}

impl Reasoning {
    /// Creates reasoning from a single step.
    ///
    /// # Examples
    ///
    /// ```
    /// use rig::completion::message::Reasoning;
    ///
    /// let reasoning = Reasoning::new("Analyzing the problem requirements");
    /// assert_eq!(reasoning.reasoning.len(), 1);
    /// assert!(reasoning.id.is_none());
    /// ```
    pub fn new(input: &str) -> Self {
        Self {
            id: None,
            reasoning: vec![input.to_string()],
        }
    }

    /// Sets an optional ID for this reasoning.
    ///
    /// # Examples
    ///
    /// ```
    /// use rig::completion::message::Reasoning;
    ///
    /// let reasoning = Reasoning::new("Step 1")
    ///     .optional_id(Some("reasoning-123".to_string()));
    /// assert_eq!(reasoning.id, Some("reasoning-123".to_string()));
    /// ```
    pub fn optional_id(mut self, id: Option<String>) -> Self {
        self.id = id;
        self
    }

    /// Sets the ID for this reasoning.
    ///
    /// # Examples
    ///
    /// ```
    /// use rig::completion::message::Reasoning;
    ///
    /// let reasoning = Reasoning::new("Step 1")
    ///     .with_id("reasoning-456".to_string());
    /// assert_eq!(reasoning.id, Some("reasoning-456".to_string()));
    /// ```
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Creates reasoning from multiple steps.
    ///
    /// # Examples
    ///
    /// ```
    /// use rig::completion::message::Reasoning;
    ///
    /// let steps = vec![
    ///     "First step".to_string(),
    ///     "Second step".to_string(),
    /// ];
    /// let reasoning = Reasoning::multi(steps);
    /// assert_eq!(reasoning.reasoning.len(), 2);
    /// ```
    pub fn multi(input: Vec<String>) -> Self {
        Self {
            id: None,
            reasoning: input,
        }
    }
}

/// Tool result content containing information about a tool call and it's resulting content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolResult {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub content: OneOrMany<ToolResultContent>,
}

/// Describes the content of a tool result, which can be text or an image.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolResultContent {
    Text(Text),
    Image(Image),
}

/// Describes a tool call with an id and function to call, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub call_id: Option<String>,
    pub function: ToolFunction,
}

/// Describes a tool function to call with a name and arguments, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ================================================================
// Base content models
// ================================================================

/// Basic text content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Text {
    pub text: String,
}

impl Text {
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { text } = self;
        write!(f, "{text}")
    }
}

/// Image content containing image data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Image {
    pub data: DocumentSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<ImageMediaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl Image {
    pub fn try_into_url(self) -> Result<String, MessageError> {
        match self.data {
            DocumentSourceKind::Url(url) => Ok(url),
            DocumentSourceKind::Base64(data) => {
                let Some(media_type) = self.media_type else {
                    return Err(MessageError::ConversionError(
                        "A media type is required to create a valid base64-encoded image URL"
                            .to_string(),
                    ));
                };

                Ok(format!(
                    "data:image/{ty};base64,{data}",
                    ty = media_type.to_mime_type()
                ))
            }
            unknown => Err(MessageError::ConversionError(format!(
                "Tried to convert unknown type to a URL: {unknown:?}"
            ))),
        }
    }
}

/// The kind of image source (to be used).
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
#[serde(tag = "type", content = "value", rename_all = "camelCase")]
#[non_exhaustive]
pub enum DocumentSourceKind {
    /// A file URL/URI.
    Url(String),
    /// A base-64 encoded string.
    Base64(String),
    /// Raw bytes
    Raw(Vec<u8>),
    /// A string (or a string literal).
    String(String),
    #[default]
    /// An unknown file source (there's nothing there).
    Unknown,
}

impl DocumentSourceKind {
    pub fn url(url: &str) -> Self {
        Self::Url(url.to_string())
    }

    pub fn base64(base64_string: &str) -> Self {
        Self::Base64(base64_string.to_string())
    }

    pub fn raw(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Raw(bytes.into())
    }

    pub fn string(input: &str) -> Self {
        Self::String(input.into())
    }

    pub fn unknown() -> Self {
        Self::Unknown
    }

    pub fn try_into_inner(self) -> Option<String> {
        match self {
            Self::Url(s) | Self::Base64(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for DocumentSourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Url(string) => write!(f, "{string}"),
            Self::Base64(string) => write!(f, "{string}"),
            Self::String(string) => write!(f, "{string}"),
            Self::Raw(_) => write!(f, "<binary data>"),
            Self::Unknown => write!(f, "<unknown>"),
        }
    }
}

/// Audio content containing audio data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Audio {
    pub data: DocumentSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<AudioMediaType>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Video content containing video data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Video {
    pub data: DocumentSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<VideoMediaType>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Document content containing document data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Document {
    pub data: DocumentSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<DocumentMediaType>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Describes the format of the content, which can be base64 or string.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ContentFormat {
    #[default]
    Base64,
    String,
    Url,
}

/// Helper enum that tracks the media type of the content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum MediaType {
    Image(ImageMediaType),
    Audio(AudioMediaType),
    Document(DocumentMediaType),
    Video(VideoMediaType),
}

/// Describes the image media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageMediaType {
    JPEG,
    PNG,
    GIF,
    WEBP,
    HEIC,
    HEIF,
    SVG,
}

/// Describes the document media type of the content. Not every provider supports every media type.
/// Includes also programming languages as document types for providers who support code running.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentMediaType {
    PDF,
    TXT,
    RTF,
    HTML,
    CSS,
    MARKDOWN,
    CSV,
    XML,
    Javascript,
    Python,
}

impl DocumentMediaType {
    pub fn is_code(&self) -> bool {
        matches!(self, Self::Javascript | Self::Python)
    }
}

/// Describes the audio media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AudioMediaType {
    WAV,
    MP3,
    AIFF,
    AAC,
    OGG,
    FLAC,
}

/// Describes the video media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum VideoMediaType {
    AVI,
    MP4,
    MPEG,
}

/// Describes the detail of the image content, which can be low, high, or auto (open-ai specific).
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    #[default]
    Auto,
}

// ================================================================
// Impl. for message models
// ================================================================

impl Message {
    /// This helper method is primarily used to extract the first string prompt from a `Message`.
    /// Since `Message` might have more than just text content, we need to find the first text.
    pub(crate) fn rag_text(&self) -> Option<String> {
        match self {
            Message::User { content } => {
                for item in content.iter() {
                    if let UserContent::Text(Text { text }) = item {
                        return Some(text.clone());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Helper constructor to make creating user messages easier.
    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::text(text)),
        }
    }

    /// Helper constructor to make creating assistant messages easier.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::text(text)),
        }
    }

    /// Helper constructor to make creating assistant messages easier.
    pub fn assistant_with_id(id: String, text: impl Into<String>) -> Self {
        Message::Assistant {
            id: Some(id),
            content: OneOrMany::one(AssistantContent::text(text)),
        }
    }

    /// Helper constructor to make creating tool result messages easier.
    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: id.into(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text(content)),
            })),
        }
    }

    pub fn tool_result_with_call_id(
        id: impl Into<String>,
        call_id: Option<String>,
        content: impl Into<String>,
    ) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: id.into(),
                call_id,
                content: OneOrMany::one(ToolResultContent::text(content)),
            })),
        }
    }
}

impl UserContent {
    /// Helper constructor to make creating user text content easier.
    pub fn text(text: impl Into<String>) -> Self {
        UserContent::Text(text.into().into())
    }

    /// Helper constructor to make creating user image content easier.
    pub fn image_base64(
        data: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        UserContent::Image(Image {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating user image content from raw unencoded bytes easier.
    pub fn image_raw(
        data: impl Into<Vec<u8>>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        UserContent::Image(Image {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            detail,
            ..Default::default()
        })
    }

    /// Helper constructor to make creating user image content easier.
    pub fn image_url(
        url: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        UserContent::Image(Image {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating user audio content easier.
    pub fn audio(data: impl Into<String>, media_type: Option<AudioMediaType>) -> Self {
        UserContent::Audio(Audio {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating user audio content from raw unencoded bytes easier.
    pub fn audio_raw(data: impl Into<Vec<u8>>, media_type: Option<AudioMediaType>) -> Self {
        UserContent::Audio(Audio {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper to create an audio resource from a URL
    pub fn audio_url(url: impl Into<String>, media_type: Option<AudioMediaType>) -> Self {
        UserContent::Audio(Audio {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper constructor to make creating user document content easier.
    /// This creates a document that assumes the data being passed in is a raw string.
    pub fn document(data: impl Into<String>, media_type: Option<DocumentMediaType>) -> Self {
        let data: String = data.into();
        UserContent::Document(Document {
            data: DocumentSourceKind::string(&data),
            media_type,
            additional_params: None,
        })
    }

    /// Helper to create a document from raw unencoded bytes
    pub fn document_raw(data: impl Into<Vec<u8>>, media_type: Option<DocumentMediaType>) -> Self {
        UserContent::Document(Document {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper to create a document from a URL
    pub fn document_url(url: impl Into<String>, media_type: Option<DocumentMediaType>) -> Self {
        UserContent::Document(Document {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper constructor to make creating user tool result content easier.
    pub fn tool_result(id: impl Into<String>, content: OneOrMany<ToolResultContent>) -> Self {
        UserContent::ToolResult(ToolResult {
            id: id.into(),
            call_id: None,
            content,
        })
    }

    /// Helper constructor to make creating user tool result content easier.
    pub fn tool_result_with_call_id(
        id: impl Into<String>,
        call_id: String,
        content: OneOrMany<ToolResultContent>,
    ) -> Self {
        UserContent::ToolResult(ToolResult {
            id: id.into(),
            call_id: Some(call_id),
            content,
        })
    }
}

impl AssistantContent {
    /// Helper constructor to make creating assistant text content easier.
    pub fn text(text: impl Into<String>) -> Self {
        AssistantContent::Text(text.into().into())
    }

    /// Helper constructor to make creating assistant tool call content easier.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall {
            id: id.into(),
            call_id: None,
            function: ToolFunction {
                name: name.into(),
                arguments,
            },
        })
    }

    pub fn tool_call_with_call_id(
        id: impl Into<String>,
        call_id: String,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall {
            id: id.into(),
            call_id: Some(call_id),
            function: ToolFunction {
                name: name.into(),
                arguments,
            },
        })
    }
}

impl ToolResultContent {
    /// Helper constructor to make creating tool result text content easier.
    pub fn text(text: impl Into<String>) -> Self {
        ToolResultContent::Text(text.into().into())
    }

    /// Helper constructor to make tool result images from a base64-encoded string.
    pub fn image_base64(
        data: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        ToolResultContent::Image(Image {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }

    /// Helper constructor to make tool result images from a base64-encoded string.
    pub fn image_raw(
        data: impl Into<Vec<u8>>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        ToolResultContent::Image(Image {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            detail,
            ..Default::default()
        })
    }

    /// Helper constructor to make tool result images from a URL.
    pub fn image_url(
        url: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        ToolResultContent::Image(Image {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }
}

/// Trait for converting between MIME types and media types.
pub trait MimeType {
    fn from_mime_type(mime_type: &str) -> Option<Self>
    where
        Self: Sized;
    fn to_mime_type(&self) -> &'static str;
}

impl MimeType for MediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        ImageMediaType::from_mime_type(mime_type)
            .map(MediaType::Image)
            .or_else(|| {
                DocumentMediaType::from_mime_type(mime_type)
                    .map(MediaType::Document)
                    .or_else(|| AudioMediaType::from_mime_type(mime_type).map(MediaType::Audio))
            })
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            MediaType::Image(media_type) => media_type.to_mime_type(),
            MediaType::Audio(media_type) => media_type.to_mime_type(),
            MediaType::Document(media_type) => media_type.to_mime_type(),
            MediaType::Video(media_type) => media_type.to_mime_type(),
        }
    }
}

impl MimeType for ImageMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "image/jpeg" => Some(ImageMediaType::JPEG),
            "image/png" => Some(ImageMediaType::PNG),
            "image/gif" => Some(ImageMediaType::GIF),
            "image/webp" => Some(ImageMediaType::WEBP),
            "image/heic" => Some(ImageMediaType::HEIC),
            "image/heif" => Some(ImageMediaType::HEIF),
            "image/svg+xml" => Some(ImageMediaType::SVG),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            ImageMediaType::JPEG => "image/jpeg",
            ImageMediaType::PNG => "image/png",
            ImageMediaType::GIF => "image/gif",
            ImageMediaType::WEBP => "image/webp",
            ImageMediaType::HEIC => "image/heic",
            ImageMediaType::HEIF => "image/heif",
            ImageMediaType::SVG => "image/svg+xml",
        }
    }
}

impl MimeType for DocumentMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "application/pdf" => Some(DocumentMediaType::PDF),
            "text/plain" => Some(DocumentMediaType::TXT),
            "text/rtf" => Some(DocumentMediaType::RTF),
            "text/html" => Some(DocumentMediaType::HTML),
            "text/css" => Some(DocumentMediaType::CSS),
            "text/md" | "text/markdown" => Some(DocumentMediaType::MARKDOWN),
            "text/csv" => Some(DocumentMediaType::CSV),
            "text/xml" => Some(DocumentMediaType::XML),
            "application/x-javascript" | "text/x-javascript" => Some(DocumentMediaType::Javascript),
            "application/x-python" | "text/x-python" => Some(DocumentMediaType::Python),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            DocumentMediaType::PDF => "application/pdf",
            DocumentMediaType::TXT => "text/plain",
            DocumentMediaType::RTF => "text/rtf",
            DocumentMediaType::HTML => "text/html",
            DocumentMediaType::CSS => "text/css",
            DocumentMediaType::MARKDOWN => "text/markdown",
            DocumentMediaType::CSV => "text/csv",
            DocumentMediaType::XML => "text/xml",
            DocumentMediaType::Javascript => "application/x-javascript",
            DocumentMediaType::Python => "application/x-python",
        }
    }
}

impl MimeType for AudioMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "audio/wav" => Some(AudioMediaType::WAV),
            "audio/mp3" => Some(AudioMediaType::MP3),
            "audio/aiff" => Some(AudioMediaType::AIFF),
            "audio/aac" => Some(AudioMediaType::AAC),
            "audio/ogg" => Some(AudioMediaType::OGG),
            "audio/flac" => Some(AudioMediaType::FLAC),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            AudioMediaType::WAV => "audio/wav",
            AudioMediaType::MP3 => "audio/mp3",
            AudioMediaType::AIFF => "audio/aiff",
            AudioMediaType::AAC => "audio/aac",
            AudioMediaType::OGG => "audio/ogg",
            AudioMediaType::FLAC => "audio/flac",
        }
    }
}

impl MimeType for VideoMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self>
    where
        Self: Sized,
    {
        match mime_type {
            "video/avi" => Some(VideoMediaType::AVI),
            "video/mp4" => Some(VideoMediaType::MP4),
            "video/mpeg" => Some(VideoMediaType::MPEG),
            &_ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            VideoMediaType::AVI => "video/avi",
            VideoMediaType::MP4 => "video/mp4",
            VideoMediaType::MPEG => "video/mpeg",
        }
    }
}

impl std::str::FromStr for ImageDetail {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(ImageDetail::Low),
            "high" => Ok(ImageDetail::High),
            "auto" => Ok(ImageDetail::Auto),
            _ => Err(()),
        }
    }
}

// ================================================================
// FromStr, From<String>, and From<&str> impls
// ================================================================

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text { text }
    }
}

impl From<&String> for Text {
    fn from(text: &String) -> Self {
        text.to_owned().into()
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl FromStr for Text {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl From<String> for Message {
    fn from(text: String) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Text(text.into())),
        }
    }
}

impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Text(text.into())),
        }
    }
}

impl From<&String> for Message {
    fn from(text: &String) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Text(text.into())),
        }
    }
}

impl From<Text> for Message {
    fn from(text: Text) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Text(text)),
        }
    }
}

impl From<Image> for Message {
    fn from(image: Image) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Image(image)),
        }
    }
}

impl From<Audio> for Message {
    fn from(audio: Audio) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Audio(audio)),
        }
    }
}

impl From<Document> for Message {
    fn from(document: Document) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Document(document)),
        }
    }
}

impl From<String> for ToolResultContent {
    fn from(text: String) -> Self {
        ToolResultContent::text(text)
    }
}

impl From<String> for AssistantContent {
    fn from(text: String) -> Self {
        AssistantContent::text(text)
    }
}

impl From<String> for UserContent {
    fn from(text: String) -> Self {
        UserContent::text(text)
    }
}

impl From<AssistantContent> for Message {
    fn from(content: AssistantContent) -> Self {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(content),
        }
    }
}

impl From<UserContent> for Message {
    fn from(content: UserContent) -> Self {
        Message::User {
            content: OneOrMany::one(content),
        }
    }
}

impl From<OneOrMany<AssistantContent>> for Message {
    fn from(content: OneOrMany<AssistantContent>) -> Self {
        Message::Assistant { id: None, content }
    }
}

impl From<OneOrMany<UserContent>> for Message {
    fn from(content: OneOrMany<UserContent>) -> Self {
        Message::User { content }
    }
}

impl From<ToolCall> for Message {
    fn from(tool_call: ToolCall) -> Self {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
        }
    }
}

impl From<ToolResult> for Message {
    fn from(tool_result: ToolResult) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(tool_result)),
        }
    }
}

impl From<ToolResultContent> for Message {
    fn from(tool_result_content: ToolResultContent) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: String::new(),
                call_id: None,
                content: OneOrMany::one(tool_result_content),
            })),
        }
    }
}

#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Specific {
        function_names: Vec<String>,
    },
}

// ================================================================
// Error types
// ================================================================

/// Error type to represent issues with converting messages to and from specific provider messages.
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Message conversion error: {0}")]
    ConversionError(String),
}

impl From<MessageError> for CompletionError {
    fn from(error: MessageError) -> Self {
        CompletionError::RequestError(error.into())
    }
}
