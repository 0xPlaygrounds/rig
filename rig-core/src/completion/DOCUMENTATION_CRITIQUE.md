# Documentation Critique for rig-core/src/completion/

This document provides a detailed critique of the documentation in the `rig-core/src/completion/` module and suggestions for improvement.

## Overall Assessment

**Strengths:**
- ✅ Follows official Rust documentation guidelines (C-EXAMPLE, C-LINK, C-FAILURE)
- ✅ Comprehensive coverage of public API
- ✅ Examples use `?` operator for error handling
- ✅ Good use of linking between types
- ✅ Clear international English

**Areas for Improvement:**
- ⚠️ Some examples are too simplistic
- ⚠️ Missing troubleshooting guidance
- ⚠️ Inconsistent depth of documentation
- ⚠️ Limited real-world scenarios
- ⚠️ Missing performance considerations

## Detailed Critiques by File

---

## 1. mod.rs

### Issues:

#### Issue 1.1: Minimal Module Documentation
**Severity:** Medium
**Location:** Lines 1-34

**Problem:**
The module-level documentation is too brief. It doesn't explain:
- Why this module exists
- How it fits into the larger Rig ecosystem
- The relationship between submodules
- Common workflows

**Current:**
```rust
//! LLM completion functionality for Rig.
//!
//! This module provides the core functionality for interacting with Large Language
//! Models (LLMs) through a unified, provider-agnostic interface.
```

**Suggestion:**
```rust
//! Core LLM completion functionality for Rig.
//!
//! This module forms the foundation of Rig's LLM interaction layer, providing
//! a unified, provider-agnostic interface for sending prompts to and receiving
//! responses from various Large Language Model providers.
//!
//! # Architecture
//!
//! The completion module is organized into two main submodules:
//!
//! - [`message`]: Defines the message format for conversations. Messages are
//!   provider-agnostic and automatically converted to each provider's specific
//!   format.
//! - [`request`]: Defines the traits and types for building completion requests,
//!   handling responses, and defining completion models.
//!
//! # Core Concepts
//!
//! ## Abstraction Levels
//!
//! Rig provides three levels of abstraction for LLM interactions:
//!
//! 1. **High-level ([`Prompt`])**: Simple one-shot prompting
//!    ```no_run
//!    # use rig::providers::openai;
//!    # use rig::completion::Prompt;
//!    # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!    let model = openai::Client::new("key").completion_model(openai::GPT_4);
//!    let response = model.prompt("Hello!").await?;
//!    # Ok(())
//!    # }
//!    ```
//!
//! 2. **Mid-level ([`Chat`])**: Multi-turn conversations with history
//!    ```no_run
//!    # use rig::providers::openai;
//!    # use rig::completion::{Chat, Message};
//!    # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!    let model = openai::Client::new("key").completion_model(openai::GPT_4);
//!    let history = vec![
//!        Message::user("What is 2+2?"),
//!        Message::assistant("4"),
//!    ];
//!    let response = model.chat("What about 3+3?", history).await?;
//!    # Ok(())
//!    # }
//!    ```
//!
//! 3. **Low-level ([`Completion`])**: Full control over request parameters
//!    ```no_run
//!    # use rig::providers::openai;
//!    # use rig::completion::Completion;
//!    # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!    let model = openai::Client::new("key").completion_model(openai::GPT_4);
//!    let request = model.completion_request("Hello")
//!        .temperature(0.7)
//!        .max_tokens(100)
//!        .build()?;
//!    let response = model.completion(request).await?;
//!    # Ok(())
//!    # }
//!    ```
//!
//! ## Provider Agnostic Design
//!
//! All completion operations work identically across providers. Simply swap
//! the client:
//!
//! ```no_run
//! # use rig::completion::Prompt;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // OpenAI
//! let openai_model = rig::providers::openai::Client::new("key")
//!     .completion_model(rig::providers::openai::GPT_4);
//!
//! // Anthropic
//! let anthropic_model = rig::providers::anthropic::Client::new("key")
//!     .completion_model(rig::providers::anthropic::CLAUDE_3_5_SONNET);
//!
//! // Same API for both!
//! let response1 = openai_model.prompt("Hello").await?;
//! let response2 = anthropic_model.prompt("Hello").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Common Patterns
//!
//! ## Error Handling
//!
//! All completion operations return `Result` types. Handle errors appropriately:
//!
//! ```no_run
//! # use rig::providers::openai;
//! # use rig::completion::{Prompt, CompletionError};
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = openai::Client::new("key").completion_model(openai::GPT_4);
//!
//! match model.prompt("Hello").await {
//!     Ok(response) => println!("Success: {}", response),
//!     Err(CompletionError::HttpError(e)) => {
//!         eprintln!("Network error: {}. Check your connection.", e);
//!     }
//!     Err(CompletionError::ProviderError(msg)) => {
//!         eprintln!("Provider error: {}. Check your API key.", msg);
//!     }
//!     Err(e) => eprintln!("Unexpected error: {}", e),
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Streaming Responses
//!
//! For long-running completions, use streaming to receive partial responses:
//!
//! ```no_run
//! # use rig::providers::openai;
//! # use rig::completion::Completion;
//! # use futures::StreamExt;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = openai::Client::new("key").completion_model(openai::GPT_4);
//! let request = model.completion_request("Write a story").build()?;
//!
//! let mut stream = model.completion_stream(request).await?;
//! while let Some(chunk) = stream.next().await {
//!     print!("{}", chunk?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - **Message cloning**: Messages implement `Clone` but may contain large
//!   multimedia content. Consider using references where possible.
//! - **Provider selection**: Different providers have different latencies,
//!   costs, and capabilities. Benchmark for your use case.
//! - **Streaming**: Use streaming for long responses to reduce perceived latency.
//!
//! # See also
//!
//! - [`crate::providers`] for available LLM provider integrations
//! - [`crate::agent`] for building autonomous agents with tools
//! - [`crate::embeddings`] for semantic search and RAG
```

---

## 2. message.rs

### Issues:

#### Issue 2.1: Module Documentation Missing Common Patterns
**Severity:** Medium
**Location:** Lines 1-55

**Problem:**
The module documentation doesn't show common patterns users will encounter, such as:
- Building messages with multiple content types
- Handling tool results
- Working with conversation history

**Suggestion:**
Add a "Common Patterns" section:

```rust
//! # Common Patterns
//!
//! ## Building Conversation History
//!
//! ```
//! use rig::completion::Message;
//!
//! let conversation = vec![
//!     Message::user("What's 2+2?"),
//!     Message::assistant("2+2 equals 4."),
//!     Message::user("And 3+3?"),
//!     Message::assistant("3+3 equals 6."),
//! ];
//! ```
//!
//! ## Multimodal Messages with Text and Images
//!
//! ```
//! use rig::completion::message::{Message, UserContent, ImageMediaType};
//! use rig::OneOrMany;
//!
//! // Image with description
//! let msg = Message::User {
//!     content: OneOrMany::many(vec![
//!         UserContent::text("Analyse this diagram and explain the architecture:"),
//!         UserContent::image_url(
//!             "https://example.com/architecture.png",
//!             Some(ImageMediaType::PNG),
//!             None
//!         ),
//!     ])
//! };
//! ```
//!
//! ## Tool Results
//!
//! ```
//! use rig::completion::Message;
//!
//! // After a tool call, provide the result
//! let tool_result = Message::tool_result(
//!     "tool-call-123",
//!     "The current temperature is 72°F"
//! );
//! ```
//!
//! # Troubleshooting
//!
//! ## "Media type required" Error
//!
//! When creating images from base64 data, you must specify the media type:
//!
//! ```compile_fail
//! # use rig::completion::message::UserContent;
//! // ❌ This will fail when converted
//! let img = UserContent::image_base64("base64data", None, None);
//! ```
//!
//! ```
//! # use rig::completion::message::{UserContent, ImageMediaType};
//! // ✅ Correct: specify media type
//! let img = UserContent::image_base64(
//!     "base64data",
//!     Some(ImageMediaType::PNG),
//!     None
//! );
//! ```
//!
//! ## Provider Doesn't Support Content Type
//!
//! Not all providers support all content types. Check provider documentation:
//!
//! - **Text**: All providers
//! - **Images**: GPT-4V, Claude 3+, Gemini Pro Vision
//! - **Audio**: OpenAI Whisper, specific models only
//! - **Tool calls**: Most modern models (GPT-4, Claude 3+, etc.)
```

#### Issue 2.2: ConvertMessage Example Too Simplistic
**Severity:** Low
**Location:** Lines 83-104

**Problem:**
The example doesn't actually implement the conversion logic. It returns dummy data,
which doesn't help users understand how to implement this trait properly.

**Current:**
```rust
/// ```
/// use rig::completion::message::{ConvertMessage, Message};
///
/// struct MyMessage {
///     role: String,
///     content: String,
/// }
///
/// impl ConvertMessage for MyMessage {
///     type Error = Box<dyn std::error::Error + Send>;
///
///     fn convert_from_message(message: Message) -> Result<Vec<Self>, Self::Error> {
///         // Custom conversion logic here
///         Ok(vec![MyMessage {
///             role: "user".to_string(),
///             content: "example".to_string(),
///         }])
///     }
/// }
/// ```
```

**Suggestion:**
```rust
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
///         write!(f, "{}", self.0)
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
///                 // Extract text from user content
///                 let text = content.iter()
///                     .filter_map(|c| match c {
///                         UserContent::Text(t) => Some(t.text.clone()),
///                         _ => None,
///                     })
///                     .collect::<Vec<_>>()
///                     .join(" ");
///
///                 if text.is_empty() {
///                     return Err(ConversionError("No text content found".to_string()));
///                 }
///
///                 Ok(vec![MyMessage {
///                     role: "user".to_string(),
///                     content: text,
///                 }])
///             }
///             Message::Assistant { content, .. } => {
///                 // Extract text from assistant content
///                 let text = content.iter()
///                     .filter_map(|c| match c {
///                         AssistantContent::Text(t) => Some(t.text.clone()),
///                         _ => None,
///                     })
///                     .collect::<Vec<_>>()
///                     .join(" ");
///
///                 if text.is_empty() {
///                     return Err(ConversionError("No text content found".to_string()));
///                 }
///
///                 Ok(vec![MyMessage {
///                     role: "assistant".to_string(),
///                     content: text,
///                 }])
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
```

#### Issue 2.3: Message Documentation Missing Serialization Format
**Severity:** Medium
**Location:** Lines 126-199

**Problem:**
Users might need to know the JSON serialization format for debugging or
integration purposes. This is especially important given the `#[serde(...)]` attributes.

**Suggestion:**
Add a "Serialization" section:

```rust
/// # Serialization
///
/// Messages serialize to JSON with a `role` tag:
///
/// ```
/// # use rig::completion::Message;
/// # use rig::completion::message::UserContent;
/// # use rig::OneOrMany;
/// let user_msg = Message::user("Hello");
/// let json = serde_json::to_string_pretty(&user_msg).unwrap();
/// // Produces:
/// // {
/// //   "role": "user",
/// //   "content": [
/// //     {
/// //       "type": "text",
/// //       "text": "Hello"
/// //     }
/// //   ]
/// // }
/// # assert!(json.contains("\"role\": \"user\""));
/// ```
///
/// Assistant messages include an optional `id`:
///
/// ```
/// # use rig::completion::Message;
/// let assistant_msg = Message::assistant_with_id(
///     "msg-123".to_string(),
///     "Hi there!"
/// );
/// let json = serde_json::to_string_pretty(&assistant_msg).unwrap();
/// // Produces:
/// // {
/// //   "role": "assistant",
/// //   "id": "msg-123",
/// //   "content": [...]
/// // }
/// # assert!(json.contains("\"id\": \"msg-123\""));
/// ```
```

#### Issue 2.4: UserContent Missing Usage Guidance
**Severity:** Medium
**Location:** Lines 145-213

**Problem:**
The documentation lists what content types exist but doesn't explain when
to use each one or their limitations.

**Suggestion:**
Add usage guidance:

```rust
/// # Choosing the Right Content Type
///
/// - **Text**: Use for all text-based user input. Universal support.
/// - **Image**: Use for visual analysis tasks. Requires vision-capable models.
///   - URLs are preferred for large images (faster, less token usage)
///   - Base64 for small images or when URLs aren't available
/// - **Audio**: Use for transcription or audio analysis. Limited provider support.
/// - **Video**: Use for video understanding. Very limited provider support.
/// - **Document**: Use for document analysis (PDFs, etc.). Provider-specific.
/// - **ToolResult**: Use only for returning tool execution results to the model.
///
/// # Size Limitations
///
/// Be aware of size limits:
/// - **Base64 images**: Typically 20MB max, counts heavily toward token limits
/// - **URLs**: Fetched by provider, usually larger limits
/// - **Documents**: Provider-specific, often 10-100 pages
///
/// # Performance Tips
///
/// - Prefer URLs over base64 for images when possible
/// - Resize images to appropriate dimensions before sending
/// - For multi-image scenarios, consider separate requests if quality degrades
```

#### Issue 2.5: Reasoning Type Lacks Context
**Severity:** Medium
**Location:** Lines 248-295

**Problem:**
The documentation doesn't explain which models support reasoning or how
reasoning differs from regular responses.

**Suggestion:**
```rust
/// Chain-of-thought reasoning from an AI model.
///
/// Some advanced models (like OpenAI's o1 series, Claude 3.5 Sonnet with
/// extended thinking) can provide explicit reasoning steps alongside their
/// responses. This allows you to understand the model's thought process.
///
/// # Model Support
///
/// As of 2024, reasoning is supported by:
/// - OpenAI o1 models (o1-preview, o1-mini)
/// - Anthropic Claude 3.5 Sonnet (with extended thinking mode)
/// - Google Gemini Pro (with chain-of-thought prompting)
///
/// Check provider documentation for the latest support.
///
/// # Use Cases
///
/// Reasoning is particularly useful for:
/// - Complex problem-solving tasks
/// - Mathematical proofs
/// - Multi-step analytical tasks
/// - Debugging and troubleshooting
/// - Transparent decision-making
///
/// # Performance Impact
///
/// Note that reasoning:
/// - Increases latency (models think longer)
/// - Increases token usage (reasoning steps count as tokens)
/// - May improve accuracy for complex tasks
///
/// # Examples
///
/// Models that support reasoning will automatically include reasoning steps:
///
/// ```no_run
/// # use rig::providers::openai;
/// # use rig::completion::Prompt;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = openai::Client::new("key");
/// let model = client.completion_model(openai::O1_PREVIEW);
///
/// // The model will include reasoning in its response
/// let response = model.prompt(
///     "Prove that the square root of 2 is irrational"
/// ).await?;
///
/// // Response includes both reasoning and final answer
/// # Ok(())
/// # }
/// ```
```

---

## 3. request.rs

### Issues:

#### Issue 3.1: Missing Architecture Explanation
**Severity:** High
**Location:** Lines 1-69

**Problem:**
The module documentation doesn't explain the relationship between the four
main traits (Prompt, Chat, Completion, CompletionModel) or when to use each.

**Suggestion:**
Add an "Architecture" section before examples:

```rust
//! # Architecture
//!
//! This module defines a layered architecture for LLM interactions:
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         User Application            │
//! └──────────┬──────────────────────────┘
//!            │
//!            ├─> Prompt (simple one-shot)
//!            ├─> Chat (multi-turn with history)
//!            └─> Completion (full control)
//!            │
//! ┌──────────┴──────────────────────────┐
//! │      CompletionModel Trait          │ ← Implemented by providers
//! └──────────┬──────────────────────────┘
//!            │
//!    ┌───────┴────────┬────────────┐
//!    │                │            │
//! OpenAI         Anthropic      Custom
//! Provider       Provider       Provider
//! ```
//!
//! ## Trait Responsibilities
//!
//! ### [`Prompt`] - High-level one-shot prompting
//! **When to use**: Simple, one-off questions without conversation history.
//!
//! **Example use cases**:
//! - Text generation
//! - Classification
//! - Summarization
//! - Translation
//!
//! ### [`Chat`] - Multi-turn conversations
//! **When to use**: Conversations that need context from previous exchanges.
//!
//! **Example use cases**:
//! - Customer support bots
//! - Interactive assistants
//! - Iterative refinement tasks
//! - Contextual follow-ups
//!
//! ### [`Completion`] - Low-level custom requests
//! **When to use**: Need fine control over parameters or custom request building.
//!
//! **Example use cases**:
//! - Custom temperature per request
//! - Token limits
//! - Provider-specific parameters
//! - Advanced configurations
//!
//! ### [`CompletionModel`] - Provider interface
//! **When to implement**: Building a custom provider integration.
//!
//! **Example use cases**:
//! - Integrating a new LLM provider
//! - Wrapping a private/self-hosted model
//! - Creating mock models for testing
//! - Building proxy/caching layers
```

#### Issue 3.2: Error Examples Don't Show Recovery
**Severity:** Medium
**Location:** Lines 98-118, 181-200

**Problem:**
Error handling examples show how to match errors but not how to recover
or retry, which is critical for production use.

**Suggestion:**
Replace current error example with:

```rust
/// # Examples
///
/// ## Basic Error Handling
///
/// ```no_run
/// use rig::completion::{CompletionError, Prompt};
/// use rig::providers::openai;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = openai::Client::new("api-key");
/// let model = client.completion_model(openai::GPT_4);
///
/// match model.prompt("Hello").await {
///     Ok(response) => println!("Success: {}", response),
///     Err(CompletionError::HttpError(e)) => {
///         eprintln!("Network error: {}. Check your connection.", e);
///     },
///     Err(CompletionError::ProviderError(msg)) if msg.contains("rate_limit") => {
///         eprintln!("Rate limited. Waiting before retry...");
///         // Implement backoff strategy
///     },
///     Err(CompletionError::ProviderError(msg)) if msg.contains("invalid_api_key") => {
///         eprintln!("Invalid API key. Check your credentials.");
///     },
///     Err(e) => eprintln!("Other error: {}", e),
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Retry with Exponential Backoff
///
/// ```no_run
/// use rig::completion::{CompletionError, Prompt};
/// use rig::providers::openai;
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
///             eprintln!("Retry {} after {:?}", retries, delay);
///             sleep(delay).await;
///         },
///         Err(e) => return Err(e.into()),
///     }
/// }
/// # }
/// ```
```

#### Issue 3.3: Document Type Needs More Context
**Severity:** Low
**Location:** Lines 241-273

**Problem:**
The `Document` type documentation doesn't explain:
- How documents differ from regular text messages
- When to use documents vs. text content
- How documents are processed by models

**Suggestion:**
```rust
/// A document that can be provided as context to an LLM.
///
/// Documents are structured pieces of text that models can reference during
/// completion. They differ from regular text messages in that they:
/// - Have unique IDs for reference
/// - Are typically formatted as files (with XML-like tags)
/// - Can include metadata via `additional_props`
/// - Are treated as "reference material" rather than conversation turns
///
/// # When to Use Documents vs. Messages
///
/// **Use Documents when**:
/// - Providing reference material (documentation, knowledge base articles)
/// - The content is factual/static rather than conversational
/// - You want the model to cite sources by ID
/// - Implementing RAG (Retrieval-Augmented Generation)
///
/// **Use Messages when**:
/// - Having a conversation
/// - The text is a question or response
/// - Building chat history
///
/// # Formatting
///
/// Documents are typically formatted as XML-like files in the prompt:
///
/// ```text
/// <file id="doc-1">
/// [metadata if any]
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
///     id: "doc-1".to_string(),
///     text: "Rust is a systems programming language.".to_string(),
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
/// metadata.insert("chapter".to_string(), "Introduction".to_string());
/// metadata.insert("last_updated".to_string(), "2024-01-01".to_string());
///
/// let doc = Document {
///     id: "rust-intro".to_string(),
///     text: "Rust is a systems programming language focused on safety...".to_string(),
///     additional_props: metadata,
/// };
/// ```
///
/// ## RAG Pattern
///
/// ```no_run
/// # use rig::providers::openai;
/// # use rig::completion::{Prompt, request::Document};
/// # use std::collections::HashMap;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = openai::Client::new("key");
/// let model = client.completion_model(openai::GPT_4);
///
/// // Retrieved from vector database
/// let relevant_docs = vec![
///     Document {
///         id: "doc-1".to_string(),
///         text: "Rust's ownership system...".to_string(),
///         additional_props: HashMap::new(),
///     },
///     Document {
///         id: "doc-2".to_string(),
///         text: "Borrowing rules...".to_string(),
///         additional_props: HashMap::new(),
///     },
/// ];
///
/// // Model will use documents as context
/// let response = model.prompt("Explain ownership").await?;
/// # Ok(())
/// # }
/// ```
```

---

## 4. Cross-Cutting Issues

### Issue 4.1: Missing Performance Documentation
**Severity:** Medium
**Applies to:** All files

**Problem:**
No documentation mentions performance characteristics, token usage,
or cost implications.

**Suggestion:**
Add performance notes where relevant:

```rust
/// # Performance Considerations
///
/// ## Token Usage
/// - Text content: ~1 token per 4 characters
/// - Images (URL): ~85-765 tokens depending on size and detail
/// - Images (base64): Same as URL plus encoding overhead
/// - Reasoning steps: Each step adds tokens
///
/// ## Latency
/// - Simple prompts: 1-3 seconds typical
/// - Complex prompts with tools: 5-15 seconds
/// - Streaming: First token in <1 second, rest stream in
///
/// ## Cost Optimization
/// - Use smaller models (GPT-3.5) for simple tasks
/// - Implement caching for repeated requests
/// - Prefer URLs over base64 for images
/// - Limit conversation history length
```

### Issue 4.2: Missing Async Context
**Severity:** Low
**Applies to:** request.rs

**Problem:**
Examples use `.await?` but don't explain the async context or runtime requirements.

**Suggestion:**
Add to module docs:

```rust
//! # Async Runtime
//!
//! All completion operations are async and require a runtime like Tokio:
//!
//! ```toml
//! [dependencies]
//! rig-core = "0.21"
//! tokio = { version = "1", features = ["full"] }
//! ```
//!
//! ```no_run
//! use rig::providers::openai;
//! use rig::completion::Prompt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = openai::Client::new("key");
//!     let model = client.completion_model(openai::GPT_4);
//!     let response = model.prompt("Hello").await?;
//!     println!("{}", response);
//!     Ok(())
//! }
//! ```
```

### Issue 4.3: No Migration or Upgrade Guidance
**Severity:** Low
**Applies to:** All files

**Problem:**
No guidance for users migrating from previous versions or other libraries.

**Suggestion:**
Add to module or crate root:

```rust
//! # Migration Guide
//!
//! ## From LangChain (Python)
//!
//! If you're familiar with LangChain, here are the equivalents:
//!
//! | LangChain | Rig |
//! |-----------|-----|
//! | `ChatOpenAI().invoke()` | `model.prompt().await` |
//! | `ConversationChain` | `model.chat(message, history).await` |
//! | `ChatPromptTemplate` | `Message::user()` + builder pattern |
//! | `create_stuff_documents_chain` | Documents + RAG pattern |
//!
//! ## From OpenAI SDK (Rust)
//!
//! Rig provides higher-level abstractions:
//!
//! ```no_run
//! // OpenAI SDK (low-level)
//! // let request = CreateChatCompletionRequestArgs::default()
//! //     .model("gpt-4")
//! //     .messages([...])
//! //     .build()?;
//! // let response = client.chat().create(request).await?;
//!
//! // Rig (high-level)
//! # use rig::providers::openai;
//! # use rig::completion::Prompt;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("key");
//! let model = client.completion_model(openai::GPT_4);
//! let response = model.prompt("Hello").await?;
//! # Ok(())
//! # }
//! ```
```

### Issue 4.4: Examples Not Runnable for Testing
**Severity:** Medium
**Applies to:** All files with examples requiring API keys

**Problem:**
Many examples use `no_run` which prevents doc tests from verifying they compile correctly.
Some examples might have compilation errors that aren't caught.

**Suggestion:**
Use mock implementations for testability:

```rust
/// # Examples
///
/// ```
/// # // Mock setup for testing
/// # struct MockClient;
/// # impl MockClient {
/// #     fn new(_key: &str) -> Self { MockClient }
/// #     fn completion_model(&self, _model: &str) -> MockModel { MockModel }
/// # }
/// # struct MockModel;
/// # impl MockModel {
/// #     async fn prompt(&self, _prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
/// #         Ok("Mock response".to_string())
/// #     }
/// # }
/// # let openai = MockClient::new("key");
/// # let GPT_4 = "gpt-4";
/// #
/// # tokio_test::block_on(async {
/// let model = openai.completion_model(GPT_4);
/// let response = model.prompt("Hello").await?;
/// println!("{}", response);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// # });
/// ```
```

---

## 5. Summary of Recommendations

### High Priority
1. **Add architecture explanations** to request.rs module docs
2. **Improve ConvertMessage example** with realistic implementation
3. **Add performance/cost documentation** throughout
4. **Add troubleshooting sections** for common issues

### Medium Priority
5. **Add "Common Patterns" sections** to all modules
6. **Document serialization formats** for key types
7. **Add usage guidance** for content types
8. **Improve error handling examples** with recovery patterns
9. **Add async runtime context** and setup instructions

### Low Priority
10. **Add migration guides** from other libraries
11. **Improve example runnability** with mocks where appropriate
12. **Add "when to use" guidance** for all major types
13. **Document provider-specific behaviors** and limitations

### Style Improvements
14. **Use consistent section headers** across all docs
15. **Add emoji/icons** sparingly for visual scanning (⚠️, ✅, 💡)
16. **Ensure all code examples** are properly formatted and tested
17. **Add "See also" sections** to improve discoverability

---

## 6. Positive Aspects to Maintain

- ✅ Excellent use of linking between types
- ✅ Clear, professional English throughout
- ✅ Good error variant documentation
- ✅ Proper use of `#[non_exhaustive]` on error enums
- ✅ Consistent formatting and structure
- ✅ Examples use `?` operator correctly
- ✅ Good separation of concerns between modules

---

## 7. Metrics

Current documentation coverage (estimated):

| Metric | Score | Target |
|--------|-------|--------|
| API coverage | 95% | 100% |
| Example quality | 70% | 90% |
| Real-world scenarios | 50% | 80% |
| Error handling | 75% | 90% |
| Performance docs | 20% | 70% |
| Troubleshooting | 30% | 80% |

**Overall documentation quality: B+ (Good, but room for improvement)**
