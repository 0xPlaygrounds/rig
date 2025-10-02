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
//!   provider-agnostic and automatically converted to each provider's specific format.
//! - [`request`]: Defines the traits and types for building completion requests,
//!   handling responses, and defining completion models.
//!
//! ## Abstraction Layers
//!
//! Rig provides three levels of abstraction for LLM interactions:
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │      User Application Code          │
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
//! ### High-level: [`Prompt`]
//!
//! Simple one-shot prompting for straightforward requests:
//!
//! ```ignore
//! # use rig::providers::openai;
//! # use rig::client::completion::CompletionClient;
//! # use rig::completion::Prompt;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("your-api-key");
//! let model = client.completion_model(openai::GPT_4);
//!
//! let response = model.prompt("Explain quantum computing").await?;
//! println!("{}", response);
//! # Ok(())
//! # }
//! ```
//!
//! **Use when:** You need a single response without conversation history.
//!
//! ### Mid-level: [`Chat`]
//!
//! Multi-turn conversations with context:
//!
//! ```ignore
//! # use rig::providers::openai;
//! # use rig::client::completion::CompletionClient;
//! # use rig::completion::{Chat, Message};
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("your-api-key");
//! let model = client.completion_model(openai::GPT_4);
//!
//! let history = vec![
//!     Message::user("What is 2+2?"),
//!     Message::assistant("2+2 equals 4."),
//! ];
//!
//! let response = model.chat("What about 3+3?", history).await?;
//! # Ok(())
//! # }
//! ```
//!
//! **Use when:** You need context from previous messages in the conversation.
//!
//! ### Low-level: [`Completion`]
//!
//! Full control over request parameters:
//!
//! ```ignore
//! # use rig::providers::openai;
//! # use rig::client::completion::CompletionClient;
//! # use rig::completion::Completion;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("your-api-key");
//! let model = client.completion_model(openai::GPT_4);
//!
//! let request = model.completion_request("Explain quantum computing")
//!     .temperature(0.7)
//!     .max_tokens(500)
//!     .build()?;
//!
//! let response = model.completion(request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! **Use when:** You need fine-grained control over temperature, tokens, or other parameters.
//!
//! # Provider-Agnostic Design
//!
//! All completion operations work identically across providers. Simply swap the client:
//!
//! ```ignore
//! # use rig::client::completion::CompletionClient;
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
//! // Same API for both
//! let response1 = openai_model.prompt("Hello").await?;
//! let response2 = anthropic_model.prompt("Hello").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Common Patterns
//!
//! ## Error Handling with Retry
//!
//! ```ignore
//! use rig::providers::openai;
//! use rig::client::completion::CompletionClient;
//! use rig::completion::{Prompt, CompletionError};
//! use std::time::Duration;
//! use tokio::time::sleep;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("your-api-key");
//! let model = client.completion_model(openai::GPT_4);
//!
//! let mut retries = 0;
//! let max_retries = 3;
//!
//! loop {
//!     match model.prompt("Hello").await {
//!         Ok(response) => {
//!             println!("{}", response);
//!             break;
//!         }
//!         Err(CompletionError::HttpError(_)) if retries < max_retries => {
//!             retries += 1;
//!             let delay = Duration::from_secs(2_u64.pow(retries));
//!             eprintln!("Network error. Retrying in {:?}...", delay);
//!             sleep(delay).await;
//!         }
//!         Err(e) => return Err(e.into()),
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Streaming Responses
//!
//! ```ignore
//! # use rig::providers::openai;
//! # use rig::client::completion::CompletionClient;
//! # use rig::completion::Completion;
//! # use futures::StreamExt;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("your-api-key");
//! let model = client.completion_model(openai::GPT_4);
//!
//! let request = model.completion_request("Write a story").build()?;
//! let mut stream = model.completion_stream(request).await?;
//!
//! while let Some(chunk) = stream.next().await {
//!     print!("{}", chunk?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Building Conversation History
//!
//! ```ignore
//! # use rig::providers::openai;
//! # use rig::client::completion::CompletionClient;
//! # use rig::completion::{Message, Chat};
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("your-api-key");
//! let model = client.completion_model(openai::GPT_4);
//!
//! let mut conversation = Vec::new();
//!
//! // First exchange
//! conversation.push(Message::user("What's 2+2?"));
//! let response1 = model.chat("What's 2+2?", conversation.clone()).await?;
//! conversation.push(Message::assistant(response1));
//!
//! // Second exchange with context
//! let response2 = model.chat("What about 3+3?", conversation.clone()).await?;
//! conversation.push(Message::assistant(response2));
//! # Ok(())
//! # }
//! ```
//!
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
//! ```ignore
//! use rig::providers::openai;
//! use rig::client::completion::CompletionClient;
//! use rig::completion::Prompt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = openai::Client::new("your-api-key");
//!     let model = client.completion_model(openai::GPT_4);
//!     let response = model.prompt("Hello").await?;
//!     println!("{}", response);
//!     Ok(())
//! }
//! ```
//!
//! # Performance Considerations
//!
//! ## Token Usage
//! - Text: ~1 token per 4 characters (English)
//! - Images (URL): 85-765 tokens depending on size and detail level
//! - Images (base64): Same as URL plus encoding overhead
//! - Each message in history adds to total token count
//!
//! ## Latency
//! - Simple prompts: 1-3 seconds typical
//! - Complex prompts with reasoning: 5-15 seconds
//! - Streaming: First token in <1 second
//!
//! ## Cost Optimization
//! - Use smaller models (GPT-3.5, Claude Haiku) for simple tasks
//! - Implement caching for repeated requests
//! - Limit conversation history length
//! - Use streaming for better perceived performance
//!
//! # See also
//!
//! - [`crate::providers`] for provider implementations (OpenAI, Anthropic, Cohere, etc.)
//! - [`crate::agent`] for building autonomous agents with tool use
//! - [`crate::embeddings`] for semantic search and RAG patterns

pub mod message;
pub mod request;

pub use message::{AssistantContent, Message, MessageError};
pub use request::*;
