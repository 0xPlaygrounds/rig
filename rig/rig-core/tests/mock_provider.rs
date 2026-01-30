//! Mock provider for testing Rig integrations without consuming tokens.
//!
//! This provider allows you to test agent features without needing to connect to
//! a third party provider. It echoes back prompts with a response counter and
//! optionally returns fixed tool function calls.
//!
//! **Note:** This module is only available in test builds (`#[cfg(test)]`).
//!
//! # Example
//! ```rust,ignore
//! use rig::providers::mock::{self, Client, ErrorMode};
//! use rig::client::{CompletionClient, Nothing};
//!
//! // Create a new mock client with default settings
//! let client = Client::default();
//!
//! // Create a completion model interface
//! let model = client.completion_model("mock-model");
//!
//! // Or configure the client with custom settings
//! let client = Client::builder()
//!     .api_key(Nothing)
//!     .enable_tool_calls(true)
//!     .build()
//!     .unwrap();
//!
//! // You can also test error scenarios
//! let error_client = Client::builder()
//!     .api_key(Nothing)
//!     .error_mode(ErrorMode::ResponseError("Test error".to_string()))
//!     .build()
//!     .unwrap();
//! ```

use crate::client::{
    self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
};
use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage, Usage};
use crate::message::{AssistantContent, Text, ToolCall, ToolFunction, UserContent};
use crate::streaming::{self, RawStreamingChoice, RawStreamingToolCall, StreamingResult};
use crate::OneOrMany;
use async_stream::stream;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ========== Configuration ==========

/// Error modes for simulating different error conditions during testing.
#[derive(Debug, Clone, Default)]
pub enum ErrorMode {
    /// No error - normal operation
    #[default]
    None,
    /// Simulates a request validation error
    RequestError(String),
    /// Simulates a response parsing error
    ResponseError(String),
    /// Simulates a provider-level error
    ProviderError(String),
}

/// Configuration for the mock provider.
#[derive(Debug, Clone, Default)]
pub struct MockConfig {
    /// Whether to include tool calls in responses
    pub enable_tool_calls: bool,
    /// Error mode for simulating errors
    pub error_mode: ErrorMode,
}

// ========== Provider Extension ==========

/// Mock provider extension
#[derive(Debug, Clone, Default)]
pub struct MockExt {
    config: MockConfig,
    response_counter: Arc<AtomicUsize>,
}

impl DebugExt for MockExt {}

impl Provider for MockExt {
    type Builder = MockBuilder;

    const VERIFY_PATH: &'static str = "";

    fn build<H>(
        builder: &client::ClientBuilder<
            Self::Builder,
            <Self::Builder as ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> crate::http_client::Result<Self> {
        Ok(Self {
            config: builder.ext().config.clone(),
            response_counter: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl<H> Capabilities<H> for MockExt {
    type Completion = Capable<CompletionModel<H>>;
    type Transcription = Nothing;
    type Embeddings = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

// ========== Provider Builder ==========

/// Builder for the mock provider
#[derive(Debug, Clone, Default)]
pub struct MockBuilder {
    config: MockConfig,
}

impl MockBuilder {
    /// Enable tool call responses
    pub fn enable_tool_calls(mut self, enable: bool) -> Self {
        self.config.enable_tool_calls = enable;
        self
    }

    /// Set the error mode for simulating errors
    pub fn error_mode(mut self, mode: ErrorMode) -> Self {
        self.config.error_mode = mode;
        self
    }
}

impl ProviderBuilder for MockBuilder {
    type Output = MockExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = "";
}

// ========== Client Types ==========

/// Type alias for the mock client
pub type Client<H = reqwest::Client> = client::Client<MockExt, H>;

/// Type alias for the mock client builder
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<MockBuilder, Nothing, H>;

impl Default for Client {
    fn default() -> Self {
        Client::builder().api_key(Nothing).build().unwrap()
    }
}

impl ProviderClient for Client {
    type Input = Nothing;

    fn from_env() -> Self {
        Client::default()
    }

    fn from_val(_: Self::Input) -> Self {
        Client::default()
    }
}

// ========== Builder Extension Methods ==========

impl<H> ClientBuilder<H> {
    /// Enable tool call responses
    pub fn enable_tool_calls(mut self, enable: bool) -> Self {
        self.ext_mut().config.enable_tool_calls = enable;
        self
    }

    /// Set the error mode for simulating errors
    pub fn error_mode(mut self, mode: ErrorMode) -> Self {
        self.ext_mut().config.error_mode = mode;
        self
    }
}

// ========== Response Types ==========

/// Response from a mock completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// The model name used
    pub model: String,
    /// The response content
    pub content: String,
    /// Whether tool calls were included
    pub has_tool_calls: bool,
    /// The response number (incremented for each response)
    pub response_number: usize,
    /// Token usage (fixed mock values)
    pub usage: MockUsage,
}

/// Mock token usage with fixed values
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MockUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl MockUsage {
    /// Create fixed mock usage values
    pub fn fixed() -> Self {
        Self {
            input_tokens: 100,
            output_tokens: 50,
        }
    }
}

/// Streaming response type for mock completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    /// The response number
    pub response_number: usize,
    /// Token usage
    pub usage: MockUsage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<Usage> {
        Some(Usage {
            input_tokens: self.usage.input_tokens,
            output_tokens: self.usage.output_tokens,
            total_tokens: self.usage.input_tokens + self.usage.output_tokens,
        })
    }
}

// ========== Completion Model ==========

/// Mock completion model
#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    /// The model name
    pub model: String,
}

impl<T> CompletionModel<T> {
    /// Create a new mock completion model
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_owned(),
        }
    }

    /// Get the current response counter value
    pub fn response_count(&self) -> usize {
        self.client.ext().response_counter.load(Ordering::SeqCst)
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: crate::http_client::HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into().as_str())
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        // Check for error mode
        match &self.client.ext().config.error_mode {
            ErrorMode::RequestError(msg) => {
                return Err(CompletionError::RequestError(
                    format!("[Mock Provider] {}", msg).into(),
                ));
            }
            ErrorMode::ResponseError(msg) => {
                return Err(CompletionError::ResponseError(format!(
                    "[Mock Provider] {}",
                    msg
                )));
            }
            ErrorMode::ProviderError(msg) => {
                return Err(CompletionError::ProviderError(format!(
                    "[Mock Provider] {}",
                    msg
                )));
            }
            ErrorMode::None => {}
        }

        // Increment and get response number
        let response_number = self
            .client
            .ext()
            .response_counter
            .fetch_add(1, Ordering::SeqCst);

        // Extract prompt text from the request
        let prompt_text = extract_prompt_text(&request);

        // Create deterministic seed from request content and counter
        let seed = create_seed(&prompt_text, response_number, &self.model);

        // Generate response content
        let content = format!("Response #{}: {}", response_number, prompt_text);

        // Build assistant content
        let mut assistant_contents = vec![AssistantContent::Text(Text {
            text: content.clone(),
        })];

        // Optionally add tool calls based on config and seeded "randomness"
        let has_tool_calls =
            self.client.ext().config.enable_tool_calls && should_include_tool_call(seed);

        if has_tool_calls {
            let tool_call = create_mock_tool_call(response_number, &prompt_text);
            assistant_contents.push(AssistantContent::ToolCall(tool_call));
        }

        let choice = OneOrMany::many(assistant_contents)
            .map_err(|_| CompletionError::ResponseError("No content provided".to_owned()))?;

        let usage = MockUsage::fixed();

        let raw_response = CompletionResponse {
            model: self.model.clone(),
            content,
            has_tool_calls,
            response_number,
            usage: usage.clone(),
        };

        Ok(completion::CompletionResponse {
            choice,
            usage: Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.input_tokens + usage.output_tokens,
            },
            raw_response,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        // Check for error mode
        match &self.client.ext().config.error_mode {
            ErrorMode::RequestError(msg) => {
                return Err(CompletionError::RequestError(
                    format!("[Mock Provider] {}", msg).into(),
                ));
            }
            ErrorMode::ResponseError(msg) => {
                return Err(CompletionError::ResponseError(format!(
                    "[Mock Provider] {}",
                    msg
                )));
            }
            ErrorMode::ProviderError(msg) => {
                return Err(CompletionError::ProviderError(format!(
                    "[Mock Provider] {}",
                    msg
                )));
            }
            ErrorMode::None => {}
        }

        // Increment and get response number
        let response_number = self
            .client
            .ext()
            .response_counter
            .fetch_add(1, Ordering::SeqCst);

        // Extract prompt text from the request
        let prompt_text = extract_prompt_text(&request);

        // Create deterministic seed
        let seed = create_seed(&prompt_text, response_number, &self.model);

        // Generate response content
        let content = format!("Response #{}: {}", response_number, prompt_text);

        // Determine if we should include tool calls
        let enable_tool_calls = self.client.ext().config.enable_tool_calls;
        let include_tool_calls = enable_tool_calls && should_include_tool_call(seed);

        // Create the streaming generator
        let stream: StreamingResult<StreamingCompletionResponse> = Box::pin(stream! {
            // Stream the content word by word
            let words: Vec<&str> = content.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                let text = if i == 0 {
                    word.to_string()
                } else {
                    format!(" {}", word)
                };
                yield Ok(RawStreamingChoice::Message(text));
            }

            // Optionally yield tool call
            if include_tool_calls {
                let tool_call = create_mock_streaming_tool_call(response_number, &prompt_text);
                yield Ok(RawStreamingChoice::ToolCall(tool_call));
            }

            // Yield final response
            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                response_number,
                usage: MockUsage::fixed(),
            }));
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}

// ========== Helper Functions ==========

/// Extract the prompt text from a completion request
fn extract_prompt_text(request: &CompletionRequest) -> String {
    // Get the last message from chat history (which is the prompt)
    if let Some(last_message) = request.chat_history.iter().last() {
        match last_message {
            crate::message::Message::User { content } => {
                // Extract text from user content
                content
                    .iter()
                    .filter_map(|c| match c {
                        UserContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            crate::message::Message::Assistant { content, .. } => {
                // Extract text from assistant content
                content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    } else {
        String::from("(empty prompt)")
    }
}

/// Create a deterministic seed from request content, response number, and model name
fn create_seed(prompt: &str, response_number: usize, model: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    response_number.hash(&mut hasher);
    model.hash(&mut hasher);
    hasher.finish()
}

/// Determine if we should include a tool call based on the seed
/// This provides deterministic "randomness" - same seed always gives same result
fn should_include_tool_call(seed: u64) -> bool {
    // Use the seed to determine if we include a tool call (50/50 based on last bit)
    seed % 2 == 0
}

/// Create a mock tool call for non-streaming responses
fn create_mock_tool_call(response_number: usize, prompt: &str) -> ToolCall {
    ToolCall::new(
        format!("mock_tool_call_{}", response_number),
        ToolFunction::new(
            format!("mock_tool_response_{}", response_number),
            serde_json::json!({
                "echo": prompt,
                "response_num": response_number
            }),
        ),
    )
}

/// Create a mock tool call for streaming responses
fn create_mock_streaming_tool_call(response_number: usize, prompt: &str) -> RawStreamingToolCall {
    RawStreamingToolCall::new(
        format!("mock_tool_call_{}", response_number),
        format!("mock_tool_response_{}", response_number),
        serde_json::json!({
            "echo": prompt,
            "response_num": response_number
        }),
    )
}

// ========== Tests ==========

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as CompletionModelTrait;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_basic_completion() {
        let client = Client::default();
        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Hello, world!")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let response = model.completion(request).await.unwrap();

        // Check that response contains the echoed prompt
        let text = response
            .choice
            .iter()
            .filter_map(|c| match c {
                AssistantContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        assert!(text.contains("Response #0"));
        assert!(text.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_response_counter_increments() {
        let client = Client::default();
        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Test")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        // First completion
        let response1 = model.completion(request.clone()).await.unwrap();
        assert_eq!(response1.raw_response.response_number, 0);

        // Second completion
        let response2 = model.completion(request.clone()).await.unwrap();
        assert_eq!(response2.raw_response.response_number, 1);

        // Third completion
        let response3 = model.completion(request).await.unwrap();
        assert_eq!(response3.raw_response.response_number, 2);
    }

    #[tokio::test]
    async fn test_tool_calls_enabled() {
        let client: Client = Client::builder()
            .api_key(Nothing)
            .enable_tool_calls(true)
            .build()
            .unwrap();

        let model = client.completion_model("test-model");

        // Run multiple completions to ensure at least one has tool calls
        // (since tool calls are deterministic based on seed)
        let mut found_tool_call = false;

        for i in 0..10 {
            let request = CompletionRequest {
                preamble: None,
                chat_history: OneOrMany::one(crate::message::Message::user(format!("Test {}", i))),
                documents: vec![],
                tools: vec![],
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
            };

            let response = model.completion(request).await.unwrap();

            if response.raw_response.has_tool_calls {
                found_tool_call = true;

                // Verify tool call structure
                let tool_calls: Vec<_> = response
                    .choice
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::ToolCall(tc) => Some(tc),
                        _ => None,
                    })
                    .collect();

                assert!(!tool_calls.is_empty());
                let tc = &tool_calls[0];
                assert!(tc.function.name.starts_with("mock_tool_response_"));
                break;
            }
        }

        assert!(
            found_tool_call,
            "Expected at least one response with tool calls"
        );
    }

    #[tokio::test]
    async fn test_tool_calls_disabled() {
        let client: Client = Client::builder()
            .api_key(Nothing)
            .enable_tool_calls(false)
            .build()
            .unwrap();

        let model = client.completion_model("test-model");

        for i in 0..5 {
            let request = CompletionRequest {
                preamble: None,
                chat_history: OneOrMany::one(crate::message::Message::user(format!("Test {}", i))),
                documents: vec![],
                tools: vec![],
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
            };

            let response = model.completion(request).await.unwrap();

            // Should never have tool calls when disabled
            assert!(!response.raw_response.has_tool_calls);
        }
    }

    #[tokio::test]
    async fn test_error_mode_request_error() {
        let client: Client = Client::builder()
            .api_key(Nothing)
            .error_mode(ErrorMode::RequestError("Test request error".to_string()))
            .build()
            .unwrap();

        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Test")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let result = model.completion(request).await;
        assert!(matches!(result, Err(CompletionError::RequestError(_))));
    }

    #[tokio::test]
    async fn test_error_mode_response_error() {
        let client: Client = Client::builder()
            .api_key(Nothing)
            .error_mode(ErrorMode::ResponseError("Test response error".to_string()))
            .build()
            .unwrap();

        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Test")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let result = model.completion(request).await;
        assert!(matches!(result, Err(CompletionError::ResponseError(_))));
    }

    #[tokio::test]
    async fn test_error_mode_provider_error() {
        let client: Client = Client::builder()
            .api_key(Nothing)
            .error_mode(ErrorMode::ProviderError("Test provider error".to_string()))
            .build()
            .unwrap();

        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Test")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let result = model.completion(request).await;
        assert!(matches!(result, Err(CompletionError::ProviderError(_))));
    }

    #[tokio::test]
    async fn test_streaming_completion() {
        let client = Client::default();
        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Hello streaming")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let mut stream = model.stream(request).await.unwrap();

        let mut text_chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(crate::streaming::StreamedAssistantContent::Text(text)) => {
                    text_chunks.push(text.text);
                }
                Ok(crate::streaming::StreamedAssistantContent::Final(_)) => {
                    // Final response received
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
                _ => {}
            }
        }

        let full_text = text_chunks.join("");
        assert!(full_text.contains("Response #0"));
        assert!(full_text.contains("Hello"));
        assert!(full_text.contains("streaming"));
    }

    #[tokio::test]
    async fn test_streaming_with_tool_calls() {
        let client: Client = Client::builder()
            .api_key(Nothing)
            .enable_tool_calls(true)
            .build()
            .unwrap();

        let model = client.completion_model("test-model");

        // Try multiple prompts to find one that generates a tool call
        let mut found_tool_call = false;

        for i in 0..10 {
            let request = CompletionRequest {
                preamble: None,
                chat_history: OneOrMany::one(crate::message::Message::user(format!(
                    "Streaming test {}",
                    i
                ))),
                documents: vec![],
                tools: vec![],
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
            };

            let mut stream = model.stream(request).await.unwrap();

            while let Some(chunk) = stream.next().await {
                if let Ok(crate::streaming::StreamedAssistantContent::ToolCall(tc)) = chunk {
                    found_tool_call = true;
                    assert!(tc.function.name.starts_with("mock_tool_response_"));
                    break;
                }
            }

            if found_tool_call {
                break;
            }
        }

        assert!(
            found_tool_call,
            "Expected at least one streaming response with tool calls"
        );
    }

    #[tokio::test]
    async fn test_fixed_token_usage() {
        let client = Client::default();
        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Test")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let response = model.completion(request).await.unwrap();

        // Check fixed token usage values
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 50);
        assert_eq!(response.usage.total_tokens, 150);
    }

    #[tokio::test]
    async fn test_deterministic_tool_calls() {
        // Same prompt with same settings should always produce same tool call behavior
        // We need fresh clients to reset the counter
        let results: Vec<bool> = futures::future::join_all((0..3).map(|_| async {
            let fresh_client: Client = Client::builder()
                .api_key(Nothing)
                .enable_tool_calls(true)
                .build()
                .unwrap();

            let model = fresh_client.completion_model("test-model");

            let request = CompletionRequest {
                preamble: None,
                chat_history: OneOrMany::one(crate::message::Message::user(
                    "Deterministic test prompt",
                )),
                documents: vec![],
                tools: vec![],
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
            };

            let response = model.completion(request).await.unwrap();
            response.raw_response.has_tool_calls
        }))
        .await;

        // All results should be the same
        assert!(
            results.iter().all(|&x| x == results[0]),
            "Tool call behavior should be deterministic"
        );
    }

    #[tokio::test]
    async fn test_counter_persists_across_clones() {
        let client = Client::default();
        let model1 = client.completion_model("test-model");
        let model2 = client.completion_model("test-model-2");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Test")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        // Call on model1
        let r1 = model1.completion(request.clone()).await.unwrap();
        assert_eq!(r1.raw_response.response_number, 0);

        // Call on model2 (should share counter with model1)
        let r2 = model2.completion(request.clone()).await.unwrap();
        assert_eq!(r2.raw_response.response_number, 1);

        // Call on model1 again
        let r3 = model1.completion(request).await.unwrap();
        assert_eq!(r3.raw_response.response_number, 2);
    }

    #[test]
    fn test_client_default() {
        let client = Client::default();
        // Should create successfully
        let _ = client.completion_model("test");
    }

    #[tokio::test]
    async fn test_streaming_error_mode() {
        let client: Client = Client::builder()
            .api_key(Nothing)
            .error_mode(ErrorMode::ProviderError("Stream error".to_string()))
            .build()
            .unwrap();

        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(crate::message::Message::user("Test")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        };

        let result = model.stream(request).await;
        assert!(matches!(result, Err(CompletionError::ProviderError(_))));
    }

    // ==================== Agent Integration Tests ====================

    mod agent_tests {
        use super::*;
        use crate::agent::AgentBuilder;
        use crate::completion::{Chat, Prompt};

        #[tokio::test]
        async fn test_agent_basic_prompt() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model).build();

            let response = agent.prompt("Hello from agent!").await.unwrap();

            // The mock provider echoes back the prompt with response number
            assert!(response.contains("Response #0"));
            assert!(response.contains("Hello from agent!"));
        }

        #[tokio::test]
        async fn test_agent_with_preamble() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You are a helpful assistant.")
                .build();

            let response = agent.prompt("What can you do?").await.unwrap();

            // Verify agent processes the response correctly
            assert!(response.contains("Response #0"));
            assert!(response.contains("What can you do?"));
        }

        #[tokio::test]
        async fn test_agent_chat_with_history() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You are a helpful assistant.")
                .build();

            let chat_history = vec![
                crate::message::Message::user("Hello!"),
                crate::message::Message::assistant("Hi there! How can I help you?"),
            ];

            let response = agent.chat("Tell me a joke", chat_history).await.unwrap();

            // The response counter should be 0 for the first completion
            assert!(response.contains("Response #0"));
            assert!(response.contains("Tell me a joke"));
        }

        #[tokio::test]
        async fn test_agent_multi_turn_conversation() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You are a helpful assistant.")
                .build();

            // First turn
            let response1 = agent.prompt("First message").await.unwrap();
            assert!(response1.contains("Response #0"));

            // Second turn
            let response2 = agent.prompt("Second message").await.unwrap();
            assert!(response2.contains("Response #1"));

            // Third turn
            let response3 = agent.prompt("Third message").await.unwrap();
            assert!(response3.contains("Response #2"));
        }

        #[tokio::test]
        async fn test_agent_with_temperature() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You are creative.")
                .temperature(0.9)
                .build();

            let response = agent.prompt("Be creative!").await.unwrap();
            assert!(response.contains("Response #0"));
        }

        #[tokio::test]
        async fn test_agent_error_handling() {
            let client: Client = Client::builder()
                .api_key(Nothing)
                .error_mode(ErrorMode::ProviderError("Agent error test".to_string()))
                .build()
                .unwrap();

            let model = client.completion_model("mock-model");
            let agent = AgentBuilder::new(model).build();

            let result = agent.prompt("This should fail").await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_agent_streaming_prompt() {
            use crate::agent::MultiTurnStreamItem;
            use crate::streaming::StreamingPrompt;

            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You are a helpful assistant.")
                .build();

            // stream_prompt returns StreamingPromptRequest which implements IntoFuture
            // We need to await it to get the actual stream
            let mut stream = agent.stream_prompt("Stream this message").await;

            let mut text_chunks = Vec::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(MultiTurnStreamItem::StreamAssistantItem(
                        crate::streaming::StreamedAssistantContent::Text(text),
                    )) => {
                        text_chunks.push(text.text);
                    }
                    Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                        // Final response received
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                    _ => {}
                }
            }

            let full_text = text_chunks.join("");
            assert!(full_text.contains("Response #0"));
            assert!(full_text.contains("Stream"));
        }

        #[tokio::test]
        async fn test_agent_with_context_documents() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("Answer based on the context provided.")
                .context("The capital of France is Paris.")
                .context("The capital of Germany is Berlin.")
                .build();

            let response = agent.prompt("What is the capital of France?").await.unwrap();
            assert!(response.contains("Response #0"));
        }

        #[tokio::test]
        async fn test_multiple_agents_same_client() {
            let client = Client::default();

            let model1 = client.completion_model("model-1");
            let model2 = client.completion_model("model-2");

            let agent1 = AgentBuilder::new(model1)
                .preamble("You are agent 1.")
                .build();

            let agent2 = AgentBuilder::new(model2)
                .preamble("You are agent 2.")
                .build();

            // Both agents share the same client, so counter increments across both
            let response1 = agent1.prompt("Agent 1 speaking").await.unwrap();
            assert!(response1.contains("Response #0"));

            let response2 = agent2.prompt("Agent 2 speaking").await.unwrap();
            assert!(response2.contains("Response #1"));

            let response3 = agent1.prompt("Agent 1 again").await.unwrap();
            assert!(response3.contains("Response #2"));
        }
    }

    // ==================== Extractor Tests ====================

    mod extractor_tests {
        use super::*;
        use crate::extractor::ExtractorBuilder;
        use schemars::JsonSchema;
        use serde::{Deserialize, Serialize};

        // Note: The mock provider doesn't actually call the "submit" tool,
        // so these tests verify the extractor can be created and used with
        // the mock provider, even if extraction fails due to no tool calls.

        #[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
        struct Person {
            name: Option<String>,
            age: Option<u8>,
        }

        #[tokio::test]
        async fn test_extractor_creation() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            // Verify extractor can be created with mock model
            let _extractor = ExtractorBuilder::<_, Person>::new(model).build();
        }

        #[tokio::test]
        async fn test_extractor_with_retries() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let extractor = ExtractorBuilder::<_, Person>::new(model)
                .retries(2)
                .build();

            // The mock provider doesn't call the submit tool, so extraction will fail
            // But this verifies the retry logic works without crashing
            let result = extractor.extract("John Doe is 30 years old").await;

            // Expected to fail since mock doesn't produce tool calls for submit
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_extractor_error_mode() {
            let client: Client = Client::builder()
                .api_key(Nothing)
                .error_mode(ErrorMode::ProviderError("Extractor error".to_string()))
                .build()
                .unwrap();

            let model = client.completion_model("mock-model");
            let extractor = ExtractorBuilder::<_, Person>::new(model).build();

            let result = extractor.extract("Test input").await;
            assert!(result.is_err());
        }
    }

    // ==================== Concurrency Tests ====================

    mod concurrency_tests {
        use super::*;
        use crate::agent::AgentBuilder;
        use crate::completion::Prompt;
        use std::sync::Arc;

        #[tokio::test]
        async fn test_concurrent_completions() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            // Run 10 completions concurrently
            let handles: Vec<_> = (0..10)
                .map(|i| {
                    let model = model.clone();
                    tokio::spawn(async move {
                        let request = CompletionRequest {
                            preamble: None,
                            chat_history: OneOrMany::one(crate::message::Message::user(format!(
                                "Concurrent request {}",
                                i
                            ))),
                            documents: vec![],
                            tools: vec![],
                            temperature: None,
                            max_tokens: None,
                            tool_choice: None,
                            additional_params: None,
                        };

                        model.completion(request).await.unwrap()
                    })
                })
                .collect();

            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.unwrap())
                .collect();

            // All completions should succeed
            assert_eq!(results.len(), 10);

            // Response numbers should be unique (0-9)
            let mut response_numbers: Vec<_> = results
                .iter()
                .map(|r| r.raw_response.response_number)
                .collect();
            response_numbers.sort();
            assert_eq!(response_numbers, (0..10).collect::<Vec<_>>());
        }

        #[tokio::test]
        async fn test_concurrent_agent_prompts() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = Arc::new(AgentBuilder::new(model).build());

            // Run 5 agent prompts concurrently
            let handles: Vec<_> = (0..5)
                .map(|i| {
                    let agent = Arc::clone(&agent);
                    tokio::spawn(async move {
                        agent.prompt(format!("Concurrent prompt {}", i)).await
                    })
                })
                .collect();

            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.unwrap().unwrap())
                .collect();

            // All prompts should succeed
            assert_eq!(results.len(), 5);

            // All responses should contain "Response #"
            for response in &results {
                assert!(response.contains("Response #"));
            }
        }

        #[tokio::test]
        async fn test_concurrent_streaming() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            // Run 5 streaming completions concurrently
            let handles: Vec<_> = (0..5)
                .map(|i| {
                    let model = model.clone();
                    tokio::spawn(async move {
                        let request = CompletionRequest {
                            preamble: None,
                            chat_history: OneOrMany::one(crate::message::Message::user(format!(
                                "Stream request {}",
                                i
                            ))),
                            documents: vec![],
                            tools: vec![],
                            temperature: None,
                            max_tokens: None,
                            tool_choice: None,
                            additional_params: None,
                        };

                        let mut stream = model.stream(request).await.unwrap();

                        let mut text_chunks = Vec::new();
                        while let Some(chunk) = stream.next().await {
                            if let Ok(crate::streaming::StreamedAssistantContent::Text(text)) =
                                chunk
                            {
                                text_chunks.push(text.text);
                            }
                        }

                        text_chunks.join("")
                    })
                })
                .collect();

            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.unwrap())
                .collect();

            // All streams should complete successfully
            assert_eq!(results.len(), 5);

            // All responses should contain "Response #"
            for response in &results {
                assert!(response.contains("Response #"));
            }
        }

        #[tokio::test]
        async fn test_thread_safety_counter() {
            let client = Client::default();

            // Create multiple models from the same client
            let models: Vec<_> = (0..5)
                .map(|i| client.completion_model(format!("model-{}", i)))
                .collect();

            // Run completions from different models concurrently
            let handles: Vec<_> = models
                .into_iter()
                .enumerate()
                .map(|(i, model)| {
                    tokio::spawn(async move {
                        let request = CompletionRequest {
                            preamble: None,
                            chat_history: OneOrMany::one(crate::message::Message::user(format!(
                                "Model {} request",
                                i
                            ))),
                            documents: vec![],
                            tools: vec![],
                            temperature: None,
                            max_tokens: None,
                            tool_choice: None,
                            additional_params: None,
                        };

                        model.completion(request).await.unwrap()
                    })
                })
                .collect();

            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.unwrap())
                .collect();

            // All completions should succeed
            assert_eq!(results.len(), 5);

            // Response numbers should be unique (counter is thread-safe)
            let mut response_numbers: Vec<_> = results
                .iter()
                .map(|r| r.raw_response.response_number)
                .collect();
            response_numbers.sort();

            // Check all numbers are unique
            let unique_count = response_numbers
                .windows(2)
                .filter(|w| w[0] != w[1])
                .count()
                + 1;
            assert_eq!(unique_count, 5);
        }

        #[tokio::test]
        async fn test_concurrent_different_clients() {
            // Create multiple independent clients
            let clients: Vec<_> = (0..3).map(|_| Client::default()).collect();

            let handles: Vec<_> = clients
                .into_iter()
                .enumerate()
                .map(|(i, client)| {
                    tokio::spawn(async move {
                        let model = client.completion_model("mock-model");
                        let agent = AgentBuilder::new(model).build();

                        // Each client has its own counter, so all should start at 0
                        let response = agent.prompt(format!("Client {} prompt", i)).await.unwrap();
                        response
                    })
                })
                .collect();

            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.unwrap())
                .collect();

            // All should succeed
            assert_eq!(results.len(), 3);

            // Each client has its own counter, so all should contain "Response #0"
            for response in &results {
                assert!(response.contains("Response #0"));
            }
        }
    }

    // ==================== Multi-turn Conversation Tests ====================

    mod multi_turn_tests {
        use super::*;
        use crate::agent::AgentBuilder;
        use crate::completion::{Chat, Prompt};

        #[tokio::test]
        async fn test_chat_history_accumulation() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You remember everything.")
                .build();

            let mut history = vec![];

            // First exchange
            let response1 = agent.chat("My name is Alice", history.clone()).await.unwrap();
            history.push(crate::message::Message::user("My name is Alice"));
            history.push(crate::message::Message::assistant(&response1));

            // Second exchange
            let response2 = agent.chat("What is my name?", history.clone()).await.unwrap();
            history.push(crate::message::Message::user("What is my name?"));
            history.push(crate::message::Message::assistant(&response2));

            // Third exchange
            let response3 = agent.chat("Thanks!", history.clone()).await.unwrap();

            // Verify responses are sequential
            assert!(response1.contains("Response #0"));
            assert!(response2.contains("Response #1"));
            assert!(response3.contains("Response #2"));
        }

        #[tokio::test]
        async fn test_complex_chat_history() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You are a helpful assistant.")
                .build();

            // Build a complex chat history
            let history = vec![
                crate::message::Message::user("Hello!"),
                crate::message::Message::assistant("Hi there! How can I help?"),
                crate::message::Message::user("What's the weather like?"),
                crate::message::Message::assistant("I don't have access to weather data."),
                crate::message::Message::user("That's okay."),
                crate::message::Message::assistant("Is there anything else I can help with?"),
            ];

            let response = agent.chat("Yes, tell me a joke", history).await.unwrap();

            // Response should be successful
            assert!(response.contains("Response #0"));
            assert!(response.contains("tell me a joke"));
        }

        #[tokio::test]
        async fn test_empty_chat_history() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model).build();

            let response = agent.chat("First message", vec![]).await.unwrap();

            assert!(response.contains("Response #0"));
            assert!(response.contains("First message"));
        }

        #[tokio::test]
        async fn test_interleaved_prompt_and_chat() {
            let client = Client::default();
            let model = client.completion_model("mock-model");

            let agent = AgentBuilder::new(model)
                .preamble("You are helpful.")
                .build();

            // Use prompt
            let r1 = agent.prompt("Prompt 1").await.unwrap();
            assert!(r1.contains("Response #0"));

            // Use chat with history
            let history = vec![crate::message::Message::user("Previous message")];
            let r2 = agent.chat("Chat message", history).await.unwrap();
            assert!(r2.contains("Response #1"));

            // Use prompt again
            let r3 = agent.prompt("Prompt 2").await.unwrap();
            assert!(r3.contains("Response #2"));
        }
    }
}
