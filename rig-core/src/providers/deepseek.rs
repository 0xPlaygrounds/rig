//! DeepSeek API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::deepseek;
//!
//! let client = deepseek::Client::new("DEEPSEEK_API_KEY");
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_CHAT);
//! ```

use futures::StreamExt;
use std::collections::HashMap;

use crate::client::{
    ClientBuilderError, CompletionClient, ProviderClient, VerifyClient, VerifyError,
};
use crate::completion::GetTokenUsage;
use crate::json_utils::merge;
use crate::message::Document;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    impl_conversion_traits, json_utils, message,
};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::openai::StreamingToolCall;

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: DEEPSEEK_API_BASE_URL,
            http_client: None,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    pub fn build(self) -> Result<Client, ClientBuilderError> {
        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            reqwest::Client::builder().build()?
        };

        Ok(Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            http_client,
        })
    }
}

#[derive(Clone)]
pub struct Client {
    pub base_url: String,
    api_key: String,
    http_client: HttpClient,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl Client {
    /// Create a new DeepSeek client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::deepseek::{ClientBuilder, self};
    ///
    /// // Initialize the DeepSeek client
    /// let deepseek = Client::builder("your-deepseek-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_> {
        ClientBuilder::new(api_key)
    }

    /// Create a new DeepSeek client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("DeepSeek client should build")
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url).bearer_auth(&self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.get(url).bearer_auth(&self.api_key)
    }
}

impl ProviderClient for Client {
    // If you prefer the environment variable approach:
    fn from_env() -> Self {
        let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set");
        Self::new(&api_key)
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Self::new(&api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    /// Creates a DeepSeek completion model with the given `model_name`.
    fn completion_model(&self, model_name: &str) -> CompletionModel {
        CompletionModel {
            client: self.clone(),
            model: model_name.to_string(),
        }
    }
}

impl VerifyClient for Client {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let response = self.get("/user/balance").send().await?;
        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
            | reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                Err(VerifyError::ProviderError(response.text().await?))
            }
            _ => {
                response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(
    AsEmbeddings,
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client
);

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

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

/// The response shape from the DeepSeek API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    // We'll match the JSON:
    pub choices: Vec<Choice>,
    pub usage: Usage,
    // you may want other fields
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub prompt_cache_hit_tokens: u32,
    pub prompt_cache_miss_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl Usage {
    fn new() -> Self {
        Self {
            completion_tokens: 0,
            prompt_tokens: 0,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: 0,
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            name: None,
        }
    }
}

impl From<message::ToolResult> for Message {
    fn from(tool_result: message::ToolResult) -> Self {
        let content = match tool_result.content.first() {
            message::ToolResultContent::Text(text) => text.text,
            message::ToolResultContent::Image(_) => String::from("[Image]"),
        };

        Message::ToolResult {
            tool_call_id: tool_result.id,
            content,
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            id: tool_call.id,
            // TODO: update index when we have it
            index: 0,
            r#type: ToolType::Function,
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                // extract tool results
                let mut messages = vec![];

                let tool_results = content
                    .clone()
                    .into_iter()
                    .filter_map(|content| match content {
                        message::UserContent::ToolResult(tool_result) => {
                            Some(Message::from(tool_result))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                messages.extend(tool_results);

                // extract text results
                let text_messages = content
                    .into_iter()
                    .filter_map(|content| match content {
                        message::UserContent::Text(text) => Some(Message::User {
                            content: text.text,
                            name: None,
                        }),
                        message::UserContent::Document(Document { data, .. }) => {
                            Some(Message::User {
                                content: data,
                                name: None,
                            })
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                messages.extend(text_messages);

                Ok(messages)
            }
            message::Message::Assistant { content, .. } => {
                let mut messages: Vec<Message> = vec![];

                // extract tool calls
                let tool_calls = content
                    .clone()
                    .into_iter()
                    .filter_map(|content| match content {
                        message::AssistantContent::ToolCall(tool_call) => {
                            Some(ToolCall::from(tool_call))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                // if we have tool calls, we add a new Assistant message with them
                if !tool_calls.is_empty() {
                    messages.push(Message::Assistant {
                        content: "".to_string(),
                        name: None,
                        tool_calls,
                    });
                }

                // extract text
                let text_content = content
                    .into_iter()
                    .filter_map(|content| match content {
                        message::AssistantContent::Text(text) => Some(Message::Assistant {
                            content: text.text,
                            name: None,
                            tool_calls: vec![],
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                messages.extend(text_content);

                Ok(messages)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    pub index: usize,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;
        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = if content.trim().is_empty() {
                    vec![]
                } else {
                    vec![completion::AssistantContent::text(content)]
                };

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = completion::Usage {
            input_tokens: response.usage.prompt_tokens as u64,
            output_tokens: response.usage.completion_tokens as u64,
            total_tokens: response.usage.total_tokens as u64,
        };

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

/// The struct implementing the `CompletionModel` trait
#[derive(Clone)]
pub struct CompletionModel {
    pub client: Client,
    pub model: String,
}

impl CompletionModel {
    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        // Build up the order of messages (context, chat_history, prompt)
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
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        completion::CompletionResponse<CompletionResponse>,
        crate::completion::CompletionError,
    > {
        let request = self.create_completion_request(completion_request)?;

        tracing::debug!("DeepSeek completion request: {request:?}");

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "DeepSeek completion: {}", t);

            match serde_json::from_str::<ApiResponse<CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self.client.post("/chat/completions").json(&request);
        send_compatible_streaming_request(builder).await
    }
}

#[derive(Deserialize, Debug)]
pub struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
    reasoning_content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.completion_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;

        Some(usage)
    }
}

pub async fn send_compatible_streaming_request(
    request_builder: reqwest::RequestBuilder,
) -> Result<
    crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
    CompletionError,
> {
    let response = request_builder.send().await?;

    if !response.status().is_success() {
        return Err(CompletionError::ProviderError(format!(
            "{}: {}",
            response.status(),
            response.text().await?
        )));
    }

    // Handle OpenAI Compatible SSE chunks
    let inner = Box::pin(async_stream::stream! {
        let mut stream = response.bytes_stream();

        let mut final_usage = Usage::new();
        let mut partial_data = None;
        let mut calls: HashMap<usize, (String, String, String)> = HashMap::new();

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
                let mut line = line.to_string();

                // If there was a remaining part, concat with current line
                if partial_data.is_some() {
                    line = format!("{}{}", partial_data.unwrap(), line);
                    partial_data = None;
                }
                // Otherwise full data line
                else {
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };

                    let data = data.trim_start();

                    // Partial data, split somewhere in the middle
                    if !line.ends_with("}") {
                        partial_data = Some(data.to_string());
                    } else {
                        line = data.to_string();
                    }
                }

                let data = serde_json::from_str::<StreamingCompletionChunk>(&line);

                let Ok(data) = data else {
                    let err = data.unwrap_err();
                    tracing::debug!("Couldn't serialize data as StreamingCompletionChunk: {:?}", err);
                    continue;
                };


                if let Some(choice) = data.choices.first() {
                    let delta = &choice.delta;


                            if !delta.tool_calls.is_empty() {
                                for tool_call in &delta.tool_calls {
                                    let function = tool_call.function.clone();
                                    // Start of tool call
                                    // name: Some(String)
                                    // arguments: None
                                    if function.name.is_some() && function.arguments.is_empty() {
                                        let id = tool_call.id.clone().unwrap_or("".to_string());

                                        calls.insert(tool_call.index, (id, function.name.clone().unwrap(), "".to_string()));
                                    }
                                    // Part of tool call
                                    // name: None or Empty String
                                    // arguments: Some(String)
                                    else if function.name.clone().is_none_or(|s| s.is_empty()) && !function.arguments.is_empty() {
                                        let Some((id, name, arguments)) = calls.get(&tool_call.index) else {
                                            tracing::debug!("Partial tool call received but tool call was never started.");
                                            continue;
                                        };

                                        let new_arguments = &function.arguments;
                                        let arguments = format!("{arguments}{new_arguments}");

                                        calls.insert(tool_call.index, (id.clone(), name.clone(), arguments));
                                    }
                                    // Entire tool call
                                    else {
                                        let id = tool_call.id.clone().unwrap_or("".to_string());
                                        let name = function.name.expect("function name should be present for complete tool call");
                                        let arguments = function.arguments;
                                        let Ok(arguments) = serde_json::from_str(&arguments) else {
                                            tracing::debug!("Couldn't serialize '{}' as a json value", arguments);
                                            continue;
                                        };

                                        yield Ok(crate::streaming::RawStreamingChoice::ToolCall {id, name, arguments, call_id: None })
                                    }
                                }
                            }

                            if let Some(content) = &delta.reasoning_content {
                                yield Ok(crate::streaming::RawStreamingChoice::Reasoning { reasoning: content.to_string(), id: None})
                            }

                            if let Some(content) = &delta.content {
                                yield Ok(crate::streaming::RawStreamingChoice::Message(content.clone()))
                            }

                }


                if let Some(usage) = data.usage {
                    final_usage = usage.clone();
                }
            }
        }

        for (_, (id, name, arguments)) in calls {
            let Ok(arguments) = serde_json::from_str(&arguments) else {
                continue;
            };

            yield Ok(crate::streaming::RawStreamingChoice::ToolCall {id, name, arguments, call_id: None });
        }

        yield Ok(crate::streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage.clone()
        }))
    });

    Ok(crate::streaming::StreamingCompletionResponse::stream(inner))
}

// ================================================================
// DeepSeek Completion API
// ================================================================

/// `deepseek-chat` completion model
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
/// `deepseek-reasoner` completion model
pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";

// Tests
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_deserialize_vec_choice() {
        let data = r#"[{
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message":{"role":"assistant","content":"Hello, world!"}
            }]"#;

        let choices: Vec<Choice> = serde_json::from_str(data).unwrap();
        assert_eq!(choices.len(), 1);
        match &choices.first().unwrap().message {
            Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_deserialize_deepseek_response() {
        let data = r#"{
            "choices":[{
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
                "message":{"role":"assistant","content":"Hello, world!"}
            }],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "total_tokens": 0
            }
        }"#;

        let jd = &mut serde_json::Deserializer::from_str(data);
        let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);
        match result {
            Ok(response) => match &response.choices.first().unwrap().message {
                Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
                _ => panic!("Expected assistant message"),
            },
            Err(err) => {
                panic!("Deserialization error at {}: {}", err.path(), err);
            }
        }
    }

    #[test]
    fn test_deserialize_example_response() {
        let data = r#"
        {
            "id": "e45f6c68-9d9e-43de-beb4-4f402b850feb",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Why don’t skeletons fight each other?  \nBecause they don’t have the guts! 😄"
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 32,
                "total_tokens": 45,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                },
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 13
            },
            "system_fingerprint": "fp_4b6881f2c5"
        }
        "#;
        let jd = &mut serde_json::Deserializer::from_str(data);
        let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);

        match result {
            Ok(response) => match &response.choices.first().unwrap().message {
                Message::Assistant { content, .. } => assert_eq!(
                    content,
                    "Why don’t skeletons fight each other?  \nBecause they don’t have the guts! 😄"
                ),
                _ => panic!("Expected assistant message"),
            },
            Err(err) => {
                panic!("Deserialization error at {}: {}", err.path(), err);
            }
        }
    }

    #[test]
    fn test_serialize_deserialize_tool_call_message() {
        let tool_call_choice_json = r#"
            {
              "finish_reason": "tool_calls",
              "index": 0,
              "logprobs": null,
              "message": {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                  {
                    "function": {
                      "arguments": "{\"x\":2,\"y\":5}",
                      "name": "subtract"
                    },
                    "id": "call_0_2b4a85ee-b04a-40ad-a16b-a405caf6e65b",
                    "index": 0,
                    "type": "function"
                  }
                ]
              }
            }
        "#;

        let choice: Choice = serde_json::from_str(tool_call_choice_json).unwrap();

        let expected_choice: Choice = Choice {
            finish_reason: "tool_calls".to_string(),
            index: 0,
            logprobs: None,
            message: Message::Assistant {
                content: "".to_string(),
                name: None,
                tool_calls: vec![ToolCall {
                    id: "call_0_2b4a85ee-b04a-40ad-a16b-a405caf6e65b".to_string(),
                    function: Function {
                        name: "subtract".to_string(),
                        arguments: serde_json::from_str(r#"{"x":2,"y":5}"#).unwrap(),
                    },
                    index: 0,
                    r#type: ToolType::Function,
                }],
            },
        };

        assert_eq!(choice, expected_choice);
    }
}
