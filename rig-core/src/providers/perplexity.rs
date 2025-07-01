//! Perplexity API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::perplexity;
//!
//! let client = perplexity::Client::new("YOUR_API_KEY");
//!
//! let llama_3_1_sonar_small_online = client.completion_model(perplexity::LLAMA_3_1_SONAR_SMALL_ONLINE);
//! ```

use crate::{
    OneOrMany,
    agent::AgentBuilder,
    completion::{self, CompletionError, MessageError, message},
    extractor::ExtractorBuilder,
    impl_conversion_traits, json_utils,
};

use crate::client::{CompletionClient, ProviderClient};
use crate::completion::CompletionRequest;
use crate::json_utils::merge;
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ================================================================
// Main Cohere Client
// ================================================================
const PERPLEXITY_API_BASE_URL: &str = "https://api.perplexity.ai";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    api_key: String,
    http_client: reqwest::Client,
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
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, PERPLEXITY_API_BASE_URL)
    }

    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            http_client: reqwest::Client::builder()
                .build()
                .expect("Perplexity reqwest client should build"),
        }
    }

    /// Use your own `reqwest::Client`.
    /// The required headers will be automatically attached upon trying to make a request.
    pub fn with_custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;

        self
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url).bearer_auth(&self.api_key)
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

impl ProviderClient for Client {
    /// Create a new Perplexity client from the `PERPLEXITY_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("PERPLEXITY_API_KEY").expect("PERPLEXITY_API_KEY not set");
        Self::new(&api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsEmbeddings,
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

// ================================================================
// Perplexity Completion API
// ================================================================
/// `sonar-pro` completion model
pub const SONAR_PRO: &str = "sonar-pro";
/// `sonar` completion model
pub const SONAR: &str = "sonar";

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub object: String,
    pub created: u64,
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Deserialize, Debug)]
pub struct Delta {
    pub role: Role,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub index: usize,
    pub finish_reason: String,
    pub message: Message,
    pub delta: Delta,
}

#[derive(Deserialize, Debug)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {}\nCompletion tokens: {} Total tokens: {}",
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        match &choice.message {
            Message {
                role: Role::Assistant,
                content,
            } => Ok(completion::CompletionResponse {
                choice: OneOrMany::one(content.clone().into()),
                raw_response: response,
            }),
            _ => Err(CompletionError::ResponseError(
                "Response contained no assistant message".to_owned(),
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
            model: model.to_string(),
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<Message> =
            completion_request
                .preamble
                .map_or_else(Vec::new, |preamble| {
                    vec![Message {
                        role: Role::System,
                        content: preamble,
                    }]
                });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Message>, _>>()?,
        );

        // Compose request
        let request = json!({
            "model": self.model,
            "messages": full_history,
            "temperature": completion_request.temperature,
        });

        let request = if let Some(ref params) = completion_request.additional_params {
            json_utils::merge(request, params.clone())
        } else {
            request
        };

        Ok(request)
    }
}

impl TryFrom<message::Message> for Message {
    type Error = MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => {
                let collapsed_content = content
                    .into_iter()
                    .map(|content| match content {
                        message::UserContent::Text(message::Text { text }) => Ok(text),
                        _ => Err(MessageError::ConversionError(
                            "Only text content is supported by Perplexity".to_owned(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n");

                Message {
                    role: Role::User,
                    content: collapsed_content,
                }
            }

            message::Message::Assistant { content, .. } => {
                let collapsed_content = content
                    .into_iter()
                    .map(|content| {
                        Ok(match content {
                            message::AssistantContent::Text(message::Text { text }) => text,
                            _ => return Err(MessageError::ConversionError(
                                "Only text assistant message content is supported by Perplexity"
                                    .to_owned(),
                            )),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n");

                Message {
                    role: Role::Assistant,
                    content: collapsed_content,
                }
            }
        })
    }
}

impl From<Message> for message::Message {
    fn from(message: Message) -> Self {
        match message.role {
            Role::User => message::Message::user(message.content),
            Role::Assistant => message::Message::assistant(message.content),

            // System messages get coerced into user messages for ease of error handling.
            // They should be handled on the outside of `Message` conversions via the preamble.
            Role::System => message::Message::user(message.content),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(completion) => {
                    tracing::info!(target: "rig",
                        "Perplexity completion token usage: {}",
                        completion.usage
                    );
                    Ok(completion.try_into()?)
                }
                ApiResponse::Err(error) => Err(CompletionError::ProviderError(error.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(request, json!({"stream": true}));

        let builder = self.client.post("/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_message() {
        let json_data = r#"
        {
            "role": "user",
            "content": "Hello, how can I help you?"
        }
        "#;

        let message: Message = serde_json::from_str(json_data).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content, "Hello, how can I help you?");
    }

    #[test]
    fn test_serialize_message() {
        let message = Message {
            role: Role::Assistant,
            content: "I am here to assist you.".to_string(),
        };

        let json_data = serde_json::to_string(&message).unwrap();
        let expected_json = r#"{"role":"assistant","content":"I am here to assist you."}"#;
        assert_eq!(json_data, expected_json);
    }

    #[test]
    fn test_message_to_message_conversion() {
        let user_message = message::Message::user("User message");
        let assistant_message = message::Message::assistant("Assistant message");

        let converted_user_message: Message = user_message.clone().try_into().unwrap();
        let converted_assistant_message: Message = assistant_message.clone().try_into().unwrap();

        assert_eq!(converted_user_message.role, Role::User);
        assert_eq!(converted_user_message.content, "User message");

        assert_eq!(converted_assistant_message.role, Role::Assistant);
        assert_eq!(converted_assistant_message.content, "Assistant message");

        let back_to_user_message: message::Message = converted_user_message.into();
        let back_to_assistant_message: message::Message = converted_assistant_message.into();

        assert_eq!(user_message, back_to_user_message);
        assert_eq!(assistant_message, back_to_assistant_message);
    }
}
