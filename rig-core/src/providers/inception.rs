use super::openai::send_compatible_streaming_request;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    extractor::ExtractorBuilder,
    json_utils::{self, merge_inplace},
    message::{self, MessageError},
    streaming::{StreamingCompletionModel, StreamingResult},
    OneOrMany,
};

const INCEPTION_API_BASE_URL: &str = "https://api.inceptionlabs.ai/v1";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str) -> Self {
        Self {
            base_url: INCEPTION_API_BASE_URL.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Content-Type",
                        "application/json"
                            .parse()
                            .expect("Content-Type should parse"),
                    );
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Authorization should parse"),
                    );
                    headers
                })
                .build()
                .expect("Inception reqwest client should build"),
        }
    }

    pub fn from_env() -> Self {
        let api_key = std::env::var("INCEPTION_API_KEY").expect("INCEPTION_API_KEY not set");
        Client::new(&api_key)
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
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

// ================================================================
// Inception Completion API
// ================================================================
/// `mercury-coder-small` completion model
pub const MERCURY_CODER_SMALL: &str = "mercury-coder-small";

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {}\nCompletion tokens: {}\nTotal tokens: {}",
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl TryFrom<message::Message> for Message {
    type Error = MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => Message {
                role: Role::User,
                content: match content.first() {
                    message::UserContent::Text(message::Text { text }) => text.clone(),
                    _ => {
                        return Err(MessageError::ConversionError(
                            "User message content must be a text message".to_string(),
                        ))
                    }
                },
            },
            message::Message::Assistant { content } => Message {
                role: Role::Assistant,
                content: match content.first() {
                    message::AssistantContent::Text(message::Text { text }) => text.clone(),
                    _ => {
                        return Err(MessageError::ConversionError(
                            "Assistant message content must be a text message".to_string(),
                        ))
                    }
                },
            },
        })
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message.role {
            Role::Assistant => {
                let content = completion::AssistantContent::text(&choice.message.content);

                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message".into(),
            )),
        }?;

        let choice = OneOrMany::one(content);

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

const MAX_TOKENS: u64 = 8192;

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model (e.g.: deepseek-ai/DeepSeek-R1)
    pub model: String,
}

impl CompletionModel {
    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        let mut messages = vec![];

        if let Some(preamble) = completion_request.preamble.clone() {
            messages.push(Message {
                role: Role::System,
                content: preamble.clone(),
            });
        }

        let prompt_message: Message = completion_request
            .prompt_with_context()
            .try_into()
            .map_err(|e: MessageError| CompletionError::RequestError(e.into()))?;

        let chat_history = completion_request
            .chat_history
            .into_iter()
            .map(|message| {
                message
                    .try_into()
                    .map_err(|e: MessageError| CompletionError::RequestError(e.into()))
            })
            .collect::<Result<Vec<Message>, _>>()?;

        messages.extend(chat_history);
        messages.push(prompt_message);

        let max_tokens = completion_request.max_tokens.unwrap_or(MAX_TOKENS);

        let request = json!({
            "model": self.model,
            "messages": messages,
            // The beta API reference doesn't mention temperature but it doesn't hurt to include it
            "temperature": completion_request.temperature,
            "max_tokens": max_tokens,
        });

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
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
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Inception completion token usage: {}",
                        response.usage
                    );

                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let mut request = self.create_completion_request(completion_request)?;

        merge_inplace(&mut request, json!({"stream": true}));

        let builder = self.client.post("/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}
