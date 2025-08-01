use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{convert::TryFrom, str::FromStr};

use crate::{
    Embed, OneOrMany,
    client::{ClientBuilderError, CompletionClient, EmbeddingsClient, ProviderClient},
    completion::{self, CompletionError, CompletionRequest, ToolDefinition},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    impl_conversion_traits, json_utils,
    message::Text,
    streaming::{self, RawStreamingChoice},
};

const FOUNDRY_API_BASE_URL: &str = "http://localhost:42069";

pub struct ClientBuilder<'a> {
    base_url: &'a str,
    http_client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            base_url: FOUNDRY_API_BASE_URL,
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
            http_client,
        })
    }
}

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
    /// Create a new Foundry client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::foundry::{ClientBuilder, self};
    ///
    /// // Initialize the Foundry client
    /// let client = Client::builder()
    ///    .build()
    /// ```
    pub fn builder() -> ClientBuilder<'static> {
        ClientBuilder::new()
    }

    /// Create a new Foundry client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new() -> Self {
        Self::builder().build().expect("Ollama client should build")
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
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
        Self::builder().base_url(&api_base).build().unwrap()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(_) = input else {
            panic!("Incorrect provider value type")
        };

        Self::new()
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

pub const COHERE_EMBED_V4_0: &str = "embed-v-4-0";
pub const COHERE_EMBED_V3_ENGLISH: &str = "Cohere-embed-v3-english";
pub const COHERE_EMBED_V3_MULTILINGUAL: &str = "Cohere-embed-v3-multilingual";
pub const OPENAI_TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";

// ---------- Embedding API ----------
#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingData {
    object: String,
    embedding: Vec<f64>,
    index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
struct Usage {
    prompt_tokens: u64,
    total_tokens: u64,
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

// ----------- Embedding Model --------------
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
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let docs: Vec<String> = documents.into_iter().collect();
        let payload = json!({
            "model": self.model,
            "input":docs,
        });
        let response = self
            .client
            .post("v1/embeddings")
            .json(&payload)
            .send()
            .await
            .map_err(|e| EmbeddingError::ResponseError(e.to_string()))?;
        if response.status().is_success() {
            let api_resp: EmbeddingResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;

            if api_resp.data.len() != docs.len() {
                return Err(EmbeddingError::ResponseError(
                    "Number of returned embeddings does not match input".into(),
                ));
            }
            Ok(api_resp
                .data
                .into_iter()
                .zip(docs.into_iter())
                .map(|(embedding_data, document)| embeddings::Embedding {
                    document,
                    vec: embedding_data.embedding,
                })
                .collect())
        } else {
            Err(EmbeddingError::ProviderError(response.text().await?))
        }
    }
}

// these i took from gemini ( review needed josh)
pub const COHERE_COMMAND_A: &str = "Cohere-command-a";
pub const COHERE_COMMAND_R_PLUS: &str = "Cohere-command-r-plus-08-2024";
pub const COHERE_COMMAND_R: &str = "Cohere-command-r-08-2024";
pub const MISTRAL_CODESTRAL: &str = "Codestral-2501";
pub const MISTRAL_MINISTRAL_3B: &str = "Ministral-3B";
pub const MISTRAL_NEMO: &str = "Mistral-Nemo";
pub const MISTRAL_SMALL: &str = "Mistral-small-2503";
pub const MISTRAL_MEDIUM: &str = "Mistral-medium-2505";
pub const MICROSOFT_PHI_4_MINI_INSTRUCT: &str = "Phi-4-mini-instruct";
pub const MICROSOFT_PHI_4_MULTIMODAL_INSTRUCT: &str = "Phi-4-multimodal-instruct";
pub const MICROSOFT_PHI_4: &str = "Phi-4";
pub const MICROSOFT_PHI_4_REASONING: &str = "Phi-4-reasoning";
pub const MICROSOFT_PHI_4_MINI_REASONING: &str = "Phi-4-mini-reasoning";
pub const OPENAI_GPT_4O: &str = "gpt-4o";
pub const OPENAI_GPT_4O_MINI: &str = "gpt-4o-mini";
pub const OPENAI_GPT_3_5_TURBO: &str = "gpt-35-turbo";
pub const MICROSOFT_PHI_3_MINI_4K_INSTRUCT: &str = "Phi-3-mini-4k-instruct";
pub const MICROSOFT_PHI_3_MINI_128K_INSTRUCT: &str = "Phi-3-mini-128k-instruct";
pub const MICROSOFT_PHI_3_SMALL_8K_INSTRUCT: &str = "Phi-3-small-8k-instruct";
pub const MICROSOFT_PHI_3_SMALL_128K_INSTRUCT: &str = "Phi-3-small-128k-instruct";

// ----------- Completions API -------------
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompletionsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Choice {
    pub index: u64,
    pub message: CompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: CompletionsUsage,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;
    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        let mut assitant_contents = Vec::new();

        // foundry only responds with an array of choices which have
        // role and content (role is always "assistant" for responses)
        for choice in resp.choices.clone() {
            assitant_contents.push(completion::AssistantContent::text(&choice.message.content));
        }

        let choice = OneOrMany::many(assitant_contents)
            .map_err(|_| CompletionError::ResponseError("No content provided".to_owned()))?;

        Ok(completion::CompletionResponse {
            choice,
            usage: rig::completion::request::Usage {
                input_tokens: resp.usage.prompt_tokens,
                output_tokens: resp.usage.completion_tokens,
                total_tokens: resp.usage.total_tokens,
            },
            raw_response: resp,
        })
    }
}

// ----------- Completion Model ----------

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
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        let mut full_history = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| vec![Message::system(&preamble)]);

        // convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(|msg| msg.try_into())
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<Message>>(),
        );

        let mut requeest_payload = json!({
            "model": self.model,
            "messages": full_history,
            "temparature": completion_request.temperature,
            "stream": false,
        });

        if !completion_request.tools.is_empty() {
            // Foundry's functions have same structure as completion::ToolDefination
            requeest_payload["functions"] = json!(
                completion_request
                    .tools
                    .into_iter()
                    .map(|tool| tool)
                    .collect::<Vec<ToolDefinition>>()
            );
        }

        tracing::debug!(target: "rig", "Chat mode payload: {}", requeest_payload);

        Ok(requeest_payload)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: String,
    pub model: String,
    pub usage: CompletionsUsage,
}

// ---------- CompletionModel Implementation ----------
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
            .post("/v1/chat/completions")
            .json(&request_payload)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        if response.status().is_success() {
            let text = response
                .text()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
            tracing::debug!(target: "rig", "Foundry chat response: {}", text);
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
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let mut request_payload = self.create_completion_request(request)?;
        json_utils::merge_inplace(&mut request_payload, json!({"stream": true}));

        let response = self
            .client
            .post("/v1/chat/completions")
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
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    let data_line = if let Some(data) = line.strip_prefix("data: "){
                        data
                    }else{
                        line
                    };

                    // stream termination like openai
                    if data_line == "[DONE]" {
                        break;
                    }

                    let Ok(response) = serde_json::from_str::<CompletionResponse>(data_line) else {
                        continue;
                    };

                    for choice in response.choices.iter() {
                        if !choice.message.content.is_empty() {
                            yield Ok(RawStreamingChoice::Message(choice.message.content.clone()));
                        }
                    }
                    if response.choices.iter().any(|choice| choice.finish_reason == "stop") {
                        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                            id: response.id.clone(),
                            object: response.object.clone(),
                            created: response.created.clone(),
                            model: response.model.clone(),
                            usage: response.usage.clone(),
                        }));
                    }
                }
            }
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "system")]
    System,
    #[serde(rename = "assistant")]
    Assistant,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    role: Role,
    content: String,
}

impl TryFrom<crate::message::Message> for Vec<Message> {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        use crate::message::Message as InternalMessage;
        match internal_msg {
            InternalMessage::User { content, .. } => {
                // Foundry doesn't support tool results in messages, so we skip them
                let non_tool_content: Vec<_> = content
                    .into_iter()
                    .filter(|content| {
                        !matches!(content, crate::message::UserContent::ToolResult(_))
                    })
                    .collect();

                let text_contents: Vec<String> = non_tool_content
                    .into_iter()
                    .filter_map(|content| match content {
                        crate::message::UserContent::Text(crate::message::Text { text }) => {
                            Some(text)
                        }
                        _ => None,
                    })
                    .collect();

                Ok(vec![Message {
                    role: Role::User,
                    content: text_contents.join(" "),
                }])
            }
            InternalMessage::Assistant { content, .. } => {
                let text_contents: Vec<String> = content
                    .into_iter()
                    .filter_map(|content| match content {
                        crate::message::AssistantContent::Text(text) => Some(text.text),
                        _ => None,
                    })
                    .collect();

                Ok(vec![Message {
                    role: Role::Assistant,
                    content: text_contents.join(" "),
                }])
            }
        }
    }
}

impl From<Message> for crate::completion::Message {
    fn from(msg: Message) -> Self {
        match msg.role {
            Role::User => crate::completion::Message::User {
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
                    text: msg.content,
                })),
            },
            Role::Assistant => crate::completion::Message::Assistant {
                id: None,
                content: OneOrMany::one(crate::completion::message::AssistantContent::Text({
                    Text { text: msg.content }
                })),
            },
            Role::System => crate::completion::Message::User {
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
                    text: msg.content,
                })),
            },
        }
    }
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_owned(),
        }
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
}

impl FromStr for UserContent {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text { text: s.to_owned() })
    }
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_chat_completion() {
        let sample_chat_response = json!({
            "id": "chatcmpl-1234567890",
            "object": "chat.completion",
            "created": "1677851234",
            "model": "Phi-4-mini-instruct-generic-cpu",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The sky is blue because of Rayleigh scattering."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
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

    #[test]
    fn test_message_conversion() {
        let provider_msg = Message {
            role: Role::User,
            content: "Test message".to_owned(),
        };
        let comp_msg: crate::completion::Message = provider_msg.into();
        match comp_msg {
            crate::completion::Message::User { content } => {
                let first_content = content.first();
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

    #[test]
    fn test_system_content_from_string() {
        let content = SystemContent::from("Test system message".to_string());
        assert_eq!(content.text, "Test system message");
        assert!(matches!(content.r#type, SystemContentType::Text));
    }

    #[test]
    fn test_system_content_from_str() {
        let content: SystemContent = "Test system message".parse().unwrap();
        assert_eq!(content.text, "Test system message");
        assert!(matches!(content.r#type, SystemContentType::Text));
    }

    #[test]
    fn test_assistant_content_from_str() {
        let content: AssistantContent = "Test assistant message".parse().unwrap();
        assert_eq!(content.text, "Test assistant message");
    }

    #[test]
    fn test_user_content_from_str() {
        let content: UserContent = "Test user message".parse().unwrap();
        match content {
            UserContent::Text { text } => {
                assert_eq!(text, "Test user message");
            }
        }
    }
}
