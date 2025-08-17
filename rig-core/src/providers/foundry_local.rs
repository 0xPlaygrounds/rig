use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashMap, convert::TryFrom, str::FromStr};

use crate::{
    Embed, OneOrMany,
    client::{ClientBuilderError, CompletionClient, EmbeddingsClient, ProviderClient},
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    impl_conversion_traits, json_utils,
    message::{self, Text},
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
    /// Create a new Foundry-Local client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::foundry::{ClientBuilder, self};
    ///
    /// // Initialize the Foundry client
    /// let client = Client::builder()
    ///     .build()
    /// ```
    pub fn builder() -> ClientBuilder<'static> {
        ClientBuilder::new()
    }

    /// Create a new Foundry-Local client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new() -> Self {
        Self::builder()
            .build()
            .expect("Foundry-local client should build")
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
        let api_base = std::env::var("FOUNDRY_LOCAL_API_BASE_URL")
            .expect("FOUNDRY_LOCAL_API_BASE_URL not set");
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: crate::completion::ToolDefinition,
}

impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        ToolDefinition {
            type_field: "function".to_owned(),
            function: tool,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

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
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct CompletionMessage {
    pub role: String,
    // Content can be null when tool_calls are present
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: CompletionsUsage,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;
    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = resp
            .choices
            .first()
            .ok_or_else(|| CompletionError::ResponseError("No choices in response".to_owned()))?;

        let assistant_contents = if let Some(tool_calls) = &choice.message.tool_calls {
            tool_calls
                .iter()
                .map(|tc| {
                    let arguments: Value = serde_json::from_str(&tc.function.arguments)
                        .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
                    Ok(completion::AssistantContent::tool_call(
                        tc.id.clone(),
                        tc.function.name.clone(),
                        arguments,
                    ))
                })
                .collect::<Result<Vec<_>, CompletionError>>()?
        } else if let Some(content) = &choice.message.content {
            vec![completion::AssistantContent::text(content)]
        } else {
            return Err(CompletionError::ResponseError(
                "Response has neither content nor tool calls".to_owned(),
            ));
        };

        let choice = OneOrMany::many(assistant_contents)
            .map_err(|_| CompletionError::ResponseError("No content provided".to_owned()))?;

        Ok(completion::CompletionResponse {
            choice,
            usage: rig::completion::Usage {
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

        let mut request_payload = json!({
            "model": self.model,
            "messages": full_history,
            "temperature": completion_request.temperature,
            "stream": false,
        });

        if !completion_request.tools.is_empty() {
            request_payload["tools"] = json!(
                completion_request
                    .tools
                    .into_iter()
                    .map(|tool| tool.into())
                    .collect::<Vec<ToolDefinition>>()
            );
        }

        tracing::debug!(target: "rig", "Chat mode payload: {}", request_payload);

        Ok(request_payload)
    }
}

// Changed StreamingCompletionResponse to handle SSE deltas
#[derive(Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponseChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamingChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamingChoice {
    pub index: u64,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DeltaMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<StreamingToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingToolCall {
    pub index: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<StreamingFunctionCall>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct StreamingFunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// Final response for streaming mode
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingFinalResponse {
    pub id: String,
    pub model: String,
}

// ---------- CompletionModel Implementation ----------
impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingFinalResponse;

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
            tracing::debug!(target: "rig", "Foundry-Local chat response: {}", text);
            let chat_resp: CompletionResponse = serde_json::from_str(&text)
                .map_err(|e| CompletionError::ResponseError(e.to_string()))?;
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
            let mut tool_calls: HashMap<u64, (Option<String>, StreamingFunctionCall)> = HashMap::new();
            let mut final_response_id = "".to_string();
            let mut final_response_model = "".to_string();


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
                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if data == "[DONE]" {
                            break;
                        }

                        let Ok(chunk) = serde_json::from_str::<StreamingCompletionResponseChunk>(data) else {
                            continue;
                        };

                        final_response_id = chunk.id;
                        final_response_model = chunk.model;


                        for choice in chunk.choices {
                            if let Some(content) = choice.delta.content {
                                yield Ok(RawStreamingChoice::Message(content));
                            }

                            if let Some(delta_tool_calls) = choice.delta.tool_calls {
                                for stc in delta_tool_calls {
                                    let entry = tool_calls.entry(stc.index).or_default();
                                    if let Some(id) = stc.id {
                                        entry.0 = Some(id);
                                    }
                                    if let Some(function) = stc.function {
                                        if let Some(name) = function.name {
                                            entry.1.name.get_or_insert_with(String::new).push_str(&name);
                                        }
                                        if let Some(args) = function.arguments {
                                            entry.1.arguments.get_or_insert_with(String::new).push_str(&args);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // yield any completed tool calls
            for (_, (id, function)) in tool_calls {
                if let (Some(id), Some(name), Some(arguments)) = (id, function.name, function.arguments) {
                     let Ok(args_json) = serde_json::from_str(&arguments) else {
                        yield Err(CompletionError::ResponseError(format!("Failed to parse tool call arguments: {}", arguments)));
                        continue;
                    };
                    yield Ok(RawStreamingChoice::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: args_json,
                        call_id: None,
                    });
                }
            }

            yield Ok(RawStreamingChoice::FinalResponse(StreamingFinalResponse {
                id: final_response_id,
                model: final_response_model,
            }));
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        // content can be null when tool_calls are present
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl TryFrom<crate::message::Message> for Vec<Message> {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        use crate::message::Message as InternalMessage;
        match internal_msg {
            InternalMessage::User { content, .. } => {
                let mut messages = Vec::new();
                let mut text_parts = Vec::new();

                for part in content {
                    match part {
                        message::UserContent::Text(text) => text_parts.push(text.text),
                        message::UserContent::ToolResult(result) => {
                            let content_string = result
                                .content
                                .into_iter()
                                .map(|c| match c {
                                    message::ToolResultContent::Text(t) => t.text,
                                    _ => "[unsupported content]".to_string(),
                                })
                                .collect::<Vec<_>>()
                                .join("\n");

                            messages.push(Message::Tool {
                                tool_call_id: result.id,
                                content: content_string,
                            });
                        }
                        _ => {}
                    }
                }

                if !text_parts.is_empty() {
                    messages.insert(
                        0,
                        Message::User {
                            content: text_parts.join("\n"),
                        },
                    );
                }

                Ok(messages)
            }
            InternalMessage::Assistant { content, .. } => {
                let mut text_content = None;
                let mut tool_calls = Vec::new();

                for part in content {
                    match part {
                        message::AssistantContent::Text(text) => {
                            text_content
                                .get_or_insert_with(String::new)
                                .push_str(&text.text);
                        }
                        message::AssistantContent::ToolCall(tc) => {
                            tool_calls.push(ToolCall {
                                id: tc.id,
                                r#type: "function".to_string(),
                                function: FunctionCall {
                                    name: tc.function.name,
                                    arguments: tc.function.arguments.to_string(),
                                },
                            });
                        }
                        _ => {}
                    }
                }

                Ok(vec![Message::Assistant {
                    content: text_content,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                }])
            }
        }
    }
}

impl From<Message> for crate::completion::Message {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User { content } | Message::System { content } => {
                crate::completion::Message::User {
                    content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
                        text: content,
                    })),
                }
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut assistant_contents = Vec::new();
                if let Some(text) = content {
                    if !text.is_empty() {
                        assistant_contents.push(message::AssistantContent::Text(Text { text }));
                    }
                }
                if let Some(tcs) = tool_calls {
                    for tc in tcs {
                        let arguments: Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| json!(tc.function.arguments));
                        assistant_contents.push(message::AssistantContent::tool_call(
                            tc.id,
                            tc.function.name,
                            arguments,
                        ));
                    }
                }

                crate::completion::Message::Assistant {
                    id: None,
                    content: OneOrMany::many(assistant_contents)
                        .unwrap_or_else(|_| OneOrMany::one(message::AssistantContent::text(""))),
                }
            }
            Message::Tool {
                tool_call_id,
                content,
            } => crate::completion::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    tool_call_id,
                    OneOrMany::one(message::ToolResultContent::text(content)),
                )),
            },
        }
    }
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self::System {
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
            "created": 1677851234,
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
    fn test_tool_call_deserialization_and_conversion() {
        let tool_call_response = json!({
            "id": "chatcmpl-9pFN3aGu2dM1ALf1IixE23qG1Wp7u",
            "object": "chat.completion",
            "created": 1720235377,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_stools_get_flight_info_1720235377043",
                                "type": "function",
                                "function": {
                                    "name": "get_flight_info",
                                    "arguments": "{\"origin_city\":\"Miami\",\"destination_city\":\"Seattle\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 83,
                "completion_tokens": 21,
                "total_tokens": 104
            }
        });

        let chat_resp: CompletionResponse = serde_json::from_value(tool_call_response).unwrap();
        let conv_resp: completion::CompletionResponse<CompletionResponse> =
            chat_resp.try_into().unwrap();

        assert_eq!(conv_resp.choice.len(), 1);
        match conv_resp.choice.first() {
            completion::AssistantContent::ToolCall(tc) => {
                assert_eq!(tc.id, "call_stools_get_flight_info_1720235377043");
                assert_eq!(tc.function.name, "get_flight_info");
                assert_eq!(
                    tc.function.arguments,
                    json!({"origin_city": "Miami", "destination_city": "Seattle"})
                );
            }
            _ => panic!("Expected a tool call"),
        }
    }

    #[test]
    fn test_message_conversion() {
        let provider_msg = Message::User {
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
    fn test_tool_result_message_conversion() {
        let rig_message = crate::message::Message::User {
            content: OneOrMany::one(crate::message::UserContent::tool_result(
                "call_123",
                OneOrMany::one(crate::message::ToolResultContent::text("Flight found")),
            )),
        };

        let provider_messages: Vec<Message> = rig_message.try_into().unwrap();
        assert_eq!(provider_messages.len(), 1);
        match &provider_messages[0] {
            Message::Tool {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "call_123");
                assert_eq!(content, "Flight found");
            }
            _ => panic!("Expected a Tool message"),
        }
    }
}
