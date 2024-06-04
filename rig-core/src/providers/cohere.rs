use std::collections::HashMap;

use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError},
    embeddings::{self, EmbeddingsBuilder},
    extractor::ExtractorBuilder,
    json_utils,
    model::ModelBuilder,
    rag::RagAgentBuilder,
    vector_store::{NoIndex, VectorStoreIndex},
};
use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main Cohere Client
// ================================================================
#[derive(Clone)]
pub struct Client(reqwest::Client);

impl Client {
    pub fn new(api_key: &str) -> Self {
        Self(
            reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("Cohere reqwest client should build"),
        )
    }

    pub fn embedding_model(&self, model: &str, input_type: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, input_type)
    }

    pub fn embeddings(&self, model: &str, input_type: &str) -> EmbeddingsBuilder<EmbeddingModel> {
        EmbeddingsBuilder::new(self.embedding_model(model, input_type))
    }

    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    pub fn model(&self, model: &str) -> ModelBuilder<CompletionModel> {
        ModelBuilder::new(self.completion_model(model))
    }

    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }

    pub fn rag_agent<C: VectorStoreIndex, T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn tool_rag_agent<T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, NoIndex, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn context_rag_agent<C: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, NoIndex> {
        RagAgentBuilder::new(self.completion_model(model))
    }
}

// ================================================================
// Cohere Embedding API
// ================================================================
#[derive(Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub response_type: Option<String>,
    pub id: String,
    pub embeddings: Vec<Vec<f64>>,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

#[derive(Deserialize)]
pub struct BilledUnits {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub search_units: u32,
    #[serde(default)]
    pub classifications: u32,
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    pub input_type: String,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 96;

    async fn embed_documents(&self, documents: Vec<String>) -> Result<Vec<embeddings::Embedding>> {
        let response = self
            .client
            .0
            .post("https://api.cohere.ai/v1/embed")
            .json(&json!({
                "model": self.model,
                "texts": documents,
                "input_type": self.input_type,
            }))
            .send()
            .await?
            .json::<EmbeddingResponse>()
            .await?;

        // let raw_response = self.client.0.post("https://api.cohere.ai/v1/embed")
        //     .json(&json!({
        //         "model": self.model,
        //         "texts": documents,
        //         "input_type": self.input_type,
        //     }))
        //     .send()
        //     .await?
        //     .json::<serde_json::Value>()
        //     .await?;

        // println!("raw_response: {}", serde_json::to_string_pretty(&raw_response).unwrap());
        // let response: EmbeddingResponse = serde_json::from_value(raw_response).unwrap();

        Ok(response
            .embeddings
            .into_iter()
            .zip(documents.into_iter())
            .map(|(embedding, document)| embeddings::Embedding {
                document,
                vec: embedding,
            })
            .collect())
    }
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, input_type: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            input_type: input_type.to_string(),
        }
    }
}

// ================================================================
// Cohere Completion API
// ================================================================
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
    pub generation_id: String,
    #[serde(default)]
    pub citations: Vec<Citation>,
    #[serde(default)]
    pub documents: Vec<Document>,
    #[serde(default)]
    pub is_search_required: Option<bool>,
    #[serde(default)]
    pub search_queries: Vec<SearchQuery>,
    #[serde(default)]
    pub search_results: Vec<SearchResult>,
    pub finish_reason: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub chat_history: Vec<ChatHistory>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Response {
    Completion(CompletionResponse),
    Error(ErrorResponse),
}

impl From<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    fn from(response: CompletionResponse) -> Self {
        let CompletionResponse {
            text, tool_calls, ..
        } = &response;

        let model_response = if !tool_calls.is_empty() {
            completion::ModelChoice::ToolCall(
                tool_calls.first().unwrap().name.clone(),
                tool_calls.first().unwrap().parameters.clone(),
            )
        } else {
            completion::ModelChoice::Message(text.clone())
        };

        completion::CompletionResponse {
            choice: model_response,
            raw_response: response,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Citation {
    pub start: u32,
    pub end: u32,
    pub text: String,
    pub document_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Document {
    pub id: String,
    #[serde(flatten)]
    pub additional_prop: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub generation_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub search_query: SearchQuery,
    pub connector: Connector,
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub error_message: Option<String>,
    #[serde(default)]
    pub continue_on_failure: bool,
}

#[derive(Debug, Deserialize)]
pub struct Connector {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatHistory {
    pub role: String,
    pub message: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Parameter {
    pub description: String,
    pub r#type: String,
    pub required: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameter_definitions: HashMap<String, Parameter>,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        fn convert_type(r#type: &serde_json::Value) -> String {
            fn convert_type_str(r#type: &str) -> String {
                match r#type {
                    "string" => "string".to_owned(),
                    "number" => "number".to_owned(),
                    "integer" => "integer".to_owned(),
                    "boolean" => "boolean".to_owned(),
                    "array" => "array".to_owned(),
                    "object" => "object".to_owned(),
                    _ => "string".to_owned(),
                }
            }
            match r#type {
                serde_json::Value::String(r#type) => convert_type_str(r#type.as_str()),
                serde_json::Value::Array(types) => convert_type_str(
                    types
                        .iter()
                        .find(|t| t.as_str() != Some("null"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("string"),
                ),
                _ => "string".to_owned(),
            }
        }

        let maybe_required = tool
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .map(|required| {
                required
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Self {
            name: tool.name,
            description: tool.description,
            parameter_definitions: tool
                .parameters
                .get("properties")
                .expect("Tool properties should exist")
                .as_object()
                .expect("Tool properties should be an object")
                .iter()
                .map(|(argname, argdef)| {
                    (
                        argname.clone(),
                        Parameter {
                            description: argdef
                                .get("description")
                                .expect("Argument description should exist")
                                .as_str()
                                .expect("Argument description should be a string")
                                .to_string(),
                            r#type: convert_type(
                                argdef.get("type").expect("Argument type should exist"),
                            ),
                            required: maybe_required.contains(&argname.as_str()),
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub message: String,
}

impl From<completion::Message> for Message {
    fn from(message: completion::Message) -> Self {
        Self {
            role: match message.role.as_str() {
                "system" => "SYSTEM".to_owned(),
                "user" => "USER".to_owned(),
                "assistant" => "CHATBOT".to_owned(),
                _ => "USER".to_owned(),
            },
            message: message.content,
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
}

impl completion::CompletionModel for CompletionModel {
    type T = CompletionResponse;

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = json!({
            "model": self.model,
            "preamble": completion_request.preamble,
            "message": completion_request.prompt,
            "documents": completion_request.documents,
            "chat_history": completion_request.chat_history.into_iter().map(Message::from).collect::<Vec<_>>(),
            "temperature": completion_request.temperature,
            "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
        });

        let cohere_response = self
            .client
            .0
            .post("https://api.cohere.ai/v1/chat")
            .json(
                &if let Some(ref params) = completion_request.additional_params {
                    json_utils::merge(request.clone(), params.clone())
                } else {
                    request.clone()
                },
            )
            .send()
            .await?
            .json()
            .await?;

        // let full_req = if let Some(ref params) = completion_request.additional_params {json_utils::merge(request, params.clone())} else {request};
        // println!("full_req: {}", serde_json::to_string_pretty(&full_req).unwrap());

        // let raw_response = self.client.0.post("https://api.cohere.ai/v1/chat")
        //     .json(&full_req)
        //     .send()
        //     .await?
        //     .json::<serde_json::Value>()
        //     .await?;

        // println!("raw_response: {}", serde_json::to_string_pretty(&raw_response).unwrap());

        match cohere_response {
            Response::Completion(completion) => Ok(completion.into()),
            Response::Error(error) => Err(CompletionError::ProviderError("Cohere".into(), error.message)),
        }
    }
}
