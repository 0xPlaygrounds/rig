// ================================================================
//! Google Gemini Completion Integration
//! https://ai.google.dev/api/generate-content
// ================================================================

/// `gemini-1.5-flash` completion model
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro` completion model
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-pro-8b` completion model
pub const GEMINI_1_5_PRO_8B: &str = "gemini-1.5-pro-8b";
/// `gemini-1.0-pro` completion model
pub const GEMINI_1_0_PRO: &str = "gemini-1.0-pro";

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::convert::TryFrom;

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    providers::gemini::client::ApiResponse,
};

use super::Client;

// =================================================================
// Gemini API Types
// =================================================================

// Define the struct for the GenerateContentResponse
#[derive(Debug, Deserialize)]
pub struct GenerateContentResponse {
    pub candidates: Vec<ContentCandidate>,
    pub prompt_feedback: Option<PromptFeedback>,
    pub usage_metadata: Option<UsageMetadata>,
}

// Define the struct for a Candidate
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentCandidate {
    pub content: Content,
    pub finish_reason: Option<FinishReason>,
    pub safety_ratings: Option<Vec<SafetyRating>>,
    pub citation_metadata: Option<CitationMetadata>,
    pub token_count: Option<i32>,
    pub avg_logprobs: Option<f64>,
    pub logprobs_result: Option<LogprobsResult>,
    pub index: Option<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Content {
    pub parts: Vec<Part>,
    pub role: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Part {
    pub text: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SafetyRating {
    pub category: HarmCategory,
    pub probability: HarmProbability,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmProbability {
    HarmProbabilityUnspecified,
    Negligible,
    Low,
    Medium,
    High,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    HarmCategoryUnspecified,
    HarmCategoryDerogatory,
    HarmCategoryToxicity,
    HarmCategoryViolence,
    HarmCategorySexually,
    HarmCategoryMedical,
    HarmCategoryDangerous,
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent,
    HarmCategoryCivicIntegrity,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: i32,
    pub cached_content_token_count: i32,
    pub candidates_token_count: i32,
    pub total_token_count: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    pub block_reason: Option<BlockReason>,
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
pub enum BlockReason {
    BlockReasonUnspecified,
    Safety,
    Other,
    Blocklist,
    ProhibitedContent,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    FinishReasonUnspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Language,
    Other,
    Blocklist,
    ProhibitedContent,
}

#[derive(Debug, Deserialize)]
pub struct CitationMetadata {
    pub citation_sources: Vec<CitationSource>,
}

#[derive(Debug, Deserialize)]
pub struct CitationSource {
    pub uri: Option<String>,
    pub start_index: Option<i32>,
    pub end_index: Option<i32>,
    pub license: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LogprobsResult {
    pub top_candidate: Vec<TopCandidate>,
    pub chosen_candidate: Vec<LogProbCandidate>,
}

#[derive(Debug, Deserialize)]
pub struct TopCandidate {
    pub candidates: Vec<LogProbCandidate>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogProbCandidate {
    pub token: String,
    pub token_id: String,
    pub log_probability: f64,
}

/// Gemini API Configuration options for model generation and outputs. Not all parameters are
/// configurable for every model. https://ai.google.dev/api/generate-content#generationconfig
#[derive(Debug, Deserialize, Serialize)]
pub struct GenerationConfig {
    /// The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop
    /// at the first appearance of a stop_sequence. The stop sequence will not be included as part of the response.
    pub stop_sequences: Option<Vec<String>>,
    /// MIME type of the generated candidate text. Supported MIME types are:
    ///     - text/plain:  (default) Text output
    ///     - application/json: JSON response in the response candidates.
    ///     - text/x.enum: ENUM as a string response in the response candidates.
    /// Refer to the docs for a list of all supported text MIME types
    pub response_mime_type: Option<String>,
    /// Output schema of the generated candidate text. Schemas must be a subset of the OpenAPI schema and can be
    /// objects, primitives or arrays. If set, a compatible responseMimeType must also  be set. Compatible MIME
    /// types: application/json: Schema for JSON response. Refer to the JSON text generation guide for more details.
    pub response_schema: Option<Schema>,
    /// Number of generated responses to return. Currently, this value can only be set to 1. If
    /// unset, this will default to 1.
    pub candidate_count: Option<i32>,
    /// The maximum number of tokens to include in a response candidate. Note: The default value varies by model, see
    /// the Model.output_token_limit attribute of the Model returned from the getModel function.
    pub max_output_tokens: Option<u64>,
    /// Controls the randomness of the output. Note: The default value varies by model, see the Model.temperature
    /// attribute of the Model returned from the getModel function. Values can range from [0.0, 2.0].
    pub temperature: Option<f64>,
    /// The maximum cumulative probability of tokens to consider when sampling. The model uses combined Top-k and
    /// Top-p (nucleus) sampling. Tokens are sorted based on their assigned probabilities so that only the most
    /// likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while
    /// Nucleus sampling limits the number of tokens based on the cumulative probability. Note: The default value
    /// varies by Model and is specified by theModel.top_p attribute returned from the getModel function. An empty
    /// topK attribute indicates that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
    pub top_p: Option<f64>,
    /// The maximum number of tokens to consider when sampling. Gemini models use Top-p (nucleus) sampling or a
    /// combination of Top-k and nucleus sampling. Top-k sampling considers the set of topK most probable tokens.
    /// Models running with nucleus sampling don't allow topK setting. Note: The default value varies by Model and is
    /// specified by theModel.top_p attribute returned from the getModel function. An empty topK attribute indicates
    /// that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
    pub top_k: Option<i32>,
    /// Presence penalty applied to the next token's logprobs if the token has already been seen in the response.
    /// This penalty is binary on/off and not dependant on the number of times the token is used (after the first).
    /// Use frequencyPenalty for a penalty that increases with each use. A positive penalty will discourage the use
    /// of tokens that have already been used in the response, increasing the vocabulary. A negative penalty will
    /// encourage the use of tokens that have already been used in the response, decreasing the vocabulary.
    pub presence_penalty: Option<f64>,
    /// Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been
    /// seen in the respponse so far. A positive penalty will discourage the use of tokens that have already been
    /// used, proportional to the number of times the token has been used: The more a token is used, the more
    /// dificult it is for the  model to use that token again increasing the vocabulary of responses. Caution: A
    /// negative penalty will encourage the model to reuse tokens proportional to the number of times the token has
    /// been used. Small negative values will reduce the vocabulary of a response. Larger negative values will cause
    /// the model to  repeating a common token until it hits the maxOutputTokens limit: "...the the the the the...".
    pub frequency_penalty: Option<f64>,
    /// If true, export the logprobs results in response.
    pub response_logprobs: Option<bool>,
    /// Only valid if responseLogprobs=True. This sets the number of top logprobs to return at each decoding step in
    /// [Candidate.logprobs_result].
    pub logprobs: Option<i32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: Some(1.0),
            max_output_tokens: Some(4096),
            stop_sequences: None,
            response_mime_type: None,
            response_schema: None,
            candidate_count: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_logprobs: None,
            logprobs: None,
        }
    }
}
/// The Schema object allows the definition of input and output data types. These types can be objects, but also
/// primitives and arrays. Represents a select subset of an OpenAPI 3.0 schema object.
/// https://ai.google.dev/api/caching#Schema
#[derive(Debug, Deserialize, Serialize)]
pub struct Schema {
    pub r#type: String,
    pub format: Option<String>,
    pub description: Option<String>,
    pub nullable: Option<bool>,
    pub r#enum: Option<Vec<String>>,
    pub max_items: Option<i32>,
    pub min_items: Option<i32>,
    pub properties: Option<HashMap<String, Schema>>,
    pub required: Option<Vec<String>>,
    pub items: Option<Box<Schema>>,
}

impl TryFrom<Value> for Schema {
    type Error = CompletionError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Some(obj) = value.as_object() {
            Ok(Schema {
                r#type: obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                format: obj.get("format").and_then(|v| v.as_str()).map(String::from),
                description: obj
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                nullable: obj.get("nullable").and_then(|v| v.as_bool()),
                r#enum: obj.get("enum").and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
                max_items: obj
                    .get("maxItems")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32),
                min_items: obj
                    .get("minItems")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32),
                properties: obj
                    .get("properties")
                    .and_then(|v| v.as_object())
                    .map(|map| {
                        map.iter()
                            .filter_map(|(k, v)| {
                                v.clone().try_into().ok().map(|schema| (k.clone(), schema))
                            })
                            .collect()
                    }),
                required: obj.get("required").and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
                items: obj
                    .get("items")
                    .map(|v| Box::new(v.clone().try_into().unwrap())),
            })
        } else {
            Err(CompletionError::ResponseError(
                "Expected a JSON object for Schema".into(),
            ))
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    pub tools: Option<Vec<Tool>>,
    pub tool_config: Option<ToolConfig>,
    /// Optional. Configuration options for model generation and outputs.
    pub generation_config: Option<GenerationConfig>,
    /// Optional. A list of unique SafetySetting instances for blocking unsafe content. This will be enforced on the
    /// [GenerateContentRequest.contents] and [GenerateContentResponse.candidates]. There should not be more than one
    /// setting for each SafetyCategory type. The API will block any contents and responses that fail to meet the
    /// thresholds set by these settings. This list overrides the default settings for each SafetyCategory specified
    /// in the safetySettings. If there is no SafetySetting for a given SafetyCategory provided in the list, the API
    /// will use the default safety setting for that category. Harm categories:
    ///     - HARM_CATEGORY_HATE_SPEECH,
    ///     - HARM_CATEGORY_SEXUALLY_EXPLICIT
    ///     - HARM_CATEGORY_DANGEROUS_CONTENT
    ///     - HARM_CATEGORY_HARASSMENT
    /// are supported.
    /// Refer to the guide for detailed information on available safety settings. Also refer to the Safety guidance
    /// to learn how to incorporate safety considerations in your AI applications.
    pub safety_settings: Option<Vec<SafetySetting>>,
    /// Optional. Developer set system instruction(s). Currently, text only.
    /// https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest
    pub system_instruction: Option<String>,
    // cachedContent: Optional<String>
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub function_declaration: FunctionDeclaration,
    pub code_execution: Option<CodeExecution>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Option<Vec<Schema>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub schema: Option<Schema>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecution {}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetySetting {
    pub category: HarmCategory,
    pub threshold: HarmBlockThreshold,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
    HarmBlockThresholdUnspecified,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone,
    Off,
}

// =================================================================
// Rig Implementation Types
// =================================================================

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
    type Response = GenerateContentResponse;

    async fn completion(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError> {
        // QUESTION: Why do Anthropic/openAi implementation differ here? OpenAI adds the preamble but Anthropic does not.

        let mut full_history = if let Some(preamble) = &completion_request.preamble {
            vec![completion::Message {
                role: "system".into(),
                content: preamble.clone(),
            }]
        } else {
            vec![]
        };

        full_history.append(&mut completion_request.chat_history);

        let prompt_with_context = completion_request.prompt_with_context();

        full_history.push(completion::Message {
            role: "user".into(),
            content: prompt_with_context,
        });

        // Handle Gemini specific parameters
        let mut generation_config =
            GenerationConfig::try_from(completion_request.additional_params.unwrap_or_default())?;

        // Set temperature from completion_request or additional_params
        if let Some(temp) = completion_request.temperature {
            generation_config.temperature = Some(temp);
        }

        // Set max_tokens from completion_request or additional_params
        if let Some(max_tokens) = completion_request.max_tokens {
            generation_config.max_output_tokens = Some(max_tokens);
        }

        let request = GenerateContentRequest {
            contents: full_history
                .into_iter()
                .map(|msg| Content {
                    parts: vec![Part { text: msg.content }],
                    role: match msg.role.as_str() {
                        "system" => Some("model".to_string()),
                        "user" => Some("user".to_string()),
                        "assistant" => Some("model".to_string()),
                        _ => None,
                    },
                })
                .collect(),
            generation_config: Some(generation_config),
            safety_settings: None,
            tools: Some(
                completion_request
                    .tools
                    .into_iter()
                    .map(Tool::from)
                    .collect(),
            ),
            tool_config: None,
            system_instruction: None,
        };

        let response = self
            .client
            .post(&format!("/v1beta/models/{}:generateContent", self.model))
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<ApiResponse<GenerateContentResponse>>()
            .await?;

        match response {
            ApiResponse::Ok(response) => Ok(response.try_into()?),
            ApiResponse::Err(err) => Err(CompletionError::ResponseError(err.message)),
        }
    }
}

impl From<completion::ToolDefinition> for Tool {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            function_declaration: FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters: None, // tool.parameters, TODO: Map Gemini
            },
            code_execution: None,
        }
    }
}

impl TryFrom<GenerateContentResponse> for completion::CompletionResponse<GenerateContentResponse> {
    type Error = CompletionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        match response.candidates.as_slice() {
            [ContentCandidate { content, .. }, ..] => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(
                    content.parts.first().unwrap().text.clone(),
                ),
                raw_response: response,
            }),
            _ => Err(CompletionError::ResponseError(
                "No candidates found in response".into(),
            )),
        }
    }
}

impl TryFrom<serde_json::Value> for GenerationConfig {
    type Error = CompletionError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        let mut config = GenerationConfig {
            temperature: None,
            max_output_tokens: None,
            stop_sequences: None,
            response_mime_type: None,
            response_schema: None,
            candidate_count: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_logprobs: None,
            logprobs: None,
        };

        fn unexpected_type_error(field: &str) -> CompletionError {
            CompletionError::ResponseError(format!("Unexpected type for field '{}'", field))
        }

        if let Some(obj) = value.as_object() {
            for (key, value) in obj.iter().filter(|(_, v)| !v.is_null()) {
                match key.as_str() {
                    "temperature" => {
                        if !value.is_null() {
                            if let Some(v) = value.as_f64() {
                                config.temperature = Some(v);
                            } else {
                                return Err(unexpected_type_error("temperature"));
                            }
                        }
                    }
                    "max_output_tokens" => {
                        if let Some(v) = value.as_u64() {
                            config.max_output_tokens = Some(v);
                        } else {
                            return Err(unexpected_type_error("max_output_tokens"));
                        }
                    }
                    "top_p" => {
                        if let Some(v) = value.as_f64() {
                            config.top_p = Some(v);
                        } else {
                            return Err(unexpected_type_error("top_p"));
                        }
                    }
                    "top_k" => {
                        if let Some(v) = value.as_i64() {
                            config.top_k = Some(v as i32);
                        } else {
                            return Err(unexpected_type_error("top_k"));
                        }
                    }
                    "candidate_count" => {
                        if let Some(v) = value.as_i64() {
                            config.candidate_count = Some(v as i32);
                        } else {
                            return Err(unexpected_type_error("candidate_count"));
                        }
                    }
                    "stop_sequences" => {
                        if let Some(v) = value.as_array() {
                            config.stop_sequences = Some(
                                v.iter()
                                    .filter_map(|s| s.as_str().map(String::from))
                                    .collect(),
                            );
                        } else {
                            return Err(unexpected_type_error("stop_sequences"));
                        }
                    }
                    "response_mime_type" => {
                        if let Some(v) = value.as_str() {
                            config.response_mime_type = Some(v.to_string());
                        } else {
                            return Err(unexpected_type_error("response_mime_type"));
                        }
                    }
                    "response_schema" => {
                        config.response_schema = Some(value.clone().try_into()?);
                    }
                    "presence_penalty" => {
                        if let Some(v) = value.as_f64() {
                            config.presence_penalty = Some(v);
                        } else {
                            return Err(unexpected_type_error("presence_penalty"));
                        }
                    }
                    "frequency_penalty" => {
                        if let Some(v) = value.as_f64() {
                            config.frequency_penalty = Some(v);
                        } else {
                            return Err(unexpected_type_error("frequency_penalty"));
                        }
                    }
                    "response_logprobs" => {
                        if let Some(v) = value.as_bool() {
                            config.response_logprobs = Some(v);
                        } else {
                            return Err(unexpected_type_error("response_logprobs"));
                        }
                    }
                    "logprobs" => {
                        if let Some(v) = value.as_i64() {
                            config.logprobs = Some(v as i32);
                        } else {
                            return Err(unexpected_type_error("logprobs"));
                        }
                    }
                    _ => {
                        tracing::warn!(
                            "Unknown GenerationConfig parameter, will be ignored: {}",
                            key
                        );
                    }
                }
            }
        } else {
            return Err(CompletionError::ResponseError(
                "Expected a JSON object for GenerationConfig".into(),
            ));
        }

        Ok(config)
    }
}
