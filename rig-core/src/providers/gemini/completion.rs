// ================================================================
//! Google Gemini Completion Integration
//! From [Gemini API Reference](https://ai.google.dev/api/generate-content)
// ================================================================

/// `gemini-1.5-flash` completion model
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro` completion model
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-pro-8b` completion model
pub const GEMINI_1_5_PRO_8B: &str = "gemini-1.5-pro-8b";
/// `gemini-1.0-pro` completion model
pub const GEMINI_1_0_PRO: &str = "gemini-1.0-pro";

use gemini_api_types::{
    Content, ContentCandidate, FunctionDeclaration, GenerateContentRequest,
    GenerateContentResponse, GenerationConfig, Part, Role, Tool,
};
use serde_json::{Map, Value};
use std::convert::TryFrom;

use crate::completion::{self, CompletionError, CompletionRequest};

use super::Client;

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

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError> {
        let mut full_history = Vec::new();
        full_history.append(&mut completion_request.chat_history);

        let prompt_with_context = completion_request.prompt_with_context();

        full_history.push(completion::Message {
            role: "user".into(),
            content: prompt_with_context,
        });

        // Handle Gemini specific parameters
        let additional_params = completion_request
            .additional_params
            .unwrap_or_else(|| Value::Object(Map::new()));
        let mut generation_config = serde_json::from_value::<GenerationConfig>(additional_params)?;

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
                    parts: vec![Part {
                        text: Some(msg.content),
                        ..Default::default()
                    }],
                    role: match msg.role.as_str() {
                        "system" => Some(Role::Model),
                        "user" => Some(Role::User),
                        "assistant" => Some(Role::Model),
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
            system_instruction: Some(Content {
                parts: vec![Part {
                    text: Some("system".to_string()),
                    ..Default::default()
                }],
                role: Some(Role::Model),
            }),
        };

        tracing::debug!("Sending completion request to Gemini API");

        let response = self
            .client
            .post(&format!("/v1beta/models/{}:generateContent", self.model))
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<GenerateContentResponse>()
            .await?;

        match response.usage_metadata {
            Some(ref usage) => tracing::info!(target: "rig",
            "Gemini completion token usage: {}",
            usage
            ),
            None => tracing::info!(target: "rig",
                "Gemini completion token usage: n/a",
            ),
        }

        tracing::debug!("Received response");

        completion::CompletionResponse::try_from(response)
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
                choice: match content.parts.first().unwrap() {
                    Part {
                        text: Some(text), ..
                    } => completion::ModelChoice::Message(text.clone()),
                    Part {
                        function_call: Some(function_call),
                        ..
                    } => {
                        let args_value = serde_json::Value::Object(
                            function_call.args.clone().unwrap_or_default(),
                        );
                        completion::ModelChoice::ToolCall(
                            function_call.name.clone(),
                            "".to_owned(),
                            args_value,
                        )
                    }
                    _ => {
                        return Err(CompletionError::ResponseError(
                            "Unsupported response by the model of type ".into(),
                        ))
                    }
                },
                raw_response: response,
            }),
            _ => Err(CompletionError::ResponseError(
                "No candidates found in response".into(),
            )),
        }
    }
}

pub mod gemini_api_types {
    use std::collections::HashMap;

    // =================================================================
    // Gemini API Types
    // =================================================================
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};

    use crate::{
        completion::CompletionError,
        providers::gemini::gemini_api_types::{CodeExecutionResult, ExecutableCode},
    };

    /// Response from the model supporting multiple candidate responses.
    /// Safety ratings and content filtering are reported for both prompt in GenerateContentResponse.prompt_feedback
    /// and for each candidate in finishReason and in safetyRatings.
    /// The API:
    ///     - Returns either all requested candidates or none of them
    ///     - Returns no candidates at all only if there was something wrong with the prompt (check promptFeedback)
    ///     - Reports feedback on each candidate in finishReason and safetyRatings.
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentResponse {
        /// Candidate responses from the model.
        pub candidates: Vec<ContentCandidate>,
        /// Returns the prompt's feedback related to the content filters.
        pub prompt_feedback: Option<PromptFeedback>,
        /// Output only. Metadata on the generation requests' token usage.
        pub usage_metadata: Option<UsageMetadata>,
        pub model_version: Option<String>,
    }

    /// A response candidate generated from the model.
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ContentCandidate {
        /// Output only. Generated content returned from the model.
        pub content: Content,
        /// Optional. Output only. The reason why the model stopped generating tokens.
        /// If empty, the model has not stopped generating tokens.
        pub finish_reason: Option<FinishReason>,
        /// List of ratings for the safety of a response candidate.
        /// There is at most one rating per category.
        pub safety_ratings: Option<Vec<SafetyRating>>,
        /// Output only. Citation information for model-generated candidate.
        /// This field may be populated with recitation information for any text included in the content.
        /// These are passages that are "recited" from copyrighted material in the foundational LLM's training data.
        pub citation_metadata: Option<CitationMetadata>,
        /// Output only. Token count for this candidate.
        pub token_count: Option<i32>,
        /// Output only.
        pub avg_logprobs: Option<f64>,
        /// Output only. Log-likelihood scores for the response tokens and top tokens
        pub logprobs_result: Option<LogprobsResult>,
        /// Output only. Index of the candidate in the list of response candidates.
        pub index: Option<i32>,
    }
    #[derive(Debug, Deserialize, Serialize)]
    pub struct Content {
        /// Ordered Parts that constitute a single message. Parts may have different MIME types.
        pub parts: Vec<Part>,
        /// The producer of the content. Must be either 'user' or 'model'.
        /// Useful to set for multi-turn conversations, otherwise can be left blank or unset.
        pub role: Option<Role>,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Model,
    }

    /// A datatype containing media that is part of a multi-part [Content] message.
    /// A Part consists of data which has an associated datatype. A Part can only contain one of the accepted types in Part.data.
    /// A Part must have a fixed IANA MIME type identifying the type and subtype of the media if the inlineData field is filled with raw bytes.
    #[derive(Debug, Default, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Part {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub inline_data: Option<Blob>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub function_call: Option<FunctionCall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub function_response: Option<FunctionResponse>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file_data: Option<FileData>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub executable_code: Option<ExecutableCode>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub code_execution_result: Option<CodeExecutionResult>,
    }

    /// Raw media bytes.
    /// Text should not be sent as raw bytes, use the 'text' field.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Blob {
        /// The IANA standard MIME type of the source data. Examples: - image/png - image/jpeg
        /// If an unsupported MIME type is provided, an error will be returned.
        pub mime_type: String,
        /// Raw bytes for media formats. A base64-encoded string.
        pub data: String,
    }

    /// A predicted FunctionCall returned from the model that contains a string representing the
    /// FunctionDeclaration.name with the arguments and their values.
    ///     #[derive(Debug, Deserialize, Serialize)]
    #[derive(Debug, Deserialize, Serialize)]
    pub struct FunctionCall {
        /// Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores
        /// and dashes, with a maximum length of 63.
        pub name: String,
        /// Optional. The function parameters and values in JSON object format.
        pub args: Option<Map<String, Value>>,
    }

    /// The result output from a FunctionCall that contains a string representing the FunctionDeclaration.name
    /// and a structured JSON object containing any output from the function is used as context to the model.
    /// This should contain the result of aFunctionCall made based on model prediction.
    #[derive(Debug, Deserialize, Serialize)]
    pub struct FunctionResponse {
        /// The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores and dashes,
        /// with a maximum length of 63.
        pub name: String,
        /// The function response in JSON object format.
        pub response: Option<HashMap<String, Value>>,
    }

    /// URI based data.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FileData {
        /// Optional. The IANA standard MIME type of the source data.
        pub mime_type: Option<String>,
        /// Required. URI.
        pub file_uri: String,
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
        pub cached_content_token_count: Option<i32>,
        pub candidates_token_count: i32,
        pub total_token_count: i32,
    }

    impl std::fmt::Display for UsageMetadata {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Prompt token count: {}\nCached content token count: {}\nCandidates token count: {}\nTotal token count: {}",
                self.prompt_token_count,
                match self.cached_content_token_count {
                    Some(count) => count.to_string(),
                    None => "n/a".to_string(),
                },
                self.candidates_token_count,
                self.total_token_count
            )
        }
    }

    /// A set of the feedback metadata the prompt specified in [GenerateContentRequest.contents](GenerateContentRequest).
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PromptFeedback {
        /// Optional. If set, the prompt was blocked and no candidates are returned. Rephrase the prompt.
        pub block_reason: Option<BlockReason>,
        /// Ratings for safety of the prompt. There is at most one rating per category.
        pub safety_ratings: Option<Vec<SafetyRating>>,
    }

    /// Reason why a prompt was blocked by the model
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum BlockReason {
        /// Default value. This value is unused.
        BlockReasonUnspecified,
        /// Prompt was blocked due to safety reasons. Inspect safetyRatings to understand which safety category blocked it.
        Safety,
        /// Prompt was blocked due to unknown reasons.
        Other,
        /// Prompt was blocked due to the terms which are included from the terminology blocklist.
        Blocklist,
        /// Prompt was blocked due to prohibited content.
        ProhibitedContent,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum FinishReason {
        /// Default value. This value is unused.
        FinishReasonUnspecified,
        /// Natural stop point of the model or provided stop sequence.
        Stop,
        /// The maximum number of tokens as specified in the request was reached.
        MaxTokens,
        /// The response candidate content was flagged for safety reasons.
        Safety,
        /// The response candidate content was flagged for recitation reasons.
        Recitation,
        /// The response candidate content was flagged for using an unsupported language.
        Language,
        /// Unknown reason.
        Other,
        /// Token generation stopped because the content contains forbidden terms.
        Blocklist,
        /// Token generation stopped for potentially containing prohibited content.
        ProhibitedContent,
        /// Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).
        Spii,
        /// The function call generated by the model is invalid.
        MalformedFunctionCall,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationMetadata {
        pub citation_sources: Vec<CitationSource>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationSource {
        pub uri: Option<String>,
        pub start_index: Option<i32>,
        pub end_index: Option<i32>,
        pub license: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
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
    /// configurable for every model. From [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
    /// ### Rig Note:
    /// Can be used to cosntruct a typesafe `additional_params` in rig::[AgentBuilder](crate::agent::AgentBuilder).
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
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
        /// This penalty is binary on/off and not dependent on the number of times the token is used (after the first).
        /// Use frequencyPenalty for a penalty that increases with each use. A positive penalty will discourage the use
        /// of tokens that have already been used in the response, increasing the vocabulary. A negative penalty will
        /// encourage the use of tokens that have already been used in the response, decreasing the vocabulary.
        pub presence_penalty: Option<f64>,
        /// Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been
        /// seen in the response so far. A positive penalty will discourage the use of tokens that have already been
        /// used, proportional to the number of times the token has been used: The more a token is used, the more
        /// difficult it is for the  model to use that token again increasing the vocabulary of responses. Caution: A
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
    /// From [Gemini API Reference](https://ai.google.dev/api/caching#Schema)
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
        /// From [Gemini API Reference](https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest)
        pub system_instruction: Option<Content>,
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
}
