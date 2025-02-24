// ================================================================
//! Google Gemini Completion Integration
//! From [Gemini API Reference](https://ai.google.dev/api/generate-content)
// ================================================================

/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
/// `gemini-1.5-flash` completion model
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro` completion model
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-pro-8b` completion model
pub const GEMINI_1_5_PRO_8B: &str = "gemini-1.5-pro-8b";
/// `gemini-1.0-pro` completion model
pub const GEMINI_1_0_PRO: &str = "gemini-1.0-pro";

use gemini_api_types::{
    Content, FunctionDeclaration, GenerateContentRequest, GenerateContentResponse,
    GenerationConfig, Part, Role, Tool,
};
use serde_json::{Map, Value};
use std::convert::TryFrom;

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    OneOrMany,
};

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

        full_history.push(completion_request.prompt_with_context());

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

        let system_instruction = completion_request.preamble.clone().map(|preamble| Content {
            parts: OneOrMany::one(preamble.into()),
            role: Some(Role::Model),
        });

        let request = GenerateContentRequest {
            contents: full_history
                .into_iter()
                .map(|msg| {
                    msg.try_into()
                        .map_err(|e| CompletionError::RequestError(Box::new(e)))
                })
                .collect::<Result<Vec<_>, _>>()?,
            generation_config: Some(generation_config),
            safety_settings: None,
            tools: Some(
                completion_request
                    .tools
                    .into_iter()
                    .map(Tool::try_from)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            tool_config: None,
            system_instruction,
        };

        tracing::debug!(
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        let response = self
            .client
            .post(&format!("/v1beta/models/{}:generateContent", self.model))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let response = response.json::<GenerateContentResponse>().await?;
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

            Ok(completion::CompletionResponse::try_from(response))
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }?
    }
}

impl TryFrom<completion::ToolDefinition> for Tool {
    type Error = CompletionError;

    fn try_from(tool: completion::ToolDefinition) -> Result<Self, Self::Error> {
        Ok(Self {
            function_declarations: FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters: Some(tool.parameters.try_into()?),
            },
            code_execution: None,
        })
    }
}

impl TryFrom<GenerateContentResponse> for completion::CompletionResponse<GenerateContentResponse> {
    type Error = CompletionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ResponseError("No response candidates in response".into())
        })?;

        let content = candidate
            .content
            .parts
            .iter()
            .map(|part| {
                Ok(match part {
                    Part::Text(text) => completion::AssistantContent::text(text),
                    Part::FunctionCall(function_call) => completion::AssistantContent::tool_call(
                        &function_call.name,
                        &function_call.name,
                        function_call.args.clone(),
                    ),
                    _ => {
                        return Err(CompletionError::ResponseError(
                            "Response did not contain a message or tool call".into(),
                        ))
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

pub mod gemini_api_types {
    use std::{collections::HashMap, convert::Infallible, str::FromStr};

    // =================================================================
    // Gemini API Types
    // =================================================================
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use crate::{
        completion::CompletionError,
        message::{self, MimeType as _},
        one_or_many::string_or_one_or_many,
        providers::gemini::gemini_api_types::{CodeExecutionResult, ExecutableCode},
        OneOrMany,
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
        #[serde(deserialize_with = "string_or_one_or_many")]
        pub parts: OneOrMany<Part>,
        /// The producer of the content. Must be either 'user' or 'model'.
        /// Useful to set for multi-turn conversations, otherwise can be left blank or unset.
        pub role: Option<Role>,
    }

    impl TryFrom<message::Message> for Content {
        type Error = message::MessageError;

        fn try_from(msg: message::Message) -> Result<Self, Self::Error> {
            Ok(match msg {
                message::Message::User { content } => Content {
                    parts: content.try_map(|c| c.try_into())?,
                    role: Some(Role::User),
                },
                message::Message::Assistant { content } => Content {
                    role: Some(Role::Model),
                    parts: content.map(|content| content.into()),
                },
            })
        }
    }

    impl TryFrom<Content> for message::Message {
        type Error = message::MessageError;

        fn try_from(content: Content) -> Result<Self, Self::Error> {
            match content.role {
                Some(Role::User) | None => Ok(message::Message::User {
                    content: content.parts.try_map(|part| {
                        Ok(match part {
                            Part::Text(text) => message::UserContent::text(text),
                            Part::InlineData(inline_data) => {
                                let mime_type =
                                    message::MediaType::from_mime_type(&inline_data.mime_type);

                                match mime_type {
                                    Some(message::MediaType::Image(media_type)) => {
                                        message::UserContent::image(
                                            inline_data.data,
                                            Some(message::ContentFormat::default()),
                                            Some(media_type),
                                            Some(message::ImageDetail::default()),
                                        )
                                    }
                                    Some(message::MediaType::Document(media_type)) => {
                                        message::UserContent::document(
                                            inline_data.data,
                                            Some(message::ContentFormat::default()),
                                            Some(media_type),
                                        )
                                    }
                                    Some(message::MediaType::Audio(media_type)) => {
                                        message::UserContent::audio(
                                            inline_data.data,
                                            Some(message::ContentFormat::default()),
                                            Some(media_type),
                                        )
                                    }
                                    _ => {
                                        return Err(message::MessageError::ConversionError(
                                            format!("Unsupported media type {:?}", mime_type),
                                        ))
                                    }
                                }
                            }
                            _ => {
                                return Err(message::MessageError::ConversionError(format!(
                                    "Unsupported gemini content part type: {:?}",
                                    part
                                )))
                            }
                        })
                    })?,
                }),
                Some(Role::Model) => Ok(message::Message::Assistant {
                    content: content.parts.try_map(|part| {
                        Ok(match part {
                            Part::Text(text) => message::AssistantContent::text(text),
                            Part::FunctionCall(function_call) => {
                                message::AssistantContent::ToolCall(function_call.into())
                            }
                            _ => {
                                return Err(message::MessageError::ConversionError(format!(
                                    "Unsupported part type: {:?}",
                                    part
                                )))
                            }
                        })
                    })?,
                }),
            }
        }
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Model,
    }

    /// A datatype containing media that is part of a multi-part [Content] message.
    /// A Part consists of data which has an associated datatype. A Part can only contain one of the accepted types in Part.data.
    /// A Part must have a fixed IANA MIME type identifying the type and subtype of the media if the inlineData field is filled with raw bytes.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub enum Part {
        Text(String),
        InlineData(Blob),
        FunctionCall(FunctionCall),
        FunctionResponse(FunctionResponse),
        FileData(FileData),
        ExecutableCode(ExecutableCode),
        CodeExecutionResult(CodeExecutionResult),
    }

    impl From<String> for Part {
        fn from(text: String) -> Self {
            Self::Text(text)
        }
    }

    impl From<&str> for Part {
        fn from(text: &str) -> Self {
            Self::Text(text.to_string())
        }
    }

    impl FromStr for Part {
        type Err = Infallible;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(s.into())
        }
    }

    impl TryFrom<message::UserContent> for Part {
        type Error = message::MessageError;

        fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
            match content {
                message::UserContent::Text(message::Text { text }) => Ok(Self::Text(text)),
                message::UserContent::ToolResult(message::ToolResult { id, content }) => {
                    let content = match content.first() {
                        message::ToolResultContent::Text(text) => text.text,
                        message::ToolResultContent::Image(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Tool result content must be text".to_string(),
                            ))
                        }
                    };
                    Ok(Part::FunctionResponse(FunctionResponse {
                        name: id,
                        response: Some(serde_json::from_str(&content).map_err(|e| {
                            message::MessageError::ConversionError(format!(
                                "Failed to parse tool response: {}",
                                e
                            ))
                        })?),
                    }))
                }
                message::UserContent::Image(message::Image {
                    data, media_type, ..
                }) => match media_type {
                    Some(media_type) => match media_type {
                        message::ImageMediaType::JPEG
                        | message::ImageMediaType::PNG
                        | message::ImageMediaType::WEBP
                        | message::ImageMediaType::HEIC
                        | message::ImageMediaType::HEIF => Ok(Self::InlineData(Blob {
                            mime_type: media_type.to_mime_type().to_owned(),
                            data,
                        })),
                        _ => Err(message::MessageError::ConversionError(format!(
                            "Unsupported image media type {:?}",
                            media_type
                        ))),
                    },
                    None => Err(message::MessageError::ConversionError(
                        "Media type for image is required for Anthropic".to_string(),
                    )),
                },
                message::UserContent::Document(message::Document {
                    data, media_type, ..
                }) => match media_type {
                    Some(media_type) => match media_type {
                        message::DocumentMediaType::PDF
                        | message::DocumentMediaType::TXT
                        | message::DocumentMediaType::RTF
                        | message::DocumentMediaType::HTML
                        | message::DocumentMediaType::CSS
                        | message::DocumentMediaType::MARKDOWN
                        | message::DocumentMediaType::CSV
                        | message::DocumentMediaType::XML => Ok(Self::InlineData(Blob {
                            mime_type: media_type.to_mime_type().to_owned(),
                            data,
                        })),
                        _ => Err(message::MessageError::ConversionError(format!(
                            "Unsupported document media type {:?}",
                            media_type
                        ))),
                    },
                    None => Err(message::MessageError::ConversionError(
                        "Media type for document is required for Anthropic".to_string(),
                    )),
                },
                message::UserContent::Audio(message::Audio {
                    data, media_type, ..
                }) => match media_type {
                    Some(media_type) => Ok(Self::InlineData(Blob {
                        mime_type: media_type.to_mime_type().to_owned(),
                        data,
                    })),
                    None => Err(message::MessageError::ConversionError(
                        "Media type for audio is required for Anthropic".to_string(),
                    )),
                },
            }
        }
    }

    impl From<message::AssistantContent> for Part {
        fn from(content: message::AssistantContent) -> Self {
            match content {
                message::AssistantContent::Text(message::Text { text }) => text.into(),
                message::AssistantContent::ToolCall(tool_call) => tool_call.into(),
            }
        }
    }

    impl From<message::ToolCall> for Part {
        fn from(tool_call: message::ToolCall) -> Self {
            Self::FunctionCall(FunctionCall {
                name: tool_call.function.name,
                args: tool_call.function.arguments,
            })
        }
    }

    /// Raw media bytes.
    /// Text should not be sent as raw bytes, use the 'text' field.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct FunctionCall {
        /// Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores
        /// and dashes, with a maximum length of 63.
        pub name: String,
        /// Optional. The function parameters and values in JSON object format.
        pub args: serde_json::Value,
    }

    impl From<FunctionCall> for message::ToolCall {
        fn from(function_call: FunctionCall) -> Self {
            Self {
                id: function_call.name.clone(),
                function: message::ToolFunction {
                    name: function_call.name,
                    arguments: function_call.args,
                },
            }
        }
    }

    impl From<message::ToolCall> for FunctionCall {
        fn from(tool_call: message::ToolCall) -> Self {
            Self {
                name: tool_call.function.name,
                args: tool_call.function.arguments,
            }
        }
    }

    /// The result output from a FunctionCall that contains a string representing the FunctionDeclaration.name
    /// and a structured JSON object containing any output from the function is used as context to the model.
    /// This should contain the result of aFunctionCall made based on model prediction.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct FunctionResponse {
        /// The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores and dashes,
        /// with a maximum length of 63.
        pub name: String,
        /// The function response in JSON object format.
        pub response: Option<HashMap<String, serde_json::Value>>,
    }

    /// URI based data.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct FileData {
        /// Optional. The IANA standard MIME type of the source data.
        pub mime_type: Option<String>,
        /// Required. URI.
        pub file_uri: String,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct SafetyRating {
        pub category: HarmCategory,
        pub probability: HarmProbability,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmProbability {
        HarmProbabilityUnspecified,
        Negligible,
        Low,
        Medium,
        High,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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
        #[serde(skip_serializing_if = "Option::is_none")]
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
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub start_index: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub end_index: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
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
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop_sequences: Option<Vec<String>>,
        /// MIME type of the generated candidate text. Supported MIME types are:
        ///     - text/plain:  (default) Text output
        ///     - application/json: JSON response in the response candidates.
        ///     - text/x.enum: ENUM as a string response in the response candidates.
        /// Refer to the docs for a list of all supported text MIME types
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_mime_type: Option<String>,
        /// Output schema of the generated candidate text. Schemas must be a subset of the OpenAPI schema and can be
        /// objects, primitives or arrays. If set, a compatible responseMimeType must also  be set. Compatible MIME
        /// types: application/json: Schema for JSON response. Refer to the JSON text generation guide for more details.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_schema: Option<Schema>,
        /// Number of generated responses to return. Currently, this value can only be set to 1. If
        /// unset, this will default to 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidate_count: Option<i32>,
        /// The maximum number of tokens to include in a response candidate. Note: The default value varies by model, see
        /// the Model.output_token_limit attribute of the Model returned from the getModel function.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_output_tokens: Option<u64>,
        /// Controls the randomness of the output. Note: The default value varies by model, see the Model.temperature
        /// attribute of the Model returned from the getModel function. Values can range from [0.0, 2.0].
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,
        /// The maximum cumulative probability of tokens to consider when sampling. The model uses combined Top-k and
        /// Top-p (nucleus) sampling. Tokens are sorted based on their assigned probabilities so that only the most
        /// likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while
        /// Nucleus sampling limits the number of tokens based on the cumulative probability. Note: The default value
        /// varies by Model and is specified by theModel.top_p attribute returned from the getModel function. An empty
        /// topK attribute indicates that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,
        /// The maximum number of tokens to consider when sampling. Gemini models use Top-p (nucleus) sampling or a
        /// combination of Top-k and nucleus sampling. Top-k sampling considers the set of topK most probable tokens.
        /// Models running with nucleus sampling don't allow topK setting. Note: The default value varies by Model and is
        /// specified by theModel.top_p attribute returned from the getModel function. An empty topK attribute indicates
        /// that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_k: Option<i32>,
        /// Presence penalty applied to the next token's logprobs if the token has already been seen in the response.
        /// This penalty is binary on/off and not dependent on the number of times the token is used (after the first).
        /// Use frequencyPenalty for a penalty that increases with each use. A positive penalty will discourage the use
        /// of tokens that have already been used in the response, increasing the vocabulary. A negative penalty will
        /// encourage the use of tokens that have already been used in the response, decreasing the vocabulary.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f64>,
        /// Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been
        /// seen in the response so far. A positive penalty will discourage the use of tokens that have already been
        /// used, proportional to the number of times the token has been used: The more a token is used, the more
        /// difficult it is for the  model to use that token again increasing the vocabulary of responses. Caution: A
        /// negative penalty will encourage the model to reuse tokens proportional to the number of times the token has
        /// been used. Small negative values will reduce the vocabulary of a response. Larger negative values will cause
        /// the model to  repeating a common token until it hits the maxOutputTokens limit: "...the the the the the...".
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f64>,
        /// If true, export the logprobs results in response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_logprobs: Option<bool>,
        /// Only valid if responseLogprobs=True. This sets the number of top logprobs to return at each decoding step in
        /// [Candidate.logprobs_result].
        #[serde(skip_serializing_if = "Option::is_none")]
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
        #[serde(skip_serializing_if = "Option::is_none")]
        pub format: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub nullable: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub r#enum: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_items: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub min_items: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub properties: Option<HashMap<String, Schema>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub required: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub items: Option<Box<Schema>>,
    }

    impl TryFrom<Value> for Schema {
        type Error = CompletionError;

        fn try_from(value: Value) -> Result<Self, Self::Error> {
            if let Some(obj) = value.as_object() {
                Ok(Schema {
                    r#type: obj
                        .get("type")
                        .and_then(|v| {
                            if v.is_string() {
                                v.as_str().map(String::from)
                            } else if v.is_array() {
                                v.as_array()
                                    .and_then(|arr| arr.first())
                                    .and_then(|v| v.as_str().map(String::from))
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default(),
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
        pub function_declarations: FunctionDeclaration,
        pub code_execution: Option<CodeExecution>,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionDeclaration {
        pub name: String,
        pub description: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parameters: Option<Schema>,
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

#[cfg(test)]
mod tests {
    use crate::message;

    use super::*;
    use serde_json::json;

    #[test]
    fn test_deserialize_message_user() {
        let raw_message = r#"{
            "parts": [
                {"text": "Hello, world!"},
                {"inlineData": {"mimeType": "image/png", "data": "base64encodeddata"}},
                {"functionCall": {"name": "test_function", "args": {"arg1": "value1"}}},
                {"functionResponse": {"name": "test_function", "response": {"result": "success"}}},
                {"fileData": {"mimeType": "application/pdf", "fileUri": "http://example.com/file.pdf"}},
                {"executableCode": {"code": "print('Hello, world!')", "language": "PYTHON"}},
                {"codeExecutionResult": {"output": "Hello, world!", "outcome": "OUTCOME_OK"}}
            ],
            "role": "user"
        }"#;

        let content: Content = {
            let jd = &mut serde_json::Deserializer::from_str(raw_message);
            serde_path_to_error::deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };
        assert_eq!(content.role, Some(Role::User));
        assert_eq!(content.parts.len(), 7);

        let parts: Vec<Part> = content.parts.into_iter().collect();

        if let Part::Text(text) = &parts[0] {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text part");
        }

        if let Part::InlineData(inline_data) = &parts[1] {
            assert_eq!(inline_data.mime_type, "image/png");
            assert_eq!(inline_data.data, "base64encodeddata");
        } else {
            panic!("Expected inline data part");
        }

        if let Part::FunctionCall(function_call) = &parts[2] {
            assert_eq!(function_call.name, "test_function");
            assert_eq!(
                function_call.args.as_object().unwrap().get("arg1").unwrap(),
                "value1"
            );
        } else {
            panic!("Expected function call part");
        }

        if let Part::FunctionResponse(function_response) = &parts[3] {
            assert_eq!(function_response.name, "test_function");
            assert_eq!(
                function_response
                    .response
                    .as_ref()
                    .unwrap()
                    .get("result")
                    .unwrap(),
                "success"
            );
        } else {
            panic!("Expected function response part");
        }

        if let Part::FileData(file_data) = &parts[4] {
            assert_eq!(file_data.mime_type.as_ref().unwrap(), "application/pdf");
            assert_eq!(file_data.file_uri, "http://example.com/file.pdf");
        } else {
            panic!("Expected file data part");
        }

        if let Part::ExecutableCode(executable_code) = &parts[5] {
            assert_eq!(executable_code.code, "print('Hello, world!')");
        } else {
            panic!("Expected executable code part");
        }

        if let Part::CodeExecutionResult(code_execution_result) = &parts[6] {
            assert_eq!(
                code_execution_result.clone().output.unwrap(),
                "Hello, world!"
            );
        } else {
            panic!("Expected code execution result part");
        }
    }

    #[test]
    fn test_deserialize_message_model() {
        let json_data = json!({
            "parts": [{"text": "Hello, user!"}],
            "role": "model"
        });

        let content: Content = serde_json::from_value(json_data).unwrap();
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
        if let Part::Text(text) = &content.parts.first() {
            assert_eq!(text, "Hello, user!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_message_conversion_user() {
        let msg = message::Message::user("Hello, world!");
        let content: Content = msg.try_into().unwrap();
        assert_eq!(content.role, Some(Role::User));
        assert_eq!(content.parts.len(), 1);
        if let Part::Text(text) = &content.parts.first() {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_message_conversion_model() {
        let msg = message::Message::assistant("Hello, user!");

        let content: Content = msg.try_into().unwrap();
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
        if let Part::Text(text) = &content.parts.first() {
            assert_eq!(text, "Hello, user!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_message_conversion_tool_call() {
        let tool_call = message::ToolCall {
            id: "test_tool".to_string(),
            function: message::ToolFunction {
                name: "test_function".to_string(),
                arguments: json!({"arg1": "value1"}),
            },
        };

        let msg = message::Message::Assistant {
            content: OneOrMany::one(message::AssistantContent::ToolCall(tool_call)),
        };

        let content: Content = msg.try_into().unwrap();
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
        if let Part::FunctionCall(function_call) = &content.parts.first() {
            assert_eq!(function_call.name, "test_function");
            assert_eq!(
                function_call.args.as_object().unwrap().get("arg1").unwrap(),
                "value1"
            );
        } else {
            panic!("Expected function call part");
        }
    }
}
