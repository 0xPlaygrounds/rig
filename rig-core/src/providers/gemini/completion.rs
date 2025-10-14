// ================================================================
//! Google Gemini Completion Integration
//! From [Gemini API Reference](https://ai.google.dev/api/generate-content)
// ================================================================
/// `gemini-2.5-pro-preview-06-05` completion model
pub const GEMINI_2_5_PRO_PREVIEW_06_05: &str = "gemini-2.5-pro-preview-06-05";
/// `gemini-2.5-pro-preview-05-06` completion model
pub const GEMINI_2_5_PRO_PREVIEW_05_06: &str = "gemini-2.5-pro-preview-05-06";
/// `gemini-2.5-pro-preview-03-25` completion model
pub const GEMINI_2_5_PRO_PREVIEW_03_25: &str = "gemini-2.5-pro-preview-03-25";
/// `gemini-2.5-flash-preview-05-20` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_05_20: &str = "gemini-2.5-flash-preview-05-20";
/// `gemini-2.5-flash-preview-04-17` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_04_17: &str = "gemini-2.5-flash-preview-04-17";
/// `gemini-2.5-pro-exp-03-25` experimental completion model
pub const GEMINI_2_5_PRO_EXP_03_25: &str = "gemini-2.5-pro-exp-03-25";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
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

use self::gemini_api_types::Schema;
use crate::message::Reasoning;
use crate::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, FunctionCallingMode, ToolConfig,
};
use crate::providers::gemini::streaming::StreamingCompletionResponse;
use crate::telemetry::SpanCombinator;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
};
use gemini_api_types::{
    Content, FunctionDeclaration, GenerateContentRequest, GenerateContentResponse, Part, PartKind,
    Role, Tool,
};
use serde_json::{Map, Value};
use std::convert::TryFrom;
use tracing::info_span;

use super::Client;

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = GenerateContentResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "generate_content",
                gen_ai.operation.name = "generate_content",
                gen_ai.provider.name = "gcp.gemini",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = create_request_body(completion_request)?;
        span.record_model_input(&request.contents);

        span.record_model_input(&request.contents);

        tracing::debug!(
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        let body = serde_json::to_vec(&request)?;

        let request = self
            .client
            .post(&format!("/v1beta/models/{}:generateContent", self.model))
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let response = self.client.send::<_, Vec<u8>>(request).await?;

        if response.status().is_success() {
            let response_body = response
                .into_body()
                .await
                .map_err(CompletionError::HttpError)?;

            let response: GenerateContentResponse = serde_json::from_slice(&response_body)?;

            match response.usage_metadata {
                Some(ref usage) => tracing::info!(target: "rig",
                "Gemini completion token usage: {}",
                usage
                ),
                None => tracing::info!(target: "rig",
                    "Gemini completion token usage: n/a",
                ),
            }

            let span = tracing::Span::current();
            span.record_model_output(&response.candidates);
            span.record_response_metadata(&response);
            span.record_token_usage(&response.usage_metadata);

            tracing::debug!(
                "Received response from Gemini API: {}",
                serde_json::to_string_pretty(&response)?
            );

            response.try_into()
        } else {
            let text = String::from_utf8_lossy(
                &response
                    .into_body()
                    .await
                    .map_err(CompletionError::HttpError)?,
            )
            .into();

            Err(CompletionError::ProviderError(text))
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
        CompletionModel::stream(self, request).await
    }
}

pub(crate) fn create_request_body(
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, CompletionError> {
    let mut full_history = Vec::new();
    full_history.extend(completion_request.chat_history);

    let additional_params = completion_request
        .additional_params
        .unwrap_or_else(|| Value::Object(Map::new()));

    let AdditionalParameters {
        mut generation_config,
        additional_params,
    } = serde_json::from_value::<AdditionalParameters>(additional_params)?;

    if let Some(temp) = completion_request.temperature {
        generation_config.temperature = Some(temp);
    }

    if let Some(max_tokens) = completion_request.max_tokens {
        generation_config.max_output_tokens = Some(max_tokens);
    }

    let system_instruction = completion_request.preamble.clone().map(|preamble| Content {
        parts: vec![preamble.into()],
        role: Some(Role::Model),
    });

    let tools = if completion_request.tools.is_empty() {
        None
    } else {
        Some(Tool::try_from(completion_request.tools)?)
    };

    let tool_config = if let Some(cfg) = completion_request.tool_choice {
        Some(ToolConfig {
            function_calling_config: Some(FunctionCallingMode::try_from(cfg)?),
        })
    } else {
        None
    };

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
        tools,
        tool_config,
        system_instruction,
        additional_params,
    };

    Ok(request)
}

impl TryFrom<completion::ToolDefinition> for Tool {
    type Error = CompletionError;

    fn try_from(tool: completion::ToolDefinition) -> Result<Self, Self::Error> {
        let parameters: Option<Schema> =
            if tool.parameters == serde_json::json!({"type": "object", "properties": {}}) {
                None
            } else {
                Some(tool.parameters.try_into()?)
            };

        Ok(Self {
            function_declarations: vec![FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters,
            }],
            code_execution: None,
        })
    }
}

impl TryFrom<Vec<completion::ToolDefinition>> for Tool {
    type Error = CompletionError;

    fn try_from(tools: Vec<completion::ToolDefinition>) -> Result<Self, Self::Error> {
        let mut function_declarations = Vec::new();

        for tool in tools {
            let parameters =
                if tool.parameters == serde_json::json!({"type": "object", "properties": {}}) {
                    None
                } else {
                    match tool.parameters.try_into() {
                        Ok(schema) => Some(schema),
                        Err(e) => {
                            let emsg = format!(
                                "Tool '{}' could not be converted to a schema: {:?}",
                                tool.name, e,
                            );
                            return Err(CompletionError::ProviderError(emsg));
                        }
                    }
                };

            function_declarations.push(FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters,
            });
        }

        Ok(Self {
            function_declarations,
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
            .map(|Part { thought, part, .. }| {
                Ok(match part {
                    PartKind::Text(text) => {
                        if let Some(thought) = thought
                            && *thought
                        {
                            completion::AssistantContent::Reasoning(Reasoning::new(text))
                        } else {
                            completion::AssistantContent::text(text)
                        }
                    }
                    PartKind::FunctionCall(function_call) => {
                        completion::AssistantContent::tool_call(
                            &function_call.name,
                            &function_call.name,
                            function_call.args.clone(),
                        )
                    }
                    _ => {
                        return Err(CompletionError::ResponseError(
                            "Response did not contain a message or tool call".into(),
                        ));
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_token_count as u64,
                output_tokens: usage.candidates_token_count.unwrap_or(0) as u64,
                total_tokens: usage.total_token_count as u64,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

pub mod gemini_api_types {
    use crate::telemetry::ProviderResponseExt;
    use std::{collections::HashMap, convert::Infallible, str::FromStr};

    // =================================================================
    // Gemini API Types
    // =================================================================
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    use crate::completion::GetTokenUsage;
    use crate::message::{DocumentSourceKind, ImageMediaType, MessageError, MimeType};
    use crate::{
        OneOrMany,
        completion::CompletionError,
        message::{self, Reasoning, Text},
        providers::gemini::gemini_api_types::{CodeExecutionResult, ExecutableCode},
    };

    #[derive(Debug, Deserialize, Serialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct AdditionalParameters {
        /// Change your Gemini request configuration.
        pub generation_config: GenerationConfig,
        /// Any additional parameters that you want.
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<serde_json::Value>,
    }

    impl AdditionalParameters {
        pub fn with_config(mut self, cfg: GenerationConfig) -> Self {
            self.generation_config = cfg;
            self
        }

        pub fn with_params(mut self, params: serde_json::Value) -> Self {
            self.additional_params = Some(params);
            self
        }
    }

    /// Response from the model supporting multiple candidate responses.
    /// Safety ratings and content filtering are reported for both prompt in GenerateContentResponse.prompt_feedback
    /// and for each candidate in finishReason and in safetyRatings.
    /// The API:
    ///     - Returns either all requested candidates or none of them
    ///     - Returns no candidates at all only if there was something wrong with the prompt (check promptFeedback)
    ///     - Reports feedback on each candidate in finishReason and safetyRatings.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentResponse {
        pub response_id: String,
        /// Candidate responses from the model.
        pub candidates: Vec<ContentCandidate>,
        /// Returns the prompt's feedback related to the content filters.
        pub prompt_feedback: Option<PromptFeedback>,
        /// Output only. Metadata on the generation requests' token usage.
        pub usage_metadata: Option<UsageMetadata>,
        pub model_version: Option<String>,
    }

    impl ProviderResponseExt for GenerateContentResponse {
        type OutputMessage = ContentCandidate;
        type Usage = UsageMetadata;

        fn get_response_id(&self) -> Option<String> {
            Some(self.response_id.clone())
        }

        fn get_response_model_name(&self) -> Option<String> {
            None
        }

        fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
            self.candidates.clone()
        }

        fn get_text_response(&self) -> Option<String> {
            let str = self
                .candidates
                .iter()
                .filter_map(|x| {
                    if x.content.role.as_ref().is_none_or(|y| y != &Role::Model) {
                        return None;
                    }

                    let res = x
                        .content
                        .parts
                        .iter()
                        .filter_map(|part| {
                            if let PartKind::Text(ref str) = part.part {
                                Some(str.to_owned())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<String>>()
                        .join("\n");

                    Some(res)
                })
                .collect::<Vec<String>>()
                .join("\n");

            if str.is_empty() { None } else { Some(str) }
        }

        fn get_usage(&self) -> Option<Self::Usage> {
            self.usage_metadata.clone()
        }
    }

    /// A response candidate generated from the model.
    #[derive(Clone, Debug, Deserialize, Serialize)]
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

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Content {
        /// Ordered Parts that constitute a single message. Parts may have different MIME types.
        #[serde(default)]
        pub parts: Vec<Part>,
        /// The producer of the content. Must be either 'user' or 'model'.
        /// Useful to set for multi-turn conversations, otherwise can be left blank or unset.
        pub role: Option<Role>,
    }

    impl TryFrom<message::Message> for Content {
        type Error = message::MessageError;

        fn try_from(msg: message::Message) -> Result<Self, Self::Error> {
            Ok(match msg {
                message::Message::User { content } => Content {
                    parts: content
                        .into_iter()
                        .map(|c| c.try_into())
                        .collect::<Result<Vec<_>, _>>()?,
                    role: Some(Role::User),
                },
                message::Message::Assistant { content, .. } => Content {
                    role: Some(Role::Model),
                    parts: content.into_iter().map(|content| content.into()).collect(),
                },
            })
        }
    }

    impl TryFrom<Content> for message::Message {
        type Error = message::MessageError;

        fn try_from(content: Content) -> Result<Self, Self::Error> {
            match content.role {
                Some(Role::User) | None => {
                    Ok(message::Message::User {
                        content: {
                            let user_content: Result<Vec<_>, _> = content.parts.into_iter()
                            .map(|Part { part, .. }| {
                                Ok(match part {
                                    PartKind::Text(text) => message::UserContent::text(text),
                                    PartKind::InlineData(inline_data) => {
                                        let mime_type =
                                            message::MediaType::from_mime_type(&inline_data.mime_type);

                                        match mime_type {
                                            Some(message::MediaType::Image(media_type)) => {
                                                message::UserContent::image_base64(
                                                    inline_data.data,
                                                    Some(media_type),
                                                    Some(message::ImageDetail::default()),
                                                )
                                            }
                                            Some(message::MediaType::Document(media_type)) => {
                                                message::UserContent::document(
                                                    inline_data.data,
                                                    Some(media_type),
                                                )
                                            }
                                            Some(message::MediaType::Audio(media_type)) => {
                                                message::UserContent::audio(
                                                    inline_data.data,
                                                    Some(media_type),
                                                )
                                            }
                                            _ => {
                                                return Err(message::MessageError::ConversionError(
                                                    format!("Unsupported media type {mime_type:?}"),
                                                ));
                                            }
                                        }
                                    }
                                    _ => {
                                        return Err(message::MessageError::ConversionError(format!(
                                            "Unsupported gemini content part type: {part:?}"
                                        )));
                                    }
                                })
                            })
                            .collect();
                            OneOrMany::many(user_content?).map_err(|_| {
                                message::MessageError::ConversionError(
                                    "Failed to create OneOrMany from user content".to_string(),
                                )
                            })?
                        },
                    })
                }
                Some(Role::Model) => Ok(message::Message::Assistant {
                    id: None,
                    content: {
                        let assistant_content: Result<Vec<_>, _> = content
                            .parts
                            .into_iter()
                            .map(|Part { thought, part, .. }| {
                                Ok(match part {
                                    PartKind::Text(text) => match thought {
                                        Some(true) => message::AssistantContent::Reasoning(
                                            Reasoning::new(&text),
                                        ),
                                        _ => message::AssistantContent::Text(Text { text }),
                                    },

                                    PartKind::FunctionCall(function_call) => {
                                        message::AssistantContent::ToolCall(function_call.into())
                                    }
                                    _ => {
                                        return Err(message::MessageError::ConversionError(
                                            format!("Unsupported part type: {part:?}"),
                                        ));
                                    }
                                })
                            })
                            .collect();
                        OneOrMany::many(assistant_content?).map_err(|_| {
                            message::MessageError::ConversionError(
                                "Failed to create OneOrMany from assistant content".to_string(),
                            )
                        })?
                    },
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

    #[derive(Debug, Default, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct Part {
        /// whether or not the part is a reasoning/thinking text or not
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thought: Option<bool>,
        /// an opaque sig for the thought so it can be reused - is a base64 string
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thought_signature: Option<String>,
        #[serde(flatten)]
        pub part: PartKind,
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<Value>,
    }

    /// A datatype containing media that is part of a multi-part [Content] message.
    /// A Part consists of data which has an associated datatype. A Part can only contain one of the accepted types in Part.data.
    /// A Part must have a fixed IANA MIME type identifying the type and subtype of the media if the inlineData field is filled with raw bytes.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub enum PartKind {
        Text(String),
        InlineData(Blob),
        FunctionCall(FunctionCall),
        FunctionResponse(FunctionResponse),
        FileData(FileData),
        ExecutableCode(ExecutableCode),
        CodeExecutionResult(CodeExecutionResult),
    }

    // This default instance is primarily so we can easily fill in the optional fields of `Part`
    // So this instance for `PartKind` (and the allocation it would cause) should be optimized away
    impl Default for PartKind {
        fn default() -> Self {
            Self::Text(String::new())
        }
    }

    impl From<String> for Part {
        fn from(text: String) -> Self {
            Self {
                thought: Some(false),
                thought_signature: None,
                part: PartKind::Text(text),
                additional_params: None,
            }
        }
    }

    impl From<&str> for Part {
        fn from(text: &str) -> Self {
            Self::from(text.to_string())
        }
    }

    impl FromStr for Part {
        type Err = Infallible;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(s.into())
        }
    }

    impl TryFrom<(ImageMediaType, DocumentSourceKind)> for PartKind {
        type Error = message::MessageError;
        fn try_from(
            (mime_type, doc_src): (ImageMediaType, DocumentSourceKind),
        ) -> Result<Self, Self::Error> {
            let mime_type = mime_type.to_mime_type().to_string();
            let part = match doc_src {
                DocumentSourceKind::Url(url) => PartKind::FileData(FileData {
                    mime_type: Some(mime_type),
                    file_uri: url,
                }),
                DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data) => {
                    PartKind::InlineData(Blob { mime_type, data })
                }
                DocumentSourceKind::Raw(_) => {
                    return Err(message::MessageError::ConversionError(
                        "Raw files not supported, encode as base64 first".into(),
                    ));
                }
                DocumentSourceKind::Unknown => {
                    return Err(message::MessageError::ConversionError(
                        "Can't convert an unknown document source".to_string(),
                    ));
                }
            };

            Ok(part)
        }
    }

    impl TryFrom<message::UserContent> for Part {
        type Error = message::MessageError;

        fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
            match content {
                message::UserContent::Text(message::Text { text }) => Ok(Part {
                    thought: Some(false),
                    thought_signature: None,
                    part: PartKind::Text(text),
                    additional_params: None,
                }),
                message::UserContent::ToolResult(message::ToolResult { id, content, .. }) => {
                    let content = match content.first() {
                        message::ToolResultContent::Text(text) => text.text,
                        message::ToolResultContent::Image(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Tool result content must be text".to_string(),
                            ));
                        }
                    };
                    // Convert to JSON since this value may be a valid JSON value
                    let result: serde_json::Value =
                        serde_json::from_str(&content).unwrap_or_else(|error| {
                            tracing::trace!(
                                ?error,
                                "Tool result is not a valid JSON, treat it as normal string"
                            );
                            json!(content)
                        });
                    Ok(Part {
                        thought: Some(false),
                        thought_signature: None,
                        part: PartKind::FunctionResponse(FunctionResponse {
                            name: id,
                            response: Some(json!({ "result": result })),
                        }),
                        additional_params: None,
                    })
                }
                message::UserContent::Image(message::Image {
                    data, media_type, ..
                }) => match media_type {
                    Some(media_type) => match media_type {
                        message::ImageMediaType::JPEG
                        | message::ImageMediaType::PNG
                        | message::ImageMediaType::WEBP
                        | message::ImageMediaType::HEIC
                        | message::ImageMediaType::HEIF => {
                            let part = PartKind::try_from((media_type, data))?;
                            Ok(Part {
                                thought: Some(false),
                                thought_signature: None,
                                part,
                                additional_params: None,
                            })
                        }
                        _ => Err(message::MessageError::ConversionError(format!(
                            "Unsupported image media type {media_type:?}"
                        ))),
                    },
                    None => Err(message::MessageError::ConversionError(
                        "Media type for image is required for Gemini".to_string(),
                    )),
                },
                message::UserContent::Document(message::Document {
                    data, media_type, ..
                }) => {
                    let Some(media_type) = media_type else {
                        return Err(MessageError::ConversionError(
                            "A mime type is required for document inputs to Gemini".to_string(),
                        ));
                    };

                    if !media_type.is_code() {
                        let mime_type = media_type.to_mime_type().to_string();

                        let part = match data {
                            DocumentSourceKind::Url(file_uri) => PartKind::FileData(FileData {
                                mime_type: Some(mime_type),
                                file_uri,
                            }),
                            DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data) => {
                                PartKind::InlineData(Blob { mime_type, data })
                            }
                            DocumentSourceKind::Raw(_) => {
                                return Err(message::MessageError::ConversionError(
                                    "Raw files not supported, encode as base64 first".into(),
                                ));
                            }
                            _ => {
                                return Err(message::MessageError::ConversionError(
                                    "Document has no body".to_string(),
                                ));
                            }
                        };

                        Ok(Part {
                            thought: Some(false),
                            part,
                            ..Default::default()
                        })
                    } else {
                        Err(message::MessageError::ConversionError(format!(
                            "Unsupported document media type {media_type:?}"
                        )))
                    }
                }

                message::UserContent::Audio(message::Audio {
                    data, media_type, ..
                }) => {
                    let Some(media_type) = media_type else {
                        return Err(MessageError::ConversionError(
                            "A mime type is required for audio inputs to Gemini".to_string(),
                        ));
                    };

                    let mime_type = media_type.to_mime_type().to_string();

                    let part = match data {
                        DocumentSourceKind::Base64(data) => {
                            PartKind::InlineData(Blob { data, mime_type })
                        }

                        DocumentSourceKind::Url(file_uri) => PartKind::FileData(FileData {
                            mime_type: Some(mime_type),
                            file_uri,
                        }),
                        DocumentSourceKind::String(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Strings cannot be used as audio files!".into(),
                            ));
                        }
                        DocumentSourceKind::Raw(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Raw files not supported, encode as base64 first".into(),
                            ));
                        }
                        DocumentSourceKind::Unknown => {
                            return Err(message::MessageError::ConversionError(
                                "Content has no body".to_string(),
                            ));
                        }
                    };

                    Ok(Part {
                        thought: Some(false),
                        part,
                        ..Default::default()
                    })
                }
                message::UserContent::Video(message::Video {
                    data,
                    media_type,
                    additional_params,
                    ..
                }) => {
                    let mime_type = media_type.map(|media_ty| media_ty.to_mime_type().to_string());

                    let part = match data {
                        DocumentSourceKind::Url(file_uri) => {
                            if file_uri.starts_with("https://www.youtube.com") {
                                PartKind::FileData(FileData {
                                    mime_type,
                                    file_uri,
                                })
                            } else {
                                if mime_type.is_none() {
                                    return Err(MessageError::ConversionError(
                                        "A mime type is required for non-Youtube video file inputs to Gemini"
                                            .to_string(),
                                    ));
                                }

                                PartKind::FileData(FileData {
                                    mime_type,
                                    file_uri,
                                })
                            }
                        }
                        DocumentSourceKind::Base64(data) => {
                            let Some(mime_type) = mime_type else {
                                return Err(MessageError::ConversionError(
                                    "A media type is expected for base64 encoded strings"
                                        .to_string(),
                                ));
                            };
                            PartKind::InlineData(Blob { mime_type, data })
                        }
                        DocumentSourceKind::String(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Strings cannot be used as audio files!".into(),
                            ));
                        }
                        DocumentSourceKind::Raw(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Raw file data not supported, encode as base64 first".into(),
                            ));
                        }
                        DocumentSourceKind::Unknown => {
                            return Err(message::MessageError::ConversionError(
                                "Media type for video is required for Gemini".to_string(),
                            ));
                        }
                    };

                    Ok(Part {
                        thought: Some(false),
                        thought_signature: None,
                        part,
                        additional_params,
                    })
                }
            }
        }
    }

    impl From<message::AssistantContent> for Part {
        fn from(content: message::AssistantContent) -> Self {
            match content {
                message::AssistantContent::Text(message::Text { text }) => text.into(),
                message::AssistantContent::ToolCall(tool_call) => tool_call.into(),
                message::AssistantContent::Reasoning(message::Reasoning { reasoning, .. }) => {
                    Part {
                        thought: Some(true),
                        thought_signature: None,
                        part: PartKind::Text(
                            reasoning.first().cloned().unwrap_or_else(|| "".to_string()),
                        ),
                        additional_params: None,
                    }
                }
            }
        }
    }

    impl From<message::ToolCall> for Part {
        fn from(tool_call: message::ToolCall) -> Self {
            Self {
                thought: Some(false),
                thought_signature: None,
                part: PartKind::FunctionCall(FunctionCall {
                    name: tool_call.function.name,
                    args: tool_call.function.arguments,
                }),
                additional_params: None,
            }
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
                call_id: None,
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
        pub response: Option<serde_json::Value>,
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

    #[derive(Debug, Deserialize, Clone, Default, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct UsageMetadata {
        pub prompt_token_count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cached_content_token_count: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidates_token_count: Option<i32>,
        pub total_token_count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thoughts_token_count: Option<i32>,
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
                match self.candidates_token_count {
                    Some(count) => count.to_string(),
                    None => "n/a".to_string(),
                },
                self.total_token_count
            )
        }
    }

    impl GetTokenUsage for UsageMetadata {
        fn token_usage(&self) -> Option<crate::completion::Usage> {
            let mut usage = crate::completion::Usage::new();

            usage.input_tokens = self.prompt_token_count as u64;
            usage.output_tokens = (self.cached_content_token_count.unwrap_or_default()
                + self.candidates_token_count.unwrap_or_default()
                + self.thoughts_token_count.unwrap_or_default())
                as u64;
            usage.total_tokens = usage.input_tokens + usage.output_tokens;

            Some(usage)
        }
    }

    /// A set of the feedback metadata the prompt specified in [GenerateContentRequest.contents](GenerateContentRequest).
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PromptFeedback {
        /// Optional. If set, the prompt was blocked and no candidates are returned. Rephrase the prompt.
        pub block_reason: Option<BlockReason>,
        /// Ratings for safety of the prompt. There is at most one rating per category.
        pub safety_ratings: Option<Vec<SafetyRating>>,
    }

    /// Reason why a prompt was blocked by the model
    #[derive(Debug, Deserialize, Serialize)]
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

    #[derive(Clone, Debug, Deserialize, Serialize)]
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

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationMetadata {
        pub citation_sources: Vec<CitationSource>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
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

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LogprobsResult {
        pub top_candidate: Vec<TopCandidate>,
        pub chosen_candidate: Vec<LogProbCandidate>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct TopCandidate {
        pub candidates: Vec<LogProbCandidate>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LogProbCandidate {
        pub token: String,
        pub token_id: String,
        pub log_probability: f64,
    }

    /// Gemini API Configuration options for model generation and outputs. Not all parameters are
    /// configurable for every model. From [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
    /// ### Rig Note:
    /// Can be used to construct a typesafe `additional_params` in rig::[AgentBuilder](crate::agent::AgentBuilder).
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
        /// Configuration for thinking/reasoning.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_config: Option<ThinkingConfig>,
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
                thinking_config: None,
            }
        }
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ThinkingConfig {
        pub thinking_budget: u32,
        pub include_thoughts: Option<bool>,
    }
    /// The Schema object allows the definition of input and output data types. These types can be objects, but also
    /// primitives and arrays. Represents a select subset of an OpenAPI 3.0 schema object.
    /// From [Gemini API Reference](https://ai.google.dev/api/caching#Schema)
    #[derive(Debug, Deserialize, Serialize, Clone)]
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

    /// Flattens a JSON schema by resolving all `$ref` references inline.
    /// It takes a JSON schema that may contain `$ref` references to definitions
    /// in `$defs` or `definitions` sections and returns a new schema with all references
    /// resolved and inlined. This is necessary for APIs like Gemini that don't support
    /// schema references.
    pub fn flatten_schema(mut schema: Value) -> Result<Value, CompletionError> {
        // extracting $defs if they exist
        let defs = if let Some(obj) = schema.as_object() {
            obj.get("$defs").or_else(|| obj.get("definitions")).cloned()
        } else {
            None
        };

        let Some(defs_value) = defs else {
            return Ok(schema);
        };

        let Some(defs_obj) = defs_value.as_object() else {
            return Err(CompletionError::ResponseError(
                "$defs must be an object".into(),
            ));
        };

        resolve_refs(&mut schema, defs_obj)?;

        // removing $defs from the final schema because we have inlined everything
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$defs");
            obj.remove("definitions");
        }

        Ok(schema)
    }

    /// Recursively resolves all `$ref` references in a JSON value by
    /// replacing them with their definitions.
    fn resolve_refs(
        value: &mut Value,
        defs: &serde_json::Map<String, Value>,
    ) -> Result<(), CompletionError> {
        match value {
            Value::Object(obj) => {
                if let Some(ref_value) = obj.get("$ref")
                    && let Some(ref_str) = ref_value.as_str()
                {
                    // "#/$defs/Person" -> "Person"
                    let def_name = parse_ref_path(ref_str)?;

                    let def = defs.get(&def_name).ok_or_else(|| {
                        CompletionError::ResponseError(format!("Reference not found: {}", ref_str))
                    })?;

                    let mut resolved = def.clone();
                    resolve_refs(&mut resolved, defs)?;
                    *value = resolved;
                    return Ok(());
                }

                for (_, v) in obj.iter_mut() {
                    resolve_refs(v, defs)?;
                }
            }
            Value::Array(arr) => {
                for item in arr.iter_mut() {
                    resolve_refs(item, defs)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Parses a JSON Schema `$ref` path to extract the definition name.
    ///
    /// JSON Schema references use URI fragment syntax to point to definitions within
    /// the same document. This function extracts the definition name from common
    /// reference patterns used in JSON Schema.
    fn parse_ref_path(ref_str: &str) -> Result<String, CompletionError> {
        if let Some(fragment) = ref_str.strip_prefix('#') {
            if let Some(name) = fragment.strip_prefix("/$defs/") {
                Ok(name.to_string())
            } else if let Some(name) = fragment.strip_prefix("/definitions/") {
                Ok(name.to_string())
            } else {
                Err(CompletionError::ResponseError(format!(
                    "Unsupported reference format: {}",
                    ref_str
                )))
            }
        } else {
            Err(CompletionError::ResponseError(format!(
                "Only fragment references (#/...) are supported: {}",
                ref_str
            )))
        }
    }

    /// Helper function to extract the type string from a JSON value.
    /// Handles both direct string types and array types (returns the first element).
    fn extract_type(type_value: &Value) -> Option<String> {
        if type_value.is_string() {
            type_value.as_str().map(String::from)
        } else if type_value.is_array() {
            type_value
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str().map(String::from))
        } else {
            None
        }
    }

    /// Helper function to extract type from anyOf, oneOf, or allOf schemas.
    /// Returns the type of the first non-null schema found.
    fn extract_type_from_composition(composition: &Value) -> Option<String> {
        composition.as_array().and_then(|arr| {
            arr.iter().find_map(|schema| {
                if let Some(obj) = schema.as_object() {
                    // Skip null types
                    if let Some(type_val) = obj.get("type")
                        && let Some(type_str) = type_val.as_str()
                        && type_str == "null"
                    {
                        return None;
                    }
                    // Extract type from this schema
                    obj.get("type").and_then(extract_type).or_else(|| {
                        if obj.contains_key("properties") {
                            Some("object".to_string())
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
        })
    }

    /// Helper function to extract the first non-null schema from anyOf, oneOf, or allOf.
    /// Returns the schema object that should be used for properties, required, etc.
    fn extract_schema_from_composition(
        composition: &Value,
    ) -> Option<serde_json::Map<String, Value>> {
        composition.as_array().and_then(|arr| {
            arr.iter().find_map(|schema| {
                if let Some(obj) = schema.as_object()
                    && let Some(type_val) = obj.get("type")
                    && let Some(type_str) = type_val.as_str()
                {
                    if type_str == "null" {
                        return None;
                    }
                    Some(obj.clone())
                } else {
                    None
                }
            })
        })
    }

    /// Helper function to infer the type of a schema object.
    /// Checks for explicit type, then anyOf/oneOf/allOf, then infers from properties.
    fn infer_type(obj: &serde_json::Map<String, Value>) -> String {
        // First, try direct type field
        if let Some(type_val) = obj.get("type")
            && let Some(type_str) = extract_type(type_val)
        {
            return type_str;
        }

        // Then try anyOf, oneOf, allOf (in that order)
        if let Some(any_of) = obj.get("anyOf")
            && let Some(type_str) = extract_type_from_composition(any_of)
        {
            return type_str;
        }

        if let Some(one_of) = obj.get("oneOf")
            && let Some(type_str) = extract_type_from_composition(one_of)
        {
            return type_str;
        }

        if let Some(all_of) = obj.get("allOf")
            && let Some(type_str) = extract_type_from_composition(all_of)
        {
            return type_str;
        }

        // Finally, infer object type if properties are present
        if obj.contains_key("properties") {
            "object".to_string()
        } else {
            String::new()
        }
    }

    impl TryFrom<Value> for Schema {
        type Error = CompletionError;

        fn try_from(value: Value) -> Result<Self, Self::Error> {
            let flattened_val = flatten_schema(value)?;
            if let Some(obj) = flattened_val.as_object() {
                // Determine which object to use for extracting properties and required fields.
                // If this object has anyOf/oneOf/allOf, we need to extract properties from the composition.
                let props_source = if obj.get("properties").is_none() {
                    if let Some(any_of) = obj.get("anyOf") {
                        extract_schema_from_composition(any_of)
                    } else if let Some(one_of) = obj.get("oneOf") {
                        extract_schema_from_composition(one_of)
                    } else if let Some(all_of) = obj.get("allOf") {
                        extract_schema_from_composition(all_of)
                    } else {
                        None
                    }
                    .unwrap_or(obj.clone())
                } else {
                    obj.clone()
                };

                Ok(Schema {
                    r#type: infer_type(obj),
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
                    properties: props_source
                        .get("properties")
                        .and_then(|v| v.as_object())
                        .map(|map| {
                            map.iter()
                                .filter_map(|(k, v)| {
                                    v.clone().try_into().ok().map(|schema| (k.clone(), schema))
                                })
                                .collect()
                        }),
                    required: props_source
                        .get("required")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        }),
                    items: obj
                        .get("items")
                        .and_then(|v| v.clone().try_into().ok())
                        .map(Box::new),
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
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Tool>,
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
        /// Additional parameters.
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<serde_json::Value>,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Tool {
        pub function_declarations: Vec<FunctionDeclaration>,
        pub code_execution: Option<CodeExecution>,
    }

    #[derive(Debug, Serialize, Clone)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionDeclaration {
        pub name: String,
        pub description: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parameters: Option<Schema>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ToolConfig {
        pub function_calling_config: Option<FunctionCallingMode>,
    }

    #[derive(Debug, Serialize, Deserialize, Default)]
    #[serde(tag = "mode", rename_all = "UPPERCASE")]
    pub enum FunctionCallingMode {
        #[default]
        Auto,
        None,
        Any {
            #[serde(skip_serializing_if = "Option::is_none")]
            allowed_function_names: Option<Vec<String>>,
        },
    }

    impl TryFrom<message::ToolChoice> for FunctionCallingMode {
        type Error = CompletionError;
        fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
            let res = match value {
                message::ToolChoice::Auto => Self::Auto,
                message::ToolChoice::None => Self::None,
                message::ToolChoice::Required => Self::Any {
                    allowed_function_names: None,
                },
                message::ToolChoice::Specific { function_names } => Self::Any {
                    allowed_function_names: Some(function_names),
                },
            };

            Ok(res)
        }
    }

    #[derive(Debug, Serialize)]
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
    use crate::{message, providers::gemini::completion::gemini_api_types::flatten_schema};

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

        if let Part {
            part: PartKind::Text(text),
            ..
        } = &parts[0]
        {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text part");
        }

        if let Part {
            part: PartKind::InlineData(inline_data),
            ..
        } = &parts[1]
        {
            assert_eq!(inline_data.mime_type, "image/png");
            assert_eq!(inline_data.data, "base64encodeddata");
        } else {
            panic!("Expected inline data part");
        }

        if let Part {
            part: PartKind::FunctionCall(function_call),
            ..
        } = &parts[2]
        {
            assert_eq!(function_call.name, "test_function");
            assert_eq!(
                function_call.args.as_object().unwrap().get("arg1").unwrap(),
                "value1"
            );
        } else {
            panic!("Expected function call part");
        }

        if let Part {
            part: PartKind::FunctionResponse(function_response),
            ..
        } = &parts[3]
        {
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

        if let Part {
            part: PartKind::FileData(file_data),
            ..
        } = &parts[4]
        {
            assert_eq!(file_data.mime_type.as_ref().unwrap(), "application/pdf");
            assert_eq!(file_data.file_uri, "http://example.com/file.pdf");
        } else {
            panic!("Expected file data part");
        }

        if let Part {
            part: PartKind::ExecutableCode(executable_code),
            ..
        } = &parts[5]
        {
            assert_eq!(executable_code.code, "print('Hello, world!')");
        } else {
            panic!("Expected executable code part");
        }

        if let Part {
            part: PartKind::CodeExecutionResult(code_execution_result),
            ..
        } = &parts[6]
        {
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
        if let Some(Part {
            part: PartKind::Text(text),
            ..
        }) = content.parts.first()
        {
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
        if let Some(Part {
            part: PartKind::Text(text),
            ..
        }) = &content.parts.first()
        {
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
        if let Some(Part {
            part: PartKind::Text(text),
            ..
        }) = &content.parts.first()
        {
            assert_eq!(text, "Hello, user!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_message_conversion_tool_call() {
        let tool_call = message::ToolCall {
            id: "test_tool".to_string(),
            call_id: None,
            function: message::ToolFunction {
                name: "test_function".to_string(),
                arguments: json!({"arg1": "value1"}),
            },
        };

        let msg = message::Message::Assistant {
            id: None,
            content: OneOrMany::one(message::AssistantContent::ToolCall(tool_call)),
        };

        let content: Content = msg.try_into().unwrap();
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
        if let Some(Part {
            part: PartKind::FunctionCall(function_call),
            ..
        }) = content.parts.first()
        {
            assert_eq!(function_call.name, "test_function");
            assert_eq!(
                function_call.args.as_object().unwrap().get("arg1").unwrap(),
                "value1"
            );
        } else {
            panic!("Expected function call part");
        }
    }

    #[test]
    fn test_vec_schema_conversion() {
        let schema_with_ref = json!({
            "type": "array",
            "items": {
                "$ref": "#/$defs/Person"
            },
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": ["string", "null"],
                            "description": "The person's first name, if provided (null otherwise)"
                        },
                        "last_name": {
                            "type": ["string", "null"],
                            "description": "The person's last name, if provided (null otherwise)"
                        },
                        "job": {
                            "type": ["string", "null"],
                            "description": "The person's job, if provided (null otherwise)"
                        }
                    },
                    "required": []
                }
            }
        });

        let result: Result<Schema, _> = schema_with_ref.try_into();

        match result {
            Ok(schema) => {
                assert_eq!(schema.r#type, "array");

                if let Some(items) = schema.items {
                    println!("item types: {}", items.r#type);

                    assert_ne!(items.r#type, "", "Items type should not be empty string!");
                    assert_eq!(items.r#type, "object", "Items should be object type");
                } else {
                    panic!("Schema should have items field for array type");
                }
            }
            Err(e) => println!("Schema conversion failed: {:?}", e),
        }
    }

    #[test]
    fn test_object_schema() {
        let simple_schema = json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            }
        });

        let schema: Schema = simple_schema.try_into().unwrap();
        assert_eq!(schema.r#type, "object");
        assert!(schema.properties.is_some());
    }

    #[test]
    fn test_array_with_inline_items() {
        let inline_schema = json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    }
                }
            }
        });

        let schema: Schema = inline_schema.try_into().unwrap();
        assert_eq!(schema.r#type, "array");

        if let Some(items) = schema.items {
            assert_eq!(items.r#type, "object");
            assert!(items.properties.is_some());
        } else {
            panic!("Schema should have items field");
        }
    }
    #[test]
    fn test_flattened_schema() {
        let ref_schema = json!({
            "type": "array",
            "items": {
                "$ref": "#/$defs/Person"
            },
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            }
        });

        let flattened = flatten_schema(ref_schema).unwrap();
        let schema: Schema = flattened.try_into().unwrap();

        assert_eq!(schema.r#type, "array");

        if let Some(items) = schema.items {
            println!("Flattened items type: '{}'", items.r#type);

            assert_eq!(items.r#type, "object");
            assert!(items.properties.is_some());
        }
    }
}
