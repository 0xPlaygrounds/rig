// ================================================================
//! Google Gemini gRPC Completion Integration
// ================================================================

/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::message::{self, MimeType, Reasoning};
use crate::telemetry::ProviderResponseExt;
use std::convert::TryFrom;

use super::Client;
use super::proto::{self, GenerateContentRequest, GenerateContentResponse};

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone, Debug)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = GenerateContentResponse;
    type StreamingResponse = super::streaming::StreamingCompletionResponse;
    type Client = super::Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError> {
        let request = create_grpc_request(self.model.clone(), completion_request)?;

        let mut grpc_client = self
            .client
            .ext()
            .grpc_client()
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let response = grpc_client
            .generate_content(request)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?
            .into_inner();

        response.try_into()
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        super::streaming::stream(self.client.clone(), self.model.clone(), request).await
    }
}

// Helper function to create gRPC request from Rig's CompletionRequest
pub(crate) fn create_grpc_request(
    model: String,
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, CompletionError> {
    let mut contents = Vec::new();

    // Convert chat history to gRPC Content messages
    for msg in completion_request.chat_history {
        contents.push(rig_message_to_grpc_content(msg)?);
    }

    // Handle system instruction (preamble)
    let system_instruction = completion_request.preamble.map(|preamble| proto::Content {
        parts: vec![proto::Part {
            thought: Some(false),
            thought_signature: None,
            data: Some(proto::part::Data::Text(preamble)),
        }],
        role: Some("model".to_string()),
    });

    // Handle generation config
    let generation_config =
        if completion_request.temperature.is_some() || completion_request.max_tokens.is_some() {
            Some(proto::GenerationConfig {
                temperature: completion_request.temperature.map(|t| t as f32),
                max_output_tokens: completion_request.max_tokens.map(|t| t as i32),
                ..Default::default()
            })
        } else {
            None
        };

    // Handle tools (functions)
    let tools = if !completion_request.tools.is_empty() {
        let function_declarations = completion_request
            .tools
            .into_iter()
            .map(|tool| proto::FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters: None, // TODO: map schema
            })
            .collect();

        vec![proto::Tool {
            function_declarations,
            code_execution: None,
        }]
    } else {
        vec![]
    };

    Ok(GenerateContentRequest {
        model: format!("models/{}", model),
        contents,
        system_instruction,
        tools,
        tool_config: None,
        safety_settings: vec![],
        generation_config,
    })
}

// Convert Rig message to gRPC Content
fn rig_message_to_grpc_content(msg: message::Message) -> Result<proto::Content, CompletionError> {
    match msg {
        message::Message::User { content } => {
            let parts = content
                .into_iter()
                .map(rig_user_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: Some("user".to_string()),
            })
        }
        message::Message::Assistant { content, .. } => {
            let parts = content
                .into_iter()
                .map(rig_assistant_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: Some("model".to_string()),
            })
        }
    }
}

// Convert Rig UserContent to gRPC Part
fn rig_user_content_to_grpc_part(
    content: message::UserContent,
) -> Result<proto::Part, CompletionError> {
    match content {
        message::UserContent::Text(message::Text { text }) => Ok(proto::Part {
            thought: Some(false),
            thought_signature: None,
            data: Some(proto::part::Data::Text(text)),
        }),
        message::UserContent::ToolResult(result) => {
            let response_text = match &result.content.first() {
                message::ToolResultContent::Text(t) => t.text.clone(),
                _ => String::new(),
            };

            Ok(proto::Part {
                thought: Some(false),
                thought_signature: None,
                data: Some(proto::part::Data::FunctionResponse(
                    proto::FunctionResponse {
                        name: result.id,
                        response: Some(response_text),
                    },
                )),
            })
        }
        message::UserContent::Image(img) => {
            let mime_type = img
                .media_type
                .map(|mt| mt.to_mime_type().to_string())
                .unwrap_or_else(|| "image/jpeg".to_string());

            let data = match img.data {
                message::DocumentSourceKind::Base64(data)
                | message::DocumentSourceKind::String(data) => data,
                _ => {
                    return Err(CompletionError::RequestError(
                        "Only base64 encoded images are supported".into(),
                    ));
                }
            };

            Ok(proto::Part {
                thought: Some(false),
                thought_signature: None,
                data: Some(proto::part::Data::InlineData(proto::Blob {
                    mime_type,
                    data,
                })),
            })
        }
        _ => Err(CompletionError::RequestError(
            "Unsupported user content type".into(),
        )),
    }
}

// Convert Rig AssistantContent to gRPC Part
fn rig_assistant_content_to_grpc_part(
    content: message::AssistantContent,
) -> Result<proto::Part, CompletionError> {
    match content {
        message::AssistantContent::Text(message::Text { text }) => Ok(proto::Part {
            thought: Some(false),
            thought_signature: None,
            data: Some(proto::part::Data::Text(text)),
        }),
        message::AssistantContent::ToolCall(tool_call) => Ok(proto::Part {
            thought: Some(false),
            thought_signature: tool_call.signature,
            data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
                name: tool_call.function.name,
                args: Some(serde_json::to_string(&tool_call.function.arguments)?),
            })),
        }),
        message::AssistantContent::Reasoning(reasoning) => Ok(proto::Part {
            thought: Some(true),
            thought_signature: reasoning.signature,
            data: Some(proto::part::Data::Text(reasoning.reasoning.join("\n"))),
        }),
        _ => Err(CompletionError::RequestError(
            "Unsupported assistant content type".into(),
        )),
    }
}

// Convert gRPC GenerateContentResponse to Rig CompletionResponse
impl TryFrom<GenerateContentResponse> for completion::CompletionResponse<GenerateContentResponse> {
    type Error = CompletionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ResponseError("No response candidates in response".into())
        })?;

        let content_ref = candidate.content.as_ref().ok_or_else(|| {
            CompletionError::ResponseError(format!(
                "Gemini candidate missing content (finish_reason={:?})",
                candidate.finish_reason
            ))
        })?;

        let mut assistant_contents = Vec::new();

        for part in &content_ref.parts {
            let assistant_content = match &part.data {
                Some(proto::part::Data::Text(text)) => {
                    if part.thought.unwrap_or(false) {
                        completion::AssistantContent::Reasoning(Reasoning::new(text))
                    } else {
                        completion::AssistantContent::text(text)
                    }
                }
                Some(proto::part::Data::FunctionCall(function_call)) => {
                    let args = function_call
                        .args
                        .as_ref()
                        .and_then(|a| serde_json::from_str(a).ok())
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    let mut tool_call = message::ToolCall::new(
                        function_call.name.clone(),
                        message::ToolFunction::new(function_call.name.clone(), args),
                    );

                    if let Some(sig) = &part.thought_signature {
                        tool_call = tool_call.with_signature(Some(sig.clone()));
                    }

                    completion::AssistantContent::ToolCall(tool_call)
                }
                _ => {
                    return Err(CompletionError::ResponseError(
                        "Response did not contain a message or tool call".into(),
                    ));
                }
            };

            assistant_contents.push(assistant_content);
        }

        let choice = OneOrMany::many(assistant_contents).map_err(|_| {
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

// Implement ProviderResponseExt for telemetry
impl ProviderResponseExt for GenerateContentResponse {
    type OutputMessage = proto::ContentCandidate;
    type Usage = proto::UsageMetadata;

    fn get_response_id(&self) -> Option<String> {
        Some(self.response_id.clone())
    }

    fn get_response_model_name(&self) -> Option<String> {
        self.model_version.clone()
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.candidates.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        self.candidates.first().and_then(|c| {
            c.content.as_ref().and_then(|content| {
                let text: Vec<String> = content
                    .parts
                    .iter()
                    .filter_map(|part| {
                        if let Some(proto::part::Data::Text(text)) = &part.data {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if text.is_empty() {
                    None
                } else {
                    Some(text.join("\\n"))
                }
            })
        })
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage_metadata.clone()
    }
}
