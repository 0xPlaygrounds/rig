use crate::types::json_utils;
use google_cloud_aiplatform_v1 as vertexai;
use rig::OneOrMany;
use rig::completion::{CompletionError, CompletionResponse, Usage};
use rig::message::{AssistantContent, Text, ToolCall, ToolFunction};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct VertexGenerateContentOutput(pub vertexai::model::GenerateContentResponse);

impl TryFrom<VertexGenerateContentOutput> for CompletionResponse<VertexGenerateContentOutput> {
    type Error = CompletionError;

    fn try_from(value: VertexGenerateContentOutput) -> Result<Self, Self::Error> {
        let response = &value.0;

        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ProviderError("No candidates in response".to_string())
        })?;

        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| CompletionError::ProviderError("No content in candidate".to_string()))?;

        let mut assistant_contents = Vec::new();

        for part in content.parts.iter() {
            if let Some(function_call) = part.function_call() {
                let args_json = function_call
                    .args
                    .as_ref()
                    .map(|s| json_utils::struct_to_json(s.clone()))
                    .unwrap_or_else(|| serde_json::json!({}));

                assistant_contents.push(AssistantContent::ToolCall(ToolCall {
                    id: function_call.name.clone(),
                    call_id: None,
                    function: ToolFunction {
                        name: function_call.name.clone(),
                        arguments: args_json,
                    },
                }));
            } else if let Some(text) = part.text() {
                assistant_contents.push(AssistantContent::Text(Text { text: text.clone() }));
            }
        }

        if assistant_contents.is_empty() {
            return Err(CompletionError::ProviderError(
                "No text or tool call content found in response".to_string(),
            ));
        }

        let choice = OneOrMany::many(assistant_contents).map_err(|e| {
            CompletionError::ProviderError(format!("Failed to create OneOrMany: {e}"))
        })?;

        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage| Usage {
                input_tokens: usage.prompt_token_count as u64,
                output_tokens: usage.candidates_token_count as u64,
                total_tokens: usage.total_token_count as u64,
            })
            .unwrap_or_default();

        Ok(CompletionResponse {
            choice,
            usage,
            raw_response: value,
        })
    }
}
