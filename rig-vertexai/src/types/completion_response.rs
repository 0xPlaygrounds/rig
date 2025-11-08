use google_cloud_aiplatform_v1 as vertexai;
use rig::OneOrMany;
use rig::completion::{CompletionError, CompletionResponse, Usage};
use rig::message::{AssistantContent, Text};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct VertexGenerateContentOutput(pub vertexai::model::GenerateContentResponse);

impl TryFrom<VertexGenerateContentOutput> for CompletionResponse<VertexGenerateContentOutput> {
    type Error = CompletionError;

    fn try_from(value: VertexGenerateContentOutput) -> Result<Self, Self::Error> {
        let response = &value.0;

        // Get the first candidate
        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ProviderError("No candidates in response".to_string())
        })?;

        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| CompletionError::ProviderError("No content in candidate".to_string()))?;

        // Extract text from parts
        let mut text_parts = Vec::new();
        for part in content.parts.iter() {
            if let Some(text) = part.text() {
                text_parts.push(AssistantContent::Text(Text { text: text.clone() }));
            }
        }

        if text_parts.is_empty() {
            return Err(CompletionError::ProviderError(
                "No text content found in response".to_string(),
            ));
        }

        let choice = OneOrMany::many(text_parts).map_err(|e| {
            CompletionError::ProviderError(format!("Failed to create OneOrMany: {e}"))
        })?;

        // Extract usage metadata
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
