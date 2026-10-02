use google_cloud_aiplatform_v1 as vertexai;
use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::providers::gemini::completion::gemini_api_types::{PromptFeedback, UsageMetadata};
use rig_core::providers::gemini::streaming::{GenerateContentChunk, GenerateContentDecoder};
use rig_core::wire::{Decoder, Flow, Out, WireEvent};
use serde_json::{Map, Value};

/// Stable descriptor name reported on normalized Vertex AI responses.
pub const PROVIDER_NAME: &str = "vertexai";

/// Decodes Vertex AI's whole `GenerateContent` reply. The reply is restated
/// as the REST chunk its JSON is and read by the Gemini API's decoder, so a
/// block's provider item is the SDK part's JSON. A streamed call re-emits
/// the same reply.
#[derive(Debug, Default)]
pub struct VertexDecoder(GenerateContentDecoder);

impl<'id> Decoder<'id, Completion, vertexai::model::GenerateContentResponse> for VertexDecoder {
    type Event = vertexai::model::GenerateContentResponse;

    fn classify(
        &self,
        frame: vertexai::model::GenerateContentResponse,
    ) -> WireEvent<vertexai::model::GenerateContentResponse> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        response: vertexai::model::GenerateContentResponse,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        // The provider's own document, captured before it is restated.
        self.0.keep_raw(serde_json::to_value(&response)?);
        Decoder::<'id, Completion>::decode(&mut self.0, rest_chunk(&response)?, out)
    }

    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        Decoder::<'id, Completion>::eof(&mut self.0, out)
    }
}

/// `response` as the REST chunk its JSON is, with the finish and block
/// reasons spelled by name.
fn rest_chunk(
    response: &vertexai::model::GenerateContentResponse,
) -> Result<GenerateContentChunk, ProviderError> {
    let candidates = response
        .candidates
        .iter()
        .map(|candidate| {
            let Value::Object(mut fields) = serde_json::to_value(candidate)? else {
                return Ok(Map::new());
            };
            // The SDK spells enums by number; REST JSON spells them by name.
            if fields.contains_key("finishReason") {
                let reason = &candidate.finish_reason;
                let name = reason
                    .name()
                    .map_or_else(|| reason.to_string(), str::to_owned);
                fields.insert("finishReason".to_owned(), name.into());
            }
            Ok(fields)
        })
        .collect::<Result<_, serde_json::Error>>()?;
    let prompt_feedback = response
        .prompt_feedback
        .as_ref()
        .and_then(|feedback| feedback.block_reason.name())
        .filter(|reason| *reason != "BLOCKED_REASON_UNSPECIFIED")
        .map(|reason| PromptFeedback {
            block_reason: serde_json::from_value(reason.into()).ok(),
            safety_ratings: None,
        });
    Ok(GenerateContentChunk {
        response_id: response.response_id.clone(),
        candidates,
        prompt_feedback,
        usage_metadata: response.usage_metadata.as_ref().map(rest_usage),
        model_version: Some(response.model_version.clone()).filter(|model| !model.is_empty()),
        error: None,
    })
}

/// Vertex's usage as the REST usage it reads as. Vertex reports the
/// tool-use prompt only per modality, so its count is the breakdown's sum;
/// every count is reported, zero included.
fn rest_usage(usage: &vertexai::model::generate_content_response::UsageMetadata) -> UsageMetadata {
    UsageMetadata {
        prompt_token_count: usage.prompt_token_count,
        cached_content_token_count: Some(usage.cached_content_token_count),
        candidates_token_count: Some(usage.candidates_token_count),
        total_token_count: usage.total_token_count,
        thoughts_token_count: Some(usage.thoughts_token_count),
        tool_use_prompt_token_count: Some(
            usage
                .tool_use_prompt_tokens_details
                .iter()
                .map(|modality| modality.token_count)
                .sum(),
        ),
        ..UsageMetadata::default()
    }
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod vertex_usage_mapping_tests;
