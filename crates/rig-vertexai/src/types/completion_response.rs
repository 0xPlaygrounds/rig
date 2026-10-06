use google_cloud_aiplatform_v1 as vertexai;
use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::providers::gemini::streaming::document::GenerateContentResponse;
use rig_core::providers::gemini::streaming::{GenerateContentChunk, GenerateContentDecoder};
use rig_core::wire::{Decoder, Flow, Out, WireEvent};
use serde_json::{Map, Value};

/// Stable descriptor name reported on normalized Vertex AI responses.
pub const PROVIDER_NAME: &str = "vertexai";

/// A Vertex AI reply's document: its responses' REST JSON, rebuilt by the
/// Gemini API's [`GenerateContentResponse`] fold. The transport reports the
/// one response a call returns, so this is fed only by a reply that
/// arrives without it.
#[derive(Debug, Default)]
pub struct VertexDocument(GenerateContentResponse);

impl rig_core::wire::document::Reassemble<vertexai::model::GenerateContentResponse>
    for VertexDocument
{
    fn absorb(&mut self, frame: &vertexai::model::GenerateContentResponse) {
        if let Ok(chunk) = rest_chunk(frame) {
            self.0.chunk(chunk);
        }
    }

    fn finish(self) -> Value {
        self.0.document()
    }
}

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
        let chunk = rest_chunk(&response)?;
        Decoder::<'id, Completion>::decode(&mut self.0, GenerateContentChunk(chunk), out)
    }

    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        Decoder::<'id, Completion>::eof(&mut self.0, out)
    }
}

/// `response` as the REST JSON Vertex AI sent: the SDK's own JSON, with
/// each enum value respelled by name, since the SDK serializes enums by
/// number. The SDK keeps the tool-use prompt count only per modality, so
/// its total is the breakdown's sum.
pub(crate) fn rest_chunk(
    response: &vertexai::model::GenerateContentResponse,
) -> Result<Map<String, Value>, ProviderError> {
    let Value::Object(mut chunk) = serde_json::to_value(response)? else {
        return Ok(Map::new());
    };
    let candidates = chunk.get_mut("candidates").and_then(Value::as_array_mut);
    for (candidate, json) in response
        .candidates
        .iter()
        .zip(candidates.into_iter().flatten())
    {
        spell(json, "finishReason", candidate.finish_reason.name());
        ratings(json, &candidate.safety_ratings);
        // A part's enums too, so a stored part reads back whatever a store
        // does to its numbers.
        let parts = candidate.content.iter().flat_map(|content| &content.parts);
        let json_parts = json
            .pointer_mut("/content/parts")
            .and_then(Value::as_array_mut);
        for (part, json) in parts.zip(json_parts.into_iter().flatten()) {
            if let (Some(code), Some(json)) =
                (part.executable_code(), json.get_mut("executableCode"))
            {
                spell(json, "language", code.language.name());
            }
            if let (Some(result), Some(json)) = (
                part.code_execution_result(),
                json.get_mut("codeExecutionResult"),
            ) {
                spell(json, "outcome", result.outcome.name());
            }
        }
    }
    if let (Some(feedback), Some(json)) =
        (&response.prompt_feedback, chunk.get_mut("promptFeedback"))
    {
        spell(json, "blockReason", feedback.block_reason.name());
        ratings(json, &feedback.safety_ratings);
    }
    if let (Some(usage), Some(json)) = (&response.usage_metadata, chunk.get_mut("usageMetadata")) {
        spell(json, "trafficType", usage.traffic_type.name());
        for (key, details) in [
            ("promptTokensDetails", &usage.prompt_tokens_details),
            ("cacheTokensDetails", &usage.cache_tokens_details),
            ("candidatesTokensDetails", &usage.candidates_tokens_details),
            (
                "toolUsePromptTokensDetails",
                &usage.tool_use_prompt_tokens_details,
            ),
        ] {
            let json = json.get_mut(key).and_then(Value::as_array_mut);
            for (detail, json) in details.iter().zip(json.into_iter().flatten()) {
                spell(json, "modality", detail.modality.name());
            }
        }
        let tool_use: i64 = usage
            .tool_use_prompt_tokens_details
            .iter()
            .map(|detail| i64::from(detail.token_count))
            .sum();
        if let (Some(json), true) = (json.as_object_mut(), tool_use > 0) {
            json.entry("toolUsePromptTokenCount")
                .or_insert_with(|| tool_use.into());
        }
    }
    Ok(chunk)
}

/// Respell the safety ratings under `json` by name.
fn ratings(json: &mut Value, ratings: &[vertexai::model::SafetyRating]) {
    let json = json.get_mut("safetyRatings").and_then(Value::as_array_mut);
    for (rating, json) in ratings.iter().zip(json.into_iter().flatten()) {
        spell(json, "category", rating.category.name());
        spell(json, "probability", rating.probability.name());
        spell(json, "severity", rating.severity.name());
    }
}

/// Replace the number at `json[key]` with `name`, when both exist.
fn spell(json: &mut Value, key: &str, name: Option<&str>) {
    if let (Some(slot), Some(name)) = (json.get_mut(key), name) {
        *slot = Value::String(name.to_owned());
    }
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod vertex_usage_mapping_tests;
