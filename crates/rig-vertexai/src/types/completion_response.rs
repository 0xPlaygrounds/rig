use google_cloud_aiplatform_v1 as vertexai;
use rig_core::error::ProviderError;
use rig_core::providers::gemini::streaming::GenerateContentChunk;
use serde_json::Value;

/// Stable descriptor name reported on normalized Vertex AI responses.
pub const PROVIDER_NAME: &str = "vertexai";

/// `response` as the REST chunk Vertex AI sent, which the Gemini API's
/// decoder and reassembler read: the SDK's own JSON, with each enum value
/// respelled by name, since the SDK serializes enums by number. The SDK
/// keeps the tool-use prompt count only per modality, so its total is the
/// breakdown's sum. A transport of the wire transcodes each reply once,
/// with this.
pub fn rest_chunk(
    response: &vertexai::model::GenerateContentResponse,
) -> Result<GenerateContentChunk, ProviderError> {
    let Value::Object(mut chunk) = serde_json::to_value(response)? else {
        return Ok(GenerateContentChunk::default());
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
    Ok(GenerateContentChunk(chunk))
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
