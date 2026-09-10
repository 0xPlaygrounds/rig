use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionError, CompletionResponse, Usage};
use rig_core::message::{
    AssistantContent, ImageDetail, ImageMediaType, MediaType, MimeType, Reasoning, Text, ToolCall,
    ToolFunction,
};
use rig_core::providers::gemini::completion::gemini_api_types::map_google_finish_reason;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct VertexGenerateContentOutput(pub vertexai::model::GenerateContentResponse);

/// Stable descriptor name reported on normalized Vertex AI responses.
pub const PROVIDER_NAME: &str = "vertexai";

/// Map Vertex AI's `finishReason` onto rig's normalized vocabulary.
///
/// Unmapped values are carried verbatim in their wire SCREAMING_SNAKE spelling
/// so a reason Vertex adds later surfaces instead of reading as a natural stop.
pub fn map_finish_reason(
    reason: &vertexai::model::candidate::FinishReason,
) -> Option<rig_core::completion::FinishReason> {
    // `name()` yields the wire form (`MALFORMED_FUNCTION_CALL`) the shared
    // Google table keys on; a value the SDK does not model falls back to
    // `Display`, which prints the raw enum value. Formatting the variant with
    // `Debug` would silently drop the underscores.
    let wire_name = reason
        .name()
        .map_or_else(|| reason.to_string(), ToOwned::to_owned);

    map_google_finish_reason(&wire_name)
}

impl TryFrom<VertexGenerateContentOutput> for CompletionResponse {
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

        // vertexai internally uses a wkt::Struct (serde_json::Map<String, serde_json::Value>) in
        // function calling args. We need to convert that to serde_json::Value for rig_core::completion type matching
        for part in content.parts.iter() {
            // Gemini "thinking" models attach an opaque `thoughtSignature` to (usually) the
            // functionCall part. It must be echoed back verbatim on subsequent turns or Vertex
            // rejects the request with INVALID_ARGUMENT ("missing a thought_signature"). We carry
            // it through rig-core's `ToolCall.signature` (base64, since it is raw bytes).
            let signature = (!part.thought_signature.is_empty())
                .then(|| BASE64.encode(&part.thought_signature));

            if let Some(function_call) = part.function_call() {
                let args_json = function_call.args.as_ref().map_or_else(
                    || serde_json::json!({}),
                    |s| serde_json::Value::Object(s.clone()),
                );

                // Vertex function calls carry no identifier: mint the
                // correlation handle — never name-as-id, which collides two
                // same-tool calls in one turn.
                assistant_contents.push(AssistantContent::ToolCall(
                    ToolCall::from_wire(
                        "",
                        ToolFunction::new(function_call.name.clone(), args_json),
                    )
                    .with_signature(signature),
                ));
            } else if let Some(text) = part.text() {
                if part.thought {
                    assistant_contents.push(AssistantContent::Reasoning(
                        Reasoning::new_with_signature(text, signature),
                    ));
                } else {
                    assistant_contents.push(AssistantContent::Text(Text::new(text.clone())));
                }
            } else if let Some(inline_data) = part.inline_data() {
                if signature.is_some() {
                    return Err(CompletionError::ResponseError(
                        "Vertex inline images with thought_signature cannot be replayed through assistant history"
                            .to_string(),
                    ));
                }

                // Assistant history cannot represent the `thought` flag on image parts, so
                // avoid replaying an internal thought image as visible assistant content.
                if part.thought {
                    continue;
                }

                let media_type = MediaType::from_mime_type(&inline_data.mime_type);
                match media_type {
                    Some(MediaType::Image(
                        media_type @ (ImageMediaType::JPEG
                        | ImageMediaType::PNG
                        | ImageMediaType::WEBP
                        | ImageMediaType::HEIC
                        | ImageMediaType::HEIF),
                    )) => {
                        assistant_contents.push(AssistantContent::image_base64(
                            BASE64.encode(&inline_data.data),
                            Some(media_type),
                            Some(ImageDetail::default()),
                        ));
                    }
                    Some(MediaType::Image(media_type)) => {
                        return Err(CompletionError::ResponseError(format!(
                            "Unsupported Vertex inline image media type {media_type:?}; it cannot be replayed through assistant history"
                        )));
                    }
                    _ => {
                        return Err(CompletionError::ResponseError(format!(
                            "Unsupported Vertex inline media type {:?}",
                            inline_data.mime_type
                        )));
                    }
                }
            } else if signature.is_some() {
                // A signature-bearing part that is neither a function call nor text (e.g. a
                // standalone "thinking" part). rig-core has no carrier for it, so it is dropped —
                // log it so a later INVALID_ARGUMENT can be traced back here rather than being silent.
                tracing::warn!(
                    "Vertex response part carries a thought_signature but is neither a function \
                     call nor text; signature dropped (no rig-core carrier)."
                );
            }
        }

        let choice = rig_core::message::require_non_empty_response(assistant_contents)?;

        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage| Usage {
                input_tokens: usage.prompt_token_count as u64,
                output_tokens: usage.candidates_token_count as u64,
                total_tokens: usage.total_token_count as u64,
                // `prompt_token_count` is documented as "still the total
                // effective prompt size... including the number of tokens in the
                // cached content", so the cached count is a *subset* of the
                // input count, matching the Gemini surface.
                cached_input_tokens: usage.cached_content_token_count as u64,
                // Vertex reports no cache-write counter.
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                // Vertex reports `thoughts_token_count`, and rig has a field for
                // it. Hardcoding zero here silently discarded the thinking spend
                // on every Vertex response — on the sibling Gemini surface it is
                // routinely the largest component of the bill.
                reasoning_tokens: usage.thoughts_token_count as u64,
            })
            .unwrap_or_default();

        let finish_reason = map_finish_reason(&candidate.finish_reason);
        let model = Some(response.model_version.clone()).filter(|model| !model.is_empty());

        Ok(CompletionResponse::new(choice, usage, PROVIDER_NAME)
            .with_optional_finish_reason(finish_reason)
            .with_optional_model(model)
            .with_optional_response_id(
                Some(response.response_id.clone()).filter(|id| !id.is_empty()),
            ))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod vertex_usage_mapping_tests;
