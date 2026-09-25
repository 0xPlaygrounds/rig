use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::Usage;
use rig_core::error::ProviderError;
use rig_core::message::{
    AssistantContent, ImageDetail, ImageMediaType, MediaType, MimeType, Reasoning, Text, ToolCall,
    ToolFunction,
};
use rig_core::operation::{AdapterOutput, Completion, ImagePart};
use rig_core::providers::gemini::completion::gemini_api_types::map_google_finish_reason;
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming::StreamFinal;
use rig_core::wire::Decoder;

/// Stable descriptor name reported on normalized Vertex AI responses.
pub const PROVIDER_NAME: &str = "vertexai";

/// The text-block `AdditionalParams` key holding Vertex AI extras for that
/// text, today the `thoughtSignature` Vertex put on the answer part. Only the
/// Vertex codec reads it, so the signature returns only to Vertex.
pub const VERTEX_TEXT_EXTRAS_KEY: &str = "vertexai";

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

/// Decodes Vertex AI's whole `GenerateContent` reply into the events a
/// stream sends for it; a streamed call re-emits the same reply.
#[derive(Default)]
pub struct VertexDecoder {
    /// The reply's document, for the response's `raw`.
    document: Option<serde_json::Value>,
}

impl Decoder<Completion, vertexai::model::GenerateContentResponse> for VertexDecoder {
    type Event = vertexai::model::GenerateContentResponse;

    fn classify(
        &self,
        frame: vertexai::model::GenerateContentResponse,
    ) -> WireEvent<vertexai::model::GenerateContentResponse> {
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, response: Self::Event, out: &mut AdapterOutput) {
        // The provider's own document, captured before the response is
        // consumed into normalized content.
        let raw = match serde_json::to_value(&response) {
            Ok(raw) => raw,
            Err(error) => return out.error(error.into()),
        };
        self.document = Some(raw.clone());
        let choice = match assistant_content(&response) {
            Ok(choice) => choice,
            Err(error) => return out.error(error),
        };
        out.content(&choice, ImagePart::Block);
        let finish_reason = response
            .candidates
            .first()
            .and_then(|candidate| map_finish_reason(&candidate.finish_reason));
        out.final_record(
            StreamFinal::new(PROVIDER_NAME, usage(&response), raw)
                .with_optional_finish_reason(finish_reason)
                .with_optional_model(
                    Some(response.model_version.clone()).filter(|model| !model.is_empty()),
                )
                .with_optional_response_id(
                    Some(response.response_id.clone()).filter(|id| !id.is_empty()),
                ),
        );
    }

    fn document(&self) -> Option<serde_json::Value> {
        self.document.clone()
    }
}

/// The assistant content of a whole reply.
fn assistant_content(
    response: &vertexai::model::GenerateContentResponse,
) -> Result<Vec<AssistantContent>, ProviderError> {
    let candidate = response
        .candidates
        .first()
        .ok_or_else(|| ProviderError::Provider("No candidates in response".to_string()))?;

    let content = candidate
        .content
        .as_ref()
        .ok_or_else(|| ProviderError::Provider("No content in candidate".to_string()))?;

    let mut assistant_contents = Vec::new();
    // Vertex function calls carry no id: the `index`-th call of the
    // response mints its own handle, so two calls in one turn never
    // share one (the position pass below is then a no-op).
    let mut tool_index = 0u64;

    for part in content.parts.iter() {
        // Preserve opaque signature bytes as base64 for exact replay.
        let signature =
            (!part.thought_signature.is_empty()).then(|| BASE64.encode(&part.thought_signature));

        if let Some(function_call) = part.function_call() {
            let args_json = function_call.args.as_ref().map_or_else(
                || serde_json::json!({}),
                |s| serde_json::Value::Object(s.clone()),
            );

            // Mint by call index, not name, so repeated calls to one tool
            // retain distinct correlation handles.
            let index = tool_index;
            tool_index += 1;
            assistant_contents.push(AssistantContent::ToolCall(
                ToolCall::from_wire_indexed(
                    "",
                    index,
                    ToolFunction::new(function_call.name.clone(), args_json),
                )
                .with_signature(signature),
            ));
        } else if let Some(text) = part.text() {
            if part.thought {
                assistant_contents.push(AssistantContent::Reasoning(
                    Reasoning::new_with_signature(text, signature).with_provider(PROVIDER_NAME),
                ));
            } else {
                // A signature on answer text returns on that text part.
                assistant_contents.push(AssistantContent::Text(Text {
                    text: text.clone(),
                    additional_params: signature.clone().and_then(|signature| {
                        rig_core::providers::gemini::text_signature_extras(
                            VERTEX_TEXT_EXTRAS_KEY,
                            signature,
                        )
                    }),
                }));
            }
        } else if let Some(inline_data) = part.inline_data() {
            if signature.is_some() {
                return Err(ProviderError::Response(
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
                    return Err(ProviderError::Response(format!(
                        "Unsupported Vertex inline image media type {media_type:?}; it cannot be replayed through assistant history"
                    )));
                }
                _ => {
                    return Err(ProviderError::Response(format!(
                        "Unsupported Vertex inline media type {:?}",
                        inline_data.mime_type
                    )));
                }
            }
        } else if signature.is_some() {
            // Unrepresentable signatures cannot survive replay; warn without
            // exposing their contents.
            tracing::warn!(
                "Vertex response part carries a thought_signature but is neither a function \
                 call nor text; signature dropped (no rig-core carrier)."
            );
        }
    }

    rig_core::message::normalize_missing_tool_call_ids(&mut assistant_contents);
    rig_core::message::require_non_empty_response(assistant_contents)
}

/// Vertex's token counts in rig's usage record.
fn usage(response: &vertexai::model::GenerateContentResponse) -> Usage {
    response
        .usage_metadata
        .as_ref()
        .map(|usage| Usage {
            input_tokens: Some(usage.prompt_token_count as u64),
            output_tokens: Some(usage.candidates_token_count as u64),
            total_tokens: Some(usage.total_token_count as u64),
            // Cached tokens are already included in prompt_token_count.
            cached_input_tokens: Some(usage.cached_content_token_count as u64),
            // Vertex reports no cache-write counter.
            cache_creation_input_tokens: None,
            tool_use_prompt_tokens: None,
            reasoning_tokens: Some(usage.thoughts_token_count as u64),
        })
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod vertex_usage_mapping_tests;
