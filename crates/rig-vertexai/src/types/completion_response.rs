use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::OneOrMany;
use rig_core::completion::{
    CompletionError, CompletionFinishReason, CompletionResponse, CompletionTerminalMetadata, Usage,
};
use rig_core::message::{
    AssistantContent, ImageDetail, ImageMediaType, MediaType, MimeType, Reasoning, Text, ToolCall,
    ToolFunction,
};
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

        if matches!(
            candidate.finish_reason,
            vertexai::model::candidate::FinishReason::MalformedFunctionCall
        ) {
            return Err(match serde_json::to_string(&value) {
                Ok(body) => CompletionError::from_provider_body(body),
                Err(error) => CompletionError::ProviderError(format!(
                    "Vertex stopped with MALFORMED_FUNCTION_CALL: {}; failed to preserve response: {error}",
                    candidate.finish_message.as_deref().unwrap_or("no details")
                )),
            });
        }

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
                let args_json = function_call
                    .args
                    .as_ref()
                    .map(|s| serde_json::Value::Object(s.clone()))
                    .unwrap_or_else(|| serde_json::json!({}));

                assistant_contents.push(AssistantContent::ToolCall(
                    ToolCall::new(
                        function_call.name.clone(),
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
                cached_input_tokens: 0, // unreported at time of writing
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default();
        let terminal_metadata = terminal_metadata_from_finish_reason(&candidate.finish_reason);

        Ok(CompletionResponse {
            choice,
            usage,
            raw_response: value,
            message_id: None,
            terminal_metadata,
        })
    }
}

pub(crate) fn terminal_metadata_from_finish_reason(
    reason: &vertexai::model::candidate::FinishReason,
) -> Option<CompletionTerminalMetadata> {
    use vertexai::model::candidate::FinishReason;

    if matches!(reason, FinishReason::Unspecified) {
        return None;
    }

    let canonical = match reason {
        FinishReason::Stop => CompletionFinishReason::Stop,
        FinishReason::MaxTokens => CompletionFinishReason::Length,
        FinishReason::Safety
        | FinishReason::Recitation
        | FinishReason::Blocklist
        | FinishReason::ProhibitedContent
        | FinishReason::Spii
        | FinishReason::ModelArmor => CompletionFinishReason::ContentFilter,
        _ => CompletionFinishReason::Unknown,
    };
    let raw_reason = reason
        .name()
        .map(str::to_owned)
        .or_else(|| reason.value().map(|value| value.to_string()));

    Some(
        CompletionTerminalMetadata::new(canonical)
            .with_raw_reason(raw_reason.unwrap_or_else(|| reason.to_string())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use google_cloud_aiplatform_v1 as vertexai;
    use rig_core::OneOrMany;
    use rig_core::message::{
        AssistantContent, DocumentSourceKind, ImageDetail, ImageMediaType, Text, ToolCall,
    };

    fn create_text_response(text: &str) -> VertexGenerateContentOutput {
        let part = vertexai::model::Part::new().set_text(text.to_string());
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new()
            .set_content(content)
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        VertexGenerateContentOutput(response)
    }

    fn create_parts_response(
        parts: impl IntoIterator<Item = vertexai::model::Part>,
    ) -> VertexGenerateContentOutput {
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts(parts);
        let candidate = vertexai::model::Candidate::new().set_content(content);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        VertexGenerateContentOutput(response)
    }

    fn inline_data_part(mime_type: &str, data: Vec<u8>) -> vertexai::model::Part {
        vertexai::model::Part::new().set_inline_data(
            vertexai::model::Blob::new()
                .set_mime_type(mime_type)
                .set_data(data),
        )
    }

    fn create_tool_call_response(
        function_name: &str,
        args: serde_json::Value,
    ) -> VertexGenerateContentOutput {
        let struct_args = match args {
            serde_json::Value::Object(map) => map,
            _ => panic!("Expected JSON object for Struct conversion"),
        };
        let function_call = vertexai::model::FunctionCall::new()
            .set_name(function_name.to_string())
            .set_args(struct_args);
        let part = vertexai::model::Part::new().set_function_call(function_call);
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new()
            .set_content(content)
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        VertexGenerateContentOutput(response)
    }

    fn create_signed_tool_call_response(
        function_name: &str,
        signature: &[u8],
    ) -> VertexGenerateContentOutput {
        let function_call = vertexai::model::FunctionCall::new()
            .set_name(function_name.to_string())
            .set_args(serde_json::Map::new());
        let part = vertexai::model::Part::new()
            .set_function_call(function_call)
            .set_thought_signature(signature.to_vec());
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new().set_content(content);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        VertexGenerateContentOutput(response)
    }

    #[test]
    fn test_tool_call_response_captures_thought_signature() {
        let raw = b"\x00\x01\x02thinking-sig\xff";
        let response: CompletionResponse<VertexGenerateContentOutput> =
            create_signed_tool_call_response("add", raw)
                .try_into()
                .unwrap();
        match response.choice.first() {
            AssistantContent::ToolCall(tc) => assert_eq!(tc.signature, Some(BASE64.encode(raw))),
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_tool_call_response_without_signature_is_none() {
        let response: CompletionResponse<VertexGenerateContentOutput> =
            create_tool_call_response("add", serde_json::json!({"x": 1}))
                .try_into()
                .unwrap();
        match response.choice.first() {
            AssistantContent::ToolCall(tc) => assert_eq!(tc.signature, None),
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn test_thought_text_response_captures_thought_signature() {
        let raw = b"\x00\x01\x02thinking-text-sig\xff";
        let part = vertexai::model::Part::new()
            .set_text("thinking text".to_string())
            .set_thought(true)
            .set_thought_signature(raw.to_vec());
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([part]);
        let candidate = vertexai::model::Candidate::new().set_content(content);
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);

        let response: CompletionResponse<VertexGenerateContentOutput> =
            VertexGenerateContentOutput(response).try_into().unwrap();

        match response.choice.first() {
            AssistantContent::Reasoning(reasoning) => {
                assert_eq!(reasoning.display_text(), "thinking text");
                assert_eq!(
                    reasoning.first_signature(),
                    Some(BASE64.encode(raw).as_str())
                );
            }
            _ => panic!("Expected Reasoning"),
        }
    }

    #[test]
    fn test_text_response_conversion() {
        let vertex_output = create_text_response("Hello, world!");
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_ok());
        let response = completion_response.unwrap();
        assert_eq!(
            response.choice,
            OneOrMany::one(AssistantContent::Text(Text::new(
                "Hello, world!".to_string()
            )))
        );
        let metadata = response
            .terminal_metadata
            .expect("Vertex finish reason should be normalized");
        assert_eq!(metadata.reason(), CompletionFinishReason::Stop);
        assert_eq!(metadata.raw_reason(), Some("STOP"));
    }

    #[test]
    fn test_tool_call_response_conversion() {
        let args = serde_json::json!({
            "x": 5,
            "y": 3
        });
        let vertex_output = create_tool_call_response("add", args.clone());
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_ok());
        let response = completion_response.unwrap();

        match response.choice.first() {
            AssistantContent::ToolCall(ToolCall { id, function, .. }) => {
                assert_eq!(id, "add");
                assert_eq!(function.name, "add");
                assert_eq!(function.arguments, args);
            }
            _ => panic!("Expected ToolCall"),
        }
    }

    #[test]
    fn malformed_function_call_remains_a_provider_error() {
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([vertexai::model::Part::new().set_text("not a valid tool call")]);
        let candidate = vertexai::model::Candidate::new()
            .set_content(content)
            .set_finish_reason(vertexai::model::candidate::FinishReason::MalformedFunctionCall)
            .set_finish_message("invalid function arguments");
        let response = VertexGenerateContentOutput(
            vertexai::model::GenerateContentResponse::new().set_candidates([candidate]),
        );

        let error = match CompletionResponse::try_from(response) {
            Ok(_) => panic!("malformed function calls must not become successful turns"),
            Err(error) => error,
        };
        assert_eq!(error.provider_response_status(), None);
        let body = error
            .provider_response_body()
            .expect("provider response should be preserved");
        assert!(body.contains("finishReason"));
        assert!(body.contains("invalid function arguments"));
    }

    #[test]
    fn inline_image_response_converts_raw_bytes_to_base64_with_mime_type() {
        let raw = vec![0, 1, 2, 255];
        let response: CompletionResponse<VertexGenerateContentOutput> =
            create_parts_response([inline_data_part("image/png", raw.clone())])
                .try_into()
                .expect("image response should convert");

        match response.choice.first() {
            AssistantContent::Image(image) => {
                assert_eq!(image.data, DocumentSourceKind::Base64(BASE64.encode(raw)));
                assert_eq!(image.media_type, Some(ImageMediaType::PNG));
                assert_eq!(image.detail, Some(ImageDetail::default()));
            }
            _ => panic!("Expected Image"),
        }
    }

    #[test]
    fn mixed_text_and_image_response_preserves_part_order() {
        let raw = vec![1, 2, 3];
        let response: CompletionResponse<VertexGenerateContentOutput> = create_parts_response([
            vertexai::model::Part::new().set_text("before"),
            inline_data_part("image/jpeg", raw.clone()),
            vertexai::model::Part::new().set_text("after"),
        ])
        .try_into()
        .expect("mixed response should convert");

        let contents: Vec<_> = response.choice.iter().collect();
        assert!(matches!(contents[0], AssistantContent::Text(text) if text.text == "before"));
        match contents[1] {
            AssistantContent::Image(image) => {
                assert_eq!(image.data, DocumentSourceKind::Base64(BASE64.encode(raw)));
                assert_eq!(image.media_type, Some(ImageMediaType::JPEG));
            }
            _ => panic!("Expected Image"),
        }
        assert!(matches!(contents[2], AssistantContent::Text(text) if text.text == "after"));
    }

    #[test]
    fn mixed_text_and_thought_image_response_keeps_only_visible_text_in_order() {
        let response: CompletionResponse<VertexGenerateContentOutput> = create_parts_response([
            vertexai::model::Part::new().set_text("before"),
            inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
            vertexai::model::Part::new().set_text("after"),
        ])
        .try_into()
        .expect("thought image should be skipped");

        let contents: Vec<_> = response.choice.iter().collect();
        assert_eq!(contents.len(), 2);
        assert!(matches!(contents[0], AssistantContent::Text(text) if text.text == "before"));
        assert!(matches!(contents[1], AssistantContent::Text(text) if text.text == "after"));
    }

    #[test]
    fn thought_image_only_response_fails_without_visible_assistant_content() {
        let result =
            CompletionResponse::<VertexGenerateContentOutput>::try_from(create_parts_response([
                inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
            ]));

        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("thought-image-only response must fail"),
        };
        assert!(matches!(error, CompletionError::ProviderError(_)));
        assert!(
            error
                .to_string()
                .contains("No text or tool call content found in response")
        );
    }

    #[test]
    fn inline_audio_and_non_image_media_are_rejected() {
        for mime_type in ["audio/wav", "application/pdf", "application/octet-stream"] {
            let result = CompletionResponse::<VertexGenerateContentOutput>::try_from(
                create_parts_response([inline_data_part(mime_type, vec![0])]),
            );
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("unsupported inline media must fail"),
            };
            assert!(matches!(error, CompletionError::ResponseError(_)));
            assert!(error.to_string().contains(mime_type));
        }
    }

    #[test]
    fn inline_gif_and_svg_images_are_rejected() {
        for mime_type in ["image/gif", "image/svg+xml"] {
            let result = CompletionResponse::<VertexGenerateContentOutput>::try_from(
                create_parts_response([inline_data_part(mime_type, vec![0])]),
            );
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("non-replayable inline image must fail"),
            };
            assert!(matches!(error, CompletionError::ResponseError(_)));
            assert!(
                error
                    .to_string()
                    .contains("Unsupported Vertex inline image media type")
            );
        }
    }

    #[test]
    fn signed_inline_image_is_rejected() {
        let part = inline_data_part("image/png", vec![0]).set_thought_signature(vec![1, 2, 3]);
        let result =
            CompletionResponse::<VertexGenerateContentOutput>::try_from(create_parts_response([
                part,
            ]));
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("signed inline image must fail"),
        };
        assert!(matches!(error, CompletionError::ResponseError(_)));
        assert!(error.to_string().contains("thought_signature"));
    }

    #[test]
    fn test_usage_metadata_conversion() {
        let mut response = create_text_response("test").0;
        let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
            .set_prompt_token_count(10)
            .set_candidates_token_count(20)
            .set_total_token_count(30);
        response = response.set_usage_metadata(usage_metadata);

        let vertex_output = VertexGenerateContentOutput(response);
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_ok());
        let response = completion_response.unwrap();
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
        assert_eq!(response.usage.total_tokens, 30);
    }

    #[test]
    fn test_empty_response_error() {
        // Create a response with no candidates
        let response = vertexai::model::GenerateContentResponse::new();
        let vertex_output = VertexGenerateContentOutput(response);
        let completion_response: Result<CompletionResponse<VertexGenerateContentOutput>, _> =
            vertex_output.try_into();

        assert!(completion_response.is_err());
    }
}
