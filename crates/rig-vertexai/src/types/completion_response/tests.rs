use super::*;
use google_cloud_aiplatform_v1 as vertexai;
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
    let serde_json::Value::Object(struct_args) = args else {
        panic!("Expected JSON object for Struct conversion")
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
    let response: CompletionResponse = create_signed_tool_call_response("add", raw)
        .try_into()
        .unwrap();
    match response.choice.first() {
        Some(AssistantContent::ToolCall(tc)) => {
            assert_eq!(tc.signature, Some(BASE64.encode(raw)));
        }
        _ => panic!("Expected ToolCall"),
    }
}

#[test]
fn test_tool_call_response_without_signature_is_none() {
    let response: CompletionResponse =
        create_tool_call_response("add", serde_json::json!({"x": 1}))
            .try_into()
            .unwrap();
    match response.choice.first() {
        Some(AssistantContent::ToolCall(tc)) => assert_eq!(tc.signature, None),
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

    let response: CompletionResponse = VertexGenerateContentOutput(response).try_into().unwrap();

    match response.choice.first() {
        Some(AssistantContent::Reasoning(reasoning)) => {
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
    let completion_response: Result<CompletionResponse, _> = vertex_output.try_into();

    assert!(completion_response.is_ok());
    let response = completion_response.unwrap();
    assert_eq!(
        response.choice,
        vec![AssistantContent::Text(Text::new(
            "Hello, world!".to_string()
        ))]
    );
}

#[test]
fn test_tool_call_response_conversion() {
    let args = serde_json::json!({
        "x": 5,
        "y": 3
    });
    let vertex_output = create_tool_call_response("add", args.clone());
    let completion_response: Result<CompletionResponse, _> = vertex_output.try_into();

    assert!(completion_response.is_ok());
    let response = completion_response.unwrap();

    match response.choice.first() {
        Some(AssistantContent::ToolCall(ToolCall {
            id,
            provider,
            function,
            ..
        })) => {
            // Vertex issues no call ids: the decode mints a unique
            // non-empty handle (never the function name) and records
            // that the provider issued nothing.
            assert!(!id.as_str().is_empty());
            assert_ne!(id, "add");
            assert_eq!(provider, &None);
            assert_eq!(function.name, "add");
            assert_eq!(function.arguments, args);
        }
        _ => panic!("Expected ToolCall"),
    }
}

#[test]
fn inline_image_response_converts_raw_bytes_to_base64_with_mime_type() {
    let raw = vec![0, 1, 2, 255];
    let response: CompletionResponse =
        create_parts_response([inline_data_part("image/png", raw.clone())])
            .try_into()
            .expect("image response should convert");

    match response.choice.first() {
        Some(AssistantContent::Image(image)) => {
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
    let response: CompletionResponse = create_parts_response([
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
    let response: CompletionResponse = create_parts_response([
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
    let result = CompletionResponse::try_from(create_parts_response([inline_data_part(
        "image/png",
        vec![1, 2, 3],
    )
    .set_thought(true)]));

    let Err(error) = result else {
        panic!("thought-image-only response must fail")
    };
    // Rejected with the shared empty-response wording via
    // `require_non_empty_response`, like every other wire.
    assert!(matches!(
        error,
        CompletionError::ResponseError(message)
            if message == rig_core::message::EMPTY_RESPONSE_ERROR
    ));
}

#[test]
fn inline_audio_and_non_image_media_are_rejected() {
    for mime_type in ["audio/wav", "application/pdf", "application/octet-stream"] {
        let result = CompletionResponse::try_from(create_parts_response([inline_data_part(
            mime_type,
            vec![0],
        )]));
        let Err(error) = result else {
            panic!("unsupported inline media must fail")
        };
        assert!(matches!(error, CompletionError::ResponseError(_)));
        assert!(error.to_string().contains(mime_type));
    }
}

#[test]
fn inline_gif_and_svg_images_are_rejected() {
    for mime_type in ["image/gif", "image/svg+xml"] {
        let result = CompletionResponse::try_from(create_parts_response([inline_data_part(
            mime_type,
            vec![0],
        )]));
        let Err(error) = result else {
            panic!("non-replayable inline image must fail")
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
    let result = CompletionResponse::try_from(create_parts_response([part]));
    let Err(error) = result else {
        panic!("signed inline image must fail")
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
    let completion_response: Result<CompletionResponse, _> = vertex_output.try_into();

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
    let completion_response: Result<CompletionResponse, _> = vertex_output.try_into();

    assert!(completion_response.is_err());
}

/// The load-bearing property behind `CompletionResponse::raw` for Vertex
/// AI: the captured value is `serde_json::to_value(&VertexGenerateContentOutput)`
/// — the SDK response as `raw_completion` returns it — and a consumer must
/// be able to read it back as the same type and get the same JSON.
/// Vertex has no cassette harness, so this is the unit-form pin: the
/// serde derives on the newtype delegate to the SDK's own (camelCase)
/// wire encoding (camelCase keys, enums as proto numbers), and fields
/// rig never normalizes (`modelVersion` is mapped, but a candidate's
/// `safetyRatings` and `avgLogprobs` are not)
/// survive both directions. Normalizing the restored value agrees with
/// normalizing the original.
#[test]
fn vertex_generate_content_output_round_trips_through_serde_json_value() {
    let part = vertexai::model::Part::new().set_text("hello".to_string());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new()
        .set_content(content)
        .set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
        .set_avg_logprobs(-0.25)
        .set_safety_ratings([vertexai::model::SafetyRating::new()
            .set_category(vertexai::model::HarmCategory::Harassment)
            .set_probability(vertexai::model::safety_rating::HarmProbability::Negligible)]);
    let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(10)
        .set_candidates_token_count(20)
        .set_total_token_count(30);
    let response = vertexai::model::GenerateContentResponse::new()
        .set_candidates([candidate])
        .set_model_version("gemini-2.5-flash-001")
        .set_response_id("resp-vertex-1")
        .set_usage_metadata(usage_metadata);
    let raw = VertexGenerateContentOutput(response);

    let value = serde_json::to_value(&raw).expect("serialize");
    assert_eq!(value["modelVersion"], "gemini-2.5-flash-001");
    assert_eq!(value["candidates"][0]["avgLogprobs"], -0.25);
    // The SDK encodes enums as their proto numbers, not their names —
    // `HARM_CATEGORY_HARASSMENT` is 3, `STOP` is 1 — so that is what the
    // capture carries; a consumer decodes it through the SDK's own enum.
    assert_eq!(value["candidates"][0]["safetyRatings"][0]["category"], 3);
    assert_eq!(value["candidates"][0]["finishReason"], 1);

    let back: VertexGenerateContentOutput =
        serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&back).expect("re-serialize"),
        value,
        "the capture must read back into VertexGenerateContentOutput and re-serialize identically"
    );
    assert_eq!(back.0, raw.0);

    let original: CompletionResponse = raw.try_into().expect("original converts");
    let restored: CompletionResponse = back.try_into().expect("restored converts");
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.model, original.model);
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(restored.model.as_deref(), Some("gemini-2.5-flash-001"));
    assert_eq!(
        restored.identity().response_id.as_deref(),
        Some("resp-vertex-1")
    );
}
