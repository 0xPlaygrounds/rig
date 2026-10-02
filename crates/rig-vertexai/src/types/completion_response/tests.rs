use super::*;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};

/// Answers every request with one scripted SDK reply.
#[derive(Clone)]
pub(crate) struct Reply(vertexai::model::GenerateContentResponse);

impl Transport<crate::completion::GenerateContent> for Reply {
    fn send(
        &self,
        _payload: crate::completion::VertexRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
    }
}

/// The reply as the unary endpoint answers it.
pub(crate) trait Complete {
    fn complete(self) -> Result<CompletionResponse, ProviderError>;
}

impl Complete for vertexai::model::GenerateContentResponse {
    fn complete(self) -> Result<CompletionResponse, ProviderError> {
        let model = Model::new(
            crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
            Reply(self),
        );
        futures::executor::block_on(model.call(CompletionRequest::new("hello")))
    }
}

pub(crate) fn complete(
    response: vertexai::model::GenerateContentResponse,
) -> Result<CompletionResponse, ProviderError> {
    response.complete()
}
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use rig_core::message::{AssistantContent, DocumentSourceKind, ImageMediaType, Text, ToolCall};
use serde_json::json;

/// `candidate` as a finished one: Vertex states why every unary reply ended.
fn finished(candidate: vertexai::model::Candidate) -> vertexai::model::Candidate {
    candidate.set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
}

/// The provider item a decoded block holds.
fn item(block: &AssistantContent) -> Option<&serde_json::Value> {
    block.native_item()
}

fn create_text_response(text: &str) -> vertexai::model::GenerateContentResponse {
    let part = vertexai::model::Part::new().set_text(text.to_string());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new()
        .set_content(content)
        .set_finish_reason(vertexai::model::candidate::FinishReason::Stop);
    vertexai::model::GenerateContentResponse::new().set_candidates([candidate])
}

fn create_parts_response(
    parts: impl IntoIterator<Item = vertexai::model::Part>,
) -> vertexai::model::GenerateContentResponse {
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts);
    let candidate = finished(vertexai::model::Candidate::new().set_content(content));
    vertexai::model::GenerateContentResponse::new().set_candidates([candidate])
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
) -> vertexai::model::GenerateContentResponse {
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
    vertexai::model::GenerateContentResponse::new().set_candidates([candidate])
}

fn create_signed_tool_call_response(
    function_name: &str,
    signature: &[u8],
) -> vertexai::model::GenerateContentResponse {
    let function_call = vertexai::model::FunctionCall::new()
        .set_name(function_name.to_string())
        .set_args(serde_json::Map::new());
    let part = vertexai::model::Part::new()
        .set_function_call(function_call)
        .set_thought_signature(signature.to_vec());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = finished(vertexai::model::Candidate::new().set_content(content));
    vertexai::model::GenerateContentResponse::new().set_candidates([candidate])
}

#[test]
fn test_tool_call_response_captures_thought_signature() {
    let raw = b"\x00\x01\x02thinking-sig\xff";
    let response: CompletionResponse = create_signed_tool_call_response("add", raw)
        .complete()
        .unwrap();
    let block = response.choice.first().expect("a call");
    assert!(matches!(block, AssistantContent::ToolCall(_)));
    assert_eq!(
        item(block),
        Some(&json!({
            "functionCall": { "name": "add", "args": {} },
            "thoughtSignature": BASE64.encode(raw),
        }))
    );
}

#[test]
fn test_tool_call_response_without_signature_is_none() {
    let response: CompletionResponse =
        create_tool_call_response("add", serde_json::json!({"x": 1}))
            .complete()
            .unwrap();
    let block = response.choice.first().expect("a call");
    assert!(matches!(block, AssistantContent::ToolCall(_)));
    assert_eq!(
        item(block).and_then(|item| item.get("thoughtSignature")),
        None
    );
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
    let candidate = finished(vertexai::model::Candidate::new().set_content(content));
    let response = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);

    let response: CompletionResponse = response.complete().unwrap();

    let Some(AssistantContent::Reasoning(reasoning)) = response.choice.first() else {
        panic!("Expected Reasoning");
    };
    assert_eq!(reasoning.text, "thinking text");
    assert_eq!(
        reasoning
            .native
            .as_ref()
            .map(|native| &native.item["thoughtSignature"]),
        Some(&json!(BASE64.encode(raw)))
    );
    assert_eq!(response.provider(), super::PROVIDER_NAME);
    assert_eq!(response.origin.api.as_str(), "vertexai.generate_content");
}

#[test]
fn test_text_response_conversion() {
    let vertex_output = create_text_response("Hello, world!");
    let completion_response: Result<CompletionResponse, _> = vertex_output.complete();

    assert!(completion_response.is_ok());
    let response = completion_response.unwrap();
    assert_eq!(
        response
            .choice
            .iter()
            .map(AssistantContent::canonical)
            .collect::<Vec<_>>(),
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
    let completion_response: Result<CompletionResponse, _> = vertex_output.complete();

    assert!(completion_response.is_ok());
    let response = completion_response.unwrap();

    match response.choice.first() {
        Some(AssistantContent::ToolCall(ToolCall { id, function, .. })) => {
            // Vertex issues no call ids: rig issues one.
            assert!(id.is_local());
            assert_eq!(function.name, "add");
            assert_eq!(function.arguments_value(), args);
        }
        _ => panic!("Expected ToolCall"),
    }
}

#[test]
fn inline_image_response_converts_raw_bytes_to_base64_with_mime_type() {
    let raw = vec![0, 1, 2, 255];
    let response: CompletionResponse =
        create_parts_response([inline_data_part("image/png", raw.clone())])
            .complete()
            .expect("image response should convert");

    match response.choice.first() {
        Some(AssistantContent::Image(image)) => {
            assert_eq!(image.data, DocumentSourceKind::Base64(BASE64.encode(raw)));
            assert_eq!(image.media_type, Some(ImageMediaType::PNG));
            assert_eq!(image.detail, None);
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
    .complete()
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

/// A thought image, inline media that is not an image, and a signed image
/// all keep their part, so each replays to Vertex as it arrived: the
/// thought image and the audio as opaque blocks, the image as an image.
#[test]
fn every_inline_part_keeps_its_part_and_replays_it() {
    let parts = [
        vertexai::model::Part::new().set_text("before"),
        inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
        inline_data_part("audio/wav", vec![0]),
        inline_data_part("image/gif", vec![0]).set_thought_signature(vec![1, 2, 3]),
        vertexai::model::Part::new().set_text("after"),
    ];
    let response = complete(create_parts_response(parts.clone())).expect("the reply decodes");
    let kinds: Vec<&str> = response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Text(_) => "text",
            AssistantContent::Opaque(_) => "opaque",
            AssistantContent::Image(_) => "image",
            AssistantContent::Reasoning(_) | AssistantContent::ToolCall(_) => "other",
        })
        .collect();
    assert_eq!(kinds, ["text", "opaque", "opaque", "image", "text"]);

    let replayed = crate::types::message::content_from_message(rig_core::message::Message::from(
        response.choice.clone(),
    ))
    .expect("the turn replays");
    assert_eq!(replayed.parts, parts);
}

#[test]
fn test_usage_metadata_conversion() {
    let mut response = create_text_response("test");
    let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(10)
        .set_candidates_token_count(20)
        .set_total_token_count(30);
    response = response.set_usage_metadata(usage_metadata);

    let vertex_output = response;
    let completion_response: Result<CompletionResponse, _> = vertex_output.complete();

    assert!(completion_response.is_ok());
    let response = completion_response.unwrap();
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(20));
    assert_eq!(response.usage.total_tokens, Some(30));
}

#[test]
fn test_empty_response_error() {
    // Create a response with no candidates
    let response = vertexai::model::GenerateContentResponse::new();
    let vertex_output = response;
    let completion_response: Result<CompletionResponse, _> = vertex_output.complete();

    assert!(completion_response.is_err());
}

/// The load-bearing property behind `CompletionResponse::raw` for Vertex
/// AI: the captured value is `serde_json::to_value(&vertexai::model::GenerateContentResponse)`
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
    let raw = response;

    let value = serde_json::to_value(&raw).expect("serialize");
    assert_eq!(value["modelVersion"], "gemini-2.5-flash-001");
    assert_eq!(value["candidates"][0]["avgLogprobs"], -0.25);
    // The SDK encodes enums as their proto numbers, not their names —
    // `HARM_CATEGORY_HARASSMENT` is 3, `STOP` is 1 — so that is what the
    // capture carries; a consumer decodes it through the SDK's own enum.
    assert_eq!(value["candidates"][0]["safetyRatings"][0]["category"], 3);
    assert_eq!(value["candidates"][0]["finishReason"], 1);

    let back: vertexai::model::GenerateContentResponse =
        serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&back).expect("re-serialize"),
        value,
        "the capture must read back into GenerateContentResponse and re-serialize identically"
    );
    assert_eq!(back, raw);

    let original: CompletionResponse = raw.clone().complete().expect("original converts");
    // The candidate's metadata stays with the turn.
    let native = original.native.as_ref().map(|native| &native.item);
    assert_eq!(
        native.map(|native| &native["avgLogprobs"]),
        Some(&json!(-0.25))
    );
    assert_eq!(
        native.map(|native| &native["safetyRatings"][0]["category"]),
        Some(&json!(3))
    );
    let restored: CompletionResponse = back.complete().expect("restored converts");
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.model(), original.model());
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(restored.model(), Some("gemini-2.5-flash-001"));
    assert_eq!(
        restored.identity().response_id.as_deref(),
        Some("resp-vertex-1")
    );
}

/// Constructed SDK values isolate id normalization without credentials.
#[test]
fn multiple_missing_call_ids_are_distinct() {
    let convert = || {
        complete(create_parts_response((0..3).map(|i| {
            vertexai::model::Part::new().set_function_call(
                vertexai::model::FunctionCall::new()
                    .set_name("same")
                    .set_args(serde_json::json!({"n":i}).as_object().unwrap().clone()),
            )
        })))
        .unwrap()
    };
    let first = convert();
    let calls: Vec<_> = first
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(
        calls
            .iter()
            .map(|call| &call.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert!(calls.iter().all(|call| call.id.provider().is_none()));
    for (i, call) in calls.iter().enumerate() {
        assert_eq!(call.function.arguments_value(), serde_json::json!({"n":i}));
    }
}

/// Vertex never issues call ids (its `FunctionCall` has no id field). The
/// `index`-th call of a response mints its own handle at the converter, so
/// two same-tool calls in one turn stay distinct and a text part between them
/// does not merge them.
#[test]
fn id_less_calls_in_one_turn_mint_distinct_handles_by_position() {
    use rig_core::message::CallId;
    let call = || {
        let function_call = vertexai::model::FunctionCall::new()
            .set_name("same".to_string())
            .set_args(serde_json::Map::new());
        vertexai::model::Part::new().set_function_call(function_call)
    };
    let build = || {
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts([
                call(),
                vertexai::model::Part::new().set_text("between".to_string()),
                call(),
                call(),
            ]);
        let candidate = finished(vertexai::model::Candidate::new().set_content(content));
        vertexai::model::GenerateContentResponse::new().set_candidates([candidate])
    };
    let first = complete(build()).expect("converts");
    let ids: Vec<_> = first
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call.id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids.len(), 3);
    assert!(ids.iter().all(CallId::is_local));
    assert_eq!(
        ids.iter().collect::<std::collections::HashSet<_>>().len(),
        3
    );
}

/// A signature on answer text stays on that text and replays on it.
#[test]
fn answer_text_signature_is_kept_and_replayed_on_its_part() {
    let raw = b"\x00\x01answer-sig\xff";
    let part = vertexai::model::Part::new()
        .set_text("the answer".to_string())
        .set_thought_signature(raw.to_vec());
    let response: CompletionResponse = create_parts_response([part]).complete().unwrap();

    let block = response.choice.first().expect("the answer text");
    assert!(matches!(block, AssistantContent::Text(text) if text.text == "the answer"));
    assert_eq!(
        item(block),
        Some(&json!({ "text": "the answer", "thoughtSignature": BASE64.encode(raw) }))
    );

    let replayed: vertexai::model::Content = crate::types::message::content_from_message(
        rig_core::message::Message::from(response.choice.clone()),
    )
    .expect("the turn replays");
    assert_eq!(replayed.parts.len(), 1);
    assert_eq!(replayed.parts[0].thought_signature.to_vec(), raw.to_vec());
    assert!(!replayed.parts[0].thought);
}
