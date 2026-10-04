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
        _payload: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
    }
}

/// The content a turn holding `choice` replays as, encoded for the model
/// the replies come from.
pub(crate) fn replay(
    choice: Vec<rig_core::message::AssistantContent>,
) -> Result<vertexai::model::Content, rig_core::error::EncodeError> {
    use rig_core::wire::{Mode, Wire};
    let mut request = CompletionRequest::new("next");
    request.chat_history = vec![rig_core::message::Message::from(choice)];
    let request = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)?;
    request
        .contents
        .into_iter()
        .next()
        .ok_or_else(|| rig_core::error::EncodeError::request("no content"))
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
use rig_core::message::{AssistantContent, Text};
use serde_json::json;

/// `candidate` as a finished one: Vertex states why every unary reply ended.
fn finished(candidate: vertexai::model::Candidate) -> vertexai::model::Candidate {
    candidate.set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
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

    let replayed = replay(response.choice.clone()).expect("the turn replays");
    assert_eq!(replayed.parts, parts);
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
    // The candidate's metadata stays in the response's raw document.
    let native = original.raw.pointer("/candidates/0");
    assert_eq!(
        native.map(|native| &native["avgLogprobs"]),
        Some(&json!(-0.25))
    );
    // The turn reads the REST JSON, which spells enums by name.
    assert_eq!(
        native.map(|native| &native["safetyRatings"][0]["category"]),
        Some(&json!("HARM_CATEGORY_HARASSMENT"))
    );
    assert_eq!(
        original.raw.pointer("/candidates/0/finishReason"),
        Some(&json!("STOP"))
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
