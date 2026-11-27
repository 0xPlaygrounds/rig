//! Wire-conformance suite for the Vertex AI `GenerateContent` typed-event wire.
//!
//! Fixture frames are already-typed SDK responses streamed through an offline
//! `PredictionService` stub into `VertexAi`'s `Transport<GenerateContent>`,
//! exercising `rest_chunk`, `rpc_error`, the Gemini decoder, and terminal
//! normalization with no network channel.

use google_cloud_aiplatform_v1 as vertexai;
use google_cloud_gax::streaming::ResponseStream;
use rig_core::completion::{CompletionRequest, FinishReason};
use rig_core::error::ProviderError;
use rig_core::test_utils::streaming_conformance::{
    InterleavedReasoningFixture, ProviderWireFixture, WireDriver, WireInput, event_frame,
    fixtures::drain,
};
use rig_vertexai::VertexAi;
use rig_vertexai::completion::GEMINI_2_5_FLASH;

#[derive(Debug)]
struct ScriptedPredictionService {
    stream: std::sync::Mutex<
        Option<google_cloud_gax::Result<ResponseStream<vertexai::model::GenerateContentResponse>>>,
    >,
}

impl vertexai::stub::PredictionService for ScriptedPredictionService {
    async fn stream_generate_content(
        &self,
        _req: vertexai::model::GenerateContentRequest,
        _options: google_cloud_gax::options::RequestOptions,
    ) -> vertexai::Result<ResponseStream<vertexai::model::GenerateContentResponse>> {
        let mut guard = self.stream.lock().map_err(|_| {
            google_cloud_gax::error::Error::io(std::io::Error::other("script lock poisoned"))
        })?;
        guard.take().unwrap_or_else(|| {
            Err(google_cloud_gax::error::Error::io(std::io::Error::other(
                "unexpected stream_generate_content call",
            )))
        })
    }
}

fn driver() -> WireDriver {
    WireDriver::new("vertexai", |chunks| {
        Box::pin(async move {
            let mut items = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                match chunk {
                    Ok(frame) => {
                        let response = frame
                            .downcast_event::<vertexai::model::GenerateContentResponse>()
                            .ok_or_else(|| {
                                ProviderError::Provider(
                                    "vertexai conformance frames must be SDK responses".to_owned(),
                                )
                            })?;
                        items.push(Ok(response.clone()));
                    }
                    Err(error) => {
                        items.push(Err(google_cloud_gax::error::Error::io(
                            std::io::Error::other(error.to_string()),
                        )));
                    }
                }
            }

            let (tx, rx) = tokio::sync::mpsc::channel(items.len().max(1));
            for item in items {
                if tx.send(item).await.is_err() {
                    return Err(ProviderError::Provider(
                        "conformance channel closed prematurely".to_owned(),
                    ));
                }
            }
            drop(tx);

            let stub = std::sync::Arc::new(ScriptedPredictionService {
                stream: std::sync::Mutex::new(Some(Ok(ResponseStream::from(rx)))),
            });
            let service =
                vertexai::client::PredictionService::from_stub::<ScriptedPredictionService>(stub);
            let client = VertexAi::builder()
                .with_project("test-project")
                .with_location("us-central1")
                .with_prediction_service(service)
                .build()
                .map_err(ProviderError::request)?;
            let stream = client
                .completion(GEMINI_2_5_FLASH)
                .stream(CompletionRequest::new("hello"))?;
            Ok(drain(stream).await)
        })
    })
}

fn response(
    parts: Vec<vertexai::model::Part>,
    finish_reason: Option<vertexai::model::candidate::FinishReason>,
) -> vertexai::model::GenerateContentResponse {
    let mut candidate = vertexai::model::Candidate::new().set_content(
        vertexai::model::Content::new()
            .set_role("model")
            .set_parts(parts),
    );
    if let Some(reason) = finish_reason {
        candidate = candidate.set_finish_reason(reason);
    }
    vertexai::model::GenerateContentResponse::new()
        .set_candidates([candidate])
        .set_model_version(GEMINI_2_5_FLASH)
        .set_response_id("resp-1")
}

fn text_part(text: &str) -> vertexai::model::Part {
    vertexai::model::Part::new().set_text(text)
}

fn thought_part(text: &str) -> vertexai::model::Part {
    vertexai::model::Part::new()
        .set_thought(true)
        .set_text(text)
}

fn function_call_part(name: &str) -> vertexai::model::Part {
    let args = serde_json::Map::from_iter([(
        "city".to_owned(),
        serde_json::Value::String("Tokyo".to_owned()),
    )]);
    vertexai::model::Part::new().set_function_call(
        vertexai::model::FunctionCall::new()
            .set_name(name)
            .set_args(args),
    )
}

fn usage(
    prompt: i32,
    candidates: i32,
    total: i32,
) -> vertexai::model::generate_content_response::UsageMetadata {
    vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(prompt)
        .set_candidates_token_count(candidates)
        .set_total_token_count(total)
}

fn terminal(
    usage_metadata: Option<vertexai::model::generate_content_response::UsageMetadata>,
) -> WireInput {
    let mut response = response(
        Vec::new(),
        Some(vertexai::model::candidate::FinishReason::Stop),
    );
    if let Some(usage_metadata) = usage_metadata {
        response = response.set_usage_metadata(usage_metadata);
    }
    event_frame(response)
}

fn fixture() -> ProviderWireFixture {
    ProviderWireFixture {
        driver: driver(),
        text_frames: vec![event_frame(response(vec![text_part("hi")], None))],
        expected_texts: vec!["hi"],
        tool_call_frames: vec![event_frame(response(
            vec![function_call_part("get_weather")],
            None,
        ))],
        expected_tool_name: "get_weather",
        // Vertex AI delivers function calls whole; arguments never stream.
        partial_tool_call_frames: None,
        terminal_frames: vec![terminal(Some(usage(5, 2, 7)))],
        expected_usage_total: 7,
        expected_finish_reason: Some(FinishReason::Stop),
        zero_usage_terminal_frames: Some(vec![terminal(None)]),
        bare_terminal_frames: None,
        // The SDK owns wire decoding, so corrupt frames surface as SDK errors.
        malformed_frame: None,
        unknown_event_frame: None,
        defective_known_frame: None,
        delta_less_prelude_frame: None,
        refusal: None,
        interleaved_reasoning: Some(InterleavedReasoningFixture {
            frames: vec![
                event_frame(response(vec![thought_part("before tool")], None)),
                event_frame(response(vec![function_call_part("get_weather")], None)),
                event_frame(response(vec![thought_part("after tool")], None)),
                terminal(Some(usage(5, 2, 7))),
            ],
            first_reasoning: "before tool",
            tool_name: "get_weather",
            second_reasoning: "after tool",
        }),
    }
}

rig_core::streaming_conformance_suite! {
    provider: "vertexai",
    fixture: fixture(),
    manifest: [zero_usage_terminal, interleaved_reasoning],
}

/// Compile-linked manifest of the wire families this binary covers.
const SUITE_FAMILIES: &[&str] = &[WIRE_FAMILY];

#[test]
fn suite_families_are_registered_wire_families() {
    for family in SUITE_FAMILIES {
        assert!(
            rig_core::test_utils::streaming_conformance::WIRE_FAMILIES.contains(family),
            "suite names wire family {family:?}, absent from WIRE_FAMILIES"
        );
    }
}
