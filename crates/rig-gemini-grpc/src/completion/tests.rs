use super::*;
use crate::proto;
use rig_core::Model;
use rig_core::streaming::CompletionStream;

/// The protobuf request `request` sends to `model`.
fn create_grpc_request(
    model: &str,
    request: CompletionRequest,
) -> Result<GenerateContentRequest, EncodeError> {
    GenerateContent::new(model).encode(request, Mode::Unary)
}

/// Answers every request with scripted protobuf replies, one per frame.
#[derive(Clone)]
pub(crate) struct Scripted(
    std::sync::Arc<std::sync::Mutex<Vec<Result<GenerateContentResponse, ProviderError>>>>,
);

impl Transport<GenerateContent> for Scripted {
    fn send(
        &self,
        _request: GenerateContentRequest,
        exchange: Exchange,
    ) -> Opening<GenerateContentResponse> {
        let replies = std::mem::take(&mut *self.0.lock().expect("script lock"));
        // A unary reply is one message, opened as the endpoint opens it.
        match (exchange.mode, replies.as_slice()) {
            (Mode::Unary, [Ok(response)]) => match unary(response.clone()) {
                Ok(opened) => Opening::ready(opened),
                Err(error) => Opening::failed(error),
            },
            _ => Opening::ready(Opened::new(futures::stream::iter(replies))),
        }
    }
}

fn scripted(
    replies: Vec<Result<GenerateContentResponse, ProviderError>>,
) -> Model<GenerateContent, Scripted> {
    Model::new(
        GenerateContent::new(GEMINI_2_5_FLASH),
        Scripted(std::sync::Arc::new(std::sync::Mutex::new(replies))),
    )
}

fn hello() -> CompletionRequest {
    rig_core::completion::CompletionRequest::new("hello")
}

/// `response` as the unary endpoint answers it.
pub(crate) fn complete(
    response: GenerateContentResponse,
) -> Result<rig_core::completion::CompletionResponse, ProviderError> {
    futures::executor::block_on(scripted(vec![Ok(response)]).call(hello()))
}

/// The stream the endpoint yields for scripted chunks.
pub(crate) fn stream_from_events(
    chunks: Vec<Result<GenerateContentResponse, ProviderError>>,
) -> CompletionStream {
    scripted(chunks).stream(hello()).expect("the stream opens")
}

// ============================================================
// rpc_error — pins the from_provider_body usage on the RPC error path
// ============================================================

/// Gemini's `FunctionResponsePart` declares inline data only, so a URL
/// image a tool returns reaches Gemini 3 in the user turn after the results.
#[test]
fn a_url_tool_result_image_follows_the_results_on_gemini_3() {
    use rig_core::message::{Image, ImageMediaType, ToolResultContent, UserContent};
    const URL: &str = "https://example.com/shot.png";
    let model = "gemini-3-flash-preview";
    let call = message::ToolCall::new(
        message::CallId::from_wire("call_1"),
        message::ToolFunction::new(
            message::ToolName::new("shoot").expect("a tool name"),
            serde_json::json!({}),
        ),
    );
    let image = Image {
        data: message::DocumentSourceKind::url(URL),
        media_type: Some(ImageMediaType::PNG),
        ..Default::default()
    };
    let history = vec![
        message::Message::user("q"),
        message::Message::Assistant(message::AssistantMessage::new(vec![
            message::AssistantContent::ToolCall(call.clone()),
        ])),
        message::Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Image(image)]),
            )],
        },
    ];
    let mut request = CompletionRequest::new("next");
    request.chat_history = rig_core::completion::adapt(&history, &GenerateContent::new(model));
    let request = create_grpc_request(model, request).expect("the history encodes");
    let data: Vec<_> = request
        .contents
        .iter()
        .flat_map(|content| &content.parts)
        .filter_map(|part| part.data.as_ref())
        .collect();
    let response = data
        .iter()
        .position(|data| matches!(data, proto::part::Data::FunctionResponse(_)))
        .expect("the result is sent");
    let file = data
        .iter()
        .position(|data| matches!(data, proto::part::Data::FileData(file) if file.file_uri == URL))
        .expect("the image is sent");
    assert!(response < file, "{data:?}");
}

/// The gRPC wire carries the model's chain-of-thought in the same `parts`
/// array as the answer, flagged by `thought` — same shape as the REST
/// wire, where reading it as output text was a live-confirmed defect.
/// There is no cassette harness for this transport (it is protobuf over
/// gRPC, not HTTP), so the wire shape is stated directly.
/// A signature on answer text stays on that text: Gemini's rules return a
/// signature inside the part that carried it, never merged into another.
/// A reply's `raw` is its REST JSON, which reads back as the same
/// message, and normalizing the restored message agrees with normalizing
/// the original. Fields rig never normalizes (`cachedContentTokenCount`,
/// the candidate's `finishMessage`) survive both directions.
#[test]
fn generate_content_response_round_trips_through_its_rest_json() {
    let raw = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![proto::Part {
                    data: Some(proto::part::Data::Text("hello".to_string())),
                    ..Default::default()
                }],
                role: "model".to_string(),
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            index: Some(0),
            finish_message: Some("done".to_string()),
            ..Default::default()
        }],
        usage_metadata: Some(proto::UsageMetadata {
            prompt_token_count: 10,
            candidates_token_count: 20,
            total_token_count: 30,
            cached_content_token_count: 4,
            ..Default::default()
        }),
        model_version: "gemini-2.5-flash".to_string(),
        response_id: "resp-grpc-1".to_string(),
        prompt_feedback: None,
    };

    let value = crate::rest::to_rest(&raw).expect("transcodes");
    assert_eq!(
        value.pointer("/usageMetadata/cachedContentTokenCount"),
        Some(&serde_json::json!(4))
    );
    assert_eq!(
        value.pointer("/candidates/0/finishMessage"),
        Some(&serde_json::json!("done"))
    );
    assert_eq!(
        value.pointer("/candidates/0/finishReason"),
        Some(&serde_json::json!("STOP"))
    );

    let back: proto::GenerateContentResponse =
        crate::rest::from_rest(value.clone()).expect("reads back");
    assert_eq!(back, raw);

    let original = complete(raw.clone()).expect("original converts");
    assert_eq!(original.raw, value, "the response's raw is its REST JSON");
    let restored = complete(back).expect("restored converts");
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(
        restored.finish_reason(),
        Some(rig_core::completion::FinishReason::Stop)
    );
}

/// The REST JSON `history` sends to `model` over gRPC, prepared as the
/// driver prepares it: the protobuf request read back as its REST JSON.
fn sent(model: &str, history: Vec<message::Message>) -> serde_json::Value {
    use rig_core::wire::Operation;
    let wire = GenerateContent::new(model);
    let request = rig_core::operation::Completion::prepare(
        CompletionRequest::from(history),
        &wire.describe(),
    )
    .expect("the history prepares");
    crate::rest::to_rest(
        &wire
            .encode(request, Mode::Unary)
            .expect("the history encodes"),
    )
    .expect("the request has REST JSON")
}

/// #2658, round 4 NEW-1: a user video with Gemini's `videoMetadata` (clip
/// offsets), which the shared encoder sends, reaches the gRPC request: the
/// proto declares every part field that encoder emits.
#[test]
fn a_user_video_keeps_its_video_metadata() {
    let video = message::Video {
        data: message::DocumentSourceKind::Url("https://www.youtube.com/watch?v=abc".to_owned()),
        media_type: Some(message::VideoMediaType::MP4),
        additional_params: Some(
            serde_json::json!({"videoMetadata": {"startOffset": "10s", "endOffset": "20s"}}),
        ),
    };
    let body = sent(
        "gemini-2.5-flash",
        vec![message::Message::User {
            content: vec![
                message::UserContent::Video(video),
                message::UserContent::text("summarise this clip"),
            ],
        }],
    );
    assert_eq!(
        body["contents"][0]["parts"][0]["videoMetadata"],
        serde_json::json!({"startOffset": "10s", "endOffset": "20s"}),
        "{body}"
    );
}

/// The shared encoder's whole request transcodes: generation, thinking and
/// image config, safety settings, hosted tools and the tool choice all reach
/// the gRPC request rather than being dropped.
#[test]
fn the_rest_request_transcodes_in_full() {
    let mut request = CompletionRequest::new("hi").tools(vec![rig_core::completion::ToolDefinition {
        name: message::ToolName::new("add").expect("a tool name"),
        description: "add".to_owned(),
        parameters: serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
    }]);
    request.tool_choice = Some(message::ToolChoice::Specific {
        function_names: vec![message::ToolName::new("add").expect("a tool name")],
    });
    request.temperature = Some(0.5);
    request.additional_params = Some(serde_json::json!({
        "generationConfig": {
            "topK": 3,
            "thinkingConfig": {"thinkingLevel": "low", "includeThoughts": true},
            "imageConfig": {"aspectRatio": "1:1"},
        },
        "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
        "tools": [{"googleSearch": {}}, {"codeExecution": {}}],
    }));
    let body = crate::rest::to_rest(
        &GenerateContent::new("gemini-3-flash-preview")
            .encode(request, Mode::Unary)
            .expect("the request transcodes"),
    )
    .expect("the request has REST JSON");
    assert_eq!(body["generationConfig"]["topK"], 3, "{body}");
    assert_eq!(body["generationConfig"]["temperature"], 0.5, "{body}");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "LOW",
        "{body}"
    );
    assert_eq!(
        body["generationConfig"]["imageConfig"]["aspectRatio"], "1:1",
        "{body}"
    );
    assert_eq!(
        body["safetySettings"][0]["threshold"], "BLOCK_NONE",
        "{body}"
    );
    assert_eq!(
        body["toolConfig"]["functionCallingConfig"]["mode"], "ANY",
        "{body}"
    );
    assert_eq!(
        body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
        serde_json::json!(["add"]),
        "{body}"
    );
    assert_eq!(
        body["tools"][1],
        serde_json::json!({"googleSearch": {}}),
        "{body}"
    );
    assert_eq!(
        body["tools"][2],
        serde_json::json!({"codeExecution": {}}),
        "{body}"
    );
}

/// Round 4 NEW-3: a streamed part that carries only a thought signature
/// joins the text before it, so the signature replays to the same model.
#[test]
fn a_signature_only_part_replays_its_signature() {
    let chunk = |parts: serde_json::Value, finish: Option<&str>| {
        let mut candidate = serde_json::json!({"content": {"role": "model", "parts": parts}});
        if let Some(finish) = finish {
            candidate["finishReason"] = serde_json::json!(finish);
        }
        crate::rest::from_rest::<GenerateContentResponse>(
            serde_json::json!({"candidates": [candidate], "modelVersion": "m"}),
        )
        .expect("a reply chunk")
    };
    let frames = vec![
        chunk(serde_json::json!([{"text": "Answer"}]), None),
        chunk(
            serde_json::json!([{"thoughtSignature": "c2ln"}]),
            Some("STOP"),
        ),
    ];
    let response = rig_core::test_utils::history::decode(
        &GenerateContent::new("gemini-3-flash-preview"),
        Mode::Streaming,
        frames,
    )
    .expect("the reply decodes");
    let history = vec![
        message::Message::user("q"),
        response.message().expect("a turn"),
        message::Message::user("n"),
    ];
    let body = sent("gemini-3-flash-preview", history);
    assert_eq!(
        body["contents"][1]["parts"][0]["thoughtSignature"], "c2ln",
        "{body}"
    );
}

/// A user part's own `mediaResolution` has no field in the gRPC `Part`
/// (Google's `google/ai/generativelanguage/v1beta/content.proto` and the
/// vendored `proto/gemini.proto` declare none; v1alpha neither), so the wire
/// refuses it loudly rather than dropping it. The REST API honours one
/// (`gemini/media_resolution/per_part_low`); Rig sends none on any wire.
#[test]
fn a_part_with_its_own_media_resolution_is_refused() {
    let content = serde_json::json!({
        "role": "user",
        "parts": [{
            "inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="},
            "mediaResolution": {"level": "MEDIA_RESOLUTION_LOW"}
        }]
    });
    let error = crate::rest::from_rest::<proto::Content>(content)
        .expect_err("a part's mediaResolution is not a gRPC field");
    assert!(
        error.to_string().contains("mediaResolution"),
        "the refusal names the field: {error}"
    );
}

/// gRPC takes the GenerateContent cells for what its proto declares: no
/// service tier, and no thinking level, which Google's proto does not name.
#[test]
fn options_take_the_cells_the_proto_declares() {
    use rig_core::completion::ReplayTarget as _;
    use rig_core::completion::{
        CompletionRequest, Effort, GenerationOptions, ServiceTier, options::Mapping,
    };
    use rig_core::wire::{Mode, Wire as _};

    let wire = GenerateContent::new("gemini-2.5-flash");
    let request = CompletionRequest::new("hi")
        .top_p(0.5)
        .seed(7)
        .stop(["END"]);
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let config = encoded
        .generation_config
        .expect("the request has a generation config");
    assert_eq!(config.top_p, Some(0.5));
    assert_eq!(config.seed, Some(7));
    assert_eq!(config.stop_sequences, vec!["END".to_owned()]);

    let answers = |options: GenerationOptions| {
        let request = CompletionRequest::new("hi").options(options);
        wire.map_options(&request, request.options.fields())
    };
    assert!(matches!(
        answers(GenerationOptions::default().service_tier(ServiceTier::Default)).service_tier,
        Mapping::Unsupported(_)
    ));
    assert!(matches!(
        answers(GenerationOptions::default().reasoning(Effort::High)).reasoning,
        Mapping::Unsupported(_)
    ));
}

/// A raw `generationConfig: null` is absent, as it was before the merge:
/// the typed fields still reach the gRPC request.
#[test]
fn a_raw_null_generation_config_keeps_the_typed_fields() {
    use rig_core::wire::{Mode, Wire as _};

    let request = CompletionRequest::new("hi")
        .temperature(0.2)
        .max_tokens(64)
        .additional_params(serde_json::json!({"generationConfig": null}));
    let encoded = GenerateContent::new("gemini-2.5-flash")
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let config = encoded
        .generation_config
        .expect("the request has a generation config");
    assert_eq!(config.temperature, Some(0.2));
    assert_eq!(config.max_output_tokens, Some(64));
}
