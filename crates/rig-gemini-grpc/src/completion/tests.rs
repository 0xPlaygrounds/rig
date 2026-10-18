use super::*;
use crate::proto;
use rig_core::Model;
use rig_core::completion::CompletionResponse;
use rig_core::streaming::CompletionStream;

/// The protobuf request `request` sends to `model`.
fn create_grpc_request(
    model: &str,
    request: CompletionRequest,
) -> Result<GenerateContentRequest, EncodeError> {
    GenerateContent::new(model).encode(request, Mode::Unary)
}

/// Answers every request with scripted protobuf replies, each transcoded
/// as the gRPC transport transcodes it.
#[derive(Clone)]
pub(crate) struct Scripted(
    std::sync::Arc<std::sync::Mutex<Vec<Result<GenerateContentResponse, ProviderError>>>>,
);

impl Transport<GenerateContent> for Scripted {
    fn send(
        &self,
        _request: GenerateContentRequest,
        exchange: Exchange,
    ) -> Opening<GenerateContentChunk> {
        let replies = std::mem::take(&mut *self.0.lock().expect("script lock"));
        // A unary reply is one message, opened as the endpoint opens it.
        match (exchange.mode, replies.as_slice()) {
            (Mode::Unary, [Ok(response)]) => match unary(response) {
                Ok(opened) => Opening::ready(opened),
                Err(error) => Opening::failed(error),
            },
            _ => Opening::ready(Opened::new(futures::stream::iter(
                replies
                    .into_iter()
                    .map(|reply| reply.and_then(|response| rest_chunk(&response))),
            ))),
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

/// RPC failures keep the status text and its code, and only the transient
/// codes retry; no HTTP status is invented.
#[test]
fn rpc_errors_carry_the_code_and_its_verdict() {
    let busy = rpc_error(&tonic::Status::resource_exhausted("quota"));
    assert!(busy.is_retryable());
    assert_eq!(
        rig_core::error::ErrorReport::from(&busy).code.as_deref(),
        Some("RESOURCE_EXHAUSTED")
    );
    let bad = rpc_error(&tonic::Status::invalid_argument("dims"));
    assert!(!bad.is_retryable());
    assert_eq!(
        rig_core::error::ErrorReport::from(&bad).code.as_deref(),
        Some("INVALID_ARGUMENT")
    );
    assert!(bad.to_string().contains("dims"), "{bad}");
    assert_eq!(bad.provider_response_status(), None);
}

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
    request.chat_history = rig_core::completion::adapt(&history, &GenerateContent::new(model))
        .into_iter()
        .map(Into::into)
        .collect();
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
        parameters: serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}), annotations: Default::default() 
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
        let response = crate::rest::from_rest::<GenerateContentResponse>(
            serde_json::json!({"candidates": [candidate], "modelVersion": "m"}),
        )
        .expect("a reply message");
        rest_chunk(&response).expect("a reply chunk")
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
/// service tier.
#[test]
fn options_take_the_cells_the_proto_declares() {
    use rig_core::completion::ReplayTarget as _;
    use rig_core::completion::{
        CompletionRequest, GenerationOptions, ServiceTier, options::Mapping,
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
}

/// A current Gemini model over gRPC has the Gemini API's catalog facts, and
/// a Gemini 3 effort reaches the proto's `thinking_level`.
#[test]
fn a_gemini_3_effort_reaches_the_thinking_level() {
    use rig_core::catalog::{Catalog, ModelFacts};
    use rig_core::completion::{CompletionRequest, Effort, GenerationOptions};
    use rig_core::wire::{Mode, Wire as _};

    let resolved = Catalog::builtin()
        .resolve("gemini-grpc/gemini-3.8-flash")
        .expect("the catalog lists the model over gRPC");
    let wire =
        GenerateContent::new("gemini-3.8-flash").with_facts(ModelFacts::new(resolved.shared()));
    let request =
        CompletionRequest::new("hi").options(GenerationOptions::default().reasoning(Effort::High));
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let level = encoded
        .generation_config
        .and_then(|config| config.thinking_config)
        .and_then(|thinking| thinking.thinking_level);
    assert_eq!(
        level,
        Some(proto::thinking_config::ThinkingLevel::High as i32)
    );
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

/// The facts the gRPC wire is given are the facts it answers with: its
/// `ReplayTarget::facts`, which the shared option rules and the cost read,
/// and its descriptor's spec, which `DynModel::spec` returns.
#[test]
fn the_wire_answers_from_the_facts_it_is_given() {
    use rig_core::catalog::{ModelFacts, ModelSpec};
    use rig_core::completion::ReplayTarget as _;
    use rig_core::wire::Wire as _;

    let vendor =
        rig_core::providers::registry::ProviderId::catalog("gemini-grpc").expect("a vendor");
    let spec = ModelSpec::new(vendor, "gemini-2.5-flash").with_max_output_tokens(1_234);
    let wire = GenerateContent::new("gemini-2.5-flash").with_facts(ModelFacts::new(spec));
    let bound = wire.facts().and_then(ModelFacts::spec);
    assert_eq!(bound.and_then(|spec| spec.max_output_tokens), Some(1_234));
    assert_eq!(
        wire.describe()
            .spec()
            .and_then(|spec| spec.max_output_tokens),
        Some(1_234)
    );
}

fn response(parts: Vec<proto::Part>, finish_reason: i32) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts,
                role: "model".to_string(),
            }),
            finish_reason,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn text_part(text: &str) -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::Text(text.to_string())),
        ..Default::default()
    }
}

/// The terminal frame of a stream: the last `GenerateContentResponse`,
/// carrying usage, a finish reason and the response id, so the stream
/// ends with a fully populated terminal record.
fn terminal_frame() -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![proto::Part {
                    data: Some(proto::part::Data::Text("!".to_string())),
                    ..Default::default()
                }],
                role: "model".to_string(),
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            ..Default::default()
        }],
        usage_metadata: Some(proto::UsageMetadata {
            prompt_token_count: 3,
            candidates_token_count: 2,
            total_token_count: 5,
            cached_content_token_count: 0,
            tool_use_prompt_token_count: 0,
            thoughts_token_count: 0,
            ..Default::default()
        }),
        model_version: "gemini-2.5-flash".to_string(),
        response_id: "resp-grpc-stream".to_string(),
        ..Default::default()
    }
}

/// Drive protobuf events through the pipeline the `Model` seam
/// uses, returning the finished response.
async fn normalized_terminal(events: Vec<proto::GenerateContentResponse>) -> CompletionResponse {
    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    stream.finish().await.expect("the stream must end")
}

/// A streamed reply's `raw` is the unary message its chunks add up to:
/// the text parts joined, the terminal chunk's finish, usage and ids. It
/// reads back into that prost message, equals the `raw` of the same reply
/// served unary, and re-normalizing it reproduces every normalized field.
#[tokio::test]
async fn streamed_raw_is_the_unary_message_the_chunks_add_up_to() {
    let terminal =
        normalized_terminal(vec![response(vec![text_part("hi")], 0), terminal_frame()]).await;
    let mut whole = terminal_frame();
    for candidate in &mut whole.candidates {
        candidate.content = Some(proto::Content {
            parts: vec![text_part("hi!")],
            role: "model".to_string(),
        });
    }

    let raw = &terminal.raw;
    let typed: proto::GenerateContentResponse =
        crate::rest::from_rest(raw.clone()).expect("raw must read back");
    assert_eq!(typed, whole);
    assert_eq!(
        *raw,
        complete(whole.clone())
            .expect("the unary reply decodes")
            .raw,
        "the streamed raw is the unary raw of the same reply"
    );
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        Some(&serde_json::json!("STOP"))
    );

    let renormalized = normalized_terminal(vec![typed]).await;
    assert_eq!(terminal.identity(), renormalized.identity());
    assert_eq!(terminal.finish_reason(), renormalized.finish_reason());
    assert_eq!(terminal.model(), renormalized.model());
    assert_eq!(terminal.usage, renormalized.usage);
    assert_eq!(
        terminal.finish_reason(),
        Some(rig_core::completion::FinishReason::Stop)
    );
    assert_eq!(terminal.model(), Some("gemini-2.5-flash"));
    assert_eq!(
        terminal.identity().response_id.as_deref(),
        Some("resp-grpc-stream")
    );
}

/// #2475: a blocked prompt's only reply carries `promptFeedback` and no
/// candidate. It is the provider's refusal, naming the reason and ratings,
/// as on REST, never a truncated reply.
#[test]
fn a_blocked_prompt_is_a_refusal_on_grpc() {
    let blocked = proto::GenerateContentResponse {
        prompt_feedback: Some(proto::PromptFeedback {
            block_reason: proto::prompt_feedback::BlockReason::Safety as i32,
            safety_ratings: vec![proto::SafetyRating {
                category: proto::HarmCategory::HateSpeech as i32,
                probability: proto::safety_rating::HarmProbability::High as i32,
                blocked: true,
            }],
        }),
        ..Default::default()
    };
    let error = complete(blocked).expect_err("a blocked prompt is no answer");
    assert!(error.report().refusal, "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains("block_reason=SAFETY")
            && message.contains("HARM_CATEGORY_HATE_SPEECH=HIGH"),
        "{message}"
    );
}

/// A part of a kind the proto does not declare arrives with no data, its
/// signature alone. The turn replays to the same model without it, never
/// as a part with no data.
#[test]
fn a_part_of_an_undeclared_kind_never_replays_without_data() {
    use rig_core::message::Message;
    let signed = proto::Part {
        data: None,
        thought_signature: b"sig".to_vec(),
        ..Default::default()
    };
    let response = complete(response(
        vec![signed, text_part("answer")],
        proto::candidate::FinishReason::Stop as i32,
    ))
    .expect("the reply decodes");
    let history = vec![
        Message::user("q"),
        Message::Assistant(response.continued(response.choice.clone())),
    ];
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    let mut request = rig_core::completion::CompletionRequest::new("next");
    request.chat_history = rig_core::completion::adapt(&history, &wire)
        .into_iter()
        .map(Into::into)
        .collect();
    let request = rig_core::wire::Wire::encode(&wire, request, rig_core::wire::Mode::Unary)
        .expect("the history encodes");
    let parts: Vec<_> = request.contents.iter().flat_map(|c| &c.parts).collect();
    assert!(parts.iter().all(|part| part.data.is_some()), "{parts:?}");
    assert!(
        parts
            .iter()
            .any(|part| part.data == Some(proto::part::Data::Text("answer".to_owned()))),
        "{parts:?}"
    );
}

/// The cited text and sources of each citation on the reply's text blocks.
fn citations(response: &CompletionResponse) -> Vec<(String, Vec<String>, Vec<Option<f32>>)> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            rig_core::message::AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .flat_map(|text| {
            text.citations().iter().map(|citation| {
                let urls = citation
                    .sources
                    .iter()
                    .map(|source| match &source.location {
                        rig_core::message::SourceLocation::Url { url } => url.clone(),
                        other => format!("{other:?}"),
                    });
                let scores = citation.sources.iter().map(|source| source.confidence);
                (
                    text.cited(citation).unwrap_or_default().to_owned(),
                    urls.collect(),
                    scores.collect(),
                )
            })
        })
        .collect()
}

/// Hand-built replies, as no recording carries grounding: the proto keeps
/// the candidate's grounding and recitation metadata, so a segment counted
/// in bytes past a multi-byte dash cites its text and sources, unary or
/// streamed with the metadata on the last chunk.
#[tokio::test]
async fn grounding_and_recitations_cite_the_answer_unary_or_streamed() {
    const ANSWER: &str = "Spain won 2\u{2013}1. Oyarzabal scored late.";
    let quoted = "Oyarzabal scored late.";
    let at = ANSWER.find(quoted).expect("the answer holds it");
    let text = |text: &str| proto::Part {
        data: Some(proto::part::Data::Text(text.to_owned())),
        ..Default::default()
    };
    let web = |uri: &str| proto::GroundingChunk {
        chunk_type: Some(proto::grounding_chunk::ChunkType::Web(
            proto::grounding_chunk::Web {
                uri: Some(uri.to_owned()),
                title: Some("example".to_owned()),
            },
        )),
    };
    let metadata = |mut response: proto::GenerateContentResponse| {
        let candidate = response.candidates.first_mut().expect("a candidate");
        candidate.grounding_metadata = Some(proto::GroundingMetadata {
            grounding_chunks: vec![web("https://example.com/a"), web("https://example.com/b")],
            grounding_supports: vec![proto::GroundingSupport {
                segment: Some(proto::Segment {
                    part_index: 0,
                    start_index: at as i32,
                    end_index: (at + quoted.len()) as i32,
                    text: quoted.to_owned(),
                }),
                grounding_chunk_indices: vec![1, 0],
                confidence_scores: vec![0.5, 0.25],
            }],
        });
        candidate.citation_metadata = Some(proto::CitationMetadata {
            citation_sources: vec![proto::CitationSource {
                start_index: None,
                end_index: Some(9),
                uri: Some("https://example.com/c".to_owned()),
                license: None,
            }],
        });
        response
    };
    let expected = vec![
        (
            quoted.to_owned(),
            vec![
                "https://example.com/b".to_owned(),
                "https://example.com/a".to_owned(),
            ],
            vec![Some(0.5), Some(0.25)],
        ),
        (
            "Spain won".to_owned(),
            vec!["https://example.com/c".to_owned()],
            vec![None],
        ),
    ];

    let unary = complete(metadata(response(vec![text(ANSWER)], 1))).expect("the reply decodes");
    assert_eq!(citations(&unary), expected);

    let (head, tail) = ANSWER.split_at(at);
    let mut stream = stream_from_events(vec![
        Ok(response(vec![text(head)], 0)),
        Ok(metadata(response(vec![text(tail)], 1))),
    ]);
    while stream.next().await.is_some() {}
    let streamed = stream.finish().await.expect("the stream completes");
    assert_eq!(citations(&streamed), expected);
}
