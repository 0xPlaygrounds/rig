use super::*;
use google_cloud_gax::error::{
    Error,
    rpc::{Code, Status},
};
use rig_core::error::ErrorKind;

/// A service error from the SDK carries its RPC code: the code is the
/// provider's own, the reply has no HTTP status, and only the codes that
/// say the server was unreachable, overloaded or cut the call short retry.
#[test]
fn rpc_codes_classify_retryability_and_keep_the_code() {
    let cells = [
        (Code::Unavailable, true),
        (Code::ResourceExhausted, true),
        (Code::DeadlineExceeded, true),
        (Code::Aborted, true),
        (Code::InvalidArgument, false),
        (Code::NotFound, false),
        (Code::PermissionDenied, false),
        (Code::Unauthenticated, false),
        (Code::FailedPrecondition, false),
        (Code::Internal, false),
        (Code::Unknown, false),
    ];
    for (code, retryable) in cells {
        let name = code.name();
        let error = Error::service(Status::default().set_code(code).set_message("boom"));
        let err = rpc_error(&error);
        assert_eq!(err.is_retryable(), retryable, "{name}");
        assert_eq!(err.provider_response_status(), None, "{name}");
        assert_eq!(
            err.provider_response_body(),
            Some(error.to_string().as_str()),
            "the provider's text is kept verbatim"
        );
        let report = err.report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{name}");
        assert_eq!(report.retryable, retryable, "{name}");
        assert_eq!(report.code.as_deref(), Some(name), "{name}");
    }
}

/// A response-less failure the SDK classifies as its own (an I/O error on
/// the connection) has no code and no status, and the same call may be
/// retried: the request never reached a decision, as on every HTTP wire.
#[test]
fn an_sdk_transport_failure_has_no_code_and_retries() {
    let error = Error::io(std::io::Error::other("connection reset"));
    let err = rpc_error(&error);
    assert!(err.is_retryable());
    assert_eq!(err.provider_response_status(), None);
    let report = err.report();
    assert_eq!(report.code, None);
    assert_eq!(
        report.provider_response.expect("kept").transient,
        Some(true)
    );
}

/// Vertex AI requires `FunctionResponse.response`. A result that is only an
/// image names it by `$ref`, the image carrying that `displayName`.
/// The encoded `contents` of a history whose one tool result holds `images`
/// copies of one image and nothing else.
fn image_tool_result_contents(images: usize) -> serde_json::Value {
    use rig_core::message::{
        AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType,
        Message, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
    };
    use rig_core::wire::Operation;
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(
            ToolName::new("shot").expect("a tool name"),
            serde_json::json!({}),
        ),
    );
    let image = Image {
        data: DocumentSourceKind::Base64("iVBORw0KGgo=".to_owned()),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let mut request = CompletionRequest::new("next");
    request.tools = vec![
        rig_core::completion::ToolDefinition {
            name: ToolName::new("shot").expect("a tool name"),
            description: "a screenshot".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            annotations: Default::default(),
        }
        .into(),
    ];
    request.chat_history = vec![
        Message::user("shoot").into(),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )]))
        .into(),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Image(image); images]),
            )],
        }
        .into(),
    ];
    let wire = GenerateContent::new("gemini-3-flash-preview");
    let request = rig_core::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    let request = wire
        .encode(request, Mode::Unary)
        .expect("the history encodes");
    serde_json::to_value(&request.contents).expect("JSON")
}

#[test]
fn an_image_only_tool_result_keeps_a_response_naming_the_image() {
    let contents = image_tool_result_contents(1);
    let response = &contents[2]["parts"][0]["functionResponse"];
    assert_eq!(
        response["response"],
        serde_json::json!({"output": {"$ref": "rig_tool_result_image_0"}}),
        "{contents:#}"
    );
    assert_eq!(
        response["parts"][0]["inlineData"]["displayName"], "rig_tool_result_image_0",
        "{contents:#}"
    );
}

#[test]
fn a_tool_result_of_several_images_names_each_in_order() {
    let contents = image_tool_result_contents(2);
    let response = &contents[2]["parts"][0]["functionResponse"];
    assert_eq!(
        response["response"],
        serde_json::json!({"output": [
            {"$ref": "rig_tool_result_image_0"},
            {"$ref": "rig_tool_result_image_1"},
        ]}),
        "{contents:#}"
    );
    assert_eq!(
        response["parts"][1]["inlineData"]["displayName"], "rig_tool_result_image_1",
        "{contents:#}"
    );
}

/// Vertex AI takes the GenerateContent cells, but its tiers: the standard
/// one is its default, and the others need headers its SDK cannot set. The
/// cells reach the SDK request, a `generationConfig` merged over the typed
/// fields.
#[test]
fn options_take_the_generate_content_cells_with_vertex_tiers() {
    use rig_core::completion::ReplayTarget as _;
    use rig_core::completion::{CompletionRequest, Effort, ServiceTier, options::Mapping};
    use rig_core::wire::{Mode, Wire as _};

    let wire = GenerateContent::new("gemini-3-flash-preview");
    let request = CompletionRequest::new("hi")
        .max_tokens(16)
        .reasoning(Effort::High)
        .top_p(0.5)
        .seed(7);
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let config = encoded
        .generation_config
        .expect("the request has a generation config");
    assert_eq!(config.top_p, Some(0.5));
    assert_eq!(config.seed, Some(7));
    assert_eq!(config.max_output_tokens, Some(16));
    assert!(config.thinking_config.is_some());
    assert_eq!(encoded.model, "gemini-3-flash-preview");

    let tier = |tier| {
        let request = CompletionRequest::new("hi").service_tier(tier);
        wire.map_options(&request, request.options.fields())
            .service_tier
    };
    assert!(matches!(tier(ServiceTier::Default), Mapping::Omit(_)));
    assert!(matches!(tier(ServiceTier::Flex), Mapping::Unsupported(_)));
    assert!(matches!(
        tier(ServiceTier::Priority),
        Mapping::Unsupported(_)
    ));
}

/// A raw `generationConfig: null` is absent, as it was before the merge:
/// the typed fields and the mapped thinking still reach the SDK request.
#[test]
fn a_raw_null_generation_config_keeps_the_typed_fields() {
    use rig_core::completion::{CompletionRequest, Effort};
    use rig_core::wire::{Mode, Wire as _};

    let request = CompletionRequest::new("hi")
        .temperature(0.2)
        .max_tokens(64)
        .reasoning(Effort::High)
        .additional_params(serde_json::json!({"generationConfig": null}));
    let encoded = GenerateContent::new("gemini-3-flash-preview")
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let config = encoded
        .generation_config
        .expect("the request has a generation config");
    assert_eq!(config.temperature, Some(0.2));
    assert_eq!(config.max_output_tokens, Some(64));
    assert!(config.thinking_config.is_some());
}

/// The facts the Vertex wire is given are the facts it answers with: its
/// `ReplayTarget::facts`, which the shared option rules and the cost read,
/// and its descriptor's spec, which `DynModel::spec` returns.
#[test]
fn the_wire_answers_from_the_facts_it_is_given() {
    use rig_core::catalog::{ModelFacts, ModelSpec};
    use rig_core::completion::ReplayTarget as _;
    use rig_core::wire::Wire as _;

    let vendor = rig_core::providers::registry::ProviderId::catalog("vertexai").expect("a vendor");
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

/// Vertex AI names its own API, and only it takes an image URL inside a
/// function response; the Gemini API takes the same image elsewhere.
#[test]
fn vertex_names_its_api_and_takes_image_urls_in_function_responses() {
    use rig_core::completion::{Media, Place, ReplayTarget};
    use rig_core::message::{DocumentSourceKind, Image, ImageMediaType};

    let vertex = GenerateContent::new(GEMINI_2_5_FLASH);
    let rest = rig_core::providers::gemini::completion::GenerateContent::new(
        rig_core::providers::gemini::GeminiConfig::new("key"),
        GEMINI_2_5_FLASH,
    );
    assert_eq!(vertex.api().as_str(), "vertexai.generate_content");
    assert_eq!(rest.api().as_str(), "gemini.generate_content");

    let image = Image {
        data: DocumentSourceKind::Url("https://example.com/a.png".to_owned()),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let in_response = Media::Image(&image, Place::ToolResult);
    assert!(vertex.encodes(GEMINI_2_5_FLASH, in_response));
    assert!(!rest.encodes(GEMINI_2_5_FLASH, in_response));
    let in_user = Media::Image(&image, Place::User);
    assert!(vertex.encodes(GEMINI_2_5_FLASH, in_user));
    assert!(rest.encodes(GEMINI_2_5_FLASH, in_user));
}

const TEST_PROJECT: &str = "test-project";
const TEST_LOCATION: &str = "us-central1";
const TEST_MODEL_VERSION: &str = "gemini-2.5-flash-001";

#[derive(Debug, Default)]
struct MockPredictionState {
    unary_calls: Vec<vertexai::model::GenerateContentRequest>,
    stream_calls: Vec<vertexai::model::GenerateContentRequest>,
    unary_reply: Option<google_cloud_gax::Result<vertexai::model::GenerateContentResponse>>,
    stream_reply: Option<
        google_cloud_gax::Result<
            google_cloud_gax::streaming::ResponseStream<vertexai::model::GenerateContentResponse>,
        >,
    >,
}

/// Offline `PredictionService` stub recording unary vs streaming RPCs.
#[derive(Debug, Default)]
struct MockPredictionService {
    state: std::sync::Mutex<MockPredictionState>,
}

impl MockPredictionService {
    fn with_unary(
        reply: google_cloud_gax::Result<vertexai::model::GenerateContentResponse>,
    ) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            state: std::sync::Mutex::new(MockPredictionState {
                unary_reply: Some(reply),
                ..MockPredictionState::default()
            }),
        })
    }

    fn with_stream(
        reply: google_cloud_gax::Result<
            google_cloud_gax::streaming::ResponseStream<vertexai::model::GenerateContentResponse>,
        >,
    ) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            state: std::sync::Mutex::new(MockPredictionState {
                stream_reply: Some(reply),
                ..MockPredictionState::default()
            }),
        })
    }

    fn unary_calls(&self) -> Vec<vertexai::model::GenerateContentRequest> {
        self.state.lock().expect("stub lock").unary_calls.clone()
    }

    fn stream_calls(&self) -> Vec<vertexai::model::GenerateContentRequest> {
        self.state.lock().expect("stub lock").stream_calls.clone()
    }
}

impl vertexai::stub::PredictionService for MockPredictionService {
    async fn generate_content(
        &self,
        req: vertexai::model::GenerateContentRequest,
        _options: google_cloud_gax::options::RequestOptions,
    ) -> vertexai::Result<
        google_cloud_gax::response::Response<vertexai::model::GenerateContentResponse>,
    > {
        let reply = {
            let mut state = self.state.lock().expect("stub lock");
            state.unary_calls.push(req);
            state
                .unary_reply
                .take()
                .expect("unexpected generate_content call")
        };
        reply.map(google_cloud_gax::response::Response::from)
    }

    async fn stream_generate_content(
        &self,
        req: vertexai::model::GenerateContentRequest,
        _options: google_cloud_gax::options::RequestOptions,
    ) -> vertexai::Result<
        google_cloud_gax::streaming::ResponseStream<vertexai::model::GenerateContentResponse>,
    > {
        let mut state = self.state.lock().expect("stub lock");
        state.stream_calls.push(req);
        state
            .stream_reply
            .take()
            .expect("unexpected stream_generate_content call")
    }
}

fn expected_model_path() -> String {
    format!(
        "projects/{TEST_PROJECT}/locations/{TEST_LOCATION}/publishers/google/models/{GEMINI_2_5_FLASH}"
    )
}

fn stub_model(
    stub: std::sync::Arc<MockPredictionService>,
) -> rig_core::Model<GenerateContent, VertexAi> {
    let service = vertexai::client::PredictionService::from_stub::<MockPredictionService>(stub);
    VertexAi::builder()
        .with_project(TEST_PROJECT)
        .with_location(TEST_LOCATION)
        .with_prediction_service(service)
        .build()
        .expect("stubbed client builds")
        .completion(GEMINI_2_5_FLASH)
}

fn sdk_chunk(json: serde_json::Value) -> vertexai::model::GenerateContentResponse {
    serde_json::from_value(json).expect("valid GenerateContentResponse JSON")
}

fn text_chunk(text: &str) -> vertexai::model::GenerateContentResponse {
    sdk_chunk(serde_json::json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "index": 0
        }]
    }))
}

#[tokio::test]
async fn unary_mode_calls_generate_content_not_stream_generate_content() {
    use rig_core::completion::FinishReason;
    use rig_core::message::AssistantContent;

    let stub = MockPredictionService::with_unary(Ok(sdk_chunk(serde_json::json!({
        "responseId": "unary-resp-1",
        "modelVersion": TEST_MODEL_VERSION,
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hello unary"}]},
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 2,
            "totalTokenCount": 6
        }
    }))));
    let model = stub_model(stub.clone());

    let response = model
        .call(CompletionRequest::new("hi"))
        .await
        .expect("unary completion succeeds");

    let unary_calls = stub.unary_calls();
    let [call] = unary_calls.as_slice() else {
        panic!("expected one unary call, got {unary_calls:?}");
    };
    assert!(stub.stream_calls().is_empty());
    assert_eq!(call.model, expected_model_path());
    assert!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(t)] if t.text == "hello unary")
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage.input_tokens, Some(4));
    assert_eq!(response.usage.output_tokens, Some(2));
    assert_eq!(response.usage.total_tokens, Some(6));
    assert_eq!(response.raw["responseId"], "unary-resp-1");
    assert_eq!(response.raw["candidates"][0]["finishReason"], "STOP");
    let recovered: vertexai::model::GenerateContentResponse =
        serde_json::from_value(response.raw).expect("raw recovers as SDK type");
    assert_eq!(recovered.response_id, "unary-resp-1");
}

#[tokio::test]
async fn streaming_mode_emits_incremental_chunks_and_reassembles_response() {
    use futures::StreamExt as _;
    use rig_core::completion::FinishReason;
    use rig_core::message::AssistantContent;
    use rig_core::streaming::{Item, PartKind, StreamEvent};

    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let stub = MockPredictionService::with_stream(Ok(
        google_cloud_gax::streaming::ResponseStream::from(rx),
    ));
    let model = stub_model(stub.clone());

    let mut stream = model
        .stream(CompletionRequest::new("check weather in Lisbon"))
        .expect("stream opens");

    // Send chunk 1 alone to prove incremental delivery before chunk 2 arrives.
    tx.send(Ok(sdk_chunk(serde_json::json!({
        "responseId": "stream-resp-1",
        "modelVersion": TEST_MODEL_VERSION,
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "thinking...", "thought": true, "thoughtSignature": "c2lnLXRob3VnaHQ="}]
            },
            "index": 0
        }]
    }))))
    .await
    .expect("send chunk 1");

    let first_start = stream.next().await.expect("start event").expect("no error");
    assert!(matches!(
        first_start,
        Item::Event(StreamEvent::Start {
            kind: PartKind::Reasoning,
            ..
        })
    ));
    let first_delta = stream
        .next()
        .await
        .expect("reasoning delta")
        .expect("no error");
    assert!(matches!(
        first_delta,
        Item::Event(StreamEvent::Reasoning { ref text, .. }) if text == "thinking..."
    ));

    tx.send(Ok(text_chunk("Hel"))).await.expect("send chunk 2");
    loop {
        if matches!(
            stream.next().await.expect("event").expect("no error"),
            Item::Event(StreamEvent::Text { ref text, .. }) if text == "Hel"
        ) {
            break;
        }
    }

    tx.send(Ok(text_chunk("lo"))).await.expect("send chunk 3");
    assert!(matches!(
        stream.next().await.expect("text delta").expect("no error"),
        Item::Event(StreamEvent::Text { ref text, .. }) if text == "lo"
    ));

    tx.send(Ok(sdk_chunk(serde_json::json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "id": "call_1",
                        "name": "lookup_weather",
                        "args": {"city": "Lisbon"}
                    },
                    "thoughtSignature": "c2lnLWNhbGw="
                }]
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 8,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 23
        }
    }))))
    .await
    .expect("send chunk 4");
    drop(tx);

    let response = stream.finish().await.expect("stream finishes cleanly");

    assert!(stub.unary_calls().is_empty());
    let stream_calls = stub.stream_calls();
    let [stream_call] = stream_calls.as_slice() else {
        panic!("expected one stream call, got {stream_calls:?}");
    };
    assert_eq!(stream_call.model, expected_model_path());
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(response.usage.input_tokens, Some(12));
    assert_eq!(response.usage.output_tokens, Some(11));
    assert_eq!(response.usage.reasoning_tokens, Some(3));
    assert_eq!(response.usage.total_tokens, Some(23));

    let [
        AssistantContent::Reasoning(reasoning),
        AssistantContent::Text(text),
        tool @ AssistantContent::ToolCall(call),
    ] = response.choice.as_slice()
    else {
        panic!(
            "expected reasoning, text, and tool call, got {:?}",
            response.choice
        );
    };
    assert_eq!(reasoning.text, "thinking...");
    assert_eq!(text.text, "Hello");
    assert_eq!(call.function.name, "lookup_weather");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"city": "Lisbon"})
    );
    assert_eq!(
        tool.native_item()
            .and_then(|item| item.get("thoughtSignature"))
            .and_then(serde_json::Value::as_str),
        Some("c2lnLWNhbGw=")
    );

    assert_eq!(response.raw["responseId"], "stream-resp-1");
    assert_eq!(response.raw["candidates"][0]["finishReason"], "STOP");
    assert_eq!(
        response.raw["candidates"][0]["content"]["parts"][1]["text"],
        "Hello"
    );
    assert_eq!(
        response.raw["candidates"][0]["content"]["parts"][2]["thoughtSignature"],
        "c2lnLWNhbGw="
    );
    let recovered: vertexai::model::GenerateContentResponse =
        serde_json::from_value(response.raw).expect("reassembled raw recovers as SDK type");
    assert_eq!(recovered.response_id, "stream-resp-1");
}

#[tokio::test]
async fn streaming_initial_rpc_error_maps_through_rpc_error() {
    let sdk_err = Error::service(
        Status::default()
            .set_code(Code::ResourceExhausted)
            .set_message("quota exceeded"),
    );
    let expected_body = sdk_err.to_string();
    let stub = MockPredictionService::with_stream(Err(sdk_err));
    let model = stub_model(stub.clone());

    let error = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream builds")
        .finish()
        .await
        .expect_err("initial RPC error surfaces");

    assert!(stub.unary_calls().is_empty());
    assert_eq!(stub.stream_calls().len(), 1);
    assert!(error.is_retryable());
    assert_eq!(error.provider_response_body(), Some(expected_body.as_str()));
    let report = error.report();
    assert_eq!(report.kind, ErrorKind::ProviderResponse);
    assert_eq!(report.code.as_deref(), Some("RESOURCE_EXHAUSTED"));
    assert!(report.retryable);
}

#[tokio::test]
async fn streaming_mid_stream_rpc_error_yields_prior_chunks_then_stops() {
    use futures::StreamExt as _;
    use rig_core::streaming::{Item, StreamEvent};

    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let sdk_err = Error::service(
        Status::default()
            .set_code(Code::Unavailable)
            .set_message("backend reset"),
    );
    let expected_body = sdk_err.to_string();

    tx.send(Ok(text_chunk("partial")))
        .await
        .expect("send partial chunk");
    tx.send(Err(sdk_err)).await.expect("send mid-stream error");
    tx.send(Ok(text_chunk("must not arrive")))
        .await
        .expect("send trailing chunk");
    drop(tx);

    let stub = MockPredictionService::with_stream(Ok(
        google_cloud_gax::streaming::ResponseStream::from(rx),
    ));
    let model = stub_model(stub);
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream opens");

    let mut texts = Vec::new();
    let mut terminal_error = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Ok(_) => {}
            Err(error) => {
                terminal_error = Some(error);
                break;
            }
        }
    }

    assert_eq!(texts, vec!["partial"]);
    let error = terminal_error.expect("mid-stream error emitted");
    assert!(error.is_retryable());
    assert_eq!(error.provider_response_body(), Some(expected_body.as_str()));
    let report = error.report();
    assert_eq!(report.kind, ErrorKind::ProviderResponse);
    assert_eq!(report.code.as_deref(), Some("UNAVAILABLE"));
    assert!(report.retryable);
    assert!(
        stream.next().await.is_none(),
        "stream terminates after the mid-stream error"
    );
}

#[tokio::test]
async fn streaming_empty_response_stream_ends_as_truncated() {
    use futures::StreamExt as _;

    let (tx, rx) = tokio::sync::mpsc::channel(1);
    drop(tx);

    let stub = MockPredictionService::with_stream(Ok(
        google_cloud_gax::streaming::ResponseStream::from(rx),
    ));
    let model = stub_model(stub.clone());
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream opens");

    let first = stream
        .next()
        .await
        .expect("empty stream surfaces Truncated error");
    assert!(
        matches!(first, Err(ProviderError::Truncated)),
        "expected Err(Truncated), got {first:?}"
    );
    assert!(
        stream.next().await.is_none(),
        "nothing follows the Truncated error"
    );
    let error = stream
        .finish()
        .await
        .expect_err("an empty stream without a terminal chunk is truncated");
    assert!(
        matches!(error, ProviderError::Truncated),
        "expected Truncated, got {error:?}"
    );
    assert!(stub.unary_calls().is_empty());
    assert_eq!(stub.stream_calls().len(), 1);
}
