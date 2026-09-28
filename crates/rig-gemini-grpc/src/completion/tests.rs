use super::*;
use rig_core::Model;
use rig_core::streaming::CompletionStream;

/// Answers every request with scripted protobuf replies, one per frame.
#[derive(Clone)]
pub(crate) struct Scripted(
    std::sync::Arc<std::sync::Mutex<Vec<Result<GenerateContentResponse, ProviderError>>>>,
);

impl Transport<GenerateContent> for Scripted {
    fn send(
        &self,
        _request: GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<GenerateContentResponse> {
        let replies = std::mem::take(&mut *self.0.lock().expect("script lock"));
        Opening::ready(Opened::new(futures::stream::iter(replies)))
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
) -> Result<completion::CompletionResponse, ProviderError> {
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

#[test]
fn rpc_error_preserves_status_text_without_http_status() {
    let status = tonic::Status::unavailable("boom");
    let expected = status.to_string();

    let err = rpc_error(&status);

    // The raw provider error text is preserved verbatim, and there is no
    // HTTP status because gRPC is a non-HTTP transport.
    assert_eq!(err.provider_response_body(), Some(expected.as_str()));
    assert_eq!(err.provider_response_status(), None);
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_with_padding() {
    assert!(matches!(
        decode_base64_bytes("_-wgVQA="),
        Ok(bytes) if bytes == vec![0xFF, 0xEC, 0x20, 0x55, 0x00]
    ));
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_no_pad() {
    assert!(matches!(
        decode_base64_bytes("_-wgVQA"),
        Ok(bytes) if bytes == vec![0xFF, 0xEC, 0x20, 0x55, 0x00]
    ));
}

#[test]
fn test_decode_base64_bytes_accepts_standard_no_pad() {
    assert!(matches!(
        decode_base64_bytes("Zg"),
        Ok(bytes) if bytes == b"f".to_vec()
    ));
}

#[test]
fn test_decode_base64_bytes_accepts_data_uri_prefix() {
    assert!(matches!(
        decode_base64_bytes("data:text/plain;base64,Zm9v"),
        Ok(bytes) if bytes == b"foo".to_vec()
    ));
}

// ============================================================
// tool_parameters_to_proto_schema — regression coverage for #1710
// ============================================================

#[test]
fn create_grpc_request_sends_the_executed_name_not_an_identifier() {
    use rig_core::message::{
        AssistantContent, ToolCall, ToolFunction, ToolResult, ToolResultContent,
    };

    let call = |wire_id: &str, name: &str| message::Message::Assistant {
        id: None,
        content: rig_core::NonEmpty::new(AssistantContent::ToolCall(ToolCall::from_wire(
            wire_id,
            ToolFunction {
                name: rig_core::message::ToolName::new(name.to_owned()).expect("tool name"),
                arguments: serde_json::json!({}),
            },
        ))),
    };
    let result = |wire_id: &str, name: &str| message::Message::User {
        content: rig_core::NonEmpty::new(message::UserContent::ToolResult(ToolResult {
            call: rig_core::message::CallId::from_wire(wire_id),
            name: rig_core::message::ToolName::new(name.to_owned()).expect("tool name"),
            content: rig_core::NonEmpty::new(ToolResultContent::text("out")),
        })),
    };

    let req = create_grpc_request(
        "gemini-2.5-flash",
        CompletionRequest {
            model: None,
            chat_history: rig_core::NonEmpty::with_rest(
                // Driver-built: the executed name travels as data (a
                // repair hook renamed the call: `sum` ran, not `add`).
                call("call_1", "add"),
                [
                    result("call_1", "sum"), // Cross-provider history with an OpenAI-shaped id —
                    // `call_abc` must never travel as the name.
                    call("call_abc", "get_weather"),
                    result("call_abc", "get_weather"),
                ],
            ),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
    )
    .expect("request build");

    // The name is the executed tool's name; the proto `id` is the
    // provider-issued call id (never rig's minted handle).
    let responses: Vec<(&str, &str)> = req
        .contents
        .iter()
        .flat_map(|content| content.parts.iter())
        .filter_map(|part| match &part.data {
            Some(proto::part::Data::FunctionResponse(fr)) => {
                Some((fr.id.as_str(), fr.name.as_str()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        responses,
        vec![("call_1", "sum"), ("call_abc", "get_weather")]
    );
}

#[test]
fn create_grpc_request_populates_tool_parameters() {
    use rig_core::completion::ToolDefinition;

    let tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Look up the current weather for a city.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
    };

    let req = create_grpc_request(
        "gemini-2.5-flash",
        CompletionRequest {
            model: None,
            chat_history: rig_core::NonEmpty::new(message::Message::user("forecast in Berlin?")),
            documents: Vec::new(),
            tools: vec![tool],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
    )
    .expect("request build");

    assert_eq!(req.tools.len(), 1);
    let tool = req.tools.first().expect("tool entry");
    let decl = tool
        .function_declarations
        .first()
        .expect("function declaration");
    assert_eq!(decl.name, "get_weather");

    // The schema goes as written, in `parameters_json_schema`.
    assert!(decl.parameters.is_none());
    let params = decl
        .parameters_json_schema
        .as_ref()
        .expect("parameters populated");
    let schema = match params.kind.as_ref() {
        Some(proto::value::Kind::StructValue(schema)) => Some(schema),
        _ => None,
    }
    .expect("an object schema");
    assert!(schema.fields.contains_key("required"));
    assert!(schema.fields.contains_key("properties"));
}

/// The gRPC wire carries the model's chain-of-thought in the same `parts`
/// array as the answer, flagged by `thought` — same shape as the REST
/// wire, where reading it as output text was a live-confirmed defect.
/// There is no cassette harness for this transport (it is protobuf over
/// gRPC, not HTTP), so the wire shape is stated directly.
/// A signature on answer text stays on that text: Gemini's rules return a
/// signature inside the part that carried it, never merged into another.
#[test]
fn a_signature_on_answer_text_stays_on_that_text() {
    let response = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![
                    proto::Part {
                        data: Some(proto::part::Data::Text("the chain".to_string())),
                        thought: true,
                        ..Default::default()
                    },
                    proto::Part {
                        data: Some(proto::part::Data::Text("answer".to_string())),
                        thought: false,
                        thought_signature: b"sig-bytes".to_vec(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            ..Default::default()
        }],
        ..Default::default()
    };

    let normalized = complete(response).expect("payload should normalize");
    assert_eq!(normalized.choice.len(), 2, "{:?}", normalized.choice);
    assert!(
        matches!(
            normalized.choice.first(),
            Some(completion::AssistantContent::Reasoning(reasoning))
                if reasoning.open(reasoning.issuer()).expect("sealed reasoning").first_signature().is_none()
        ),
        "the reasoning stays unsigned: {:?}",
        normalized.choice
    );
    let signature = match normalized.choice.get(1) {
        Some(completion::AssistantContent::Text(text)) => text
            .signature
            .as_ref()
            .and_then(|signature| signature.open(&ISSUER))
            .map(|signature| signature.signature.as_str()),
        _ => None,
    }
    .expect("the answer text carries its signature");

    // It replays on the answer part.
    let request = create_grpc_request(
        "gemini-3-flash-preview",
        CompletionRequest {
            model: None,
            chat_history: rig_core::NonEmpty::with_rest(
                message::Message::user("q"),
                [
                    message::Message::Assistant {
                        id: None,
                        content: rig_core::NonEmpty::from_vec(normalized.choice.clone())
                            .expect("non-empty"),
                    },
                    message::Message::user("again"),
                ],
            ),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
    )
    .expect("request build");
    let answer = request
        .contents
        .get(1)
        .expect("the assistant turn")
        .parts
        .iter()
        .find(|part| matches!(&part.data, Some(proto::part::Data::Text(text)) if text == "answer"))
        .expect("the answer part");
    assert_eq!(answer.thought_signature, b"sig-bytes".to_vec());
    assert!(!signature.is_empty());
}

/// The load-bearing property behind `CompletionResponse::raw` for the
/// gRPC provider: the captured value is
/// `serde_json::to_value(&GenerateContentResponse)` — the prost message
/// `raw_completion` returns, with the serde derives `build.rs` attaches to
/// every generated type — and a consumer must be able to read it back as
/// the same message and get the same JSON. There is no cassette harness
/// for gRPC, so this is the unit-form pin. Fields rig never normalizes
/// (`cached_content_token_count` under `usage_metadata`, the candidate's
/// `finish_message`) survive both directions, and normalizing the
/// restored message agrees with normalizing the original.
#[test]
fn generate_content_response_round_trips_through_serde_json_value() {
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
        }],
        usage_metadata: Some(proto::UsageMetadata {
            prompt_token_count: 10,
            candidates_token_count: 20,
            total_token_count: 30,
            cached_content_token_count: 4,
        }),
        model_version: "gemini-2.5-flash".to_string(),
        response_id: "resp-grpc-1".to_string(),
        prompt_feedback: None,
    };

    let value = serde_json::to_value(&raw).expect("serialize");
    assert_eq!(
        value.pointer("/usage_metadata/cached_content_token_count"),
        Some(&serde_json::json!(4))
    );
    assert_eq!(
        value.pointer("/candidates/0/finish_message"),
        Some(&serde_json::json!("done"))
    );
    assert_eq!(
        value.pointer("/model_version"),
        Some(&serde_json::json!("gemini-2.5-flash"))
    );

    let back: proto::GenerateContentResponse =
        serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&back).expect("re-serialize"),
        value,
        "the capture must read back into GenerateContentResponse and re-serialize identically"
    );
    assert_eq!(back, raw);

    let original = complete(raw.clone()).expect("original converts");
    assert_eq!(original.raw, value, "the response's raw is the capture");
    let restored = complete(back).expect("restored converts");
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.model, original.model);
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(
        restored.identity().response_id.as_deref(),
        Some("resp-grpc-1")
    );
    assert_eq!(
        restored.finish_reason(),
        Some(completion::FinishReason::Stop)
    );
}

/// Constructed protobuf input pins missing-ID collisions without a gRPC service.
#[test]
fn missing_call_ids_remain_distinct_and_do_not_collide_with_explicit_ids() {
    let wire = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: (0..3)
                    .map(|i| proto::Part {
                        data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
                            id: if i == 1 {
                                "tool-0".into()
                            } else {
                                String::new()
                            },
                            name: "same".into(),
                            ..Default::default()
                        })),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            ..Default::default()
        }],
        ..Default::default()
    };
    let first = complete(wire).unwrap();
    let calls: Vec<_> = first
        .choice
        .iter()
        .filter_map(|item| match item {
            message::AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 3);
    let (first, second, third) = (
        calls.first().unwrap(),
        calls.get(1).unwrap(),
        calls.get(2).unwrap(),
    );
    assert!(first.id.is_local() && third.id.is_local());
    assert_ne!(first.id, third.id);
    assert_eq!(second.id, rig_core::message::CallId::from_wire("tool-0"));
    assert!(calls.first().unwrap().id.provider().is_none());
    assert_eq!(
        calls
            .get(1)
            .unwrap()
            .id
            .provider()
            .as_ref()
            .unwrap()
            .call_id,
        "tool-0"
    );
    assert!(calls.get(2).unwrap().id.provider().is_none());
}

/// gRPC replies carry no HTTP status; the status code is the provider's
/// own code and decides retryability: the four codes that say "the server
/// was unreachable, overloaded or cut the call short" retry, every other
/// code says the call itself is wrong. The report carries the code by its
/// canonical name.
#[test]
fn rpc_codes_classify_retryability_and_keep_the_code() {
    use rig_core::error::ErrorKind;
    let cells = [
        (tonic::Code::Unavailable, true, "UNAVAILABLE"),
        (tonic::Code::ResourceExhausted, true, "RESOURCE_EXHAUSTED"),
        (tonic::Code::DeadlineExceeded, true, "DEADLINE_EXCEEDED"),
        (tonic::Code::Aborted, true, "ABORTED"),
        (tonic::Code::InvalidArgument, false, "INVALID_ARGUMENT"),
        (tonic::Code::NotFound, false, "NOT_FOUND"),
        (tonic::Code::PermissionDenied, false, "PERMISSION_DENIED"),
        (tonic::Code::Unauthenticated, false, "UNAUTHENTICATED"),
        (
            tonic::Code::FailedPrecondition,
            false,
            "FAILED_PRECONDITION",
        ),
        (tonic::Code::Unimplemented, false, "UNIMPLEMENTED"),
        (tonic::Code::Internal, false, "INTERNAL"),
        (tonic::Code::Unknown, false, "UNKNOWN"),
        (tonic::Code::Cancelled, false, "CANCELLED"),
    ];
    for (code, retryable, name) in cells {
        let err = rpc_error(&tonic::Status::new(code, "boom"));
        assert_eq!(err.is_retryable(), retryable, "{name}");
        assert_eq!(err.provider_response_status(), None, "{name}");
        let report = err.report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{name}");
        assert_eq!(report.retryable, retryable, "{name}");
        assert_eq!(report.http_status, None, "{name}");
        assert_eq!(report.code.as_deref(), Some(name), "{name}");
        let response = report
            .provider_response
            .expect("the reply rides on the report");
        assert_eq!(response.code.as_deref(), Some(name));
        assert_eq!(response.transient, Some(retryable));
    }
}

#[test]
fn only_gemini_reasoning_is_replayed() {
    use base64::Engine;
    use rig_core::message::{AssistantContent, Reasoning};

    let signature = |bytes: &[u8]| base64::prelude::BASE64_STANDARD.encode(bytes);
    let reasoning = |text: &str, bytes: &[u8], issuer: &str| {
        AssistantContent::Reasoning(
            Reasoning::new_with_signature(text, Some(signature(bytes))).sealed(issuer.to_owned()),
        )
    };
    let req = create_grpc_request(
        "gemini-2.5-flash",
        CompletionRequest {
            model: None,
            chat_history: rig_core::NonEmpty::with_rest(
                message::Message::user("What is 2 + 2?"),
                [
                    message::Message::Assistant {
                        id: None,
                        content: rig_core::NonEmpty::with_rest(
                            reasoning("grpc thought", b"grpc", REASONING_ISSUER),
                            [
                                reasoning(
                                    "rest thought",
                                    b"rest",
                                    rig_core::providers::gemini::completion::PROVIDER_NAME,
                                ),
                                reasoning("anthropic thought", b"anthropic", "anthropic"),
                                AssistantContent::text("4"),
                            ],
                        ),
                    },
                    message::Message::user("And 3 + 3?"),
                ],
            ),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
    )
    .expect("request build");

    // The Gemini service issued both the gRPC and the REST reasoning.
    let signatures: Vec<&[u8]> = req
        .contents
        .iter()
        .flat_map(|content| content.parts.iter())
        .filter(|part| part.thought)
        .map(|part| part.thought_signature.as_slice())
        .collect();
    assert_eq!(signatures, vec![b"grpc".as_slice(), b"rest".as_slice()]);
}

/// Answers with one text reply and keeps every request it was given.
#[derive(Clone, Default)]
struct Recording(std::sync::Arc<std::sync::Mutex<Vec<GenerateContentRequest>>>);

impl Transport<GenerateContent> for Recording {
    fn send(
        &self,
        request: GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<GenerateContentResponse> {
        self.0.lock().expect("recording lock").push(request);
        let reply = GenerateContentResponse {
            candidates: vec![crate::proto::Candidate {
                content: Some(crate::proto::Content {
                    parts: vec![text_part("6".to_owned())],
                    role: "model".to_owned(),
                }),
                finish_reason: crate::proto::candidate::FinishReason::Stop as i32,
                ..Default::default()
            }],
            ..Default::default()
        };
        Opening::ready(Opened::new(futures::stream::iter([Ok(reply)])))
    }
}

/// Through the driver, which scopes history to the wire's replay issuers
/// before encoding, Gemini's own reasoning still reaches the request.
#[test]
fn the_driver_replays_gemini_reasoning_to_the_grpc_wire() {
    use base64::Engine;
    use rig_core::message::{AssistantContent, Reasoning};

    let signature = |bytes: &[u8]| base64::prelude::BASE64_STANDARD.encode(bytes);
    let reasoning = |text: &str, bytes: &[u8], issuer: &str| {
        AssistantContent::Reasoning(
            Reasoning::new_with_signature(text, Some(signature(bytes))).sealed(issuer.to_owned()),
        )
    };
    let mut request = hello();
    request.chat_history = rig_core::NonEmpty::with_rest(
        message::Message::user("What is 2 + 2?"),
        [
            message::Message::Assistant {
                id: None,
                content: rig_core::NonEmpty::with_rest(
                    reasoning("gemini thought", b"gemini", REASONING_ISSUER),
                    [
                        reasoning("anthropic thought", b"anthropic", "anthropic"),
                        AssistantContent::text("4"),
                    ],
                ),
            },
            message::Message::user("And 3 + 3?"),
        ],
    );
    let recording = Recording::default();
    futures::executor::block_on(
        Model::new(GenerateContent::new(GEMINI_2_5_FLASH), recording.clone()).call(request),
    )
    .expect("the call succeeds");

    let sent = recording.0.lock().expect("recording lock");
    let signatures: Vec<&[u8]> = sent
        .first()
        .expect("one request")
        .contents
        .iter()
        .flat_map(|content| content.parts.iter())
        .filter(|part| part.thought)
        .map(|part| part.thought_signature.as_slice())
        .collect();
    assert_eq!(signatures, vec![b"gemini".as_slice()]);
}
