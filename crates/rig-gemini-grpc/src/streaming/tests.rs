use super::*;
use base64::Engine as _;
use rig_core::message::{Reasoning, ReasoningContent};
use rig_core::streaming::StreamedAssistantContent;

fn thought_part(text: &str, signature: &[u8]) -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::Text(text.to_string())),
        thought: true,
        thought_signature: signature.to_vec(),
        ..Default::default()
    }
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

/// Drive protobuf events through the full normalized path and collect the
/// Reasoning blocks the consumer sees.
async fn reasoning_blocks(events: Vec<proto::GenerateContentResponse>) -> Vec<Reasoning> {
    let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
    let mut blocks = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Reasoning { reasoning, .. } =
            item.expect("stream item should be ok")
        {
            blocks.push(reasoning);
        }
    }
    blocks
}

// Streaming parity with the unary conversion (completion.rs
// `Reasoning::new_with_signature` + base64): a signed thought part must
// reach the normalized stream as a completed signed Reasoning block that
// restates the accumulated thought text.
#[tokio::test]
async fn signed_thought_part_restates_accumulated_text_with_signature() {
    let signature_bytes = b"opaque-signature".as_slice();
    let events = vec![
        response(vec![thought_part("think1 ", b"")], 0),
        response(
            vec![thought_part("think2", signature_bytes)],
            proto::candidate::FinishReason::Stop as i32,
        ),
    ];

    let blocks = reasoning_blocks(events).await;
    let signed = blocks
        .last()
        .expect("the signed part must yield a Reasoning block");
    assert_eq!(
        signed.content,
        vec![ReasoningContent::Text {
            text: "think1 think2".to_string(),
            // The expected encoding is the unary path's: standard base64
            // over the wire's signature bytes.
            signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
        }]
    );
}

// The wire's real signed shape: the signature rides a trailing EMPTY
// thought part. It must still emit a signed block so the signature
// survives into chat history (signature-only case).
#[tokio::test]
async fn signature_on_empty_trailer_part_still_carries_the_signature() {
    let signature_bytes = b"trailer-signature".as_slice();
    let events = vec![
        response(vec![thought_part("thinking...", b"")], 0),
        response(
            vec![thought_part("", signature_bytes)],
            proto::candidate::FinishReason::Stop as i32,
        ),
    ];

    let blocks = reasoning_blocks(events).await;
    let signed = blocks
        .last()
        .expect("the signed trailer must yield a Reasoning block");
    assert_eq!(
        signed.content,
        vec![ReasoningContent::Text {
            text: "thinking...".to_string(),
            signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
        }]
    );
}

// Signature with no thought text anywhere in the stream: the signed block
// still surfaces (empty text) rather than dropping the signature.
#[tokio::test]
async fn signature_without_any_thought_text_still_surfaces() {
    let signature_bytes = b"lone-signature".as_slice();
    let events = vec![response(
        vec![thought_part("", signature_bytes)],
        proto::candidate::FinishReason::Stop as i32,
    )];

    let blocks = reasoning_blocks(events).await;
    let signed = blocks
        .last()
        .expect("a lone signature must yield a Reasoning block");
    assert_eq!(
        signed.content,
        vec![ReasoningContent::Text {
            text: String::new(),
            signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
        }]
    );
}

fn function_call_part(name: &str, id: &str) -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
            name: name.to_string(),
            args: None,
            id: id.to_string(),
        })),
        ..Default::default()
    }
}

// Two id-less calls to the same tool in one turn are two distinct calls,
// correlated by order rather than by the tool name.
//
// That is all this pins. The per-stream minter also gives each call its own
// stream key now, where the fixed `Tool.for_wire_index(0)` key gave both the
// same one — but no assertion here can tell the two apart: a whole tool call
// is emitted immediately with a freshly generated `internal_call_id`, so the
// shared key never collided anything downstream. It was a latent identity
// bug, not an observable one, and pinning it would mean asserting on
// internal keys.
#[tokio::test]
async fn two_id_less_function_calls_stay_distinct() {
    let events = vec![response(
        vec![
            function_call_part("get_weather", ""),
            function_call_part("get_weather", ""),
        ],
        proto::candidate::FinishReason::Stop as i32,
    )];

    let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
    let mut correlators = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::ToolCall {
            tool_call,
            internal_call_id,
        } = item.expect("stream item should be ok")
        {
            assert_eq!(tool_call.function.name, "get_weather");
            correlators.push(internal_call_id);
        }
    }

    assert_eq!(correlators.len(), 2, "correlators: {correlators:?}");
    assert_ne!(
        correlators.first(),
        correlators.last(),
        "each call needs its own correlator"
    );
}

// ---- #2258 H4: tool-protocol finish reasons must fail the turn ----

fn failed_response(
    reason: proto::candidate::FinishReason,
    finish_message: Option<&str>,
) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![],
                role: "model".to_string(),
            }),
            finish_reason: reason as i32,
            finish_message: finish_message.map(str::to_owned),
            ..Default::default()
        }],
        ..Default::default()
    }
}

struct Drained {
    errors: Vec<String>,
    reached_terminal: bool,
    text: String,
}

async fn drain(events: Vec<proto::GenerateContentResponse>) -> Drained {
    let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
    let mut drained = Drained {
        errors: Vec::new(),
        reached_terminal: false,
        text: String::new(),
    };

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamedAssistantContent::Final(_)) => drained.reached_terminal = true,
            Ok(StreamedAssistantContent::Text(text)) => drained.text.push_str(&text.text),
            Ok(_) => {}
            Err(error) => drained.errors.push(error.to_string()),
        }
    }

    drained
}

// The gRPC surface only set `is_final` on a nonzero finish reason, so an
// aborted tool protocol read as a completed turn. It must now fail, as
// the REST surface always has.
#[tokio::test]
async fn malformed_function_call_fails_the_stream_with_no_terminal() {
    let drained = drain(vec![failed_response(
        proto::candidate::FinishReason::MalformedFunctionCall,
        Some("could not parse the function call"),
    )])
    .await;

    assert_eq!(drained.errors.len(), 1, "errors: {:?}", drained.errors);
    let error = drained.errors.first().expect("one error");
    assert!(
        error.contains("MALFORMED_FUNCTION_CALL")
            && error.contains("could not parse the function call"),
        "error should name the reason and carry finish_message: {error}"
    );
    assert!(
        !drained.reached_terminal,
        "a failed turn must not synthesize a terminal record"
    );
}

#[tokio::test]
async fn unexpected_and_too_many_tool_calls_also_fail_the_stream() {
    for reason in [
        proto::candidate::FinishReason::UnexpectedToolCall,
        proto::candidate::FinishReason::TooManyToolCalls,
    ] {
        let drained = drain(vec![failed_response(reason, None)]).await;
        assert_eq!(
            drained.errors.len(),
            1,
            "{} should fail the stream",
            reason.as_str_name()
        );
        assert!(!drained.reached_terminal);
    }
}

// Everything after the in-band failure is dead: the adapter latches
// `failed` and reports `is_finished`, so a later genuine terminal cannot
// dress the aborted turn up as complete.
#[tokio::test]
async fn frames_after_a_tool_protocol_failure_are_not_interpreted() {
    let drained = drain(vec![
        failed_response(proto::candidate::FinishReason::MalformedFunctionCall, None),
        response(
            vec![proto::Part {
                data: Some(proto::part::Data::Text("recovered?".to_string())),
                ..Default::default()
            }],
            proto::candidate::FinishReason::Stop as i32,
        ),
    ])
    .await;

    assert_eq!(drained.errors.len(), 1, "errors: {:?}", drained.errors);
    assert!(drained.text.is_empty(), "text: {:?}", drained.text);
    assert!(!drained.reached_terminal);
}

// Ordinary terminals are untouched by the new gate.
#[tokio::test]
async fn non_tool_protocol_finish_reasons_still_complete_the_turn() {
    let drained = drain(vec![response(
        vec![proto::Part {
            data: Some(proto::part::Data::Text("done".to_string())),
            ..Default::default()
        }],
        proto::candidate::FinishReason::Stop as i32,
    )])
    .await;

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    assert_eq!(drained.text, "done");
    assert!(drained.reached_terminal);
}

// The unary path routes through the same helper, so the two surfaces
// report an aborted tool protocol with the same message.
#[test]
fn unary_and_streaming_report_the_same_tool_protocol_error() {
    let response = failed_response(
        proto::candidate::FinishReason::TooManyToolCalls,
        Some("budget exhausted"),
    );

    let expected = super::super::completion::tool_protocol_finish_reason_error(
        proto::candidate::FinishReason::TooManyToolCalls as i32,
        Some("budget exhausted"),
    )
    .expect("the helper must produce an error")
    .to_string();

    match rig_core::completion::CompletionResponse::try_from(response) {
        Err(err) => assert_eq!(err.to_string(), expected),
        Ok(_) => panic!("the unary path must fail on a tool-protocol finish reason"),
    }
}

// The streaming path maps both the initial `stream_generate_content` RPC
// failure and any per-item iteration error through `rpc_error`. Pin that the
// mapping preserves the provider's status text and exposes no HTTP status.
#[test]
fn stream_rpc_error_preserves_status_text_without_http_status() {
    let status = tonic::Status::unavailable("boom");
    let expected = status.to_string();

    let err = super::super::completion::rpc_error(&status);

    assert_eq!(err.provider_response_body(), Some(expected.as_str()));
    assert_eq!(err.provider_response_status(), None);
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
        }),
        model_version: "gemini-2.5-flash".to_string(),
        response_id: "resp-grpc-stream".to_string(),
        ..Default::default()
    }
}

/// Drive protobuf events through the pipeline the `CompletionModel` seam
/// uses, returning the terminal record.
async fn normalized_terminal(
    events: Vec<proto::GenerateContentResponse>,
) -> streaming::StreamFinal {
    let raw = run_wire_stream(
        futures::stream::iter(events.into_iter().map(Ok)),
        GrpcAdapter::default(),
    );
    collect_terminal(normalize_grpc_stream(Box::pin(raw))).await
}

async fn collect_terminal(normalized: streaming::StreamingResult) -> streaming::StreamFinal {
    let mut stream = streaming::StreamingCompletionResponse::stream(
        super::super::completion::PROVIDER_NAME,
        normalized,
    );
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    stream
        .response
        .expect("the stream must end with a terminal record")
}

/// The events-first seam captures like the request-driven one: its
/// terminal `raw` is the same terminal `GenerateContentResponse` the
/// model's `stream()` would attach, because both funnel through
/// `normalize_grpc_stream`.
#[tokio::test]
async fn stream_from_events_terminal_carries_raw() {
    let mut stream = stream_from_events(futures::stream::iter(
        vec![response(vec![text_part("hi")], 0), terminal_frame()]
            .into_iter()
            .map(Ok),
    ));
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let terminal = stream.response.expect("terminal record");

    let raw = &terminal.raw;
    let typed: proto::GenerateContentResponse =
        serde_json::from_value(raw.clone()).expect("raw must deserialize");
    assert_eq!(typed, terminal_frame());
    assert_eq!(terminal.usage.total_tokens, 5);
}

/// The load-bearing streaming capture property at the seam
/// `CompletionModel::stream` routes through: the terminal's `raw` is
/// Gemini's own terminal `GenerateContentResponse` — it deserializes back
/// into that prost message and re-serializes identically — and
/// re-normalizing that capture reproduces every normalized field. The
/// raw `finish_reason` number and the last frame's text are only readable
/// off the capture.
#[tokio::test]
async fn terminal_raw_round_trips_into_the_terminal_type() {
    let terminal =
        normalized_terminal(vec![response(vec![text_part("hi")], 0), terminal_frame()]).await;

    let raw = &terminal.raw;
    let typed: proto::GenerateContentResponse =
        serde_json::from_value(raw.clone()).expect("raw must deserialize");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "the capture must be exactly what the terminal type serializes to"
    );
    assert_eq!(typed, terminal_frame());
    assert_eq!(
        raw.pointer("/candidates/0/finish_reason"),
        Some(&serde_json::json!(
            proto::candidate::FinishReason::Stop as i32
        ))
    );

    // Feeding the capture back through the same pipeline tells the same
    // story as the terminal the stream produced.
    let renormalized = collect_terminal(normalize_grpc_stream(Box::pin(futures::stream::iter(
        vec![Ok(streaming::RawStreamingChoice::FinalResponse(typed))],
    ))))
    .await;
    assert_eq!(terminal.identity(), renormalized.identity());
    assert_eq!(terminal.finish_reason, renormalized.finish_reason);
    assert_eq!(terminal.model, renormalized.model);
    assert_eq!(terminal.usage, renormalized.usage);
    assert_eq!(
        terminal.finish_reason,
        Some(rig_core::completion::FinishReason::Stop)
    );
    assert_eq!(terminal.model.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(
        terminal.identity().response_id.as_deref(),
        Some("resp-grpc-stream")
    );
}
