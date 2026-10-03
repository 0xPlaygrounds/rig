use super::*;
use crate::completion::tests::{complete, stream_from_events};
use base64::Engine as _;
use futures::StreamExt;
use rig_core::completion::CompletionResponse;
use rig_core::message::{AssistantContent, Reasoning};
use rig_core::streaming::{Item, StreamEvent};
use serde_json::json;

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
    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    let mut blocks = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End {
            content: AssistantContent::Reasoning(reasoning),
            ..
        }) = item.expect("stream item should be ok")
        {
            blocks.push(reasoning);
        }
    }
    blocks
}

/// The REST part a signed thought block holds: the joined text and the
/// signature as standard base64 over the wire's bytes.
fn signed_thought(text: &str, signature: &[u8]) -> serde_json::Value {
    json!({
        "text": text,
        "thought": true,
        "thoughtSignature": base64::engine::general_purpose::STANDARD.encode(signature),
    })
}

// Consecutive thought parts continue one block, which holds the merged
// part with the signature.
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
    assert_eq!(signed.text, "think1 think2");
    assert_eq!(
        signed.native.as_ref().map(|native| &native.item),
        Some(&signed_thought("think1 think2", signature_bytes))
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
        signed.native.as_ref().map(|native| &native.item),
        Some(&signed_thought("thinking...", signature_bytes))
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
    assert!(signed.text.is_empty());
    assert_eq!(
        signed.native.as_ref().map(|native| &native.item),
        Some(&signed_thought("", signature_bytes))
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
// is emitted immediately under its own block id, so the
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

    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    let mut correlators = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End {
            content: AssistantContent::ToolCall(tool_call),
            ..
        }) = item.expect("stream item should be ok")
        {
            assert_eq!(tool_call.function.name, "get_weather");
            correlators.push(tool_call.id.clone());
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
    failed: bool,
    text: String,
}

async fn drain(events: Vec<proto::GenerateContentResponse>) -> Drained {
    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    let mut drained = Drained {
        errors: Vec::new(),
        reached_terminal: false,
        failed: false,
        text: String::new(),
    };

    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => drained.text.push_str(&text),
            Ok(_) => {}
            Err(error) => drained.errors.push(error.to_string()),
        }
    }
    if let Ok(response) = stream.finish().await {
        drained.reached_terminal = true;
        drained.failed = response.stop().is_failure();
    }

    drained
}

// A tool-protocol finish ends the turn as a failure, which is never
// replayed, rather than failing the reply.
#[tokio::test]
async fn tool_protocol_finishes_end_the_turn_as_a_failure() {
    for reason in [
        proto::candidate::FinishReason::MalformedFunctionCall,
        proto::candidate::FinishReason::UnexpectedToolCall,
        proto::candidate::FinishReason::TooManyToolCalls,
    ] {
        let drained = drain(vec![failed_response(
            reason,
            Some("the call was malformed"),
        )])
        .await;
        assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
        assert!(
            drained.reached_terminal && drained.failed,
            "{}",
            reason.as_str_name()
        );
    }
}

// Everything after a failure finish is dead, so a later genuine terminal
// cannot dress the failed turn up as complete.
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

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    assert!(drained.text.is_empty(), "text: {:?}", drained.text);
    assert!(drained.reached_terminal && drained.failed);
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

// The unary path decodes through the same decoder, so the two surfaces
// end a tool-protocol failure alike.
#[tokio::test]
async fn unary_and_streaming_report_the_same_tool_protocol_failure() {
    let response = failed_response(
        proto::candidate::FinishReason::TooManyToolCalls,
        Some("budget exhausted"),
    );

    let streamed = drain(vec![response.clone()]).await;
    let unary = complete(response).expect("a failed turn is a reply");
    assert!(streamed.failed);
    assert!(unary.stop().is_failure());
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

/// The events-first seam captures like the request-driven one: its
/// terminal `raw` is the same terminal `GenerateContentResponse` the
/// model's `stream()` would attach, because both funnel through the
/// adapter's `terminal_record`.
#[tokio::test]
async fn stream_from_events_terminal_carries_raw() {
    let mut stream = stream_from_events(
        vec![response(vec![text_part("hi")], 0), terminal_frame()]
            .into_iter()
            .map(Ok)
            .collect(),
    );
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let terminal = stream.finish().await.expect("the stream ends");

    let raw = &terminal.raw;
    let typed: proto::GenerateContentResponse =
        crate::rest::from_rest(raw.clone()).expect("raw must read back");
    assert_eq!(typed, terminal_frame());
    assert_eq!(terminal.usage.total_tokens, Some(5));
}

/// The load-bearing streaming capture property at the seam
/// `Model::stream` routes through: the terminal's `raw` is
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
        crate::rest::from_rest(raw.clone()).expect("raw must read back");
    assert_eq!(
        crate::rest::to_rest(&typed).expect("transcodes"),
        *raw,
        "the capture is exactly the terminal message's REST JSON"
    );
    assert_eq!(typed, terminal_frame());
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        Some(&serde_json::json!("STOP"))
    );

    // Feeding the capture back through the same pipeline tells the same
    // story as the terminal the stream produced.
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

/// A streamed turn names this wire as its origin: the Gemini API, as the
/// REST wire does, and this provider, so it replays only here.
#[tokio::test]
async fn the_stream_names_this_wire_as_its_origin() {
    let terminal = normalized_terminal(vec![
        response(vec![thought_part("hmm", b"sig")], 0),
        terminal_frame(),
    ])
    .await;
    assert_eq!(terminal.provider(), super::super::completion::PROVIDER_NAME);
    assert_eq!(terminal.origin.api.as_str(), "gemini.generate_content");
    assert!(matches!(
        terminal.choice.first(),
        Some(AssistantContent::Reasoning(_))
    ));
}

/// The streamed twin of a signed answer: the text, then an empty part
/// carrying the signature, then more text. The parts continue one text
/// block, which holds the merged part with the signature.
#[tokio::test]
async fn a_trailing_signed_part_continues_the_text_it_follows() {
    let mut signed = text_part("");
    signed.thought_signature = b"sig".to_vec();
    let mut stream = stream_from_events(
        vec![
            response(vec![text_part("289")], 0),
            response(vec![signed], 0),
            terminal_frame(),
        ]
        .into_iter()
        .map(Ok)
        .collect(),
    );
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let choice = stream.finish().await.expect("terminal").choice;
    assert_eq!(
        choice,
        vec![
            AssistantContent::text("289!")
                .with_native(json!({ "text": "289!", "thoughtSignature": "c2ln" }))
        ]
    );
}

/// Safety ratings and citations reach the response's `raw`, since a turn
/// keeps provider data on its blocks only.
#[tokio::test]
async fn safety_ratings_and_citations_reach_raw() {
    let mut frame = terminal_frame();
    if let Some(candidate) = frame.candidates.first_mut() {
        candidate.safety_ratings = vec![proto::SafetyRating {
            category: proto::HarmCategory::Harassment as i32,
            probability: proto::safety_rating::HarmProbability::Negligible as i32,
            blocked: false,
        }];
        candidate.citation_metadata = Some(proto::CitationMetadata {
            citation_sources: vec![proto::CitationSource {
                start_index: Some(0),
                end_index: Some(4),
                uri: Some("https://example.com".to_owned()),
                license: None,
            }],
        });
    }
    let terminal = normalized_terminal(vec![frame]).await;
    let native = terminal
        .raw
        .pointer("/candidates/0")
        .cloned()
        .expect("the terminal candidate");
    assert_eq!(
        native.get("safetyRatings"),
        Some(&json!([{ "category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE" }]))
    );
    assert_eq!(
        native.pointer("/citationMetadata/citationSources/0/uri"),
        Some(&json!("https://example.com"))
    );
}

/// Code execution's language and outcome are proto enums; the REST JSON
/// spells them by name.
#[test]
fn code_execution_parts_restate_with_rest_enum_names() {
    let content = proto::Content {
        parts: vec![
            proto::Part {
                data: Some(proto::part::Data::ExecutableCode(proto::ExecutableCode {
                    language: proto::executable_code::Language::Python as i32,
                    code: "print(1)".to_owned(),
                })),
                ..Default::default()
            },
            proto::Part {
                data: Some(proto::part::Data::CodeExecutionResult(
                    proto::CodeExecutionResult {
                        outcome: proto::code_execution_result::Outcome::Ok as i32,
                        output: "1\n".to_owned(),
                    },
                )),
                ..Default::default()
            },
        ],
        role: "model".to_owned(),
    };
    assert_eq!(
        crate::rest::to_rest(&content).expect("transcodes"),
        json!({"role": "model", "parts": [
            {"executableCode": {"language": "PYTHON", "code": "print(1)"}},
            {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1\n"}},
        ]})
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
    let wire = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH);
    let mut request = rig_core::completion::CompletionRequest::new("next");
    request.chat_history = rig_core::completion::adapt(&history, &wire);
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
