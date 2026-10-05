use super::*;
use crate::completion::tests::{complete, stream_from_events};
use futures::StreamExt;
use rig_core::completion::CompletionResponse;
use rig_core::streaming::{Item, StreamEvent};

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

// ---- #2258 H4: tool-protocol finish reasons must fail the turn ----

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
