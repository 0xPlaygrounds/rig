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
