use super::*;
use crate::completion::CompletionRequest;
use serde_json::json;

/// The chat wire bound to `http_client`: the model every streamed case
/// drives.
fn cohere_model<H: Clone>(
    http_client: H,
) -> crate::driver::Model<crate::providers::cohere::Chat, H> {
    crate::driver::Model::new(
        crate::providers::cohere::CohereConfig::new("test-key")
            .completion(crate::providers::cohere::COMMAND_R_08_2024),
        http_client,
    )
}

/// What one streamed reply over `events` yielded, and what it finished
/// with.
struct Replied {
    items: Vec<Result<crate::streaming::Item<crate::streaming::StreamEvent>, ProviderError>>,
    outcome: Result<crate::completion::CompletionResponse, ProviderError>,
}

impl Replied {
    fn texts(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Ok(crate::streaming::Item::Event(crate::streaming::StreamEvent::Text {
                    text,
                    ..
                })) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn error(&self) -> Option<&ProviderError> {
        self.items.iter().find_map(|item| item.as_ref().err())
    }
}

async fn replied(events: &[&str]) -> Replied {
    use futures::StreamExt;

    let sse_bytes = bytes::Bytes::from(
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    );
    let model = cohere_model(crate::test_utils::MockStreamingClient { sse_bytes });
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream should open");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    Replied {
        items,
        outcome: stream.finish().await,
    }
}

fn classify(data: &str) -> crate::wire::WireEvent<StreamingEvent> {
    wire::classify_tagged_frame(data, "type", |event_type| {
        KNOWN_EVENT_TYPES.contains(&event_type)
    })
}

#[test]
fn classify_known_event_decodes() {
    let frame = json!({
        "type": "content-delta",
        "delta": {"message": {"content": {"text": "hi"}}},
    })
    .to_string();
    assert!(matches!(
        classify(&frame),
        crate::wire::WireEvent::Known(StreamingEvent::ContentDelta { .. })
    ));
}

#[test]
fn classify_unknown_event_type_is_unknown() {
    let frame = json!({"type": "debug-trace"}).to_string();
    assert!(matches!(
        classify(&frame),
        crate::wire::WireEvent::Unknown { event_type, .. } if event_type == "debug-trace"
    ));
}

#[test]
fn classify_invalid_json_is_corrupt() {
    assert!(matches!(
        classify("{not json"),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

#[test]
fn classify_known_event_with_defective_payload_is_corrupt() {
    let frame = json!({"type": "content-delta", "delta": 42}).to_string();
    assert!(matches!(
        classify(&frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

#[tokio::test]
async fn stream_terminal_record_is_normalized() {
    let replied = replied(&[
        r#"{"type":"message-start","id":"msg_1"}"#,
        r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"MAX_TOKENS","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
    ])
    .await;
    let response = replied.outcome.expect("the reply ended");
    assert_eq!(
        response.provider(),
        crate::providers::cohere::completion::PROVIDER_NAME
    );
    assert_eq!(response.response_id(), Some("msg_1"));
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(4));
    assert_eq!(response.usage.total_tokens, Some(14));
    // Cohere's stream never names the model.
    assert_eq!(response.model(), None);
}

#[tokio::test]
async fn truncated_stream_does_not_synthesize_an_end() {
    // No `message-end`: the stream was cut off mid-response.
    let replied = replied(&[
        r#"{"type":"message-start","id":"msg_1"}"#,
        r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
    ])
    .await;
    assert_eq!(replied.texts(), ["hi"]);
    assert!(
        matches!(replied.outcome, Err(ProviderError::Truncated)),
        "EOF without message-end is truncation: {:?}",
        replied.outcome
    );
}

#[tokio::test]
async fn a_malformed_frame_ends_the_reply() {
    // A malformed frame between valid content and the genuine end ends the
    // reply with its error: the later end is never read.
    let replied = replied(&[
        r#"{"type":"message-start","id":"msg_1"}"#,
        r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
        "{not json",
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
    ])
    .await;
    assert_eq!(replied.texts(), ["hi"]);
    assert!(
        replied.error().is_some(),
        "the malformed frame reaches the consumer"
    );
    assert!(replied.outcome.is_err());
}

#[tokio::test]
async fn known_event_with_malformed_field_is_surfaced_as_an_error() {
    // A known `type` whose payload fails the full parse (text should be a
    // string) is a data-level defect, not a forward-compatibility event.
    let replied = replied(&[
        r#"{"type":"message-start","id":"msg_1"}"#,
        r#"{"type":"content-delta","delta":{"message":{"content":{"text":42}}}}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
    ])
    .await;
    let error = replied.error().expect("the defect is an error item");
    assert_eq!(error.kind(), crate::error::ErrorKind::Json, "{error:?}");
    assert!(replied.outcome.is_err(), "the defect ended the reply");
}

#[tokio::test]
async fn unknown_event_type_is_skipped_and_the_end_still_arrives() {
    // An invented `type` is an event this client doesn't know yet: it is
    // passed through for forward compatibility, not surfaced as an error.
    let replied = replied(&[
        r#"{"type":"message-start","id":"msg_1"}"#,
        r#"{"type":"debug-trace","delta":{"whatever":true}}"#,
        r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
    ])
    .await;
    assert!(replied.error().is_none(), "{:?}", replied.items);
    assert_eq!(replied.texts(), ["hi"]);
    assert_eq!(
        replied
            .outcome
            .expect("the reply ended")
            .usage
            .output_tokens,
        Some(4)
    );
}

#[tokio::test]
async fn message_end_without_delta_still_ends_the_reply() {
    // `message-end` with no payload is still the provider completing the
    // turn; the reply ends with unreported usage.
    let replied = replied(&[
        r#"{"type":"message-start","id":"msg_1"}"#,
        r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
        r#"{"type":"message-end"}"#,
    ])
    .await;
    assert_eq!(replied.texts(), ["hi"]);
    let response = replied
        .outcome
        .expect("message-end without a delta is still the end");
    assert_eq!(response.usage, crate::completion::Usage::default());
    assert_eq!(response.finish_reason(), None);
    assert_eq!(response.response_id(), Some("msg_1"));
}

#[tokio::test]
async fn thinking_deltas_aggregate_into_one_reasoning_part_before_the_text() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    // Cohere v2 reasoning models stream `content-delta` frames carrying
    // `thinking` before the answer's `text` frames (documented `thinking`
    // deltas; #2258 F8 — previously these fell through the `text` guard
    // and the thought text was lost).
    let replied = replied(&[
        r#"{"type":"message-start","id":"msg_1"}"#,
        r#"{"type":"content-start","index":0,"delta":{"message":{"content":{"type":"thinking","thinking":""}}}}"#,
        r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"thinking":"step one, "}}}}"#,
        r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"thinking":"step two"}}}}"#,
        r#"{"type":"content-end","index":0}"#,
        r#"{"type":"content-start","index":1,"delta":{"message":{"content":{"type":"text","text":""}}}}"#,
        r#"{"type":"content-delta","index":1,"delta":{"message":{"content":{"text":"answer"}}}}"#,
        r#"{"type":"content-end","index":1}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
    ])
    .await;
    let reasoning_deltas: Vec<&str> = replied
        .items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::Reasoning { text, .. })) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning_deltas, ["step one, ", "step two"]);

    let parts = replied.outcome.expect("the reply ended").choice;
    assert_eq!(parts.len(), 2, "one reasoning part, one text part");
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Reasoning(reasoning)) if reasoning.text == "step one, step two"
    ));
    assert!(matches!(
        parts.get(1),
        Some(AssistantContent::Text(text)) if text.text == "answer"
    ));
}

#[tokio::test]
async fn errored_stream_does_not_synthesize_an_end() {
    use crate::test_utils::HttpErrorStreamingClient;
    use futures::StreamExt;

    let model = cohere_model(HttpErrorStreamingClient::new(
        http::StatusCode::TOO_MANY_REQUESTS,
        r#"{"message":"slow down"}"#,
    ));
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream should open");
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        saw_error |= item.is_err();
    }
    assert!(saw_error, "the transport failure must reach the consumer");
    assert!(
        stream.finish().await.is_err(),
        "a failed stream must not be reported as a successful, zero-usage completion"
    );
}

#[test]
fn test_message_content_delta_deserialization() {
    let json = json!({
        "type": "content-delta",
        "delta": {
            "message": {
                "content": {
                    "text": "Hello world"
                }
            }
        }
    });

    let event: StreamingEvent = serde_json::from_value(json).unwrap();
    match event {
        StreamingEvent::ContentDelta { delta, .. } => {
            let message = delta.unwrap().message;
            assert_eq!(message["content"]["text"], "Hello world");
        }
        _ => panic!("Expected ContentDelta"),
    }
}

#[test]
fn test_tool_call_start_deserialization() {
    let json = json!({
        "type": "tool-call-start",
        "delta": {
            "message": {
                "tool_calls": {
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{"
                    }
                }
            }
        }
    });

    let event: StreamingEvent = serde_json::from_value(json).unwrap();
    match event {
        StreamingEvent::ToolCallStart { delta, .. } => {
            let message = delta.unwrap().message;
            assert_eq!(message["tool_calls"]["id"], "call_123");
            assert_eq!(message["tool_calls"]["function"]["name"], "get_weather");
        }
        _ => panic!("Expected ToolCallStart"),
    }
}

#[test]
fn test_tool_call_delta_deserialization() {
    let json = json!({
        "type": "tool-call-delta",
        "delta": {
            "message": {
                "tool_calls": {
                    "function": {
                        "arguments": "\"location\""
                    }
                }
            }
        }
    });

    let event: StreamingEvent = serde_json::from_value(json).unwrap();
    match event {
        StreamingEvent::ToolCallDelta { delta, .. } => {
            let message = delta.unwrap().message;
            assert_eq!(
                message["tool_calls"]["function"]["arguments"],
                "\"location\""
            );
        }
        _ => panic!("Expected ToolCallDelta"),
    }
}

#[test]
fn test_tool_call_end_deserialization() {
    let json = json!({
        "type": "tool-call-end"
    });

    let event: StreamingEvent = serde_json::from_value(json).unwrap();
    match event {
        StreamingEvent::ToolCallEnd { .. } => {
            // Success
        }
        _ => panic!("Expected ToolCallEnd"),
    }
}

#[test]
fn test_message_end_with_usage_deserialization() {
    let json = json!({
        "type": "message-end",
        "delta": {
            "usage": {
                "tokens": {
                    "input_tokens": 100,
                    "output_tokens": 50
                }
            }
        }
    });

    let event: StreamingEvent = serde_json::from_value(json).unwrap();
    match event {
        StreamingEvent::MessageEnd { delta } => {
            assert!(delta.is_some());
            let usage = delta.unwrap().usage.unwrap();
            let tokens = usage.tokens.unwrap();
            assert_eq!(tokens.input_tokens, Some(100.0));
            assert_eq!(tokens.output_tokens, Some(50.0));
        }
        _ => panic!("Expected MessageEnd"),
    }
}

#[test]
fn test_streaming_event_order() {
    // Test that a typical sequence of events deserializes correctly
    let events = vec![
        json!({"type": "message-start"}),
        json!({"type": "content-start"}),
        json!({
            "type": "content-delta",
            "delta": {
                "message": {
                    "content": {
                        "text": "Sure, "
                    }
                }
            }
        }),
        json!({
            "type": "content-delta",
            "delta": {
                "message": {
                    "content": {
                        "text": "I can help with that."
                    }
                }
            }
        }),
        json!({"type": "content-end"}),
        json!({"type": "tool-plan-delta"}),
        json!({
            "type": "tool-call-start",
            "delta": {
                "message": {
                    "tool_calls": {
                        "id": "call_abc",
                        "function": {
                            "name": "search",
                            "arguments": ""
                        }
                    }
                }
            }
        }),
        json!({
            "type": "tool-call-delta",
            "delta": {
                "message": {
                    "tool_calls": {
                        "function": {
                            "arguments": "{\"query\":"
                        }
                    }
                }
            }
        }),
        json!({
            "type": "tool-call-delta",
            "delta": {
                "message": {
                    "tool_calls": {
                        "function": {
                            "arguments": "\"Rust\"}"
                        }
                    }
                }
            }
        }),
        json!({"type": "tool-call-end"}),
        json!({
            "type": "message-end",
            "delta": {
                "usage": {
                    "tokens": {
                        "input_tokens": 50,
                        "output_tokens": 25
                    }
                }
            }
        }),
    ];

    for (i, event_json) in events.iter().enumerate() {
        let result = serde_json::from_value::<StreamingEvent>(event_json.clone());
        assert!(
            result.is_ok(),
            "Failed to deserialize event at index {}: {:?}",
            i,
            result.err()
        );
    }
}

/// A `tool-call-start` whose id is empty gets an id rig issues: two such
/// calls in one stream stay distinct calls, each with its own arguments.
#[tokio::test]
async fn empty_tool_call_ids_are_minted_not_keyed_on_the_empty_string() {
    let call = |n: u32| {
        [
            r#"{"type":"tool-call-start","delta":{"message":{"tool_calls":{"id":"","function":{"name":"add","arguments":""}}}}}"#.to_owned(),
            format!(
                r#"{{"type":"tool-call-delta","delta":{{"message":{{"tool_calls":{{"function":{{"arguments":"{{\"n\":{n}}}"}}}}}}}}}}"#
            ),
            r#"{"type":"tool-call-end"}"#.to_owned(),
        ]
    };
    let events: Vec<String> = std::iter::once(r#"{"type":"message-start","id":"msg_1"}"#.to_owned())
        .chain(call(1))
        .chain(call(2))
        .chain(std::iter::once(
            r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}}"#.to_owned(),
        ))
        .collect();
    let events: Vec<&str> = events.iter().map(String::as_str).collect();
    let response = replied(&events).await.outcome.expect("the reply ended");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2, "two calls: {calls:?}");
    assert_ne!(
        calls[0].id, calls[1].id,
        "each id-less call is its own call"
    );
    assert!(
        calls.iter().all(|call| call.id.provider().is_none()),
        "{calls:?}"
    );
    assert_eq!(calls[0].function.arguments_value(), json!({"n": 1}));
    assert_eq!(calls[1].function.arguments_value(), json!({"n": 2}));
}

/// Every event of this wire has a sample, so a new one fails to compile
/// until it is numbered here and fails until a frame decodes to it.
#[test]
fn every_streaming_event_has_a_sample() {
    let index = |event: &StreamingEvent| match event {
        StreamingEvent::MessageStart { .. } => 0,
        StreamingEvent::ContentStart { .. } => 1,
        StreamingEvent::ContentDelta { .. } => 2,
        StreamingEvent::ContentEnd { .. } => 3,
        StreamingEvent::ToolPlanDelta { .. } => 4,
        StreamingEvent::ToolCallStart { .. } => 5,
        StreamingEvent::ToolCallDelta { .. } => 6,
        StreamingEvent::ToolCallEnd { .. } => 7,
        StreamingEvent::CitationStart { .. } => 8,
        StreamingEvent::CitationEnd => 9,
        StreamingEvent::MessageEnd { .. } => 10,
    };
    let samples: Vec<StreamingEvent> = KNOWN_EVENT_TYPES
        .iter()
        .map(|kind| serde_json::from_value(json!({"type": kind})).expect("each event decodes"))
        .collect();
    crate::test_utils::history::assert_every_variant(&samples, index, 11);
}

/// The events of one turn: thinking, an item kind rig has never seen, text,
/// and a call carrying a field rig has never seen, with `id` when given.
fn invented_events(id: Option<&str>) -> Vec<String> {
    [
        json!({"type": "message-start", "id": "msg_1", "delta": {"message": {"role": "assistant"}}}),
        json!({"type": "content-start", "index": 0, "delta": {"message": {"content": {"type": "thinking", "thinking": "plan"}}}}),
        json!({"type": "content-end", "index": 0}),
        json!({"type": "content-start", "index": 1, "delta": {"message": {"content": {"type": "x-probe", "payload": {"kept": true}}}}}),
        json!({"type": "content-end", "index": 1}),
        json!({"type": "content-start", "index": 2, "delta": {"message": {"content": {"type": "text", "text": "adding"}}}}),
        json!({"type": "content-end", "index": 2}),
        {
            let mut call = json!({"type": "function", "function": {"name": "add", "arguments": ""}, "x_call_probe": 1});
            if let Some(id) = id {
                call["id"] = json!(id);
            }
            json!({"type": "tool-call-start", "index": 0, "delta": {"message": {"tool_calls": call}}})
        },
        json!({"type": "tool-call-delta", "index": 0, "delta": {"message": {"tool_calls": {"function": {"arguments": "{\"x\":1}"}}}}}),
        json!({"type": "tool-call-end", "index": 0}),
        json!({"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}}),
    ]
    .iter()
    .map(serde_json::Value::to_string)
    .collect()
}

/// A streamed call Cohere sent without an id is kept, with an id rig issues.
#[tokio::test]
async fn a_streamed_call_without_an_id_is_kept() {
    let events = invented_events(None);
    let events: Vec<&str> = events.iter().map(String::as_str).collect();
    let response = replied(&events).await.outcome.expect("the reply ended");
    let calls: Vec<_> = response.tool_calls().collect();
    let [call] = calls.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert!(call.id.is_local());
    assert_eq!(call.function.arguments_value(), json!({"x": 1}));
}

/// An item kind rig has never seen is kept as an item that replays, and a
/// field it has never seen on a call is kept on the call: both survive
/// decoding in both modes and go back to the model that sent them.
#[test]
fn an_invented_item_and_call_field_survive_decode_and_replay() {
    use crate::message::{AssistantContent, AssistantMessage, Message};
    use crate::wire::{Mode, Wire, WireFrame};

    let wire = crate::providers::cohere::CohereConfig::new("k").completion("command");
    let streamed: Vec<WireFrame> = invented_events(Some("add_1"))
        .into_iter()
        .map(WireFrame::Text)
        .collect();
    let whole = WireFrame::Text(
        json!({
            "id": "msg_1",
            "finish_reason": "TOOL_CALL",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "x-probe", "payload": {"kept": true}},
                    {"type": "text", "text": "adding"},
                ],
                "tool_calls": [{"type": "function", "function": {"name": "add", "arguments": "{\"x\":1}"}, "x_call_probe": 1, "id": "add_1"}],
            },
        })
        .to_string(),
    );
    crate::test_utils::history::assert_restated_agrees(&wire, [whole.clone()], streamed.clone());
    for (mode, frames) in [(Mode::Unary, vec![whole]), (Mode::Streaming, streamed)] {
        let response =
            crate::test_utils::history::decode(&wire, mode, frames).expect("the reply decodes");
        assert!(
            response.choice.iter().any(|block| matches!(
                block,
                AssistantContent::Opaque(opaque) if opaque.replay && opaque.kind() == Some("x-probe")
            )),
            "{mode:?}: {:?}",
            response.choice
        );
        let turn = AssistantMessage {
            content: response.choice.clone(),
            ..response.head()
        };
        let call = turn.tool_calls().next().expect("the call").clone();
        let mut request = CompletionRequest::new("and then?");
        request.chat_history = crate::completion::history::adapt(
            &[
                Message::user("add one"),
                Message::Assistant(turn),
                Message::tool_result(call.id.clone(), call.function.name.clone(), "1"),
                Message::user("and then?"),
            ],
            &wire,
        );
        let encoded = wire.encode(request, Mode::Unary).expect("encodes");
        let crate::wire::Body::Bytes(bytes) = encoded.request.body() else {
            panic!("a JSON body");
        };
        let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
        let replayed = &body["messages"][1];
        assert_eq!(replayed["content"][1]["payload"]["kept"], true, "{mode:?}");
        assert_eq!(replayed["tool_calls"][0]["x_call_probe"], 1, "{mode:?}");
        assert_eq!(replayed["tool_calls"][0]["id"], "add_1", "{mode:?}");
    }
}
