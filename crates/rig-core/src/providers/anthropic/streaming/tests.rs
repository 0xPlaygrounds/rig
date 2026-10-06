use super::super::completion::{CLAUDE_OPUS_4_8, CLAUDE_OPUS_5_5, CLAUDE_SONNET_4_6};
use super::*;
use crate::completion::CompletionRequest;
use crate::completion::Message as RigMessage;
use crate::completion::request::Document as RigDocument;
use crate::driver::{Decoded, decode_events};
use crate::message::{AssistantContent, Opaque, Reasoning, StopReason};
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::wire::Mode;
use serde_json::json;

/// A fresh decoder, for its classifier.
fn adapter() -> MessagesDecoder {
    MessagesDecoder::new(false)
}

/// Decode `events` through one decoder as one reply, then EOF.
fn decode(events: impl IntoIterator<Item = MessagesEvent>) -> Decoded<Completion> {
    decode_events!(MessagesDecoder::new(false), "anthropic", events)
}

/// The event one wire frame classifies to.
fn classified(frame: &str) -> MessagesEvent {
    let crate::wire::WireEvent::Known(event) = adapter().classify(WireFrame::Text(frame.into()))
    else {
        panic!("{frame} must classify Known");
    };
    event
}

/// The response the stream of `frames` folds into on the Anthropic wire.
fn streamed(frames: &[Value]) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
}

/// The frames of one content block: its start, its deltas and its stop.
fn block(index: usize, start: Value, deltas: &[Value]) -> Vec<Value> {
    std::iter::once(json!({"type": "content_block_start", "index": index, "content_block": start}))
        .chain(
            deltas.iter().map(
                |delta| json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ),
        )
        .chain([json!({"type": "content_block_stop", "index": index})])
        .collect()
}

/// A stream of `blocks` ending with `stop_reason`.
fn reply(blocks: Vec<Vec<Value>>, stop_reason: &str) -> Vec<Value> {
    std::iter::once(json!({"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": CLAUDE_SONNET_4_6,
        "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 1}
    }}))
    .chain(blocks.into_iter().flatten())
    .chain([json!({"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
        "usage": {"output_tokens": 5}})])
    .collect()
}

/// The provider item a decoded block holds.
fn item(content: &AssistantContent) -> Option<&Value> {
    match content {
        AssistantContent::Opaque(Opaque { item, .. }) => Some(item),
        content => content.native_item(),
    }
}

/// The streaming request body the [`Messages`](super::super::wire::Messages)
/// wire encodes — the shared typed request plus the streaming-only patches.
///
/// Read back off the encoded HTTP request rather than rebuilt here: the
/// wire's `encode` is the only statement of that body now, so a cell that
/// pins the body pins the thing that runs.
fn built_streaming_body(
    model: &str,
    request: CompletionRequest,
    strict_tools: bool,
) -> Result<Value, ProviderError> {
    use crate::wire::{Body, Mode, Wire};

    let wire =
        crate::providers::anthropic::wire::AnthropicConfig::new("test-key").completion(model);
    let wire = if strict_tools {
        wire.with_strict_tools()
    } else {
        wire
    };
    let request = <Completion as crate::wire::Operation>::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, Mode::Streaming)?;
    match encoded.request.body() {
        Body::Bytes(bytes) => Ok(serde_json::from_slice(bytes)?),
        Body::Multipart(_) => Err(ProviderError::request("the Messages endpoint takes JSON")),
    }
}

#[test]
fn test_streaming_tool_build_marks_final_combined_tool() {
    let request = CompletionRequest::new(RigMessage::user("Hi"))
        .max_tokens(64)
        .tools(vec![crate::completion::ToolDefinition {
            name: crate::message::ToolName::new("rig_tool").expect("tool name"),
            description: "Rig tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }])
        .additional_params(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"}
            }]
        }));
    let wire = AnthropicConfig::new("test-key")
        .completion(CLAUDE_SONNET_4_6)
        .with_prompt_caching();
    let body = streamed_body(&wire, request);
    let tools = body["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["name"], "provider_tool");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

/// The streaming body `wire` encodes for `request`.
fn streamed_body(
    wire: &crate::providers::anthropic::Messages,
    request: CompletionRequest,
) -> Value {
    use crate::wire::Wire;
    crate::test_utils::json_body(
        &wire
            .encode(request, Mode::Streaming)
            .expect("the request encodes")
            .request,
    )
}

#[test]
fn streaming_request_keeps_documents_after_leading_system_messages() {
    let request = CompletionRequest::from(vec![
        RigMessage::system("System prompt"),
        RigMessage::assistant("Earlier assistant turn"),
        RigMessage::system("Mid-conversation instruction"),
        RigMessage::user("Prompt"),
    ])
    .max_tokens(64)
    .documents(vec![RigDocument {
        id: "doc1".to_string(),
        text: "Document text.".to_string(),
        additional_props: Default::default(),
    }]);

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    assert_eq!(body["system"][0]["text"], "System prompt");
    assert_eq!(body["system"].as_array().map(Vec::len), Some(1));
    let messages = body["messages"]
        .as_array()
        .expect("messages should be array");
    // The misplaced mid-conversation instruction follows the user turn.
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[3]["role"], "system");
    assert_eq!(messages[0]["role"], "user");
    assert!(
        messages[0].to_string().contains("<file id: doc1>"),
        "document message should follow top-level system: {messages:?}"
    );
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc1>"))
            .count(),
        1,
        "document message should appear exactly once: {messages:?}"
    );
}

#[test]
fn streaming_body_is_blocking_body_plus_stream_flag_and_carries_output_schema() {
    let schema: schemars::Schema = serde_json::from_value(json!({
        "title": "WeatherResponse",
        "type": "object",
        "properties": { "city": { "type": "string" } }
    }))
    .expect("schema should deserialize");

    let request = CompletionRequest::from(vec![
        RigMessage::system("You are helpful"),
        RigMessage::user("What's the weather?"),
    ])
    .temperature(0.5)
    .max_tokens(64)
    .output_schema(schema);

    let streaming_body = built_streaming_body(CLAUDE_OPUS_4_8, request.clone(), false)
        .expect("streaming request body should build");

    // The streaming endpoint flag is set.
    assert_eq!(streaming_body["stream"], serde_json::Value::Bool(true));

    // Regression: `output_schema` now reaches the streaming wire as
    // `output_config` (the hand-rolled body dropped it entirely, so this
    // assertion would have failed before the typed-request unification).
    assert_eq!(
        streaming_body["output_config"]["format"]["type"],
        "json_schema"
    );
    assert!(
        streaming_body["output_config"]["format"]["schema"].is_object(),
        "streaming body must carry the structured-output schema: {streaming_body}"
    );

    // Unification invariant: the streaming body is exactly the blocking body
    // (built via the same typed request) plus `stream: true`. Pins the two
    // wire formats together so a future edit can't reintroduce drift.
    let wire = crate::providers::anthropic::wire::AnthropicConfig::new("test-key")
        .completion(CLAUDE_OPUS_4_8);
    let mut expected = crate::test_utils::json_body(
        &crate::wire::Wire::encode(&wire, request, Mode::Unary)
            .expect("blocking request body should build")
            .request,
    );
    expected
        .as_object_mut()
        .expect("body is an object")
        .insert("stream".to_string(), serde_json::Value::Bool(true));

    assert_eq!(streaming_body, expected);
}

#[test]
fn streaming_body_drops_tool_choice_when_no_tools_are_advertised() {
    // The typed request serializes a caller-set `tool_choice` regardless of
    // whether tools are present, but the streaming path has always emitted
    // `tool_choice` *only* alongside a non-empty tool set (Anthropic rejects it
    // otherwise). A `tool_choice` set with no tools must not reach the wire.
    let request = CompletionRequest::new(RigMessage::user("Hi"))
        .max_tokens(64)
        .tool_choice(crate::message::ToolChoice::Auto);

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    assert!(
        body.get("tool_choice").is_none(),
        "tool_choice must be omitted when no tools are advertised: {body}"
    );
    assert!(body.get("tools").is_none());
}

/// Signature fragments concatenate onto the opening signature, as pi
/// assembles them, and the thinking item holds the whole block.
#[test]
fn thinking_assembles_its_text_and_signature_into_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "thinking", "thinking": "", "signature": "open-"}),
            &[
                json!({"type": "thinking_delta", "thinking": "Let me "}),
                json!({"type": "thinking_delta", "thinking": "think."}),
                json!({"type": "signature_delta", "signature": "sig_"}),
                json!({"type": "signature_delta", "signature": "end"}),
            ],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
        panic!("one reasoning block: {:?}", response.choice);
    };
    assert_eq!(reasoning.text, "Let me think.");
    assert!(!reasoning.redacted);
    assert_eq!(
        item(&response.choice[0]),
        Some(
            &json!({"type": "thinking", "thinking": "Let me think.", "signature": "open-sig_end"})
        )
    );
}

#[test]
fn redacted_thinking_is_a_redacted_block_holding_its_payload() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "redacted_thinking", "data": "redacted_blob"}),
            &[],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::Reasoning(Reasoning { text, redacted, .. })] =
        response.choice.as_slice()
    else {
        panic!("one reasoning block: {:?}", response.choice);
    };
    assert!(text.is_empty() && *redacted);
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "redacted_thinking", "data": "redacted_blob"}))
    );
}

/// A call's item holds the input its fragments assembled, and keeps every
/// field the provider sent, `caller` included.
#[test]
fn a_call_assembles_its_input_into_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {},
                "caller": {"type": "direct"}}),
            &[
                json!({"type": "input_json_delta", "partial_json": "{\"location\":"}),
                json!({"type": "input_json_delta", "partial_json": "\"Paris\"}"}),
            ],
        )],
        "tool_use",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.id.to_string(), "toolu_1");
    assert_eq!(
        call.function.arguments_value(),
        json!({"location": "Paris"})
    );
    assert_eq!(
        item(&response.choice[0]),
        Some(
            &json!({"type": "tool_use", "id": "toolu_1", "name": "lookup",
            "input": {"location": "Paris"}, "caller": {"type": "direct"}})
        )
    );
    assert_eq!(response.stop(), StopReason::ToolUse);
}

/// A stream that never states its stop reason is truncated, whatever else
/// it ended with: a `message_delta` without one, or `message_stop` alone.
#[test]
fn a_stream_without_its_stop_reason_never_ends_successfully() {
    let mut frames = reply(
        vec![block(0, json!({"type": "text", "text": "partial"}), &[])],
        "end_turn",
    );
    if let Some(delta) = frames.last_mut() {
        delta["delta"]["stop_reason"] = Value::Null;
    }
    frames.push(json!({"type": "message_stop"}));
    assert!(matches!(streamed(&frames), Err(ProviderError::Truncated)));
}

#[test]
fn input_to_a_text_block_fails_the_reply() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "text", "text": ""}),
            &[json!({"type": "input_json_delta", "partial_json": "{}"})],
        )],
        "end_turn",
    );
    assert!(streamed(&frames).is_err());
}

/// Citations arrive as deltas onto a text block that opened with `null` or
/// an empty list, and land on its item in order.
#[test]
fn citation_deltas_land_on_the_text_item() {
    let citation = json!({"type": "char_location", "cited_text": "The grass is green.",
        "document_index": 0, "document_title": "Example", "start_char_index": 0,
        "end_char_index": 20});
    for opening in [json!(null), json!([])] {
        let frames = reply(
            vec![block(
                0,
                json!({"type": "text", "text": "", "citations": opening}),
                &[
                    json!({"type": "citations_delta", "citation": citation}),
                    json!({"type": "text_delta", "text": "the grass is green"}),
                ],
            )],
            "end_turn",
        );
        let response = streamed(&frames).expect("the reply folds");
        let [AssistantContent::Text(text)] = response.choice.as_slice() else {
            panic!("one text block: {:?}", response.choice);
        };
        assert_eq!(text.text, "the grass is green");
        assert_eq!(
            response.choice[0]
                .native_item()
                .and_then(|item| item.get("citations")),
            Some(&json!([citation]))
        );
    }
}

/// pi's fallback rule: a leading `fallback` block marks the model that
/// took over and never replays; one after output began fails the reply.
#[test]
fn a_fallback_block_is_kept_first_and_an_error_after_output() {
    let fallback = json!({"type": "fallback", "model": "claude-opus-4-8"});
    let text = json!({"type": "text", "text": ""});
    let leading = streamed(&reply(
        vec![
            block(0, fallback.clone(), &[]),
            block(
                1,
                text.clone(),
                &[json!({"type": "text_delta", "text": "hi"})],
            ),
        ],
        "end_turn",
    ))
    .expect("a leading fallback is legal");
    assert!(matches!(
        leading.choice.first(),
        Some(AssistantContent::Opaque(Opaque { replay: false, .. }))
    ));
    let late = streamed(&reply(
        vec![
            block(0, text, &[json!({"type": "text_delta", "text": "hi"})]),
            block(1, fallback, &[]),
        ],
        "end_turn",
    ));
    assert!(late.is_err(), "{late:?}");
}

/// Classification is the only policy site. An unmodeled *top-level* event
/// type is `Unknown` (driver: warn + skip); a `ping` is Known; and a known
/// tag whose payload this client cannot decode is `Corrupt`, never silently
/// demoted to an ignorable unknown. An unmodeled *nested* delta type is
/// Known and lands in its item.
#[test]
fn classify_dispatches_on_the_known_event_list() {
    let adapter = adapter();

    let frame = WireFrame::Text(r#"{"type":"something_new_from_anthropic","field":"x"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Unknown { event_type, .. }
            if event_type == "something_new_from_anthropic"
    ));

    let frame = WireFrame::Text(r#"{"type":"ping"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Known(event) if event.kind() == "ping"
    ));

    let frame = WireFrame::Text("{not json".into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// A terminal `message_delta` carrying only the output count (Anthropic's
/// older shape, and Messages gateways) keeps `message_start`'s cache
/// counters, so input still counts the cached prefix: 10 uncached, 6 read
/// and 4 written.
#[test]
fn cache_usage_from_message_start_survives_output_only_terminal_delta() {
    let frames = [
        r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":4,"cache_read_input_tokens":6,"cache_creation":{"ephemeral_1h_input_tokens":3,"ephemeral_5m_input_tokens":1}}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#,
    ];
    let decoded = crate::driver::feed_frames!(
        MessagesDecoder::new(false),
        document::TerminalRecord::default(),
        "anthropic",
        frames.map(|frame| WireFrame::Text(frame.to_owned()))
    );
    let response = decoded.outcome.expect("the message_delta ends the reply");
    assert_eq!(response.usage.input_tokens, Some(10 + 6 + 4));
    assert_eq!(response.usage.cache_creation_input_tokens, Some(4));
    assert_eq!(response.usage.cached_input_tokens, Some(6));
    assert_eq!(response.usage.output_tokens, Some(3));
    assert_eq!(response.usage.total_tokens, Some(23));
    let usage = &response.raw["usage"];
    assert_eq!(usage["cache_creation_input_tokens"], 4);
    assert_eq!(usage["cache_read_input_tokens"], 6);
    assert_eq!(usage["cache_creation"]["ephemeral_1h_input_tokens"], 3);
}

/// A `content_block_delta` whose `delta` omits `type` is malformed, not
/// novel: silently skipping it would turn a compat gateway's untagged
/// text delta into a successful *empty* completion, so it fails the reply
/// (#2258 B5). So does a known delta whose text is not text, since the
/// block is built from it.
#[test]
fn a_delta_without_its_type_or_text_fails_the_reply() {
    for delta in [
        json!({"text": "hello"}),
        json!({"type": "text_delta", "text": 42}),
    ] {
        let frames = reply(
            vec![block(
                0,
                json!({"type": "text", "text": ""}),
                std::slice::from_ref(&delta),
            )],
            "end_turn",
        );
        assert!(streamed(&frames).is_err(), "{delta}");
    }
}

/// Bedrock-compat quirk: `message_start` without a message body is a
/// Known no-op, not a corrupt frame.
#[test]
fn message_start_with_null_message_is_a_known_noop() {
    let decoded = decode([classified(r#"{"type":"message_start","message":null}"#)]);
    assert!(
        decoded.events().is_empty(),
        "a message-less message_start is a no-op"
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission {
    use super::super::super::completion::CLAUDE_SONNET_4_6;
    use crate::providers::anthropic::wire::AnthropicConfig;
    use crate::streaming::{Item, StreamEvent};
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    const MESSAGE_START: &str = r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}"#;
    const TEXT_START: &str =
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
    const TEXT_DELTA: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
    const MESSAGE_DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#;

    fn sse(frames: &[&str]) -> bytes::Bytes {
        bytes::Bytes::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        )
    }

    /// The texts the stream yielded, whether an error item came, and what
    /// it finished with.
    async fn collect(
        sse_bytes: bytes::Bytes,
    ) -> (
        Vec<String>,
        bool,
        Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
    ) {
        let bound = crate::driver::Model::new(
            AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
            MockStreamingClient { sse_bytes },
        );
        let request = crate::completion::CompletionRequest::new("hello");
        let stream = bound.stream(request).expect("stream should open");
        drain(stream).await
    }

    async fn drain(
        mut stream: crate::streaming::CompletionStream,
    ) -> (
        Vec<String>,
        bool,
        Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
    ) {
        let mut texts = Vec::new();
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }
        (texts, saw_error, stream.finish().await)
    }

    #[tokio::test]
    async fn errored_stream_forwards_the_error_and_no_end() {
        use crate::test_utils::SequencedStreamingHttpClient;

        // A transport failure injected into the byte stream after some
        // content must be forwarded and must not be papered over with a
        // synthesized end.
        let bound = crate::driver::Model::new(
            AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
            SequencedStreamingHttpClient::new(vec![
                Ok(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])),
                Err(crate::http_client::Error::non_success_with_details(
                    http::StatusCode::BAD_GATEWAY,
                    http::HeaderMap::new(),
                    "connection reset".to_string(),
                )),
            ]),
        );
        let request = crate::completion::CompletionRequest::new("hello");
        let stream = bound.stream(request).expect("stream should open");
        let (texts, saw_error, finished) = drain(stream).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the transport failure must reach the consumer");
        assert!(finished.is_err(), "a failed stream has no response");
    }

    /// `input_tokens` precedence between `message_start` and the terminal
    /// `message_delta`, across all three wire splits at once.
    ///
    /// Not a cassette test: one recording can only witness whichever split
    /// the endpoint it was recorded against happens to use, and the defect
    /// here is the *precedence rule* relating three of them — the gateway
    /// split, Anthropic proper, and the inverse. The gateway split is also
    /// covered end-to-end by the recorded
    /// `anthropic::cassette::streaming::gateway_reports_input_tokens_on_message_delta`;
    /// this pins the two cases a single recording structurally cannot show
    /// beside it.
    #[tokio::test]
    async fn input_tokens_prefer_the_terminal_delta_and_fall_back_to_message_start() {
        fn message_start(input_tokens: usize) -> String {
            format!(
                r#"{{"type":"message_start","message":{{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{input_tokens},"output_tokens":0}}}}}}"#
            )
        }
        fn message_delta(input_tokens: usize) -> String {
            format!(
                r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"input_tokens":{input_tokens},"output_tokens":3}}}}"#
            )
        }

        for (start, delta, expected, case) in [
            // OpenRouter's Anthropic Messages shape: `message_start`
            // reports a placeholder zero and the real prompt size lands on
            // the terminal `message_delta`.
            (
                message_start(0),
                message_delta(9),
                9,
                "a gateway reporting the prompt size on message_delta must reach the consumer",
            ),
            // A delta that omits `input_tokens` entirely — the Bedrock-compat
            // and older/leaner shapes. (Not current Anthropic, which sends
            // the count on both frames; that case is the one below, since
            // the two always agree.)
            (
                message_start(5),
                MESSAGE_DELTA.to_owned(),
                5,
                "a delta without input_tokens falls back to message_start",
            ),
            // Anthropic proper: both frames carry the same count.
            (
                message_start(5),
                message_delta(5),
                5,
                "agreeing frames report that count",
            ),
            // The inverse split: a zero on the delta must not erase the
            // real count `message_start` already gave us.
            (
                message_start(5),
                message_delta(0),
                5,
                "a zero on the delta must not erase the message_start count",
            ),
        ] {
            let (_texts, _saw_error, finished) =
                collect(sse(&[&start, TEXT_START, TEXT_DELTA, &delta])).await;

            let response = finished.expect("the turn must complete");
            assert_eq!(response.usage.input_tokens, Some(expected), "{case}");
        }
    }
}

/// The Messages projection, driven through [`crate::driver`].
///
/// The projector rides on the wire's [`crate::wire::Encoded`], and the HTTP
/// transport calls it on every raw payload — the rejection body included — so
/// these cells exercise the wire and the driver together rather than the
/// projector in isolation. That is the only way the *closure* facts
/// (`Started`, `Response`, `Finished`) are observable at all: they belong to
/// the attempt, not to the payload.
mod projection {

    use std::sync::Arc;

    use crate::completion::CompletionRequest;
    use crate::observe::{
        Action, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
        AdapterEvent, ObservationLog, Subject,
    };
    use crate::providers::anthropic::wire::{AnthropicConfig, Messages};
    use crate::test_utils::RecordingHttpClient;

    fn adapter_events(log: &ObservationLog) -> Vec<AdapterEvent> {
        log.trace()
            .observations
            .iter()
            .filter_map(|o| match &o.action {
                Action::Adapter { observation } => Some(observation.event.clone()),
                _ => None,
            })
            .collect()
    }

    fn context(log: &Arc<ObservationLog>) -> AdapterContext {
        AdapterContext::new(log.clone(), Subject::default(), "call")
    }

    fn wire() -> Messages {
        AnthropicConfig::new("test-key").completion("claude-test")
    }

    fn request() -> CompletionRequest {
        CompletionRequest::new("hello").max_tokens(64)
    }

    /// A rejected Messages call: the envelope's type and message, and the
    /// closure with the one funnel's classification.
    #[tokio::test]
    async fn messages_rejection_projects_the_envelope() {
        let http = RecordingHttpClient::with_error(
            http::StatusCode::SERVICE_UNAVAILABLE,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        );
        let log = Arc::new(ObservationLog::default());
        let error = crate::driver::Model::new(wire(), http.clone())
            .call_observed(request(), context(&log))
            .await
            .expect_err("the transport rejects the call");
        assert!(error.is_retryable());
        let events = adapter_events(&log);
        assert!(
            matches!(&events[0], AdapterEvent::Started { method, route } if method == "POST" && route == "/v1/messages")
        );
        assert!(events.contains(&AdapterEvent::Response { status: 503 }));
        assert!(events.contains(&AdapterEvent::ErrorEnvelope {
            error: AdapterErrorEnvelope {
                code: None,
                status: Some("overloaded_error".into()),
                message: Some("Overloaded".into()),
            }
        }));
        assert_eq!(
            events.last(),
            Some(&AdapterEvent::Finished {
                ending: AdapterEnding::Error {
                    boundary: AdapterErrorBoundary::ProviderResponse,
                    kind: "provider_response".into(),
                    status: Some(503),
                    retryable: true,
                }
            })
        );
    }
}
/// One sample per known Messages event: a new event fails here until it
/// has a sample.
#[test]
fn every_messages_event_has_a_sample() {
    let samples: Vec<MessagesEvent> = [
        r#"{"type":"message_start","message":null}"#,
        r#"{"type":"message","id":"msg_1","role":"assistant","model":"m","content":[],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}"#,
        r#"{"type":"message_stop"}"#,
        r#"{"type":"ping"}"#,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    ]
    .into_iter()
    .map(classified)
    .collect();
    let index = |event: &MessagesEvent| {
        KNOWN_EVENT_TYPES
            .iter()
            .position(|kind| *kind == event.kind())
            .unwrap_or(usize::MAX)
    };
    crate::test_utils::history::assert_every_variant(&samples, index, KNOWN_EVENT_TYPES.len());
}

/// An item type rig has never seen, and a field it has never seen on a
/// known item, survive decoding whole and streamed, and go back verbatim
/// to the model that produced them.
#[test]
fn an_invented_item_and_field_survive_decode_and_same_model_replay() {
    use crate::wire::{Operation, Wire};

    let invented = json!({"type": "frobnicate", "payload": {"x": 1}});
    let text = json!({"type": "text", "text": "hi", "sparkle": true});
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let whole = json!({
        "type": "message", "id": "msg_1", "role": "assistant", "model": CLAUDE_SONNET_4_6,
        "content": [invented, text], "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 1}
    });
    let unary = crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Unary,
        [WireFrame::Text(whole.to_string())],
        whole.clone(),
    )
    .expect("the whole reply folds");
    let stream = streamed(&reply(
        vec![
            block(0, invented.clone(), &[]),
            block(
                1,
                json!({"type": "text", "text": "", "sparkle": true}),
                &[json!({"type": "text_delta", "text": "hi"})],
            ),
        ],
        "end_turn",
    ))
    .expect("the stream folds");
    assert_eq!(unary.message(), stream.message());

    for response in [unary, stream] {
        let turn = response.message().expect("an assistant turn");
        let request = CompletionRequest::from(vec![
            RigMessage::user("hello"),
            turn,
            RigMessage::user("again"),
        ]);
        let request = Completion::prepare(request, &wire.describe()).expect("the history adapts");
        let encoded = wire
            .encode(request, Mode::Unary)
            .expect("the request encodes");
        let body = crate::test_utils::json_body(&encoded.request);
        assert_eq!(body["messages"][1]["content"], json!([invented, text]));
    }
}

/// The response the whole message `message` folds into on the Anthropic
/// wire.
fn whole(message: Value) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Unary,
        [WireFrame::Text(message.to_string())],
        Value::Null,
    )
}

/// A reply with no content folds the same whole or streamed, whatever its
/// stop reason: pi accepts empty content, and the stop reason decides how
/// the turn ends.
#[test]
fn an_empty_reply_folds_the_same_whole_or_streamed() {
    for reason in [
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "model_context_window_exceeded",
        "tool_use",
        "pause_turn",
        "refusal",
        "sensitive",
    ] {
        let streamed = streamed(&reply(vec![], reason)).map(|response| response.stop());
        let whole = whole(json!({
            "type": "message", "id": "msg_1", "role": "assistant", "model": CLAUDE_SONNET_4_6,
            "content": [], "stop_reason": reason, "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 5}
        }))
        .map(|response| response.stop());
        assert!(streamed.is_ok(), "{reason}: {streamed:?}");
        assert_eq!(
            whole.as_ref().ok(),
            streamed.as_ref().ok(),
            "{reason}: {whole:?}"
        );
    }
}

/// Anthropic A6: a stop reason states every block still open complete, as
/// pi keeps a signature whatever `content_block_stop` did: thinking whose
/// stop never came keeps its signed item, and so does a call whose input is
/// whole. A call whose input is still cut stays open, and the end of the
/// reply closes it unfinished.
#[test]
fn a_stop_reason_states_every_open_block_complete() {
    let without_stops = |reason: &str, input: &str| {
        let mut frames = reply(Vec::new(), reason);
        let end = frames.pop().expect("the message_delta");
        frames.extend([
            json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan"}}),
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"}}),
            json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_a", "name": "f", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": input}}),
            end,
        ]);
        frames
    };
    let response = streamed(&without_stops("tool_use", "{\"x\":1}")).expect("the reply folds");
    assert_eq!(response.stop(), StopReason::ToolUse);
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "thinking", "thinking": "plan", "signature": "sig"}))
    );
    assert_eq!(
        item(&response.choice[1]),
        Some(&json!({"type": "tool_use", "id": "toolu_a", "name": "f", "input": {"x": 1}}))
    );

    let cut = streamed(&without_stops("max_tokens", "{\"x\":")).expect("the reply folds");
    assert_eq!(cut.stop(), StopReason::Length);
    assert!(item(&cut.choice[0]).is_some(), "{:?}", cut.choice);
    assert!(item(&cut.choice[1]).is_none(), "{:?}", cut.choice);
    let unfinished = streamed(&without_stops("tool_use", "{\"x\":")).expect("the reply folds");
    assert!(unfinished.stop().is_failure(), "{:?}", unfinished.stop());
}

/// A gateway that skips `content_block_start` (seen on Anthropic-format
/// dialects) still streams the block: its first delta opens it.
#[test]
fn a_delta_for_a_block_that_never_started_opens_it() {
    let frames = reply(
        vec![vec![
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "hi"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ]],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply decodes");
    assert!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hi"),
        "{:?}",
        response.choice
    );
}

// Progress-update thinking blocks, in the documented shape of a turn that
// answers a `tool_result` on Claude Opus 5.5: a reasoning block, a
// progress-update block, then the `tool_use` it introduces. Both thinking
// blocks are empty unless the display shows updates. No recording holds a
// progress-update block, so the shapes are built here.
// Source: <https://platform.claude.com/docs/en/build-with-claude/thinking#progress-updates>

const PROGRESS_UPDATE_TEXT: &str = "Confirmed the retry path never refreshes the expired token. Editing auth.py to add the refresh call.";

fn progress_update_content(update_text: &str) -> Value {
    json!([
        {"type": "thinking", "thinking": "", "signature": "EqMBCkYICxIM-reasoning"},
        {"type": "thinking", "thinking": update_text, "signature": "Es8CCkYICxIM-update"},
        {
            "type": "tool_use",
            "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "name": "edit_file",
            "input": {"path": "auth.py", "content": "..."}
        }
    ])
}

/// The turn decoded from the whole message or its stream on Opus 5.5.
fn progress_update_reply(update_text: &str, mode: Mode) -> crate::completion::CompletionResponse {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_5_5);
    let frames: Vec<Value> = match mode {
        Mode::Unary => vec![json!({
            "type": "message", "id": "msg_progress", "model": CLAUDE_OPUS_5_5,
            "role": "assistant", "content": progress_update_content(update_text),
            "stop_reason": "tool_use", "stop_sequence": null,
            "usage": {"input_tokens": 120, "output_tokens": 80}
        })],
        _ => {
            let thinking = |index, text: &str, signature: &str| {
                block(
                    index,
                    json!({"type": "thinking", "thinking": "", "signature": ""}),
                    &[
                        json!({"type": "thinking_delta", "thinking": text}),
                        json!({"type": "signature_delta", "signature": signature}),
                    ],
                )
            };
            std::iter::once(json!({"type": "message_start", "message": {
                "type": "message", "id": "msg_progress", "model": CLAUDE_OPUS_5_5,
                "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 120, "output_tokens": 1}
            }}))
            .chain(thinking(0, "", "EqMBCkYICxIM-reasoning"))
            .chain(thinking(1, update_text, "Es8CCkYICxIM-update"))
            .chain(block(
                2,
                json!({"type": "tool_use", "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                    "name": "edit_file", "input": {}}),
                &[json!({"type": "input_json_delta",
                    "partial_json": "{\"path\": \"auth.py\", \"content\": \"...\"}"})],
            ))
            .chain([
                json!({"type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                    "usage": {"output_tokens": 80}}),
                json!({"type": "message_stop"}),
            ])
            .collect()
        }
    };
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        mode,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
    .expect("the reply folds")
}

/// Ported from #2616: both thinking blocks survive decoding unmerged and in
/// order, whole and streamed, under either display, and the turn replays to
/// its model exactly as received.
#[test]
fn progress_update_blocks_survive_decoding_and_replay_under_every_display() {
    use crate::wire::{Operation, Wire};

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_5_5);
    for update_text in ["", PROGRESS_UPDATE_TEXT] {
        for mode in [Mode::Unary, Mode::Streaming] {
            let response = progress_update_reply(update_text, mode);
            let items: Vec<Value> = response.choice.iter().filter_map(item).cloned().collect();
            assert_eq!(
                Value::Array(items),
                progress_update_content(update_text),
                "{mode:?}: each block decodes as received"
            );
            let turn = response.message().expect("an assistant turn");
            let mut request = CompletionRequest::from(vec![
                RigMessage::user("The login test fails after an hour of uptime."),
                turn,
            ]);
            request.chat_history.push(RigMessage::tool_result(
                crate::message::CallId::from_wire("toolu_01D7FLrfh4GYq7yT1ULFeyMV"),
                crate::message::ToolName::new("edit_file").expect("a tool name"),
                "saved",
            ));
            request.tools = vec![crate::completion::ToolDefinition {
                name: crate::message::ToolName::new("edit_file").expect("a tool name"),
                description: "Edit a file.".to_owned(),
                parameters: json!({"type": "object", "properties": {}}),
            }];
            let request =
                Completion::prepare(request, &wire.describe()).expect("the history adapts");
            let encoded = wire
                .encode(request, Mode::Unary)
                .expect("the request encodes");
            let body = crate::test_utils::json_body(&encoded.request);
            assert_eq!(
                body["messages"][1]["content"],
                progress_update_content(update_text),
                "the {mode:?} turn replays exactly as received"
            );
        }
    }
}
