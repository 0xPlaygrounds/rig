use super::*;
use crate::completion::{AMAZON_NOVA_LITE, ANTHROPIC_CLAUDE_SONNET_4_6, Converse};
use rig_core::completion::CompletionResponse;
use rig_core::message::{AssistantContent, Opaque, StopReason};
use rig_core::test_utils::history::decode;
use rig_core::wire::Mode;

pub(crate) const CLAUDE: &str = ANTHROPIC_CLAUDE_SONNET_4_6;
pub(crate) const NOVA: &str = AMAZON_NOVA_LITE;

pub(crate) fn usage() -> Value {
    json!({ "inputTokens": 3, "outputTokens": 1, "totalTokens": 4 })
}

/// A whole reply holding `content`.
pub(crate) fn document(content: Vec<Value>, stop_reason: &str) -> Value {
    json!({
        "output": { "message": { "role": "assistant", "content": content } },
        "stopReason": stop_reason,
        "usage": usage(),
        "metrics": { "latencyMs": 5 },
    })
}

pub(crate) fn delta(index: usize, delta: Value) -> Value {
    json!({ "contentBlockDelta": { "contentBlockIndex": index, "delta": delta } })
}

pub(crate) fn start(index: usize, start: Value) -> Value {
    json!({ "contentBlockStart": { "contentBlockIndex": index, "start": start } })
}

pub(crate) fn stop(index: usize) -> Value {
    json!({ "contentBlockStop": { "contentBlockIndex": index } })
}

pub(crate) fn ended(stop_reason: &str) -> [Value; 2] {
    [
        json!({ "messageStop": { "stopReason": stop_reason } }),
        json!({ "metadata": { "usage": usage(), "metrics": { "latencyMs": 5 } } }),
    ]
}

pub(crate) fn reasoning(text: &str, signature: Option<&str>) -> Value {
    let mut reasoning = json!({ "text": text });
    if let Some(signature) = signature {
        reasoning["signature"] = json!(signature);
    }
    json!({ "reasoningContent": { "reasoningText": reasoning } })
}

pub(crate) fn tool_use(id: &str, name: &str, input: Value) -> Value {
    json!({ "toolUse": { "toolUseId": id, "name": name, "input": input } })
}

pub(crate) fn hosted() -> [Value; 2] {
    [
        json!({ "toolUse": {
            "toolUseId": "srv_1", "name": "nova_grounding", "input": { "q": "harbor" }, "type": "server_tool_use",
        } }),
        json!({ "toolResult": { "toolUseId": "srv_1", "content": [{ "text": "nine" }] } }),
    ]
}

/// The response a whole reply holding `content` decodes to on `model`.
pub(crate) fn whole(model: &str, content: Vec<Value>, stop_reason: &str) -> CompletionResponse {
    let frames = [ConverseFrame::Whole(document(content, stop_reason))];
    decode(&Converse::new(model), Mode::Unary, frames).expect("the whole reply decodes")
}

/// The response a stream of `events` decodes to on `model`.
pub(crate) fn streamed(
    model: &str,
    events: Vec<Value>,
) -> Result<CompletionResponse, ProviderError> {
    let frames = events.into_iter().map(ConverseFrame::Event);
    decode(&Converse::new(model), Mode::Streaming, frames)
}

/// A reply holding every block kind a turn keeps an item for.
pub(crate) fn rich() -> Vec<Value> {
    let [used, result] = hosted();
    vec![
        reasoning("let me think", Some("sig-abc")),
        json!({ "reasoningContent": { "redactedContent": "AGNpcGhlcnRleHT/" } }),
        json!({ "citationsContent": {
            "content": [{ "text": "The harbor opens at nine." }],
            "citations": [{ "title": "hours", "location": { "documentChar": { "documentIndex": 0, "start": 0, "end": 24 } } }],
        } }),
        used,
        result,
        json!({ "text": "Calling." }),
        tool_use("tooluse_1", "lookup", json!({ "q": "harbor" })),
    ]
}

/// Streamed signed thinking keeps its text and signature, and signature-only
/// thinking keeps its item; empty thinking is no content; unsigned
/// reasoning keeps no item, since it is rebuilt from its text.
#[test]
fn reasoning_keeps_an_item_only_when_signed_or_redacted() {
    let thought = |index, field: &str, text: &str| {
        delta(index, json!({ "reasoningContent": { field: text } }))
    };
    let mut events = vec![
        thought(0, "text", "let me "),
        thought(0, "text", "think"),
        thought(0, "signature", "sig-"),
        thought(0, "signature", "1"),
        stop(0),
        thought(1, "signature", "only"),
        stop(1),
        thought(2, "text", ""),
        stop(2),
        thought(3, "text", "unsigned"),
        stop(3),
    ];
    events.extend(ended("end_turn"));
    let response = streamed(CLAUDE, events).expect("decodes");
    let items: Vec<_> = response
        .choice
        .iter()
        .map(|block| block.native_item().cloned())
        .collect();
    assert_eq!(
        items,
        [
            Some(reasoning("let me think", Some("sig-1"))),
            Some(reasoning("", Some("only"))),
            None,
        ]
    );
    assert_eq!(response.choice[2], AssistantContent::reasoning("unsigned"));
}

/// Redacted reasoning arrives in base64 chunks whose bytes join before they
/// are encoded once.
#[test]
fn redacted_reasoning_encodes_its_bytes_once() {
    let chunk = |bytes: &[u8]| {
        delta(
            0,
            json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(bytes) } }),
        )
    };
    let mut events = vec![chunk(b"\x00c"), chunk(b"iph"), chunk(b"er\xff"), stop(0)];
    events.extend(ended("end_turn"));
    let response = streamed(CLAUDE, events).expect("decodes");
    assert_eq!(
        response.choice[0].native_item(),
        Some(
            &json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(b"\x00cipher\xff") } })
        )
    );
}

/// #2250, #989: streamed client calls are each kept, one with no input at
/// all, and each keeps its `toolUse` with the input its arguments were
/// read from. A call that never names its tool is dropped.
#[test]
fn streamed_calls_are_all_kept_with_their_items() {
    let mut events = vec![
        delta(0, json!({ "text": "calling" })),
        stop(0),
        start(
            1,
            json!({ "toolUse": { "toolUseId": "t1", "name": "add" } }),
        ),
        delta(1, json!({ "toolUse": { "input": "{\"x\":" } })),
        delta(1, json!({ "toolUse": { "input": "1}" } })),
        stop(1),
        start(
            2,
            json!({ "toolUse": { "toolUseId": "t2", "name": "now" } }),
        ),
        stop(2),
        delta(3, json!({ "toolUse": { "input": "{}" } })),
        stop(3),
    ];
    events.extend(ended("tool_use"));
    let response = streamed(CLAUDE, events).expect("decodes");
    let calls: Vec<_> = response
        .tool_calls()
        .map(|call| (call.id.wire().into_owned(), call.function.arguments_value()))
        .collect();
    assert_eq!(
        calls,
        [
            ("t1".to_owned(), json!({ "x": 1 })),
            ("t2".to_owned(), json!({}))
        ]
    );
    assert_eq!(
        response.choice[1].native_item(),
        Some(&tool_use("t1", "add", json!({ "x": 1 })))
    );
    assert_eq!(response.stop(), StopReason::ToolUse);
}

/// An image arrives with its start and its bytes and is written whole at
/// its stop; one Converse sent no bytes for does not replay.
#[test]
fn image_and_tool_result_deltas_assemble_at_their_stop() {
    let mut events = vec![
        start(0, json!({ "image": { "format": "png" } })),
        delta(0, json!({ "image": { "source": { "bytes": "cG5n" } } })),
        stop(0),
        start(
            1,
            json!({ "toolResult": { "toolUseId": "srv_1", "status": "success" } }),
        ),
        delta(1, json!({ "toolResult": [{ "text": "nine" }] })),
        delta(1, json!({ "toolResult": [{ "json": { "opens": 9 } }] })),
        stop(1),
        start(2, json!({ "image": { "format": "heic" } })),
        stop(2),
    ];
    events.extend(ended("end_turn"));
    let response = streamed(NOVA, events).expect("decodes");
    assert!(matches!(&response.choice[0], AssistantContent::Image(image)
        if image.data == DocumentSourceKind::Base64("cG5n".to_owned())));
    assert_eq!(
        response.choice[1],
        AssistantContent::Opaque(Opaque {
            item: json!({ "toolResult": {
                "toolUseId": "srv_1", "status": "success",
                "content": [{ "text": "nine" }, { "json": { "opens": 9 } }],
            } }),
            replay: true,
        })
    );
    assert!(matches!(
        &response.choice[2],
        AssistantContent::Opaque(Opaque { replay: false, .. })
    ));
}

/// A stream that ends at its metadata without `messageStop` names no
/// reason, so it fails, as pi's "Bedrock stream ended without a stop
/// reason" does.
#[test]
fn a_stream_without_a_stop_reason_fails() {
    let events = vec![
        delta(0, json!({ "text": "half an ans" })),
        ended("end_turn")[1].clone(),
    ];
    let response = streamed(NOVA, events).expect("decodes");
    assert!(response.stop().is_failure(), "{:?}", response.stop());
}

/// An exception Bedrock sends mid-stream fails the reply with its type as
/// the code.
#[test]
fn an_in_band_exception_fails_the_stream() {
    let events = vec![
        delta(0, json!({ "text": "par" })),
        json!({ "throttlingException": { "message": "slow down" } }),
    ];
    let error = streamed(NOVA, events).expect_err("the stream fails");
    assert_eq!(error.report().code.as_deref(), Some("ThrottlingException"));
    assert!(error.is_retryable());
}

/// The document the Converse reassembler rebuilds from `events`.
fn rebuilt(events: &[Value]) -> Value {
    let mut document = document::ConverseOutput::default();
    for event in events {
        rig_core::wire::document::Reassemble::absorb(
            &mut document,
            &ConverseFrame::Event(event.clone()),
        );
    }
    rig_core::wire::document::Reassemble::finish(document)
}

/// `event` with the padding ConverseStream adds to every event.
fn padded(mut event: Value) -> Value {
    if let Some(payload) = event
        .as_object_mut()
        .and_then(|event| event.values_mut().next())
        .and_then(Value::as_object_mut)
    {
        payload.insert("p".to_owned(), json!("abcdefgh"));
    }
    event
}

/// A stream rebuilds the `ConverseOutput` a unary call returns, block by
/// block: signed and redacted reasoning, cited text, a hosted tool's use
/// and result, text and a client call. The message-level fields land where
/// the unary reply has them, and the stream's padding is dropped.
#[test]
fn a_stream_rebuilds_the_unary_converse_output() {
    let thought =
        |field: &str, text: &str| delta(0, json!({ "reasoningContent": { field: text } }));
    let redacted = BASE64_STANDARD
        .decode("AGNpcGhlcnRleHT/")
        .expect("the fixture is base64");
    let (head, tail) = redacted.split_at(4);
    let chunk = |bytes: &[u8]| {
        delta(
            1,
            json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(bytes) } }),
        )
    };
    let events: Vec<Value> = [
        json!({ "messageStart": { "role": "assistant" } }),
        thought("text", "let me "),
        thought("text", "think"),
        thought("signature", "sig-abc"),
        stop(0),
        chunk(head),
        chunk(tail),
        stop(1),
        delta(2, json!({ "text": "The harbor opens at nine." })),
        delta(2, json!({ "citation": { "title": "hours", "location": { "documentChar": { "documentIndex": 0, "start": 0, "end": 24 } } } })),
        stop(2),
        start(3, json!({ "toolUse": { "toolUseId": "srv_1", "name": "nova_grounding", "type": "server_tool_use" } })),
        delta(3, json!({ "toolUse": { "input": "{\"q\": \"harbor\"}" } })),
        stop(3),
        start(4, json!({ "toolResult": { "toolUseId": "srv_1" } })),
        delta(4, json!({ "toolResult": [{ "text": "nine" }] })),
        stop(4),
        delta(5, json!({ "text": "Calling." })),
        stop(5),
        start(6, json!({ "toolUse": { "toolUseId": "tooluse_1", "name": "lookup" } })),
        delta(6, json!({ "toolUse": { "input": "{\"q\":" } })),
        delta(6, json!({ "toolUse": { "input": " \"harbor\"}" } })),
        stop(6),
        json!({ "messageStop": { "stopReason": "tool_use" } }),
        json!({ "metadata": { "usage": usage(), "metrics": { "latencyMs": 5 } } }),
    ]
    .into_iter()
    .map(padded)
    .collect();
    let unary = document(rich(), "tool_use");
    assert_eq!(rebuilt(&events), unary);

    let response = streamed(CLAUDE, events).expect("decodes");
    assert_eq!(response.raw, unary);
    assert_eq!(response.raw, whole(CLAUDE, rich(), "tool_use").raw);
}

/// The fields only `messageStop` and `metadata` carry land at the top
/// level, as the unary reply states them, and a block that opens with a
/// delta needs no start.
#[test]
fn message_level_fields_land_where_the_unary_reply_has_them() {
    let events = vec![
        json!({ "messageStart": { "role": "assistant", "p": "ab" } }),
        delta(0, json!({ "text": "hi" })),
        stop(0),
        json!({ "messageStop": { "stopReason": "end_turn", "additionalModelResponseFields": { "x": 1 }, "p": "abc" } }),
        json!({ "metadata": {
            "usage": usage(),
            "metrics": { "latencyMs": 5 },
            "trace": { "promptRouter": { "invokedModelId": NOVA } },
            "performanceConfig": { "latency": "optimized" },
            "serviceTier": { "type": "priority" },
            "p": "abcd",
        } }),
    ];
    let response = streamed(NOVA, events).expect("decodes");
    assert_eq!(
        response.raw,
        json!({
            "output": { "message": { "role": "assistant", "content": [{ "text": "hi" }] } },
            "stopReason": "end_turn",
            "additionalModelResponseFields": { "x": 1 },
            "usage": usage(),
            "metrics": { "latencyMs": 5 },
            "trace": { "promptRouter": { "invokedModelId": NOVA } },
            "performanceConfig": { "latency": "optimized" },
            "serviceTier": { "type": "priority" },
        })
    );
}

/// An image's chunks join as bytes, a tool result's content collects, a
/// call with no input has an empty object, and a stream cut by an
/// exception rebuilds what arrived.
#[test]
fn images_results_and_cut_streams_rebuild_what_arrived() {
    let events = vec![
        start(0, json!({ "image": { "format": "png" } })),
        delta(
            0,
            json!({ "image": { "source": { "bytes": BASE64_STANDARD.encode(b"pn") } } }),
        ),
        delta(
            0,
            json!({ "image": { "source": { "bytes": BASE64_STANDARD.encode(b"g") } } }),
        ),
        start(
            1,
            json!({ "toolResult": { "toolUseId": "srv_1", "status": "success" } }),
        ),
        delta(1, json!({ "toolResult": [{ "text": "nine" }] })),
        delta(1, json!({ "toolResult": [{ "json": { "opens": 9 } }] })),
        start(
            2,
            json!({ "toolUse": { "toolUseId": "t2", "name": "now" } }),
        ),
        json!({ "throttlingException": { "message": "slow down" } }),
    ];
    assert_eq!(
        rebuilt(&events),
        json!({ "output": { "message": { "content": [
            { "image": { "format": "png", "source": { "bytes": BASE64_STANDARD.encode(b"png") } } },
            { "toolResult": { "toolUseId": "srv_1", "status": "success",
                "content": [{ "text": "nine" }, { "json": { "opens": 9 } }] } },
            { "toolUse": { "toolUseId": "t2", "name": "now", "input": {} } },
        ] } } })
    );
    assert_eq!(rebuilt(&[]), Value::Null);
}

/// Rig's input counts Bedrock's cache reads and writes, and its total is
/// input plus output.
#[test]
fn usage_counts_cache_reads_and_writes_in_input() {
    let counted = super::usage(&json!({
        "inputTokens": 10, "outputTokens": 5, "totalTokens": 15,
        "cacheReadInputTokens": 100, "cacheWriteInputTokens": 7,
    }));
    assert_eq!(counted.input_tokens, Some(117));
    assert_eq!(counted.output_tokens, Some(5));
    assert_eq!(counted.total_tokens, Some(122));
    assert_eq!(counted.cached_input_tokens, Some(100));
    assert_eq!(counted.cache_creation_input_tokens, Some(7));
}
