use super::*;
use crate::completion::{AMAZON_NOVA_LITE, ANTHROPIC_CLAUDE_SONNET_4_6, Converse};
use rig_core::completion::{CompletionResponse, Message};
use rig_core::message::{AssistantContent, Opaque, StopReason};
use rig_core::test_utils::history::{assert_restated_agrees, decode};
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

/// The events of a whole reply holding `content`, as the decoder restates it.
pub(crate) fn events(content: &[Value], stop_reason: &str) -> Vec<Value> {
    let mut events = vec![json!({ "messageStart": { "role": "assistant" } })];
    for (index, block) in content.iter().enumerate() {
        events.extend(restated(index, block));
        events.push(stop(index));
    }
    events.extend(ended(stop_reason));
    events
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

/// Each kept block holds its Converse JSON as Bedrock sent it, and a whole
/// reply and its restatement as a stream fold into the same turn.
#[test]
fn kept_blocks_hold_their_converse_json_in_both_modes() {
    let content = rich();
    let response = whole(CLAUDE, content.clone(), "tool_use");
    let items: Vec<Option<Value>> = response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Opaque(opaque) => Some(opaque.item.clone()),
            block => block.native_item().cloned(),
        })
        .collect();
    let mut expected: Vec<Option<Value>> = content.iter().cloned().map(Some).collect();
    expected[5] = None;
    assert_eq!(items, expected);
    assert_restated_agrees(
        &Converse::new(CLAUDE),
        [ConverseFrame::Whole(document(content.clone(), "tool_use"))],
        events(&content, "tool_use")
            .into_iter()
            .map(ConverseFrame::Event),
    );
}

/// A hosted tool's use is an opaque item that replays, not a call Rig runs.
#[test]
fn a_hosted_tool_use_is_not_a_client_call() {
    let response = whole(NOVA, hosted().to_vec(), "end_turn");
    assert_eq!(response.tool_calls().count(), 0);
    assert!(
        response
            .choice
            .iter()
            .all(|block| matches!(block, AssistantContent::Opaque(Opaque { replay: true, .. })))
    );
}

/// A block keeps its item only once it stopped: the end closes the rest
/// with none.
#[test]
fn a_block_keeps_its_item_only_at_its_stop() {
    let events = vec![
        delta(0, json!({ "reasoningContent": { "text": "half" } })),
        delta(0, json!({ "reasoningContent": { "signature": "sig" } })),
        ended("max_tokens")[0].clone(),
        ended("max_tokens")[1].clone(),
    ];
    let response = streamed(CLAUDE, events).expect("decodes");
    assert!(response.choice[0].native_item().is_none(), "{response:?}");
    assert_eq!(response.stop(), StopReason::Length);
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

/// Streamed citations make a cited block holding the whole text.
#[test]
fn streamed_citations_are_kept() {
    let mut events = vec![
        delta(0, json!({ "text": "The harbor " })),
        delta(0, json!({ "citation": { "title": "hours" } })),
        delta(0, json!({ "text": "opens at nine." })),
        stop(0),
    ];
    events.extend(ended("end_turn"));
    let response = streamed(NOVA, events).expect("decodes");
    assert_eq!(response.text(), "The harbor opens at nine.");
    assert_eq!(
        response.choice[0].native_item(),
        Some(&json!({ "citationsContent": {
            "content": [{ "text": "The harbor opens at nine." }],
            "citations": [{ "title": "hours" }],
        } }))
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

/// Input that is not a JSON object keeps the call, with its text, and no
/// item: the call is rebuilt from its arguments.
#[test]
fn malformed_input_keeps_the_call_and_no_item() {
    let mut events = vec![
        start(
            0,
            json!({ "toolUse": { "toolUseId": "t1", "name": "add" } }),
        ),
        delta(0, json!({ "toolUse": { "input": "{\"x\": tru" } })),
        stop(0),
    ];
    events.extend(ended("tool_use"));
    let response = streamed(CLAUDE, events).expect("decodes");
    let call = response.tool_calls().next().expect("a call");
    assert!(call.function.invalid_arguments.is_some());
    assert!(response.choice[0].native_item().is_none());
    // A whole reply whose input is `null` sends `{}` back.
    let response = whole(NOVA, vec![tool_use("t2", "add", Value::Null)], "tool_use");
    assert_eq!(
        response
            .tool_calls()
            .next()
            .map(|call| call.function.arguments_value()),
        Some(json!({}))
    );
    assert!(response.choice[0].native_item().is_none());
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

/// Every stop reason Converse documents ends a turn as documented; a
/// stream that ends at its metadata without `messageStop` names no reason
/// and fails, as pi's "Bedrock stream ended without a stop reason" does.
#[test]
fn every_stop_reason_maps() {
    for (reason, stop) in [
        ("end_turn", StopReason::Stop),
        ("stop_sequence", StopReason::Stop),
        ("tool_use", StopReason::ToolUse),
        ("max_tokens", StopReason::Length),
        ("model_context_window_exceeded", StopReason::Length),
    ] {
        let response = whole(NOVA, vec![json!({ "text": "x" })], reason);
        assert_eq!(response.stop(), stop, "{reason}");
    }
    for reason in [
        "guardrail_intervened",
        "content_filtered",
        "malformed_model_output",
        "malformed_tool_use",
        "x_rig_invented",
    ] {
        let response = whole(NOVA, vec![json!({ "text": "x" })], reason);
        assert!(response.stop().is_failure(), "{reason}");
    }
    let response = streamed(
        NOVA,
        vec![
            delta(0, json!({ "text": "half an ans" })),
            ended("end_turn")[1].clone(),
        ],
    )
    .expect("decodes");
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

/// A stream's `raw` is its message-level events as Bedrock sent them.
#[test]
fn a_streams_raw_is_bedrocks_json() {
    let events = vec![
        json!({ "messageStart": { "role": "assistant" } }),
        delta(0, json!({ "text": "hi" })),
        stop(0),
        json!({ "messageStop": { "stopReason": "end_turn", "additionalModelResponseFields": { "x": 1 } } }),
        json!({ "metadata": { "usage": usage(), "trace": { "promptRouter": { "invokedModelId": NOVA } } } }),
    ];
    let response = streamed(NOVA, events.clone()).expect("decodes");
    assert_eq!(
        response.raw,
        json!({
            "messageStart": events[0]["messageStart"],
            "messageStop": events[3]["messageStop"],
            "metadata": events[4]["metadata"],
        })
    );
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

/// A field of an unexpected type never fails a reply, and an invented block
/// replays to the model that sent it.
#[test]
fn a_reply_reads_leniently() {
    let response = whole(
        CLAUDE,
        vec![
            json!({ "text": 7 }),
            json!({ "reasoningContent": { "reasoningText": { "text": "t", "signature": 7 } } }),
            json!({ "x_rig_invented": { "a": 1 } }),
        ],
        "end_turn",
    );
    assert_eq!(response.choice.len(), 2, "{:?}", response.choice);
    assert!(response.choice[0].native_item().is_none());
    assert_eq!(
        response.choice[1],
        AssistantContent::Opaque(Opaque {
            item: json!({ "x_rig_invented": { "a": 1 } }),
            replay: true
        })
    );
    let turn = response.message();
    assert!(matches!(turn, Some(Message::Assistant(_))));
}
