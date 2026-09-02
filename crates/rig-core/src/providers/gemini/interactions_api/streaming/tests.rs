use super::*;
use serde_json::json;

#[test]
fn test_streaming_completion_response_has_model_version() {
    let response = StreamingCompletionResponse {
        usage: None,
        interaction: None,
        model_version: Some("gemini-2.5-pro-preview-05-06".to_string()),
    };

    assert_eq!(
        response.model_version.as_deref(),
        Some("gemini-2.5-pro-preview-05-06")
    );

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: StreamingCompletionResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(
        deserialized.model_version.as_deref(),
        Some("gemini-2.5-pro-preview-05-06")
    );
}

#[test]
fn test_content_delta_text_event() {
    let event_json = json!({
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "type": "text",
            "text": "Hello"
        }
    });

    let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
    let InteractionSseEvent::StepDelta { delta, .. } = event else {
        panic!("expected step delta");
    };

    let choice = content_delta_to_choice(delta, &mut streaming::SyntheticIds::tool())
        .expect("choice should exist");
    match choice {
        crate::streaming::RawStreamingChoice::Message(text) => {
            assert_eq!(text, "Hello");
        }
        other => panic!("unexpected choice: {other:?}"),
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::gemini::Client;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    // Content deltas then EOF without `interaction.completed`: the
    // truncated stream must deliver its content but never a synthesized
    // terminal record.
    let sse_bytes = bytes::Bytes::from(
        [r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    );

    let client = Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient { sse_bytes })
        .build()
        .expect("build client")
        .interactions_api();
    let model = client.completion_model("gemini-2.5-pro");
    let request = model.completion_request("hello").build();
    let mut stream = crate::completion::CompletionModel::stream(&model, request)
        .await
        .expect("stream should open");

    let mut texts = Vec::new();
    let mut saw_terminal = false;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamedAssistantContent::Text(text) => texts.push(text.text),
            StreamedAssistantContent::Final(_) => saw_terminal = true,
            _ => {}
        }
    }

    assert_eq!(texts, ["hi"]);
    assert!(
        !saw_terminal,
        "EOF without interaction.completed must not synthesize a terminal record"
    );
    assert!(stream.response.is_none());
}

/// Drive Interactions SSE frames through the full normalized path and
/// collect what the consumer sees, in order.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
async fn drive_frames(
    frames: &[&str],
) -> (
    Vec<Result<crate::streaming::StreamedAssistantContent, String>>,
    crate::streaming::StreamingCompletionResponse,
) {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::gemini::Client;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let sse_bytes = bytes::Bytes::from(
        frames
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    );
    let client = Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient { sse_bytes })
        .build()
        .expect("build client")
        .interactions_api();
    let model = client.completion_model("gemini-2.5-pro");
    let request = model.completion_request("hello").build();
    let mut stream = crate::completion::CompletionModel::stream(&model, request)
        .await
        .expect("stream should open");

    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.map_err(|error| error.to_string()));
    }
    (items, stream)
}

/// A `model_output` step interleaving text and a function call in one
/// step's `content`: every convertible item must surface, in wire
/// order. `find_map` kept only the first — a `function_call` following
/// text in the same step silently vanished.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_model_output_step_yields_every_convertible_item() {
    use crate::streaming::StreamedAssistantContent;

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":0,"step":{"type":"model_output","content":[{"type":"text","text":"answer: "},{"type":"function_call","name":"add","arguments":{"x":1},"id":"fc_9"}]}}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let mut texts = Vec::new();
    let mut calls = Vec::new();
    for item in &items {
        match item {
            Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text.clone()),
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                calls.push(tool_call.clone());
            }
            _ => {}
        }
    }
    assert_eq!(texts, ["answer: "], "the text survives, got {items:?}");
    assert_eq!(
        calls.len(),
        1,
        "the function_call after text must also survive, got {items:?}"
    );
    let call = calls.first().expect("one call");
    assert_eq!(call.function.name, "add");
    assert_eq!(call.function.arguments, serde_json::json!({"x": 1}));
}

/// A `step.start` that announces non-empty arguments AND fragments the
/// real payload across `arguments_delta` events: the deltas are the
/// arguments. Concatenating the announce payload with the fragments
/// yields `{..}{..}` — unparseable under the step's Error policy, so
/// the call was lost outright.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn announce_arguments_never_concatenate_with_fragments() {
    use crate::streaming::StreamedAssistantContent;

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{"x":1},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"x\":1}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_calls.len(),
        1,
        "the announced-then-fragmented call must survive, got {items:?}"
    );
    assert_eq!(
        tool_calls.first().expect("one call").function.arguments,
        serde_json::json!({"x": 1}),
        "streamed fragments are the arguments; the announce payload is not prepended"
    );
}

/// A partial announce with NO fragments: the announce payload is the
/// only arguments the wire sent, so it finalizes the call
/// (replace-if-no-deltas).
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn announce_arguments_finalize_a_call_with_no_fragments() {
    use crate::streaming::StreamedAssistantContent;

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{"x":7},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(tool_calls.len(), 1, "got {items:?}");
    assert_eq!(
        tool_calls.first().expect("one call").function.arguments,
        serde_json::json!({"x": 7})
    );
}

/// Interactions is a single-identifier wire: its `fc_…` id must land
/// in `provider.call_id` with `item_id` empty. Filling both slots
/// fabricated a Responses-shaped dual identity whose fake item id
/// passed the foreign-id guard on cross-provider replay.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_streamed_call_carries_a_single_wire_identity() {
    use crate::streaming::StreamedAssistantContent;

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"x\":1}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => Some(tool_call),
            _ => None,
        })
        .collect();
    let provider = tool_calls
        .first()
        .expect("one call")
        .provider
        .as_ref()
        .expect("the wire issued an id");
    assert_eq!(provider.call_id, "fc_1");
    assert_eq!(
        provider.item_id, None,
        "a single-identifier wire must not fabricate a dual identity"
    );
}

/// A `step.stop` that never arrives must not lose the call: the wire
/// announced it (`step.start`), streamed its full arguments
/// (`arguments_delta`), and proved the turn finished
/// (`interaction.completed`). Before this fix the assembly stayed open,
/// `finish` never ran (terminal return), and the accumulator's
/// end-of-stream clear dropped the whole call — the agent then treated
/// a tool-calling turn as plain text.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_missing_step_stop_does_not_lose_the_announced_call() {
    use crate::streaming::StreamedAssistantContent;

    let (items, stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{},"id":"fc_1","name":"get_weather","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"city\":\"Paris\"}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_calls.len(),
        1,
        "the announced call must survive the missing step.stop, got {items:?}"
    );
    let tool_call = tool_calls.first().expect("one call");
    assert_eq!(tool_call.function.name, "get_weather");
    assert_eq!(
        tool_call.function.arguments,
        serde_json::json!({"city": "Paris"}),
        "the streamed argument fragments finalize the call"
    );
    assert_eq!(tool_call.id, "fc_1");

    // The turn completed normally: the terminal record survives too.
    assert!(stream.response.is_some());
    let aggregated_calls = stream
        .choice
        .iter()
        .filter(|content| matches!(content, crate::message::AssistantContent::ToolCall(_)))
        .count();
    assert_eq!(
        aggregated_calls, 1,
        "the call reaches the aggregated choice"
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn provider_error_event_ends_the_stream_without_draining_later_frames() {
    use crate::streaming::StreamedAssistantContent;

    // A provider `error` event, then more frames: well-formed content, an
    // unknown frame, and a terminal `interaction.completed`. The error
    // must be the LAST item — the driver stops reading (`is_finished`),
    // so nothing after it is interpreted or passed through as `Unknown`.
    let (items, stream) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#,
        r#"{"event_type":"error","error":{"code":"internal","message":"boom"}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"dead"}}"#,
        r#"{"event_type":"something.future","payload":{"x":1}}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let error_position = items
        .iter()
        .position(|item| item.is_err())
        .expect("the provider error must reach the consumer");
    assert_eq!(
        error_position,
        items.len() - 1,
        "the in-band error must end the stream: no later text, Unknown passthrough, or terminal; got {items:?}"
    );
    assert!(
        items.iter().any(|item| matches!(
            item,
            Ok(StreamedAssistantContent::Text(text)) if text.text == "hi"
        )),
        "content before the error must survive"
    );
    assert!(stream.response.is_none());
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn thought_signature_completes_the_accumulated_reasoning_block() {
    use crate::streaming::StreamedAssistantContent;

    // Text-then-signature: the signed block must restate the full
    // accumulated thought text and carry the signature; the aggregated
    // choice keeps it (superseding the deltas), alongside the later text.
    let (items, stream) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think1 "}}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think2"}}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-abc"}}"#,
        r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
    ])
    .await;

    let signed = items
        .iter()
        .find_map(|item| match item {
            Ok(StreamedAssistantContent::Reasoning { reasoning, .. }) => Some(reasoning.clone()),
            _ => None,
        })
        .expect("the signature must yield a completed Reasoning block");
    assert_eq!(
        signed.content,
        vec![crate::completion::message::ReasoningContent::Text {
            text: "think1 think2".to_string(),
            signature: Some("sig-abc".to_string()),
        }],
        "the signed block must restate the accumulated text with the signature"
    );

    // The aggregated choice keeps exactly one reasoning part carrying the
    // signature — the signed restatement superseded the deltas.
    let aggregated: Vec<_> = stream
        .choice
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(aggregated.len(), 1, "got {:?}", stream.choice);
    assert_eq!(
        aggregated.first().map(|r| r.content.clone()),
        Some(signed.content)
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn signature_only_thought_still_carries_the_signature() {
    use crate::streaming::StreamedAssistantContent;

    // Signature with no preceding thought-summary text: the signature is
    // the provider's replay-validated payload and must still survive as a
    // signed (empty-text) Reasoning block.
    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-only"}}"#,
        r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
    ])
    .await;

    let signed = items
        .iter()
        .find_map(|item| match item {
            Ok(StreamedAssistantContent::Reasoning { reasoning, .. }) => Some(reasoning.clone()),
            _ => None,
        })
        .expect("a signature-only block must still yield a signed Reasoning");
    assert_eq!(
        signed.content,
        vec![crate::completion::message::ReasoningContent::Text {
            text: String::new(),
            signature: Some("sig-only".to_string()),
        }]
    );
}

#[test]
fn test_content_delta_function_call_event() {
    let event_json = json!({
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "type": "function_call",
            "name": "get_weather",
            "arguments": {"location": "Paris"},
            "id": "call-1"
        }
    });

    let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
    let InteractionSseEvent::StepDelta { delta, .. } = event else {
        panic!("expected step delta");
    };

    let choice = content_delta_to_choice(delta, &mut streaming::SyntheticIds::tool())
        .expect("choice should exist");
    match choice {
        crate::streaming::RawStreamingChoice::ToolCall(call) => {
            assert_eq!(call.name, "get_weather");
            // Single-identifier wire: the id travels as `tool_id` only.
            // Filling `call_id` too would take the dual-wire arm and
            // fabricate an item id the wire never issued.
            assert_eq!(call.tool_id.as_deref(), Some("call-1"));
            assert_eq!(call.call_id, None);
        }
        other => panic!("unexpected choice: {other:?}"),
    }
}
