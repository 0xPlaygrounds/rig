use super::*;
use crate::{
    message::Message,
    streaming::{StreamFinal, StreamedAssistantContent, ToolCallDeltaContent},
};
use futures::StreamExt;

fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user(prompt)],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[tokio::test]
async fn completion_consumes_scripted_turns_and_records_requests() {
    let model = MockCompletionModel::new([
        MockTurn::text("first").with_message_id("msg_1"),
        MockTurn::tool_call("tool_1", "calculator", serde_json::json!({"x": 1}))
            .with_call_id("call_1"),
    ]);

    let first = model
        .completion(request("hello"))
        .await
        .expect("first scripted turn should succeed");
    assert_eq!(first.message_id.as_deref(), Some("msg_1"));
    assert!(matches!(
        first.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "first"
    ));

    let second = model
        .completion(request("use a tool"))
        .await
        .expect("second scripted turn should succeed");
    assert!(matches!(
        second.choice.first(),
        Some(AssistantContent::ToolCall(tool_call))
            if tool_call.id == "tool_1"
                && tool_call
                    .provider
                    .as_ref()
                    .is_some_and(|provider| provider.call_id == "call_1")
    ));

    assert_eq!(model.request_count(), 2);
    assert_eq!(model.requests().len(), 2);
}

/// The mock behaves like a real seam: a scripted raw payload rides on the
/// normalized response unconditionally, and a turn that scripted none
/// reports `raw: Value::Null` — the mock never invents a payload, so `Value::Null`
/// here means "no provider record was scripted behind this turn".
#[tokio::test]
async fn completion_attaches_scripted_raw_and_reports_null_when_unscripted() {
    let payload = serde_json::json!({"provider_only": "kept", "id": "resp_1"});
    let model = MockCompletionModel::new([
        MockTurn::text("first").with_raw(payload.clone()),
        MockTurn::text("second"),
    ]);

    let scripted = model
        .completion(request("hello"))
        .await
        .expect("first scripted turn should succeed");
    assert_eq!(scripted.raw, payload);

    let unscripted = model
        .completion(request("hello"))
        .await
        .expect("second scripted turn should succeed");
    assert!(unscripted.raw.is_null());

    assert_eq!(model.requests().len(), 2);
}

/// The streaming half of the same contract: the scripted terminal goes
/// through `normalize_stream`, so the terminal's `raw` is the scripted
/// terminal record serialized (the mock's own terminal type is
/// `StreamFinal`).
#[tokio::test]
async fn stream_terminal_raw_is_the_scripted_terminal_serialized() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response(Usage {
            input_tokens: 1,
            output_tokens: 2,
            total_tokens: 3,
            ..Usage::new()
        }),
    ]]);

    let mut stream = model
        .stream(request("hello"))
        .await
        .expect("stream should open");
    while stream.next().await.is_some() {}
    let terminal = stream.response.expect("terminal record");
    let raw = &terminal.raw;
    let typed: StreamFinal = serde_json::from_value(raw.clone()).expect("terminal type");
    assert_eq!(typed.usage.total_tokens, 3);
    assert!(
        typed.raw.is_null(),
        "the scripted terminal itself carried no raw (Value::Null)"
    );
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "the capture must be exactly what the scripted terminal serializes to"
    );
    assert_eq!(terminal.usage.total_tokens, 3);
}

#[tokio::test]
async fn missing_completion_turn_returns_provider_error() {
    let model = MockCompletionModel::default();

    let err = model
        .completion(request("hello"))
        .await
        .expect_err("missing turn should error");

    assert!(matches!(
        err,
        CompletionError::ProviderError(message)
            if message.contains("no scripted completion turn")
    ));
}

#[tokio::test]
async fn stream_yields_scripted_events_and_records_requests() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::message_id("msg_stream"),
        MockStreamEvent::text("hel"),
        MockStreamEvent::text("lo"),
        MockStreamEvent::tool_call_name_delta("tool_1", "calculator"),
        MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1}"),
        MockStreamEvent::tool_call("tool_1", "calculator", serde_json::json!({"x": 1}))
            .with_call_id("call_1"),
        MockStreamEvent::final_response_with_total_tokens(7),
    ]]);

    let mut stream = model
        .stream(request("stream"))
        .await
        .expect("stream should be created");

    let mut text = String::new();
    let mut saw_name_delta = false;
    let mut saw_arguments_delta = false;
    let mut saw_tool_call = false;
    let mut saw_final = false;

    while let Some(item) = stream.next().await {
        match item.expect("stream event should succeed") {
            StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
            StreamedAssistantContent::ToolCallDelta { content, .. } => match content {
                ToolCallDeltaContent::Name(name) => {
                    saw_name_delta = name == "calculator";
                }
                ToolCallDeltaContent::Delta(arguments) => {
                    saw_arguments_delta = arguments == "{\"x\":1}";
                }
            },
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                saw_tool_call = tool_call
                    .provider
                    .as_ref()
                    .is_some_and(|provider| provider.call_id == "call_1");
            }
            StreamedAssistantContent::Final(response) => {
                saw_final = matches!(
                    response.usage,
                    Usage {
                        total_tokens: 7,
                        ..
                    }
                );
            }
            _ => {}
        }
    }

    assert_eq!(text, "hello");
    assert!(saw_name_delta);
    assert!(saw_arguments_delta);
    assert!(saw_tool_call);
    assert!(saw_final);
    assert_eq!(stream.message_id.as_deref(), Some("msg_stream"));
    assert_eq!(model.request_count(), 1);
}

#[tokio::test]
async fn stream_error_event_is_returned() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error("boom")]]);
    let mut stream = model
        .stream(request("stream"))
        .await
        .expect("stream should be created");

    let err = stream
        .next()
        .await
        .expect("stream should yield one event")
        .expect_err("scripted event should error");

    assert!(matches!(
        err,
        CompletionError::ProviderError(message) if message == "boom"
    ));
}

#[test]
fn a_script_is_serde_in_and_serde_out() {
    let turns = vec![
        MockTurn::text("hello"),
        MockTurn::tool_call("tc1", "add", serde_json::json!({"x": 1})),
        MockTurn::error("boom"),
    ];
    let json = serde_json::to_string(&turns).expect("turns serialize");
    let restored: Vec<MockTurn> = serde_json::from_str(&json).expect("turns deserialize");
    assert_eq!(restored, turns);

    let model = MockCompletionModel::from_turns(restored);
    assert_eq!(model.script(), turns);
    assert_eq!(model.stream_script(), Vec::<Vec<MockStreamEvent>>::new());

    let stream_turns = vec![vec![
        MockStreamEvent::Text("hi".into()),
        MockStreamEvent::FinalResponse(super::super::streaming::mock_final(Usage::new())),
    ]];
    let json = serde_json::to_string(&stream_turns).expect("stream turns serialize");
    let restored: Vec<Vec<MockStreamEvent>> =
        serde_json::from_str(&json).expect("stream turns deserialize");
    assert_eq!(restored, stream_turns);
    let model = MockCompletionModel::from_stream_turns(restored);
    assert_eq!(model.stream_script(), stream_turns);
    assert!(model.script().is_empty());
}
