use super::*;
use crate::{
    error::ErrorKind,
    operation::Finish,
    streaming::{Item, StreamEvent},
};
use futures::StreamExt;

#[tokio::test]
async fn completion_consumes_scripted_turns_and_records_requests() {
    let model = MockCompletionModel::from_turns([
        MockTurn::text("first"),
        MockTurn::tool_call("tool_1", "calculator", serde_json::json!({"x": 1}))
            .with_call_id("call_1"),
    ]);

    let first = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect("first scripted turn should succeed");
    assert!(matches!(
        first.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "first"
    ));

    let second = model
        .call(CompletionRequest::new("use a tool"))
        .await
        .expect("second scripted turn should succeed");
    assert!(matches!(
        second.choice.first(),
        Some(AssistantContent::ToolCall(tool_call))
            if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_1")
    ));

    assert_eq!(model.request_count(), 2);
    assert_eq!(model.requests().len(), 2);
}

/// The streaming half of the same contract: the mock's document is its
/// scripted end, so the response's `raw` is that end serialized.
#[tokio::test]
async fn stream_terminal_raw_is_the_scripted_terminal_serialized() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response(Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(3),
            ..Usage::default()
        }),
    ]]);

    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream should open");
    while stream.next().await.is_some() {}
    let response = stream.finish().await.expect("the reply ended");
    let typed: Finish = serde_json::from_value(response.raw.clone()).expect("the end");
    assert_eq!(
        typed,
        super::super::streaming::mock_final(typed.usage),
        "the capture is the scripted end"
    );
    assert_eq!(response.usage.total_tokens, Some(3));
}

#[tokio::test]
async fn missing_completion_turn_returns_provider_error() {
    let model = MockCompletionModel::from_turns([]);

    let err = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect_err("missing turn should error");

    assert!(matches!(
        err,
        ProviderError::Provider(message)
            if message.contains("no scripted completion turn")
    ));
}

#[tokio::test]
async fn stream_yields_scripted_events_and_records_requests() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("hel"),
        MockStreamEvent::text("lo"),
        MockStreamEvent::tool_call_name_delta("call_1", "calculator"),
        MockStreamEvent::tool_call_arguments_delta("call_1", "{\"x\":1}"),
        MockStreamEvent::tool_call_end("call_1"),
        MockStreamEvent::final_response_with_total_tokens(7),
    ]]);

    let mut stream = model
        .stream(CompletionRequest::new("stream"))
        .expect("stream should be created");

    let mut text = String::new();
    let mut arguments = None;
    let mut call = None;
    while let Some(item) = stream.next().await {
        match item.expect("stream event should succeed") {
            Item::Event(StreamEvent::Text { text: chunk, .. }) => text.push_str(&chunk),
            Item::Event(StreamEvent::Arguments { json, .. }) => arguments = Some(json),
            Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            }) => call = Some(tool_call),
            _ => {}
        }
    }

    assert_eq!(text, "hello");
    assert_eq!(arguments.as_deref(), Some("{\"x\":1}"));
    let call = call.expect("the call ended");
    assert_eq!(call.id.to_string(), "call_1");
    assert_eq!(call.function.name, "calculator");
    assert_eq!(call.function.arguments_value(), serde_json::json!({"x": 1}));
    let response = stream.finish().await.expect("the reply ended");
    assert_eq!(response.usage.total_tokens, Some(7));
    assert_eq!(model.request_count(), 1);
}

#[tokio::test]
async fn stream_error_event_is_returned() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error("boom")]]);
    let mut stream = model
        .stream(CompletionRequest::new("stream"))
        .expect("stream should be created");

    let err = stream
        .next()
        .await
        .expect("stream should yield one event")
        .expect_err("scripted event should error");

    assert_eq!(err.kind(), ErrorKind::Provider);
    assert_eq!(err.to_string(), "ProviderError: boom");
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

    let stream_turns = vec![vec![
        MockStreamEvent::Text("hi".into()),
        MockStreamEvent::FinalResponse(super::super::streaming::mock_final(Usage::default())),
    ]];
    let json = serde_json::to_string(&stream_turns).expect("stream turns serialize");
    let restored: Vec<Vec<MockStreamEvent>> =
        serde_json::from_str(&json).expect("stream turns deserialize");
    assert_eq!(restored, stream_turns);
    let model = MockCompletionModel::from_stream_turns(restored);
    assert!(model.script().is_empty());
}

/// A scripted model priced as a catalog model prices its replies' usage at
/// the spec's prices, streamed or not, as the provider's own wire does.
#[tokio::test]
async fn a_priced_script_prices_its_usage_like_a_real_wire() {
    use crate::catalog::{ModelSpec, Pricing};
    use crate::providers::registry::ProviderId;
    use crate::test_utils::MockStreamEvent;

    let vendor = ProviderId::catalog("deepseek").expect("a known vendor");
    let spec = ModelSpec::new(vendor, "scripted").with_pricing(Pricing::new(1.0, 2.0));
    let usage = Usage::new()
        .input_tokens(1_000_000)
        .output_tokens(1_000_000);
    let unary = MockCompletionModel::from_turns([MockTurn::text("hi").with_usage(usage)])
        .priced_as(spec.clone());
    let reply = unary
        .call(CompletionRequest::new("hello"))
        .await
        .expect("a scripted reply");
    assert_eq!(reply.usage.cost.map(|cost| cost.total), Some(3.0));
    assert_eq!(reply.origin.provider.as_str(), "deepseek");

    let streamed = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hi"),
        MockStreamEvent::final_response(usage),
    ]])
    .priced_as(spec);
    let reply = streamed
        .stream(CompletionRequest::new("hello"))
        .expect("a scripted stream")
        .finish()
        .await
        .expect("a streamed reply");
    assert_eq!(reply.usage.cost.map(|cost| cost.total), Some(3.0));
    // Unpriced, the same script states no cost.
    let plain = MockCompletionModel::from_turns([MockTurn::text("hi").with_usage(usage)]);
    let reply = plain
        .call(CompletionRequest::new("hello"))
        .await
        .expect("a reply");
    assert_eq!(reply.usage.cost, None);
}
