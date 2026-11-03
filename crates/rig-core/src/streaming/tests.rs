use super::*;
use crate::test_utils::{MockCompletionModel, MockStreamEvent};
use futures::StreamExt;

/// The stream a scripted reply opens.
fn stream_of(events: Vec<MockStreamEvent>) -> CompletionStream {
    MockCompletionModel::from_stream_turns([events])
        .stream("hi")
        .expect("the stream opens")
}

/// Every item of `stream`, read to its end.
async fn items_of(stream: &mut CompletionStream) -> Vec<Result<Item<StreamEvent>, ProviderError>> {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    items
}

#[tokio::test]
async fn an_error_is_the_last_item_and_finish_returns_it() {
    let mut stream = stream_of(vec![
        MockStreamEvent::text("kept"),
        MockStreamEvent::error("aborted by the provider"),
        MockStreamEvent::text("never read"),
    ]);
    let items = items_of(&mut stream).await;
    let Some(Err(yielded)) = items.last() else {
        panic!("the error is the last item: {items:?}");
    };
    assert!(yielded.to_string().contains("aborted"), "{yielded}");
    let partial = stream.partial();
    assert_eq!(partial.text(), "kept", "the text taken so far stays");
    let finished = stream.finish().await.expect_err("the error stands");
    assert_eq!(finished.to_string(), yielded.to_string());
}

/// A reply the provider did not end is a failed turn by construction: its
/// message is never replayed, whatever it holds.
#[tokio::test]
async fn a_partial_turn_after_an_error_is_marked_failed() {
    use crate::message::{Message, StopReason};
    let mut stream = stream_of(vec![
        MockStreamEvent::text("answer"),
        MockStreamEvent::tool_call("call_1", "lookup", serde_json::json!({"q": 1})),
        MockStreamEvent::error("mid-stream failure"),
    ]);
    let _ = items_of(&mut stream).await;
    let Some(Message::Assistant(turn)) = stream.partial().message() else {
        panic!("the partial turn has content");
    };
    assert!(
        matches!(&turn.stop, Some(StopReason::Error(error)) if error.contains("mid-stream failure")),
        "{:?}",
        turn.stop
    );
}

/// A usage-only frame leaves the relay before the transport sends anything
/// more.
#[tokio::test]
async fn anthropic_partial_usage_reaches_the_relay_before_content() {
    use crate::providers::anthropic::streaming::MessagesDecoder;
    use crate::providers::anthropic::wire::Quirks;
    use crate::wire::{Decoder, Out, WireEvent, WireFrame};

    let shared = Arc::new(Mutex::new(Shared::new(ReplyFold::relayed("anthropic"))));
    let writer = Arc::clone(&shared);
    let reading = async_stream::stream! {
        let mut decoder = MessagesDecoder::new(Quirks::anthropic());
        for frame in [
            serde_json::json!({"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-sonnet-4-6", "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }}),
            serde_json::json!({"type": "message_delta", "delta": {}, "usage": {"output_tokens": 3}}),
        ] {
            if let WireEvent::Known(event) = decoder.classify(WireFrame::Text(frame.to_string())) {
                let result = decoder.decode(event, Out::new(&writer));
                assert!(result.is_ok(), "{result:?}");
            }
            yield ();
        }
        futures::future::pending::<()>().await;
    };
    let stream = CompletionStream::new(
        Box::pin(reading),
        shared,
        tracing::Span::none(),
        "anthropic",
    );
    let mut relay = stream.into_relay();
    assert!(matches!(relay.next().await, Some(Ok(Relayed::Origin(_)))));
    for output in [0, 3] {
        let item = tokio::time::timeout(std::time::Duration::from_secs(1), relay.next()).await;
        assert!(
            matches!(item, Ok(Some(Ok(Relayed::Usage(usage))))
            if usage.input_tokens == Some(10) && usage.output_tokens == Some(output)),
            "{item:?}"
        );
    }
}

/// Intermediate relays retain reports even when the provider has not ended the reply.
#[tokio::test]
async fn partial_usage_survives_a_relay_round_trip() {
    let usage = crate::completion::Usage::new()
        .input_tokens(10)
        .output_tokens(2);
    let events =
        futures::stream::iter([Ok(Relayed::Usage(usage))]).chain(futures::stream::pending());
    let mut relay = CompletionStream::relay("test", Box::pin(events)).into_relay();
    assert!(matches!(relay.next().await, Some(Ok(Relayed::Origin(_)))));
    assert_eq!(relay.next().await, Some(Ok(Relayed::Usage(usage))));
}
