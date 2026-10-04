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
