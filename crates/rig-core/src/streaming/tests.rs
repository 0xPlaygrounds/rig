use super::*;
use crate::completion::{FinishReason, Usage};
use crate::error::ErrorKind;
use crate::message::{AssistantContent, ReasoningContent};
use crate::operation::Finish;
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

fn usage(total: u64) -> Usage {
    Usage {
        total_tokens: Some(total),
        ..Usage::default()
    }
}

#[tokio::test]
async fn finish_folds_the_parts_with_the_providers_end() {
    let mut stream = stream_of(vec![
        MockStreamEvent::text("Hello"),
        MockStreamEvent::text(" world"),
        MockStreamEvent::FinalResponse(
            Finish::new(usage(7))
                .with_reason(FinishReason::Stop)
                .with_optional_provider_request_id(Some("req-1".to_owned())),
        ),
    ]);
    let items = items_of(&mut stream).await;
    assert!(items.iter().all(Result::is_ok), "{items:?}");
    let response = stream.finish().await.expect("the reply ended");
    assert_eq!(response.text(), "Hello world");
    assert_eq!(response.usage.total_tokens, Some(7));
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.provider_request_id.as_deref(), Some("req-1"));
}

#[tokio::test]
async fn a_reply_without_its_end_is_truncated() {
    let mut stream = stream_of(vec![MockStreamEvent::text("cut")]);
    let items = items_of(&mut stream).await;
    assert!(
        matches!(items.last(), Some(Err(ProviderError::Truncated))),
        "the truncation is the last item: {items:?}"
    );
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
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

#[tokio::test]
async fn partial_keeps_the_parts_that_ended_before_an_error() {
    let mut stream = stream_of(vec![
        MockStreamEvent::text("answer"),
        MockStreamEvent::tool_call("call_1", "lookup", serde_json::json!({"q": 1})),
        MockStreamEvent::error("mid-stream failure"),
    ]);
    let _ = items_of(&mut stream).await;
    let partial = stream.partial();
    assert_eq!(partial.choice.len(), 2, "{:?}", partial.choice);
    assert_eq!(partial.text(), "answer");
}

#[tokio::test]
async fn usage_the_provider_did_not_report_stays_unreported() {
    let mut stream = stream_of(vec![
        MockStreamEvent::text("hi"),
        MockStreamEvent::final_response_with_default_usage(),
    ]);
    let _ = items_of(&mut stream).await;
    let response = stream.finish().await.expect("the reply ended");
    assert!(!response.usage.is_reported(), "{:?}", response.usage);
}

#[tokio::test]
async fn an_unmodeled_payload_reaches_the_consumer_but_not_the_choice() {
    let payload = serde_json::json!({"type": "web_search_call", "id": "ws_1"});
    let mut stream = stream_of(vec![
        MockStreamEvent::unknown(payload.clone()),
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]);
    let items = items_of(&mut stream).await;
    assert!(
        items.iter().any(|item| matches!(
            item,
            Ok(Item::Unknown(unknown)) if unknown.value() == &payload
        )),
        "{items:?}"
    );
    let response = stream.finish().await.expect("the reply ended");
    assert_eq!(response.choice, vec![AssistantContent::text("done")]);
}

#[tokio::test]
async fn the_choice_is_in_the_order_its_parts_started() {
    let mut stream = stream_of(vec![
        MockStreamEvent::text("before"),
        MockStreamEvent::tool_call("call_1", "lookup", serde_json::json!({})),
        MockStreamEvent::reasoning_delta("thinking"),
        MockStreamEvent::text("after"),
        MockStreamEvent::final_response_with_default_usage(),
    ]);
    let _ = items_of(&mut stream).await;
    let response = stream.finish().await.expect("the reply ended");
    let kinds: Vec<&str> = response
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::Text(_) => "text",
            AssistantContent::ToolCall(_) => "call",
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::Image(_) => "image",
        })
        .collect();
    assert_eq!(kinds, ["text", "call", "reasoning", "text"]);
    let reasoning = response
        .choice
        .iter()
        .find_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => reasoning.open(reasoning.issuer()).cloned(),
            _ => None,
        })
        .expect("the reasoning opens for its issuer");
    assert_eq!(
        reasoning.content,
        vec![ReasoningContent::Text {
            text: "thinking".to_owned(),
            signature: None,
        }]
    );
}

#[tokio::test]
async fn a_stop_that_carried_a_tool_call_is_reported_as_tool_calls() {
    let mut stream = stream_of(vec![
        MockStreamEvent::tool_call("call_1", "lookup", serde_json::json!({})),
        MockStreamEvent::FinalResponse(
            Finish::new(Usage::default()).with_reason(FinishReason::Stop),
        ),
    ]);
    let _ = items_of(&mut stream).await;
    let response = stream.finish().await.expect("the reply ended");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

#[tokio::test]
async fn polling_a_drained_stream_again_yields_nothing() {
    let mut stream = stream_of(vec![
        MockStreamEvent::text("once"),
        MockStreamEvent::final_response_with_default_usage(),
    ]);
    let _ = items_of(&mut stream).await;
    assert!(stream.next().await.is_none());
    let response = stream.finish().await.expect("the reply ended");
    assert_eq!(response.text(), "once");
}

#[tokio::test]
async fn a_relayed_stream_finishes_with_the_origins_response() {
    let origin = stream_of(vec![
        MockStreamEvent::text("relayed"),
        MockStreamEvent::final_response(usage(3)),
    ]);
    let mut relayed = Streamed::relay("mock", origin.into_relay());
    let items = items_of(&mut relayed).await;
    assert!(items.iter().all(Result::is_ok), "{items:?}");
    let response = relayed.finish().await.expect("the origin ended");
    assert_eq!(response.text(), "relayed");
    assert_eq!(response.usage.total_tokens, Some(3));
}

#[tokio::test]
async fn a_relay_cut_short_is_truncated() {
    let origin = stream_of(vec![MockStreamEvent::text("cut")]);
    let events = origin
        .into_relay()
        .filter(|item| futures::future::ready(matches!(item, Ok(Relayed::Item(_)))));
    let mut relayed = Streamed::relay("mock", Box::pin(events));
    let _ = items_of(&mut relayed).await;
    let error = relayed.finish().await.expect_err("no response arrived");
    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
}

#[tokio::test]
async fn a_relayed_error_keeps_its_kind() {
    let origin = stream_of(vec![MockStreamEvent::error("provider down")]);
    let mut relayed = Streamed::relay("mock", origin.into_relay());
    let items = items_of(&mut relayed).await;
    let Some(Err(error)) = items.last() else {
        panic!("the error is relayed: {items:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Provider);
}
