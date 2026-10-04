//! Canonical streaming-grammar coverage for Cohere's Chat dialect, asserted
//! through the *normalized* path: the aggregated
//! [`Streamed::finish`](rig::streaming::Streamed::finish) response, the terminal `CompletionResponse`
//! record, usage, and finish reason.

use futures::StreamExt;
use rig::completion::CompletionResponse;
use rig::completion::FinishReason;
use rig::message::{AssistantContent, Reasoning, ToolCall, ToolChoice};
use rig::streaming::Item;
use rig::streaming::StreamEvent;

use super::super::{
    CASSETTE_MODEL,
    support::{IntegerSubtract, with_cohere_cassette},
};
use rig::completion::CompletionRequest;

struct StreamRun {
    text: String,
    reasoning_delta: String,
    reasoning_blocks: Vec<Reasoning>,
    tool_calls: Vec<ToolCall>,
    choice: Vec<AssistantContent>,
    response: Option<CompletionResponse>,
}

async fn drain_stream(mut stream: rig::streaming::CompletionStream) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        reasoning_delta: String::new(),
        reasoning_blocks: Vec::new(),
        tool_calls: Vec::new(),
        choice: vec![AssistantContent::text("")],
        response: None,
    };

    let mut raw_items = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        raw_items.push(Ok(item.clone()));
        match item {
            Item::Event(StreamEvent::Text { text, .. }) => run.text.push_str(&text),
            Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            }) => {
                run.reasoning_blocks.push(reasoning);
            }
            Item::Event(StreamEvent::Reasoning { text, .. }) => {
                run.reasoning_delta.push_str(&text);
            }
            Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            }) => run.tool_calls.push(tool_call),
            Item::Event(StreamEvent::Start { .. })
            | Item::Event(StreamEvent::Arguments { .. })
            | Item::Event(StreamEvent::End { .. })
            | Item::Unknown(_) => {}
        }
    }
    let response = stream.finish().await.expect("the stream ends");

    run.choice = response.choice.clone();
    // The shared lifecycle validator runs over every recorded turn this
    // suite drains (#2258 C1).
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(&raw_items, &run.choice);
    run.response = Some(response.clone());
    run
}

#[tokio::test]
async fn none_tool_choice_streams_text() {
    with_cohere_cassette(
        "streaming_grammar/none_tool_choice_streams_text",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let request =
                CompletionRequest::new("Calculate 8 - 3. Answer directly without calling a tool.")
                    .tool(rig::tool::tool_definition(&IntegerSubtract))
                    .tool_choice(ToolChoice::None)
                    .max_tokens(32);
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "NONE should stream text");
            assert!(run.tool_calls.is_empty(), "NONE must suppress tool calls");
            assert_eq!(
                run.response
                    .as_ref()
                    .and_then(|response| response.finish_reason()),
                Some(FinishReason::Stop)
            );
        },
    )
    .await;
}
