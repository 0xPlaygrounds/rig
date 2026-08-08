//! Canonical streaming-grammar coverage for the Cohere v2 chat wire, asserted
//! through the *normalized* path: the aggregated
//! [`StreamingCompletionResponse::choice`], the terminal [`StreamFinal`]
//! record, usage, and finish reason.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{AssistantContent, Reasoning, ReasoningContent};
use rig::prelude::*;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use super::super::support::with_cohere_cassette;

/// Cohere's reasoning-capable Command model, which streams `thinking`-bearing
/// content deltas before the answer text (the F8 cohere variant).
const REASONING_MODEL: &str = "command-a-reasoning-08-2025";

struct StreamRun {
    text: String,
    reasoning_delta: String,
    reasoning_blocks: Vec<Reasoning>,
    finals: Vec<StreamFinal>,
    choice: OneOrMany<AssistantContent>,
    response: Option<StreamFinal>,
}

async fn drain_stream(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        reasoning_delta: String::new(),
        reasoning_blocks: Vec::new(),
        finals: Vec::new(),
        choice: OneOrMany::one(AssistantContent::text("")),
        response: None,
    };

    let mut raw_items = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        raw_items.push(Ok(item.clone()));
        match item {
            StreamedAssistantContent::Text(text) => run.text.push_str(&text.text),
            StreamedAssistantContent::Reasoning { reasoning, .. } => {
                run.reasoning_blocks.push(reasoning)
            }
            StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                run.reasoning_delta.push_str(&reasoning);
            }
            StreamedAssistantContent::Final(response) => run.finals.push(response),
            _ => {}
        }
    }

    run.choice = stream.choice.clone();
    // The shared lifecycle validator runs over every recorded turn this
    // suite drains (#2258 C1).
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(&raw_items, &run.choice);
    run.response = stream.response.clone();
    run
}

/// A `thinking`-bearing stream: the thinking deltas aggregate into one
/// reasoning part that survives as a discrete sibling of the answer text.
#[tokio::test]
async fn thinking_stream_keeps_reasoning_and_text_discrete() {
    with_cohere_cassette("streaming_grammar/thinking_stream", |client| async move {
        let model = client.completion_model(REASONING_MODEL);
        let request = model
            .completion_request(
                "How many positive integers n < 100 are divisible by 6? \
                 Think it through, then answer with just the number.",
            )
            .max_tokens(1024)
            .build();
        let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

        assert!(!run.text.trim().is_empty(), "turn should produce text");
        assert_eq!(
            run.finals.len(),
            1,
            "stream should yield exactly one terminal record"
        );
        let terminal = run
            .response
            .as_ref()
            .expect("aggregated stream should retain the terminal record");
        assert_eq!(
            terminal.finish_reason.as_ref(),
            Some(&FinishReason::Stop),
            "unexpected finish reason"
        );
        assert!(
            terminal.usage.total_tokens > 0,
            "terminal record should carry non-zero usage, got {:?}",
            terminal.usage
        );
        assert!(
            !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
            "reasoning model should surface thinking on the stream"
        );

        // Discrete parts: the reasoning part and the text part live
        // side-by-side in the aggregated choice.
        let reasoning_text: String = run
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Reasoning(reasoning) => Some(reasoning.content.iter()),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                ReasoningContent::Text { text, .. } => Some(text.as_str()),
                ReasoningContent::Summary(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !reasoning_text.is_empty(),
            "aggregated choice should retain the thinking text, got {:?}",
            run.choice
        );
        let aggregated_text: String = run
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            aggregated_text, run.text,
            "aggregated text should match the streamed text exactly"
        );
    })
    .await;
}
