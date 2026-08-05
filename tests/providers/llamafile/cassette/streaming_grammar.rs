//! Canonical streaming-grammar coverage for the llama.cpp-family
//! OpenAI-compatible chat wire, asserted through the *normalized* path: the
//! aggregated [`StreamingCompletionResponse::choice`], the terminal
//! [`StreamFinal`] record, usage, IDs, and finish reason — real recorded wire
//! traffic, not synthetic chunks.
//!
//! Recorded against Ollama's OpenAI-compat endpoint (`/v1`) with `qwen3:4b`
//! pulled — the compat-family wire that streams each tool call as a single
//! complete `tool_calls` delta carrying both an `id` and an `index`. These
//! recordings pin what this wire class actually sends: index-keyed slots WITH
//! wire ids (`call_*`), so the grammar must keep the wire's ids rather than
//! mint `tool-{index}` identities (minting is reserved for genuinely id-less
//! events; that branch stays pinned by the synthetic Responses-wire corpus
//! and the native-Ollama `streaming_grammar` cassettes).
//!
//! Re-record with (local Ollama daemon with `qwen3:4b` pulled, no key needed):
//! `RIG_PROVIDER_TEST_MODE=record cargo test --test llamafile streaming_grammar -- --test-threads=1`
//!
//! Cassette IDs are scrub placeholders (`call_`/`chatcmpl-` prefixes are
//! preserved); assertions derive expected IDs from the recorded turn and
//! never mint literal IDs. The `/no_think` prompt prefix is qwen3's soft
//! switch that suppresses its reasoning channel, keeping the recordings
//! focused on the tool-call grammar.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{AssistantContent, ToolCall};
use rig::prelude::*;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use super::super::cassette_support::with_llamafile_cassette;
use crate::support::{AlphaSignal, BetaSignal, TWO_TOOL_STREAM_PREAMBLE};

/// Chat model used by these recordings (see the module docs).
const MODEL: &str = "qwen3:4b";

struct StreamRun {
    text: String,
    tool_calls: Vec<ToolCall>,
    finals: Vec<StreamFinal>,
    choice: OneOrMany<AssistantContent>,
    response: Option<StreamFinal>,
}

async fn drain_stream(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        tool_calls: Vec::new(),
        finals: Vec::new(),
        choice: OneOrMany::one(AssistantContent::text("")),
        response: None,
    };

    while let Some(item) = stream.next().await {
        match item.expect("stream item should be ok") {
            StreamedAssistantContent::Text(text) => run.text.push_str(&text.text),
            StreamedAssistantContent::ToolCall { tool_call, .. } => run.tool_calls.push(tool_call),
            StreamedAssistantContent::Final(response) => run.finals.push(response),
            _ => {}
        }
    }

    run.choice = stream.choice.clone();
    run.response = stream.response.clone();
    run
}

fn assert_terminal(run: &StreamRun, expected_finish: FinishReason) {
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
        Some(&expected_finish),
        "unexpected finish reason"
    );
    assert!(
        terminal.usage.total_tokens > 0,
        "terminal record should carry non-zero usage, got {:?}",
        terminal.usage
    );
}

/// A single streamed tool call on the compat wire: the call arrives as one
/// complete `tool_calls` delta with a wire `id` and `index`; the normalized
/// path keeps the wire id (no `tool-{index}` mint) and assembles the
/// arguments uncorrupted.
#[tokio::test]
async fn single_tool_call_keeps_the_wire_id() {
    with_llamafile_cassette("streaming_grammar/single_tool_call", |client| async move {
        let model = client.completion_model(MODEL);
        let request = model
            .completion_request("/no_think Call `lookup_harbor_label` now.")
            .preamble(
                "You have lookup tools. When asked to call a tool, call it immediately."
                    .to_string(),
            )
            .tool(rig::tool::tool_definition(&AlphaSignal))
            .build();
        let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

        assert_terminal(&run, FinishReason::ToolCalls);
        let streamed = run
            .tool_calls
            .iter()
            .find(|call| call.function.name == "lookup_harbor_label")
            .expect("stream should yield the lookup_harbor_label call");
        // This wire sends real ids on its tool_calls deltas; the grammar
        // must keep them verbatim — minting is only for id-less events.
        // The prefix survives cassette scrubbing; the value is derived
        // from the recorded turn.
        assert!(
            streamed.id.starts_with("call_"),
            "the compat wire's call_* id must survive normalization un-minted, got {}",
            streamed.id
        );
        assert!(
            streamed.function.arguments.is_object(),
            "arguments must assemble into an object, got {:?}",
            streamed.function.arguments
        );
        let aggregated = run
            .choice
            .iter()
            .find_map(|content| match content {
                AssistantContent::ToolCall(call) if call.function.name == "lookup_harbor_label" => {
                    Some(call)
                }
                _ => None,
            })
            .expect("aggregated choice should keep the tool call");
        assert_eq!(aggregated.id, streamed.id, "id should aggregate unchanged");
    })
    .await;
}

/// Multiple tool calls in one turn on the compat wire: every call keeps its
/// own wire id and its own uncorrupted arguments — distinct index-keyed slots
/// never bleed into each other.
#[tokio::test]
async fn parallel_tool_calls_stay_distinct() {
    with_llamafile_cassette(
        "streaming_grammar/parallel_tool_calls",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = model
                .completion_request(
                    "/no_think Call `lookup_harbor_label` and `lookup_orchard_label` now, both \
                     of them together in this single reply, before writing any text. Emit the \
                     two tool calls in one turn — do not wait for results between them.",
                )
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            let aggregated: Vec<&ToolCall> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(call) => Some(call),
                    _ => None,
                })
                .collect();
            for name in ["lookup_harbor_label", "lookup_orchard_label"] {
                let streamed = run
                    .tool_calls
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("stream should yield a {name} call"));
                let aggregated_call = aggregated
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("aggregated choice should keep the {name} call"));
                assert_eq!(
                    aggregated_call.id, streamed.id,
                    "{name} id should aggregate"
                );
                assert!(
                    streamed.id.starts_with("call_"),
                    "{name} should keep the wire's call_* id, got {}",
                    streamed.id
                );
                assert!(
                    streamed.function.arguments.is_object(),
                    "{name} arguments must assemble into an object, got {:?}",
                    streamed.function.arguments
                );
            }
            // The model may repeat a tool within the turn; the grammar
            // contract is that every streamed call survives as its own
            // aggregated part with a distinct identity.
            assert!(run.tool_calls.len() >= 2, "turn should stream both calls");
            assert_eq!(
                aggregated.len(),
                run.tool_calls.len(),
                "every streamed call should survive as its own aggregated part"
            );
            let mut ids: Vec<&str> = run.tool_calls.iter().map(|call| call.id.as_str()).collect();
            ids.sort_unstable();
            let total = ids.len();
            ids.dedup();
            assert_eq!(
                ids.len(),
                total,
                "parallel calls must keep distinct identities"
            );
        },
    )
    .await;
}
