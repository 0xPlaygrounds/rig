//! Recordings first made for the memory matrix (`CLAUDE_SONNET_4_6`,
//! temperature 0), kept for the requests they pin: the zero-argument tool
//! schema of the acceptance index and a unary/streamed raw parity pair. The
//! memory carried no request bytes, so each serves its program with the
//! caller keeping the history: a second run continues from the first run's
//! messages.

use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, PromptResponse};
use rig::effect::EffectFamily;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_test_support::cassette_models::AnthropicModels;

use super::super::support::with_anthropic_cassette;
use crate::goldens::families;
use crate::support::{
    AlphaSignal, BASIC_PREAMBLE, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
};

const PROMPT: &str = "Reply with the single word: ready.";
const SECOND_PROMPT: &str = "Now reply with the single word: again.";

async fn final_response(stream: &mut rig::agent::StreamingResult) -> PromptResponse {
    let mut response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(last) = item.expect("the stream yields") {
            response = Some(last);
        }
    }
    response.expect("a final response")
}

/// Two runs, the second starting from the first run's messages; one
/// completion each.
async fn two_runs(client: AnthropicModels, streamed: bool) {
    let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
    let agent = rig::AgentBuilder::new(client.completion(CLAUDE_SONNET_4_6))
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .record_to(recorder.clone())
        .build();
    let mut history = Vec::new();
    for prompt in [PROMPT, SECOND_PROMPT] {
        let runner = agent.prompt(prompt).history(history.clone());
        let response = if streamed {
            final_response(&mut runner.max_turns(8).stream()).await
        } else {
            runner.max_turns(3).await.expect("the agent answers")
        };
        assert!(!response.output().is_empty());
        history.extend(response.messages);
    }
    assert_eq!(
        families(&recorder.take()),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
}

#[tokio::test]
async fn a_second_run_continues_from_the_first_runs_messages() {
    with_anthropic_cassette("corpus_memory/two_runs", |client| async move {
        two_runs(client, false).await;
    })
    .await;
}

#[tokio::test]
async fn a_second_streamed_run_continues_from_the_first_runs_messages() {
    with_anthropic_cassette("corpus_memory/two_runs_streamed", |client| async move {
        two_runs(client, true).await;
    })
    .await;
}

/// Serial serving and two tool calls in one turn: both results reach the
/// final request.
#[tokio::test]
async fn serial_serving_runs_two_tool_calls_of_one_turn() {
    with_anthropic_cassette("corpus_memory/serial_two_tools", |client| async move {
        let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
        let agent = rig::AgentBuilder::new(client.completion(CLAUDE_SONNET_4_6))
            .name("golden")
            .preamble(TWO_TOOL_STREAM_PREAMBLE)
            .temperature(0.0)
            .tool(AlphaSignal)
            .tool(BetaSignal)
            .record_to(recorder.clone())
            .build();
        let mut stream = agent.prompt(TWO_TOOL_STREAM_PROMPT).max_turns(8).stream();
        let response = final_response(&mut stream).await;
        drop(stream);
        assert!(!response.output().is_empty());
        assert_eq!(
            families(&recorder.take()),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Tool,
                EffectFamily::Completion,
            ]
        );
    })
    .await;
}
