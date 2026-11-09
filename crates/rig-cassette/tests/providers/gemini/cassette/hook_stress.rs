//! Golden effect log for a multi-turn hook-system workflow recorded against
//! real Gemini: two dependent tool turns on a wire that carries no tool-call
//! ids.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::providers::gemini;
use rig_cassette::agent::AgentReplayExt;

use super::super::agent_run_support::{Add, CHAIN_PREAMBLE, Subtract};
use super::super::support::with_gemini_cassette;

/// Golden `gemini_tool_call_turns`: two tool turns on a wire that carries
/// no tool-call ids. Every id in the log is minted from the block that
/// assembled the call, so the record is the same on every run — the proof
/// that nothing the engine mints is random.
#[tokio::test]
async fn tool_call_turns_effect_log_is_the_golden_fixture() {
    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .name("stress-agent")
                    .preamble(CHAIN_PREAMBLE)
                    .temperature(0.0)
                    .tool(Add)
                    .tool(Subtract)
                    .record_to(recorder.clone())
                    .build();
            let mut stream = agent
                .prompt(
                    "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .stream();
            let mut saw_final = false;
            while let Some(item) = stream.next().await {
                if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                    saw_final = true;
                }
            }
            assert!(saw_final, "the stream must yield a FinalResponse");
            let log = agent.stamp(recorder.take());
            let tool_ids: Vec<&rig::message::CallId> = log
                .records
                .iter()
                .filter_map(|record| match &record.outcome {
                    Ok(rig::effect::Answered::Completion(response)) => Some(response),
                    _ => None,
                })
                .flat_map(|response| response.choice.iter())
                .filter_map(|content| match content {
                    rig::message::AssistantContent::ToolCall(call) => Some(&call.id),
                    _ => None,
                })
                .collect();
            assert!(!tool_ids.is_empty(), "the program calls tools");
            assert!(
                tool_ids.iter().all(|id| id.is_local()),
                "every id-less wire call is named by its block: {tool_ids:?}"
            );
            crate::goldens::golden_effects("gemini_tool_call_turns", &log);
        },
    )
    .await;
}
