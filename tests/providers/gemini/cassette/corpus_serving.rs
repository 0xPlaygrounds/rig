//! Matrix C of the effect corpus, the Gemini cell: two tool-call turns on
//! an id-less wire under serial serving. The `hook_stress` cassette the
//! `gemini_tool_call_turns` golden records from serves this program too:
//! the policy changes how the bus serves, not what the program asks.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::prelude::*;
use rig::providers::gemini;

use super::super::hook_stress_support::CHAIN_PREAMBLE;
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract};
use crate::goldens::families;

#[tokio::test]
async fn two_turns_serial_effect_log_is_the_golden_fixture() {
    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .configure_bus(rig::serve::ServingPolicy {
                    serial_per_handler: true,
                    ..rig::serve::ServingPolicy::default()
                })
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(CountingAdd::default())
                .tool(CountingSubtract::default())
                .record_effects()
                .build();
            let mut stream = agent
                .stream_prompt(
                    "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .stream()
                .await;
            let mut saw_final = false;
            while let Some(item) = stream.next().await {
                if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                    saw_final = true;
                }
            }
            drop(stream);
            assert!(saw_final, "the stream yields a final response");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(log.header.bus.map(|bus| bus.serial_per_handler), Some(true));
            crate::goldens::golden_effects("gemini_serving_two_turns_serial", &log);
        },
    )
    .await;
}
