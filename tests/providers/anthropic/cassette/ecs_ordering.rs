//! Native Anthropic tool-batch observations and recorded follow-up requests.

use super::{
    super::support::with_anthropic_cassette,
    streaming_tools::assert_cassette_groups_multiple_tool_results,
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_observation::{install_observers, observation},
    support::{
        ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
        TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive,
    },
};
use rig::{prelude::*, providers::anthropic};

#[tokio::test]
async fn streaming_tools_batches_multiple_tool_results_in_one_followup_message() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tools_batches_multiple_tool_results_in_one_followup_message",
        |client| async move {
            let mut ecs = EcsAgent::new(
                client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6),
                TWO_TOOL_STREAM_PREAMBLE,
                1,
            );
            ecs.tool(AlphaSignal);
            ecs.tool(BetaSignal);
            install_observers(&mut ecs);
            ecs.prompt_with_max_turns(TWO_TOOL_STREAM_PROMPT, true, Some(8))
                .await;
            let observed = observation(&ecs);
            assert!(
                observed.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observed.errors
            );
            assert!(
                observed.got_final_response,
                "stream should emit a final response"
            );
            assert!(
                observed.tool_results >= 2,
                "expected at least 2 tool-result events, got {}",
                observed.tool_results
            );
            for expected_tool in ["lookup_harbor_label", "lookup_orchard_label"] {
                assert!(
                    observed.tool_calls.iter().any(|name| name == expected_tool),
                    "expected tool call for {expected_tool}, saw {:?}",
                    observed.tool_calls
                );
            }
            assert_contains_all_case_insensitive(
                observed
                    .final_response_text
                    .as_deref()
                    .expect("stream should produce a final response string"),
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
    // Keep this outside the wrapper so record mode has written the fixture.
    assert_cassette_groups_multiple_tool_results(
        "streaming_tools/streaming_tools_batches_multiple_tool_results_in_one_followup_message",
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}
