//! Native counterparts of the provider turn-termination matrix.

use super::super::support::with_openai_turn_metadata_cassette;
use super::turn_termination_matrix::{
    ROOMY_CAP, TOOL_PREAMBLE, TOOL_PROMPT, assert_recorded_wire_reason,
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_termination::{self, NativeProbe as TurnTerminationProbe},
    support::Adder,
};
use rig::completion::FinishReason;
use rig::providers::openai;
use rig_ecs::agent::{MaxTokens, Temperature};

#[tokio::test]
async fn streaming_tool_turn_reports_tool_calls() {
    rig_test_support::goldens::world_golden_test(
        async {
            {
                const SCENARIO: &str =
                    "turn_termination_matrix/streaming_tool_turn_reports_tool_calls";
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();

                with_openai_turn_metadata_cassette(
                    "turn_termination_matrix/streaming_tool_turn_reports_tool_calls",
                    |client| async move {
                        let mut ecs = EcsAgent::new(
                            client.openai.chat(openai::GPT_4O_MINI),
                            TOOL_PREAMBLE,
                            1,
                        );
                        ecs.app
                            .world_mut()
                            .entity_mut(ecs.agent)
                            .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                        ecs.tool(Adder);
                        ecs_termination::install(&mut ecs, probe, None);
                        ecs.prompt_with_max_turns(TOOL_PROMPT, true, Some(3)).await;
                    },
                )
                .await;

                assert_eq!(
                    observed.first_reason(),
                    Some(FinishReason::ToolCalls),
                    "streaming must resolve the tool turn exactly as blocking does"
                );
                assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
                assert_recorded_wire_reason(SCENARIO, "tool_calls");
            }
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "openai_termination_streaming_tool_turn_reports_tool_calls",
                log,
            )
        },
    )
    .await
}
