//! Native counterparts of the provider turn-termination matrix.

use super::super::support::with_doubleword_cassette;
use super::turn_termination_matrix::{
    CONCISE_PREAMBLE, MODEL, RETRY_PROMPT, ROOMY_CAP, TINY_CAP, TOOL_PREAMBLE, TOOL_PROMPT,
    assert_recorded_wire_reason, recorded_request_caps,
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_termination::{
        self, NativeEscalation as EscalateCapOnTruncation, NativeProbe as TurnTerminationProbe,
    },
    support::Adder,
};
use rig::completion::FinishReason;
use rig_ecs::agent::{AdditionalParams, MaxTokens, Temperature};

#[tokio::test]
async fn streaming_tool_turn_reports_tool_calls() {
    rig_test_support::goldens::world_golden_test(
        async {
            {
                const SCENARIO: &str =
                    "turn_termination_matrix/streaming_tool_turn_reports_tool_calls";
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();

                with_doubleword_cassette(
                    "turn_termination_matrix/streaming_tool_turn_reports_tool_calls",
                    |client| async move {
                        let mut ecs = EcsAgent::new(client.completion(MODEL), TOOL_PREAMBLE, 1);
                        ecs.app.world_mut().entity_mut(ecs.agent).insert((
                            Temperature(Some(0.0)),
                            MaxTokens(Some(ROOMY_CAP)),
                            AdditionalParams(Some(
                                serde_json::json!({ "reasoning_effort": "none" }),
                            )),
                        ));
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
                "doubleword_termination_streaming_tool_turn_reports_tool_calls",
                log,
            )
        },
    )
    .await
}

#[tokio::test]
async fn streaming_escalating_retry_reports_each_attempts_own_cap() {
    rig_test_support::goldens::world_golden_test(async {

    {
        const SCENARIO: &str =
            "turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap";
        let probe = TurnTerminationProbe::default();
        let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
        let observed = probe.clone();
        let escalations = escalate.clone();

        with_doubleword_cassette(
            "turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap",
            |client| async move {
                let mut ecs = EcsAgent::new(client.completion(MODEL), CONCISE_PREAMBLE, 1);
                ecs.app.world_mut().entity_mut(ecs.agent).insert((
                    Temperature(Some(0.0)),
                    MaxTokens(Some(64)),
                    AdditionalParams(Some(serde_json::json!({ "reasoning_effort": "none" }))),
                ));
                ecs_termination::install(&mut ecs, probe, Some(escalate));
                ecs.prompt_with_max_turns(RETRY_PROMPT, true, Some(2)).await;
            },
        )
        .await;

        assert_eq!(
            observed.observations(),
            vec![
                (Some(FinishReason::Length), Some(TINY_CAP)),
                (Some(FinishReason::Stop), Some(ROOMY_CAP)),
            ],
            "the streaming surface must escalate and report identically to blocking"
        );
        assert_eq!(escalations.escalations(), vec![ROOMY_CAP]);
        assert_eq!(recorded_request_caps(SCENARIO), vec![TINY_CAP, ROOMY_CAP]);
    }

}, |log| rig_test_support::goldens::world_golden_effects("doubleword_termination_streaming_escalating_retry_reports_each_attempts_own_cap", log)).await
}
