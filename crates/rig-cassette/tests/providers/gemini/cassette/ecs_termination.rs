//! Native counterparts of the provider turn-termination matrix.

use super::super::support::with_gemini_turn_metadata_cassette;
use super::turn_termination_matrix::{
    CONCISE_PREAMBLE, TINY_CAP, TRUNCATING_PROMPT, assert_recorded_request_cap,
    assert_recorded_wire_reason, no_thinking,
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_termination::{self, NativeProbe as TurnTerminationProbe},
};
use rig::completion::FinishReason;
use rig::providers::gemini;
use rig_ecs::agent::{AdditionalParams, MaxTokens, Temperature};

#[tokio::test]
async fn blocking_truncated_turn_reports_length_and_cap() {
    rig_test_support::goldens::world_golden_test(
        async {
            {
                const SCENARIO: &str =
                    "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap";
                let probe = TurnTerminationProbe::default();
                let observed = probe.clone();

                with_gemini_turn_metadata_cassette(
                    "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap",
                    |client| async move {
                        let mut ecs = EcsAgent::new(
                            client.completion(gemini::completion::GEMINI_2_5_FLASH),
                            CONCISE_PREAMBLE,
                            1,
                        );
                        ecs.app
                            .world_mut()
                            .entity_mut(ecs.agent)
                            .insert((Temperature(Some(0.0)), MaxTokens(Some(TINY_CAP))));
                        ecs.app
                            .world_mut()
                            .entity_mut(ecs.agent)
                            .insert(AdditionalParams(Some(no_thinking())));
                        ecs_termination::install(&mut ecs, probe, None);
                        ecs.prompt_with_max_turns(TRUNCATING_PROMPT, false, None)
                            .await;
                    },
                )
                .await;

                assert_eq!(
                    observed.first_reason(),
                    Some(FinishReason::Length),
                    "the wire `MAX_TOKENS` must reach the hook as FinishReason::Length"
                );
                assert_eq!(
                    observed.first_max_tokens(),
                    Some(TINY_CAP),
                    "the hook must report the cap this attempt actually ran under"
                );
                assert!(
                    observed
                        .first_reason()
                        .is_some_and(|reason| reason.truncated_output()),
                    "a truncated turn must satisfy the portable retry predicate"
                );
                assert_recorded_wire_reason(SCENARIO, "MAX_TOKENS");
                assert_recorded_request_cap(SCENARIO, TINY_CAP);
            }
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "gemini_termination_blocking_truncated_turn_reports_length_and_cap",
                log,
            )
        },
    )
    .await
}
