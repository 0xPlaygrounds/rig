//! Native counterparts of the provider turn-termination matrix.

use super::super::support::with_openai_turn_metadata_cassette;
use super::turn_termination_matrix::{
    CONCISE_PREAMBLE, RETRY_PROMPT, ROOMY_CAP, SHORT_PROMPT, TINY_CAP, TOOL_PREAMBLE, TOOL_PROMPT,
    TRUNCATING_PROMPT, assert_recorded_request_cap, assert_recorded_wire_reason,
    recorded_request_caps, recorded_wire_reasons,
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_termination::{
        self, NativeEscalation as EscalateCapOnTruncation, NativeProbe as TurnTerminationProbe,
    },
    support::Adder,
};
use rig::completion::FinishReason;
use rig::prelude::*;
use rig::providers::openai;
use rig_ecs::agent::{MaxTokens, Temperature};

#[tokio::test]
async fn blocking_truncated_turn_reports_length_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
                    CONCISE_PREAMBLE,
                    1,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(TINY_CAP))));
                ecs_termination::install(&mut ecs, probe, None);
                ecs.prompt_with_max_turns(TRUNCATING_PROMPT, false, None)
                    .await;
            },
        )
        .await;

        assert_eq!(
            observed.first_reason(),
            Some(FinishReason::Length),
            "the wire `length` must reach the hook as FinishReason::Length"
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
        assert_recorded_wire_reason(SCENARIO, "length");
        assert_recorded_request_cap(SCENARIO, TINY_CAP);
    }
}

#[tokio::test]
async fn streaming_truncated_turn_reports_length_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
                    CONCISE_PREAMBLE,
                    1,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(TINY_CAP))));
                ecs_termination::install(&mut ecs, probe, None);
                ecs.prompt_with_max_turns(TRUNCATING_PROMPT, true, None)
                    .await;
            },
        )
        .await;

        assert_eq!(
            observed.first_reason(),
            Some(FinishReason::Length),
            "the streaming surface must report the same reason as the blocking one"
        );
        assert_eq!(observed.first_max_tokens(), Some(TINY_CAP));
        assert_recorded_wire_reason(SCENARIO, "length");
        assert_recorded_request_cap(SCENARIO, TINY_CAP);
    }
}

#[tokio::test]
async fn blocking_completed_turn_reports_stop_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
                    CONCISE_PREAMBLE,
                    1,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                ecs_termination::install(&mut ecs, probe, None);
                ecs.prompt_with_max_turns(SHORT_PROMPT, false, None).await;
            },
        )
        .await;

        assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
        assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
        assert!(
            !observed
                .first_reason()
                .is_some_and(|reason| reason.truncated_output()),
            "a completed turn must not satisfy the retry predicate"
        );
        assert_recorded_wire_reason(SCENARIO, "stop");
    }
}

#[tokio::test]
async fn streaming_completed_turn_reports_stop_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
                    CONCISE_PREAMBLE,
                    1,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                ecs_termination::install(&mut ecs, probe, None);
                ecs.prompt_with_max_turns(SHORT_PROMPT, true, None).await;
            },
        )
        .await;

        assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
        assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
        assert_recorded_wire_reason(SCENARIO, "stop");
    }
}

#[tokio::test]
async fn blocking_tool_turn_reports_tool_calls() {
    {
        const SCENARIO: &str = "turn_termination_matrix/blocking_tool_turn_reports_tool_calls";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/blocking_tool_turn_reports_tool_calls",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
                    TOOL_PREAMBLE,
                    1,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(ROOMY_CAP))));
                ecs.tool(Adder);
                ecs_termination::install(&mut ecs, probe, None);
                ecs.prompt_with_max_turns(TOOL_PROMPT, false, Some(3)).await;
            },
        )
        .await;

        assert_eq!(
            observed.first_reason(),
            Some(FinishReason::ToolCalls),
            "the turn that issued the tool call must read as ToolCalls"
        );
        assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
        assert!(
            !observed
                .first_reason()
                .is_some_and(|reason| reason.truncated_output()),
            "a tool turn must not satisfy the retry predicate"
        );
        assert_recorded_wire_reason(SCENARIO, "tool_calls");
    }
}

#[tokio::test]
async fn streaming_tool_turn_reports_tool_calls() {
    {
        const SCENARIO: &str = "turn_termination_matrix/streaming_tool_turn_reports_tool_calls";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/streaming_tool_turn_reports_tool_calls",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
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
}

#[tokio::test]
async fn blocking_escalating_retry_reports_each_attempts_own_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/blocking_escalating_retry_reports_each_attempts_own_cap";
        let probe = TurnTerminationProbe::default();
        let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
        let observed = probe.clone();
        let escalations = escalate.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/blocking_escalating_retry_reports_each_attempts_own_cap",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
                    CONCISE_PREAMBLE,
                    1,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(64))));
                ecs_termination::install(&mut ecs, probe, Some(escalate));
                ecs.prompt_with_max_turns(RETRY_PROMPT, false, Some(2))
                    .await;
            },
        )
        .await;

        assert_eq!(
            observed.observations(),
            vec![
                (Some(FinishReason::Length), Some(TINY_CAP)),
                (Some(FinishReason::Stop), Some(ROOMY_CAP)),
            ],
            "each attempt must report its own post-patch cap, never the agent's baseline of 64"
        );
        assert_eq!(escalations.escalations(), vec![ROOMY_CAP]);
        assert_eq!(escalations.retries(), 1);

        // ...and the recorded traffic corroborates it: two calls, the caps the
        // hook chose, and the two reasons in order.
        assert_eq!(recorded_request_caps(SCENARIO), vec![TINY_CAP, ROOMY_CAP]);
        assert_eq!(
            recorded_wire_reasons(SCENARIO),
            vec!["length".to_owned(), "stop".to_owned()]
        );
    }
}

#[tokio::test]
async fn streaming_escalating_retry_reports_each_attempts_own_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap";
        let probe = TurnTerminationProbe::default();
        let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
        let observed = probe.clone();
        let escalations = escalate.clone();

        with_openai_turn_metadata_cassette(
            "turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap",
            |client| async move {
                let mut ecs = EcsAgent::new(
                    client
                        .completions_api()
                        .completion_model(openai::GPT_4O_MINI),
                    CONCISE_PREAMBLE,
                    1,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert((Temperature(Some(0.0)), MaxTokens(Some(64))));
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
}
