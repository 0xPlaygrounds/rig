//! Truncated provider tool calls never dispatch through native agent execution.
use super::truncation_matrix::*;
use crate::{ecs_agent::EcsAgent, ecs_observation};
use rig::prelude::*;
use rig_ecs::agent::{AdditionalParams, DefaultMaxTurns, MaxTokens};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[tokio::test]
async fn agent_blocking_truncated_call_is_not_invoked() {
    const SCENARIO: &str = "truncation_matrix/agent_blocking_truncated_call_is_not_invoked";
    super::support::with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_blocking_truncated_call_is_not_invoked",
        |client| async move {
            let invocations = Arc::new(AtomicUsize::new(0));
            let mut ecs = EcsAgent::new(client.completion_model(MODEL), TOOL_PREAMBLE, 1);
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                DefaultMaxTurns(Some(1)),
                MaxTokens(Some(32)),
                AdditionalParams(Some(non_thinking_params())),
            ));
            ecs.tool(FileReport {
                invocations: invocations.clone(),
            });
            ecs_observation::install_observers(&mut ecs);
            ecs.prompt_with_max_turns(INCIDENT_PROMPT, false, None)
                .await;
            assert!(
                ecs_observation::observation(&ecs).tool_calls.is_empty(),
                "truncated calls must not materialise"
            );
            assert_eq!(
                invocations.load(Ordering::SeqCst),
                0,
                "an incomplete call must never be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("native truncated-tool safety case should replay");
    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn agent_streaming_truncated_call_is_not_invoked() {
    const SCENARIO: &str = "truncation_matrix/agent_streaming_truncated_call_is_not_invoked";
    super::support::with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_streaming_truncated_call_is_not_invoked",
        |client| async move {
            let invocations = Arc::new(AtomicUsize::new(0));
            let mut ecs = EcsAgent::new(client.completion_model(MODEL), TOOL_PREAMBLE, 1);
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                DefaultMaxTurns(None),
                MaxTokens(Some(32)),
                AdditionalParams(Some(non_thinking_params())),
            ));
            ecs.tool(FileReport {
                invocations: invocations.clone(),
            });
            ecs_observation::install_observers(&mut ecs);
            ecs.prompt_with_max_turns(INCIDENT_PROMPT, true, Some(1))
                .await;
            assert!(
                ecs_observation::observation(&ecs).tool_calls.is_empty(),
                "truncated calls must not materialise"
            );
            assert_eq!(
                invocations.load(Ordering::SeqCst),
                0,
                "an incomplete call must never be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("native truncated-tool safety case should replay");
    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn agent_blocking_empty_arguments_on_length_are_not_invoked() {
    const SCENARIO: &str =
        "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked";
    super::support::with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked",
        |client| async move {
            let invocations = Arc::new(AtomicUsize::new(0));
            let mut ecs = EcsAgent::new(client.completion_model(MODEL), TOOL_PREAMBLE, 1);
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                DefaultMaxTurns(Some(1)),
                MaxTokens(Some(16)),
                AdditionalParams(Some(non_thinking_params())),
            ));
            ecs.tool(ZeroArgumentFileReport {
                invocations: invocations.clone(),
            });
            ecs_observation::install_observers(&mut ecs);
            ecs.prompt_with_max_turns(INCIDENT_PROMPT, false, None)
                .await;
            assert!(
                ecs_observation::observation(&ecs).tool_calls.is_empty(),
                "truncated calls must not materialise"
            );
            assert_eq!(
                invocations.load(Ordering::SeqCst),
                0,
                "an incomplete call must never be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("native truncated-tool safety case should replay");
    assert_eq!(recorded_blocking_arguments(SCENARIO), vec![String::new()]);
    assert_eq!(recorded_blocking_finish_reason(SCENARIO), "length");
}

#[tokio::test]
async fn agent_streaming_empty_arguments_on_length_are_not_invoked() {
    const SCENARIO: &str =
        "truncation_matrix/agent_streaming_empty_arguments_on_length_are_not_invoked";
    super::support::with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_streaming_empty_arguments_on_length_are_not_invoked",
        |client| async move {
            let invocations = Arc::new(AtomicUsize::new(0));
            let mut ecs = EcsAgent::new(client.completion_model(MODEL), TOOL_PREAMBLE, 1);
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                DefaultMaxTurns(None),
                MaxTokens(Some(16)),
                AdditionalParams(Some(non_thinking_params())),
            ));
            ecs.tool(ZeroArgumentFileReport {
                invocations: invocations.clone(),
            });
            ecs_observation::install_observers(&mut ecs);
            ecs.prompt_with_max_turns(INCIDENT_PROMPT, true, Some(1))
                .await;
            assert!(
                ecs_observation::observation(&ecs).tool_calls.is_empty(),
                "truncated calls must not materialise"
            );
            assert_eq!(
                invocations.load(Ordering::SeqCst),
                0,
                "an incomplete call must never be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("native truncated-tool safety case should replay");
    assert_eq!(recorded_stream_arguments(SCENARIO), vec![String::new()]);
    assert_eq!(recorded_stream_finish_reason(SCENARIO), "length");
}
