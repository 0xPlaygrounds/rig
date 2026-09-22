//! Truncated provider tool calls never dispatch through native agent execution.
use super::truncation_matrix::*;
use crate::{ecs_agent::EcsAgent, ecs_observation};
use rig::completion::FinishReason;
use rig::error::ErrorKind;
use rig_ecs::{
    agent::{AdditionalParams, DefaultMaxTurns, Failure, MaxTokens},
    systems::RunCommands,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[tokio::test]
async fn agent_blocking_truncated_call_is_not_invoked() {
    rig_test_support::goldens::world_golden_test(
        async {
            const SCENARIO: &str = "truncation_matrix/agent_blocking_truncated_call_is_not_invoked";
            super::support::with_deepseek_truncation_cassette_result(
                "truncation_matrix/agent_blocking_truncated_call_is_not_invoked",
                |client| async move {
                    let invocations = Arc::new(AtomicUsize::new(0));
                    let mut ecs = EcsAgent::new(client.completion(MODEL), TOOL_PREAMBLE, 1);
                    ecs.app.world_mut().entity_mut(ecs.agent).insert((
                        DefaultMaxTurns(Some(1)),
                        MaxTokens(Some(32)),
                        AdditionalParams(Some(non_thinking_params())),
                    ));
                    ecs.tool(FileReport {
                        invocations: invocations.clone(),
                    });
                    ecs_observation::install_observers(&mut ecs);
                    let run =
                        ecs.app
                            .world_mut()
                            .spawn_run(ecs.agent, &[], INCIDENT_PROMPT, false, None);
                    // The truncated turn reaches the loop and, answerless under
                    // `Length`, fails as rig-agent's does (CONTRACT §4): a response
                    // error naming the reason, never the provider's reply.
                    match ecs.wait_for_outcome(run).await {
                        Err(Failure::Provider(report)) => {
                            assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
                            assert!(
                                report
                                    .message
                                    .contains(&FinishReason::Length.no_answer_message()),
                                "{report:?}"
                            );
                        }
                        other => panic!("the answerless truncated turn fails the run: {other:?}"),
                    }
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
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "deepseek_truncation_agent_blocking_truncated_call_is_not_invoked",
                log,
            )
        },
    )
    .await
}

#[tokio::test]
async fn agent_streaming_truncated_call_is_not_invoked() {
    rig_test_support::goldens::world_golden_test(
        async {
            const SCENARIO: &str =
                "truncation_matrix/agent_streaming_truncated_call_is_not_invoked";
            super::support::with_deepseek_truncation_cassette_result(
                "truncation_matrix/agent_streaming_truncated_call_is_not_invoked",
                |client| async move {
                    let invocations = Arc::new(AtomicUsize::new(0));
                    let mut ecs = EcsAgent::new(client.completion(MODEL), TOOL_PREAMBLE, 1);
                    ecs.app.world_mut().entity_mut(ecs.agent).insert((
                        DefaultMaxTurns(None),
                        MaxTokens(Some(32)),
                        AdditionalParams(Some(non_thinking_params())),
                    ));
                    ecs.tool(FileReport {
                        invocations: invocations.clone(),
                    });
                    ecs_observation::install_observers(&mut ecs);
                    let run = ecs.app.world_mut().spawn_run(
                        ecs.agent,
                        &[],
                        INCIDENT_PROMPT,
                        true,
                        Some(1),
                    );
                    // The truncated turn reaches the loop and, answerless under
                    // `Length`, fails as rig-agent's does (CONTRACT §4): a response
                    // error naming the reason, never the provider's reply.
                    match ecs.wait_for_outcome(run).await {
                        Err(Failure::Provider(report)) => {
                            assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
                            assert!(
                                report
                                    .message
                                    .contains(&FinishReason::Length.no_answer_message()),
                                "{report:?}"
                            );
                        }
                        other => panic!("the answerless truncated turn fails the run: {other:?}"),
                    }
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
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "deepseek_truncation_agent_streaming_truncated_call_is_not_invoked",
                log,
            )
        },
    )
    .await
}

#[tokio::test]
async fn agent_blocking_empty_arguments_on_length_are_not_invoked() {
    rig_test_support::goldens::world_golden_test(
        async {
            const SCENARIO: &str =
                "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked";
            super::support::with_deepseek_truncation_cassette_result(
                "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked",
                |client| async move {
                    let invocations = Arc::new(AtomicUsize::new(0));
                    let mut ecs = EcsAgent::new(client.completion(MODEL), TOOL_PREAMBLE, 1);
                    ecs.app.world_mut().entity_mut(ecs.agent).insert((
                        DefaultMaxTurns(Some(1)),
                        MaxTokens(Some(16)),
                        AdditionalParams(Some(non_thinking_params())),
                    ));
                    ecs.tool(ZeroArgumentFileReport {
                        invocations: invocations.clone(),
                    });
                    ecs_observation::install_observers(&mut ecs);
                    let run =
                        ecs.app
                            .world_mut()
                            .spawn_run(ecs.agent, &[], INCIDENT_PROMPT, false, None);
                    // The truncated turn reaches the loop and, answerless under
                    // `Length`, fails as rig-agent's does (CONTRACT §4): a response
                    // error naming the reason, never the provider's reply.
                    match ecs.wait_for_outcome(run).await {
                        Err(Failure::Provider(report)) => {
                            assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
                            assert!(
                                report
                                    .message
                                    .contains(&FinishReason::Length.no_answer_message()),
                                "{report:?}"
                            );
                        }
                        other => panic!("the answerless truncated turn fails the run: {other:?}"),
                    }
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
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "deepseek_truncation_agent_blocking_empty_arguments_on_length_are_not_invoked",
                log,
            )
        },
    )
    .await
}

#[tokio::test]
async fn agent_streaming_empty_arguments_on_length_are_not_invoked() {
    rig_test_support::goldens::world_golden_test(
        async {
            const SCENARIO: &str =
                "truncation_matrix/agent_streaming_empty_arguments_on_length_are_not_invoked";
            super::support::with_deepseek_truncation_cassette_result(
                "truncation_matrix/agent_streaming_empty_arguments_on_length_are_not_invoked",
                |client| async move {
                    let invocations = Arc::new(AtomicUsize::new(0));
                    let mut ecs = EcsAgent::new(client.completion(MODEL), TOOL_PREAMBLE, 1);
                    ecs.app.world_mut().entity_mut(ecs.agent).insert((
                        DefaultMaxTurns(None),
                        MaxTokens(Some(16)),
                        AdditionalParams(Some(non_thinking_params())),
                    ));
                    ecs.tool(ZeroArgumentFileReport {
                        invocations: invocations.clone(),
                    });
                    ecs_observation::install_observers(&mut ecs);
                    let run = ecs.app.world_mut().spawn_run(
                        ecs.agent,
                        &[],
                        INCIDENT_PROMPT,
                        true,
                        Some(1),
                    );
                    // The truncated turn reaches the loop and, answerless under
                    // `Length`, fails as rig-agent's does (CONTRACT §4): a response
                    // error naming the reason, never the provider's reply.
                    match ecs.wait_for_outcome(run).await {
                        Err(Failure::Provider(report)) => {
                            assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
                            assert!(
                                report
                                    .message
                                    .contains(&FinishReason::Length.no_answer_message()),
                                "{report:?}"
                            );
                        }
                        other => panic!("the answerless truncated turn fails the run: {other:?}"),
                    }
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
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "deepseek_truncation_agent_streaming_empty_arguments_on_length_are_not_invoked",
                log,
            )
        },
    )
    .await
}
