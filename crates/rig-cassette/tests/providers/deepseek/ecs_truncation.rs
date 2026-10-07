//! A truncated provider tool call whose arguments never parse never runs
//! under native agent execution; one cut before its arguments begin states
//! an empty object, as pi reads it.
use super::truncation_matrix::*;
use crate::{ecs_agent::EcsAgent, ecs_observation};
use rig_ecs::{
    agent::{DefaultMaxTurns, Failure, MaxTokens, Options},
    systems::RunCommands,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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
                        Options(non_thinking()),
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
                    // A call cut before its first argument token states an empty
                    // object, as pi's tolerant parse reads blank arguments, so the
                    // zero-argument tool runs once; the one-turn budget then ends
                    // the run.
                    match ecs.wait_for_outcome(run).await {
                        Err(Failure::MaxTurns { limit: 1 }) => {}
                        other => {
                            panic!("the empty call runs, then the budget ends the run: {other:?}")
                        }
                    }
                    assert_eq!(
                        invocations.load(Ordering::SeqCst),
                        1,
                        "blank arguments are an empty object"
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
