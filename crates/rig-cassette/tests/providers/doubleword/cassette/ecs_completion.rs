//! Native ECS provider completions preserving the original cassette assertions.
use crate::doubleword::DEFAULT_MODEL;
use crate::doubleword::support::with_doubleword_cassette;
use crate::ecs_agent::EcsAgent;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_doubleword_cassette("agent/completion_smoke", |client| async move {
                let mut ecs = EcsAgent::new(client.completion(DEFAULT_MODEL), BASIC_PREAMBLE, 1);
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(DefaultMaxTurns(None));
                let response = ecs.prompt(BASIC_PROMPT, false).await;
                assert_nonempty_response(&response);
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "doubleword_completion_completion_smoke",
                log,
            )
        },
    )
    .await
}
