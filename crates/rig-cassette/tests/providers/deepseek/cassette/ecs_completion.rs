//! Native ECS provider completions preserving the original cassette assertions.
use crate::deepseek::support::with_deepseek_cassette;
use crate::ecs_agent::EcsAgent;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::providers::deepseek;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_deepseek_cassette("agent/completion_smoke", |client| async move {
                let mut ecs = EcsAgent::new(
                    client.completion(deepseek::DEEPSEEK_V4_FLASH),
                    BASIC_PREAMBLE,
                    1,
                );
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
                "deepseek_completion_completion_smoke",
                log,
            )
        },
    )
    .await
}
