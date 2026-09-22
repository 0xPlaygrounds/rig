//! Native ECS provider completions preserving the original cassette assertions.
use crate::ecs_agent::EcsAgent;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use crate::xai::support::with_xai_cassette;
use rig::providers::xai;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_xai_cassette("agent/completion_smoke", |client| async move {
                let mut ecs = EcsAgent::new(client.completion(xai::GROK_3_MINI), BASIC_PREAMBLE, 1);
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
            rig_test_support::goldens::world_golden_effects("xai_completion_completion_smoke", log)
        },
    )
    .await
}
