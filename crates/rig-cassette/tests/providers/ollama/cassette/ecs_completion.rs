//! Native ECS provider completions preserving the original cassette assertions.
use crate::ecs_agent::EcsAgent;
use crate::ollama::support::with_ollama_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig_ecs::agent::DefaultMaxTurns;
const MODEL: &str = "qwen3:4b";
#[tokio::test]
async fn completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_ollama_cassette("agent/completion_smoke", |client| async move {
                let mut ecs = EcsAgent::new(client.completion(MODEL), BASIC_PREAMBLE, 1);
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(DefaultMaxTurns(None));
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(rig_ecs::agent::AdditionalParams(Some(
                        serde_json::json!({ "reasoning_effort": "none" }),
                    )));
                let response = ecs.prompt(BASIC_PROMPT, false).await;
                assert_nonempty_response(&response);
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "ollama_completion_completion_smoke",
                log,
            )
        },
    )
    .await
}
/// Guards the token limit on the wire: the recorded request carries
/// `"max_tokens":24`, and the cassette matcher compares request bodies, so a
/// request that dropped it would stop matching. The recorded response
/// finishes on `"length"`, the daemon confirming it honored the budget.
#[tokio::test]
async fn completion_respects_max_tokens() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_ollama_cassette("agent/max_tokens", |client| async move {
                let mut ecs = EcsAgent::new(client.completion(MODEL), BASIC_PREAMBLE, 1);
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(DefaultMaxTurns(None));
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(rig_ecs::agent::MaxTokens(Some(24)));
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(rig_ecs::agent::AdditionalParams(Some(
                        serde_json::json!({ "reasoning_effort": "none" }),
                    )));
                let response = ecs.prompt(BASIC_PROMPT, false).await;
                assert_nonempty_response(&response);
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "ollama_completion_completion_respects_max_tokens",
                log,
            )
        },
    )
    .await
}
