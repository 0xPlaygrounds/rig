//! Native ECS provider completions preserving the original cassette assertions.
use crate::ecs_agent::EcsAgent;
use crate::perplexity::support::with_perplexity_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::providers::perplexity;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_perplexity_cassette("agent/completion_smoke", |client| async move {
                let mut ecs =
                    EcsAgent::new(client.completion(perplexity::SONAR), BASIC_PREAMBLE, 1);
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(DefaultMaxTurns(None));
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(rig_ecs::agent::Temperature(Some(0.2)));
                let response = ecs.prompt(BASIC_PROMPT, false).await;
                assert_nonempty_response(&response);
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "perplexity_completion_completion_smoke",
                log,
            )
        },
    )
    .await
}
#[tokio::test]
async fn completion_with_perplexity_options() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_perplexity_cassette(
                "agent/completion_with_perplexity_options",
                |client| async move {
                    let mut ecs = EcsAgent::new(
                        client.completion(perplexity::SONAR),
                        "Answer briefly and include the date or time context if relevant.",
                        1,
                    );
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert(DefaultMaxTurns(None));
                    ecs.app.world_mut().entity_mut(ecs.agent).insert(
                        rig_ecs::agent::AdditionalParams(Some(serde_json::json!(
                            { "return_related_questions" : true, "search_context_size" :
                            "low" }
                        ))),
                    );
                    let response = ecs
                .prompt(
                    "Name one notable recent development in Rust programming language tooling.",
                    false,
                )
                .await;
                    assert_nonempty_response(&response);
                },
            )
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "perplexity_completion_completion_with_perplexity_options",
                log,
            )
        },
    )
    .await
}
