//! Native ECS provider completions preserving the original cassette assertions.
use crate::bedrock::support::with_bedrock_cassette;
use crate::ecs_agent::EcsAgent;
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number,
};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::bedrock;
use rig::prelude::*;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_bedrock_cassette("agent/completion_smoke", |client| async move {
                let mut ecs = EcsAgent::new(
                    client.completion(bedrock::completion::AMAZON_NOVA_LITE),
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
                "bedrock_completion_completion_smoke",
                log,
            )
        },
    )
    .await
}
#[tokio::test]
async fn tool_roundtrip_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_bedrock_cassette("agent/tool_roundtrip_smoke", |client| async move {
                let mut ecs = EcsAgent::new(
                    client.completion(bedrock::completion::AMAZON_NOVA_LITE),
                    STREAMING_TOOLS_PREAMBLE,
                    2,
                );
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(rig_ecs::agent::MaxTokens(Some(1024)));
                ecs.tool(Adder);
                ecs.tool(Subtract);
                let response = ecs.prompt(STREAMING_TOOLS_PROMPT, false).await;
                assert_mentions_expected_number(&response, -3);
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "bedrock_completion_tool_roundtrip_smoke",
                log,
            )
        },
    )
    .await
}
#[tokio::test]
async fn prompt_caching_completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_bedrock_cassette(
                "agent/prompt_caching_completion_smoke",
                |client| async move {
                    let model = client
                        .completion(bedrock::completion::AMAZON_NOVA_LITE)
                        .with_prompt_caching();
                    let mut ecs = EcsAgent::new(model, BASIC_PREAMBLE, 1);
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert(DefaultMaxTurns(None));
                    let response = ecs.prompt(BASIC_PROMPT, false).await;
                    assert_nonempty_response(&response);
                },
            )
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "bedrock_completion_prompt_caching_completion_smoke",
                log,
            )
        },
    )
    .await
}
