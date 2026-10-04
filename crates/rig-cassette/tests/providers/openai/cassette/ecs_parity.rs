//! Provider-adapter execution through native ECS, using the original agent
//! fixtures. The original tests remain independent baseline executions.

use rig::providers::openai;
use rig_ecs::agent::MaxTokens;

use super::super::support::with_openai_cassette;
use crate::{
    ecs_agent::EcsAgent,
    support::{
        Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
        assert_mentions_expected_number,
    },
};

#[tokio::test]
async fn streaming_tools_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_openai_cassette(
                "streaming_tools/streaming_tools_smoke",
                |client| async move {
                    let mut ecs = EcsAgent::new(
                        client.openai.completion(openai::GPT_4O),
                        STREAMING_TOOLS_PREAMBLE,
                        2,
                    );
                    ecs.tool(Adder);
                    ecs.tool(Subtract);
                    assert_mentions_expected_number(
                        &ecs.prompt(STREAMING_TOOLS_PROMPT, true).await,
                        -3,
                    );
                },
            )
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "openai_parity_streaming_tools_smoke",
                log,
            )
        },
    )
    .await
}

#[tokio::test]
async fn example_streaming_with_tools() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_openai_cassette(
                "streaming_tools/example_streaming_with_tools",
                |client| async move {
                    let mut ecs = EcsAgent::new(
                        client.openai.completion(openai::GPT_4O),
                        "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question and answer in a full sentence.",
                        2,
                    );
                    ecs.app
                        .world_mut()
                        .entity_mut(ecs.agent)
                        .insert(MaxTokens(Some(1024)));
                    ecs.tool(Adder);
                    ecs.tool(Subtract);
                    assert_mentions_expected_number(&ecs.prompt("Calculate 2 - 5", true).await, -3);
                },
            )
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "openai_parity_example_streaming_with_tools",
                log,
            )
        },
    )
    .await
}
