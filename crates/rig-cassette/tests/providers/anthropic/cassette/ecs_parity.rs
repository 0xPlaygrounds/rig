//! Provider-adapter execution through native ECS, using the original agent
//! fixtures. The original tests remain independent baseline executions.

use rig::providers::anthropic;
use rig_ecs::bus::Streamed;

use super::super::support::with_anthropic_cassette;
use crate::{
    ecs_agent::EcsAgent,
    support::{
        Adder, BASIC_PREAMBLE, BASIC_PROMPT, STREAMING_PREAMBLE, STREAMING_PROMPT,
        STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
        assert_mentions_expected_number, assert_nonempty_response,
    },
};

#[tokio::test]
async fn completion_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_anthropic_cassette("agent/completion_smoke", |client| async move {
                let mut ecs = EcsAgent::new(
                    client.completion(anthropic::completion::CLAUDE_SONNET_4_6),
                    BASIC_PREAMBLE,
                    1,
                );
                assert_nonempty_response(&ecs.prompt(BASIC_PROMPT, false).await);
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "anthropic_parity_completion_smoke",
                log,
            )
        },
    )
    .await
}

#[tokio::test]
async fn streaming_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_anthropic_cassette("streaming/streaming_smoke", |client| async move {
                let mut ecs = EcsAgent::new(
                    client.completion(anthropic::completion::CLAUDE_SONNET_4_6),
                    STREAMING_PREAMBLE,
                    1,
                );
                assert_nonempty_response(&ecs.prompt(STREAMING_PROMPT, true).await);
                let mut streams = ecs.app.world_mut().query::<&Streamed>();
                let stream = streams
                    .single(ecs.app.world())
                    .expect("one completion stream");
                let final_event = stream
                    .events
                    .iter()
                    .rev()
                    .find_map(|event| match event {
                        rig::streaming::StreamEvent::Final(final_event) => Some(final_event),
                        _ => None,
                    })
                    .expect("provider terminal stream record");
                assert_eq!(final_event.provider, "anthropic");
                assert!(final_event.usage.total_tokens.is_some_and(|n| n > 0));
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects("anthropic_parity_streaming_smoke", log)
        },
    )
    .await
}

#[tokio::test]
async fn streaming_tools_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_anthropic_cassette(
                "streaming_tools/streaming_tools_smoke",
                |client| async move {
                    let mut ecs = EcsAgent::new(
                        client.completion(anthropic::completion::CLAUDE_SONNET_4_6),
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
                "anthropic_parity_streaming_tools_smoke",
                log,
            )
        },
    )
    .await
}
