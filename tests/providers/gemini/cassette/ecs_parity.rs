//! Provider-adapter execution through native ECS, using the original agent
//! fixtures. The original tests remain independent baseline executions.

use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig::{prelude::*, providers::gemini};
use rig_ecs::{
    agent::{AdditionalParams, MaxTokens, Temperature},
    bus::Streamed,
};

use super::super::support::with_gemini_cassette;
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
    with_gemini_cassette("agent/completion_smoke", |client| async move {
        let mut ecs = EcsAgent::new(
            client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
            BASIC_PREAMBLE,
            1,
        );
        assert_nonempty_response(&ecs.prompt(BASIC_PROMPT, false).await);
    })
    .await;
}

#[tokio::test]
async fn streaming_smoke() {
    with_gemini_cassette("streaming/streaming_smoke", |client| async move {
        let mut ecs = EcsAgent::new(
            client.completion_model(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            STREAMING_PREAMBLE,
            1,
        );
        let config = GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                thinking_budget: None,
                thinking_level: Some(ThinkingLevel::Medium),
                include_thoughts: Some(true),
            }),
            ..Default::default()
        };
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(AdditionalParams(Some(
                serde_json::to_value(AdditionalParameters::default().with_config(config))
                    .expect("thinking configuration"),
            )));
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
        assert!(final_event.usage.total_tokens > 0);
    })
    .await;
}

#[tokio::test]
async fn streaming_tools_smoke() {
    with_gemini_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let mut ecs = EcsAgent::new(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                STREAMING_TOOLS_PREAMBLE,
                1,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(AdditionalParams(Some(
                    serde_json::to_value(
                        AdditionalParameters::default().with_config(GenerationConfig::default()),
                    )
                    .expect("tool configuration"),
                )));
            ecs.tool(Adder);
            ecs.tool(Subtract);
            assert_mentions_expected_number(
                &ecs.prompt_with_max_turns(STREAMING_TOOLS_PROMPT, true, Some(3))
                    .await,
                -3,
            );
        },
    )
    .await;
}

#[tokio::test]
async fn example_streaming_prompt() {
    with_gemini_cassette("streaming/example_streaming_prompt", |client| async move {
        let mut ecs = EcsAgent::new(
            client.completion_model(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            "Be precise and concise.",
            1,
        );
        let config = GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                thinking_budget: None,
                thinking_level: Some(ThinkingLevel::Medium),
                include_thoughts: Some(true),
            }),
            ..Default::default()
        };
        ecs.app.world_mut().entity_mut(ecs.agent).insert((
            Temperature(Some(0.5)),
            AdditionalParams(Some(
                serde_json::to_value(AdditionalParameters::default().with_config(config))
                    .expect("thinking configuration"),
            )),
        ));
        assert_nonempty_response(
            &ecs.prompt(
                "When and where and what type is the next solar eclipse?",
                true,
            )
            .await,
        );
    })
    .await;
}

#[tokio::test]
async fn example_streaming_with_tools() {
    with_gemini_cassette(
        "streaming_tools/example_streaming_with_tools",
        |client| async move {
            let mut ecs = EcsAgent::new(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question.",
                1,
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                MaxTokens(Some(1024)),
                AdditionalParams(Some(
                    serde_json::to_value(
                        AdditionalParameters::default().with_config(GenerationConfig::default()),
                    )
                    .expect("tool configuration"),
                )),
            ));
            ecs.tool(Adder);
            ecs.tool(Subtract);
            assert_mentions_expected_number(
                &ecs.prompt_with_max_turns("Calculate 2 - 5", true, Some(3))
                    .await,
                -3,
            );
        },
    )
    .await;
}
