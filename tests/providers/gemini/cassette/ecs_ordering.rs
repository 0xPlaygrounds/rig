//! Native Gemini streamed tool-ordering contract.

use super::super::support::with_gemini_cassette;
use crate::{
    ecs_agent::EcsAgent,
    ecs_observation::{install_observers, observation},
    support::{
        ALPHA_SIGNAL_OUTPUT, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
        assert_tool_call_precedes_later_text,
    },
};
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig::{prelude::*, providers::gemini};
use rig_ecs::agent::AdditionalParams;

#[tokio::test]
async fn streaming_tools_emit_tool_call_before_later_text() {
    with_gemini_cassette(
        "streaming_tools/streaming_tools_emit_tool_call_before_later_text",
        |client| async move {
            let mut ecs = EcsAgent::new(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                ORDERED_TOOL_STREAM_PREAMBLE,
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
            ecs.tool(AlphaSignal);
            install_observers(&mut ecs);
            ecs.prompt_with_max_turns(ORDERED_TOOL_STREAM_PROMPT, true, Some(5))
                .await;
            assert_tool_call_precedes_later_text(
                observation(&ecs),
                "lookup_harbor_label",
                &[ALPHA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_tools_surface_two_distinct_tool_calls_before_final_answer() {
    use crate::support::{
        BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
        assert_two_tool_roundtrip_contract,
    };
    with_gemini_cassette(
        "streaming_tools/streaming_tools_surface_two_distinct_tool_calls_before_final_answer",
        |client| async move {
            let mut ecs = EcsAgent::new(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                TWO_TOOL_STREAM_PREAMBLE,
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
            ecs.tool(AlphaSignal);
            ecs.tool(BetaSignal);
            install_observers(&mut ecs);
            ecs.prompt_with_max_turns(TWO_TOOL_STREAM_PROMPT, true, Some(8))
                .await;
            assert_two_tool_roundtrip_contract(
                observation(&ecs),
                &["lookup_harbor_label", "lookup_orchard_label"],
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}
