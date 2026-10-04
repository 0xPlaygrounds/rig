//! Native OpenAI streamed tool-ordering contract and its negative control.

use super::super::support::with_openai_cassette;
use crate::{
    ecs_agent::EcsAgent,
    ecs_observation::{install_observers, observation},
    support::{
        ALPHA_SIGNAL_OUTPUT, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
        assert_tool_call_precedes_later_text,
    },
};
use rig::providers::openai;

#[tokio::test]
async fn responses_stream_preserves_tool_result_flow() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_openai_cassette(
                "streaming_tools/responses_stream_preserves_tool_result_flow",
                |client| async move {
                    let mut ecs = EcsAgent::new(
                        client.openai.completion(openai::GPT_4O),
                        ORDERED_TOOL_STREAM_PREAMBLE,
                        1,
                    );
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
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "openai_ordering_responses_stream_preserves_tool_result_flow",
                log,
            )
        },
    )
    .await
}
