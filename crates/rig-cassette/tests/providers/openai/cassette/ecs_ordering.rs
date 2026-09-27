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

/// Synthetic negative control: text before the tool must remain visible to
/// the ordering assertion even when the eventual answer is correct.
#[tokio::test]
#[should_panic(expected = "expected a tool call before later text")]
async fn pre_tool_text_is_not_hidden_by_final_answer() {
    rig_test_support::goldens::capture_world_programs(async {
        use rig::test_utils::{MockCompletionModel, MockStreamEvent, mock_final};
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("premature text"),
                MockStreamEvent::tool_call("call", "lookup_harbor_label", serde_json::json!({})),
                MockStreamEvent::FinalResponse(mock_final(rig::completion::Usage::default())),
            ],
            vec![
                MockStreamEvent::text(ALPHA_SIGNAL_OUTPUT),
                MockStreamEvent::FinalResponse(mock_final(rig::completion::Usage::default())),
            ],
        ]);
        let mut ecs = EcsAgent::new(model, ORDERED_TOOL_STREAM_PREAMBLE, 1);
        ecs.tool(AlphaSignal);
        install_observers(&mut ecs);
        ecs.prompt_with_max_turns(ORDERED_TOOL_STREAM_PROMPT, true, Some(5))
            .await;
        rig_test_support::goldens::world_golden_effects(
            "openai_ordering_pre_tool_text_is_not_hidden_by_final_answer",
            &ecs.effect_log(),
        );
        assert_tool_call_precedes_later_text(
            observation(&ecs),
            "lookup_harbor_label",
            &[ALPHA_SIGNAL_OUTPUT],
        );
    })
    .await
}
