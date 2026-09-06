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
use rig::{prelude::*, providers::openai};

#[tokio::test]
async fn responses_stream_preserves_tool_result_flow() {
    with_openai_cassette(
        "streaming_tools/responses_stream_preserves_tool_result_flow",
        |client| async move {
            let mut ecs = EcsAgent::new(
                client.completion_model(openai::GPT_4O),
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
}

/// Synthetic negative control: text before the tool must remain visible to
/// the ordering assertion even when the eventual answer is correct.
#[tokio::test]
#[should_panic(expected = "expected a tool call before later text")]
async fn pre_tool_text_is_not_hidden_by_final_answer() {
    use rig::test_utils::{MockCompletionModel, MockStreamEvent, mock_final};
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("premature text"),
            MockStreamEvent::tool_call("call", "lookup_harbor_label", serde_json::json!({})),
            MockStreamEvent::FinalResponse(mock_final(rig::completion::Usage::new())),
        ],
        vec![
            MockStreamEvent::text(ALPHA_SIGNAL_OUTPUT),
            MockStreamEvent::FinalResponse(mock_final(rig::completion::Usage::new())),
        ],
    ]);
    let mut ecs = EcsAgent::new(model, ORDERED_TOOL_STREAM_PREAMBLE, 1);
    ecs.tool(AlphaSignal);
    install_observers(&mut ecs);
    ecs.prompt_with_max_turns(ORDERED_TOOL_STREAM_PROMPT, true, Some(5))
        .await;
    assert_tool_call_precedes_later_text(
        observation(&ecs),
        "lookup_harbor_label",
        &[ALPHA_SIGNAL_OUTPUT],
    );
}
