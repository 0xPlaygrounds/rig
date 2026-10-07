//! The ECS contract matrix on the OpenAI Chat Completions wire (`gpt-5-mini`, no temperature: the gpt-5 family takes only its default; the route `gpt-5-nano`): every cell of
//! `tests/common/ecs_matrix/cells.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the same recording as its producer in
//! `corpus_matrix_chat.rs`, and asserted against the cell,
//! then by its graph, its cut and its despawn (the driver is
//! `tests/common/ecs_matrix/world.rs`). This file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, world::run_world};

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: wire_matrix_case;
    #[ignore = "gpt-5-mini low reported zero reasoning tokens on the second turn in all three attempts; record-openai-chat-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix_chat/shaping_thinking_second_turn", shaping_thinking_second_turn_18, "openai_matrix_chat_shaping_thinking_second_turn");
    #[tokio::test]
    #[ignore = "gpt-5-mini low reported zero reasoning tokens on the tool turn in attempts 1, 2 and 3; record-openai-chat-tool-unary-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_unary: ("reasoning_matrix_chat/tool_unary", reasoning_tool_unary_19, "openai_matrix_chat_reasoning_tool_unary");
    #[tokio::test]
    #[ignore = "gpt-5-mini low reported zero reasoning tokens on the tool turn in all three attempts; record-openai-chat-tool-streamed-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_streamed: ("reasoning_matrix_chat/tool_streamed", reasoning_tool_streamed_20, "openai_matrix_chat_reasoning_tool_streamed");
}

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::Chat>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat(rig::providers::openai::GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
        options: None,
    }
}

#[ignore = "gpt-5-mini at reasoning_effort low reported zero reasoning tokens in attempts 1, 2 and 3 (2026-09-13, record-openai-chat-text-unary-attempt-{1,2,3}.log); exhausted the prompt's three-attempt limit"]
#[tokio::test]
async fn reasoning_text_unary() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette("reasoning_matrix_chat/text_unary", |client| async move {
            run_world(
                &reasoning_wire(&client),
                &cells::REASONING_TEXT_UNARY,
                |log| {
                    crate::goldens::world_golden_effects(
                        "openai_matrix_chat_reasoning_text_unary",
                        log,
                    )
                },
            )
            .await;
        })
        .await;
    })
    .await
}
