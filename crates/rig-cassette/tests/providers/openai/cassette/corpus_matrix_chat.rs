//! The ECS contract matrix's producers on the OpenAI Chat Completions wire (`gpt-5-mini`, no temperature: the gpt-5 family takes only its default; the route `gpt-5-nano`): every cell of
//! `tests/common/ecs_matrix/cells.rs` on rig-agent's builder, recorded once
//! and replayed as the golden `openai_chat_<cell>` the world cells in
//! `ecs_matrix_chat.rs` are compared to. The driver is
//! `tests/common/ecs_matrix/agent.rs`; this file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.
//!
//! Every scenario here is a new recording under `corpus_matrix_chat/`; a cell of
//! the grid missing from this file reuses a recording the corpus already
//! had, whose producer stays where it is.

use rig::providers::openai::{GPT_5_MINI, GPT_5_NANO};

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::Chat>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat(GPT_5_MINI),
        route: Some(client.openai.chat(GPT_5_NANO)),
        temperature: None,
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    host_custom_at_start_streamed: ("corpus_matrix_chat/host_custom_at_start_streamed", cells::HOST_CUSTOM_AT_START_STREAMED, "openai_chat_host_custom_at_start_streamed");
    #[tokio::test]
    output_tool_choice_specific_output: ("corpus_matrix_chat/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT, "openai_chat_output_tool_choice_specific_output");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix_chat/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "openai_chat_shaping_preamble_second_turn");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix_chat/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "openai_chat_shaping_active_tools_none_second_turn");
    #[tokio::test]
    causal_completion_serial: ("corpus_matrix_chat/causal_completion_serial", cells::CAUSAL_COMPLETION_SERIAL, "openai_chat_causal_completion_serial");
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: reasoning_wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_tool_thinking: ("corpus_matrix_chat/output_tool_thinking", cells::OUTPUT_TOOL_THINKING, "openai_chat_output_tool_thinking");
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: wire_matrix_case;
    #[ignore = "gpt-5-mini low reported zero reasoning tokens on the second turn in all three attempts; record-openai-chat-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix_chat/shaping_thinking_second_turn", shaping_thinking_second_turn_9);
    #[tokio::test]
    #[ignore = "gpt-5-mini low reported zero reasoning tokens on the tool turn in attempts 1, 2 and 3; record-openai-chat-tool-unary-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_unary: ("reasoning_matrix_chat/tool_unary", reasoning_tool_unary_10);
    #[tokio::test]
    #[ignore = "gpt-5-mini low reported zero reasoning tokens on the tool turn in all three attempts; record-openai-chat-tool-streamed-attempt-{1,2,3}.log; three attempts exhausted"]
    reasoning_tool_streamed: ("reasoning_matrix_chat/tool_streamed", reasoning_tool_streamed_11);
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
    with_openai_cassette("reasoning_matrix_chat/text_unary", |client| async move {
        run_agent(
            &reasoning_wire(&client),
            &cells::REASONING_TEXT_UNARY,
            |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
        )
        .await;
    })
    .await;
}
