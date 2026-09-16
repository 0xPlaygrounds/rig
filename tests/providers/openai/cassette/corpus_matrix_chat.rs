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

use rig::completion::CompletionModel;
use rig::providers::openai::{GPT_5_MINI, GPT_5_NANO};

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat(GPT_5_MINI),
        route: Some(client.openai.chat(GPT_5_NANO)),
        temperature: None,
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    endings_tool_dispatch_cancelled: ("corpus_matrix_chat/endings_tool_dispatch_cancelled", cells::ENDINGS_TOOL_DISPATCH_CANCELLED, "openai_chat_endings_tool_dispatch_cancelled");
    #[tokio::test]
    endings_tool_outcome_cancelled: ("corpus_matrix_chat/endings_tool_outcome_cancelled", cells::ENDINGS_TOOL_OUTCOME_CANCELLED, "openai_chat_endings_tool_outcome_cancelled");
    #[tokio::test]
    endings_answer_outcome_cancelled: ("corpus_matrix_chat/endings_answer_outcome_cancelled", cells::ENDINGS_ANSWER_OUTCOME_CANCELLED, "openai_chat_endings_answer_outcome_cancelled");
    #[tokio::test]
    endings_turn_finished_stop: ("corpus_matrix_chat/endings_turn_finished_stop", cells::ENDINGS_TURN_FINISHED_STOP, "openai_chat_endings_turn_finished_stop");
    #[tokio::test]
    endings_answer_turn_stop: ("corpus_matrix_chat/endings_answer_turn_stop", cells::ENDINGS_ANSWER_TURN_STOP, "openai_chat_endings_answer_turn_stop");
    #[tokio::test]
    endings_text_delta_stop: ("corpus_matrix_chat/endings_text_delta_stop", cells::ENDINGS_TEXT_DELTA_STOP, "openai_chat_endings_text_delta_stop");
    #[tokio::test]
    endings_tool_call_delta_stop: ("corpus_matrix_chat/endings_tool_call_delta_stop", cells::ENDINGS_TOOL_CALL_DELTA_STOP, "openai_chat_endings_tool_call_delta_stop");
    #[tokio::test]
    endings_tool_dispatch_cancelled_streamed: ("corpus_matrix_chat/endings_tool_dispatch_cancelled_streamed", cells::ENDINGS_TOOL_DISPATCH_CANCELLED_STREAMED, "openai_chat_endings_tool_dispatch_cancelled_streamed");
    #[tokio::test]
    endings_turn_finished_stop_streamed: ("corpus_matrix_chat/endings_turn_finished_stop_streamed", cells::ENDINGS_TURN_FINISHED_STOP_STREAMED, "openai_chat_endings_turn_finished_stop_streamed");
    #[tokio::test]
    endings_tool_outcome_cancelled_streamed: ("corpus_matrix_chat/endings_tool_outcome_cancelled_streamed", cells::ENDINGS_TOOL_OUTCOME_CANCELLED_STREAMED, "openai_chat_endings_tool_outcome_cancelled_streamed");
    #[tokio::test]
    hooks_observe_everything: ("corpus_matrix_chat/hooks_observe_everything", cells::HOOKS_OBSERVE_EVERYTHING, "openai_chat_hooks_observe_everything");
    #[tokio::test]
    hooks_patch_tool_args: ("corpus_matrix_chat/hooks_patch_tool_args", cells::HOOKS_PATCH_TOOL_ARGS, "openai_chat_hooks_patch_tool_args");
    #[tokio::test]
    hooks_patch_tool_args_streamed: ("corpus_matrix_chat/hooks_patch_tool_args_streamed", cells::HOOKS_PATCH_TOOL_ARGS_STREAMED, "openai_chat_hooks_patch_tool_args_streamed");
    #[tokio::test]
    hooks_deny_tool: ("corpus_matrix_chat/hooks_deny_tool", cells::HOOKS_DENY_TOOL, "openai_chat_hooks_deny_tool");
    #[tokio::test]
    hooks_deny_tool_streamed: ("corpus_matrix_chat/hooks_deny_tool_streamed", cells::HOOKS_DENY_TOOL_STREAMED, "openai_chat_hooks_deny_tool_streamed");
    #[tokio::test]
    hooks_replace_tool_result: ("corpus_matrix_chat/hooks_replace_tool_result", cells::HOOKS_REPLACE_TOOL_RESULT, "openai_chat_hooks_replace_tool_result");
    #[tokio::test]
    hooks_replace_answer: ("corpus_matrix_chat/hooks_replace_answer", cells::HOOKS_REPLACE_ANSWER, "openai_chat_hooks_replace_answer");
    #[tokio::test]
    hooks_preamble_override: ("corpus_matrix_chat/hooks_preamble_override", cells::HOOKS_PREAMBLE_OVERRIDE, "openai_chat_hooks_preamble_override");
    #[tokio::test]
    hooks_demand_done: ("corpus_matrix_chat/hooks_demand_done", cells::HOOKS_DEMAND_DONE, "openai_chat_hooks_demand_done");
    #[tokio::test]
    hooks_lookup_before_run: ("corpus_matrix_chat/hooks_lookup_before_run", cells::HOOKS_LOOKUP_BEFORE_RUN, "openai_chat_hooks_lookup_before_run");
    #[tokio::test]
    hooks_two_hooks: ("corpus_matrix_chat/hooks_two_hooks", cells::HOOKS_TWO_HOOKS, "openai_chat_hooks_two_hooks");
    #[tokio::test]
    host_custom_at_start: ("corpus_matrix_chat/host_custom_at_start", cells::HOST_CUSTOM_AT_START, "openai_chat_host_custom_at_start");
    #[tokio::test]
    host_custom_at_completion_call: ("corpus_matrix_chat/host_custom_at_completion_call", cells::HOST_CUSTOM_AT_COMPLETION_CALL, "openai_chat_host_custom_at_completion_call");
    #[tokio::test]
    host_custom_at_outcome: ("corpus_matrix_chat/host_custom_at_outcome", cells::HOST_CUSTOM_AT_OUTCOME, "openai_chat_host_custom_at_outcome");
    #[tokio::test]
    host_custom_at_settled: ("corpus_matrix_chat/host_custom_at_settled", cells::HOST_CUSTOM_AT_SETTLED, "openai_chat_host_custom_at_settled");
    #[tokio::test]
    host_custom_start_and_settled: ("corpus_matrix_chat/host_custom_start_and_settled", cells::HOST_CUSTOM_START_AND_SETTLED, "openai_chat_host_custom_start_and_settled");
    #[tokio::test]
    host_custom_twice_serial: ("corpus_matrix_chat/host_custom_twice_serial", cells::HOST_CUSTOM_TWICE_SERIAL, "openai_chat_host_custom_twice_serial");
    #[tokio::test]
    host_custom_twice_concurrent: ("corpus_matrix_chat/host_custom_twice_concurrent", cells::HOST_CUSTOM_TWICE_CONCURRENT, "openai_chat_host_custom_twice_concurrent");
    #[tokio::test]
    host_custom_at_start_streamed: ("corpus_matrix_chat/host_custom_at_start_streamed", cells::HOST_CUSTOM_AT_START_STREAMED, "openai_chat_host_custom_at_start_streamed");
    #[tokio::test]
    host_custom_at_outcome_streamed: ("corpus_matrix_chat/host_custom_at_outcome_streamed", cells::HOST_CUSTOM_AT_OUTCOME_STREAMED, "openai_chat_host_custom_at_outcome_streamed");
    #[tokio::test]
    host_custom_unserved: ("corpus_matrix_chat/host_custom_unserved", cells::HOST_CUSTOM_UNSERVED, "openai_chat_host_custom_unserved");
    #[tokio::test]
    serving_serial_concurrency_one: ("corpus_matrix_chat/serving_serial_concurrency_one", cells::SERVING_SERIAL_CONCURRENCY_ONE, "openai_chat_serving_serial_concurrency_one");
    #[tokio::test]
    serving_concurrent_concurrency_one: ("corpus_matrix_chat/serving_concurrent_concurrency_one", cells::SERVING_CONCURRENT_CONCURRENCY_ONE, "openai_chat_serving_concurrent_concurrency_one");
    #[tokio::test]
    serving_concurrent_concurrency_two: ("corpus_matrix_chat/serving_concurrent_concurrency_two", cells::SERVING_CONCURRENT_CONCURRENCY_TWO, "openai_chat_serving_concurrent_concurrency_two");
    #[tokio::test]
    serving_concurrent_concurrency_two_events: ("corpus_matrix_chat/serving_concurrent_concurrency_two_events", cells::SERVING_CONCURRENT_CONCURRENCY_TWO_EVENTS, "openai_chat_serving_concurrent_concurrency_two_events");
    #[tokio::test]
    serving_capacity_one: ("corpus_matrix_chat/serving_capacity_one", cells::SERVING_CAPACITY_ONE, "openai_chat_serving_capacity_one");
    #[tokio::test]
    serving_serial_memory_tools: ("corpus_matrix_chat/serving_serial_memory_tools", cells::SERVING_SERIAL_MEMORY_TOOLS, "openai_chat_serving_serial_memory_tools");
    #[tokio::test]
    serving_model_route: ("corpus_matrix_chat/serving_model_route", cells::SERVING_MODEL_ROUTE, "openai_chat_serving_model_route");
    #[tokio::test]
    serving_model_route_unselected: ("corpus_matrix_chat/serving_model_route_unselected", cells::SERVING_MODEL_ROUTE_UNSELECTED, "openai_chat_serving_model_route_unselected");
    #[tokio::test]
    serving_host_bus: ("corpus_matrix_chat/serving_host_bus", cells::SERVING_HOST_BUS, "openai_chat_serving_host_bus");
    #[tokio::test]
    serving_host_bus_streamed: ("corpus_matrix_chat/serving_host_bus_streamed", cells::SERVING_HOST_BUS_STREAMED, "openai_chat_serving_host_bus_streamed");
    #[tokio::test]
    layers_deny_tool: ("corpus_matrix_chat/layers_deny_tool", cells::LAYERS_DENY_TOOL, "openai_chat_layers_deny_tool");
    #[tokio::test]
    layers_patch_tool_args: ("corpus_matrix_chat/layers_patch_tool_args", cells::LAYERS_PATCH_TOOL_ARGS, "openai_chat_layers_patch_tool_args");
    #[tokio::test]
    layers_replace_tool_result: ("corpus_matrix_chat/layers_replace_tool_result", cells::LAYERS_REPLACE_TOOL_RESULT, "openai_chat_layers_replace_tool_result");
    #[tokio::test]
    layers_two_layers: ("corpus_matrix_chat/layers_two_layers", cells::LAYERS_TWO_LAYERS, "openai_chat_layers_two_layers");
    #[tokio::test]
    layers_host_deny_over_host_bus: ("corpus_matrix_chat/layers_host_deny_over_host_bus", cells::LAYERS_HOST_DENY_OVER_HOST_BUS, "openai_chat_layers_host_deny_over_host_bus");
    #[tokio::test]
    layers_patch_beneath_hook_patch: ("corpus_matrix_chat/layers_patch_beneath_hook_patch", cells::LAYERS_PATCH_BENEATH_HOOK_PATCH, "openai_chat_layers_patch_beneath_hook_patch");
    #[tokio::test]
    layers_memory_load_replaced: ("corpus_matrix_chat/layers_memory_load_replaced", cells::LAYERS_MEMORY_LOAD_REPLACED, "openai_chat_layers_memory_load_replaced");
    #[tokio::test]
    memory_clear_at_start: ("corpus_matrix_chat/memory_clear_at_start", cells::MEMORY_CLEAR_AT_START, "openai_chat_memory_clear_at_start");
    #[tokio::test]
    memory_clear_at_settled: ("corpus_matrix_chat/memory_clear_at_settled", cells::MEMORY_CLEAR_AT_SETTLED, "openai_chat_memory_clear_at_settled");
    #[tokio::test]
    memory_two_runs: ("corpus_matrix_chat/memory_two_runs", cells::MEMORY_TWO_RUNS, "openai_chat_memory_two_runs");
    #[tokio::test]
    memory_two_runs_streamed: ("corpus_matrix_chat/memory_two_runs_streamed", cells::MEMORY_TWO_RUNS_STREAMED, "openai_chat_memory_two_runs_streamed");
    #[tokio::test]
    memory_clear_at_settled_two_runs: ("corpus_matrix_chat/memory_clear_at_settled_two_runs", cells::MEMORY_CLEAR_AT_SETTLED_TWO_RUNS, "openai_chat_memory_clear_at_settled_two_runs");
    #[tokio::test]
    memory_clear_at_start_two_runs: ("corpus_matrix_chat/memory_clear_at_start_two_runs", cells::MEMORY_CLEAR_AT_START_TWO_RUNS, "openai_chat_memory_clear_at_start_two_runs");
    #[tokio::test]
    memory_history_bypass: ("corpus_matrix_chat/memory_history_bypass", cells::MEMORY_HISTORY_BYPASS, "openai_chat_memory_history_bypass");
    #[tokio::test]
    memory_host_bus_memory: ("corpus_matrix_chat/memory_host_bus_memory", cells::MEMORY_HOST_BUS_MEMORY, "openai_chat_memory_host_bus_memory");
    #[tokio::test]
    memory_serial_two_tools: ("corpus_matrix_chat/memory_serial_two_tools", cells::MEMORY_SERIAL_TWO_TOOLS, "openai_chat_memory_serial_two_tools");
    #[tokio::test]
    memory_failing_append: ("corpus_matrix_chat/memory_failing_append", cells::MEMORY_FAILING_APPEND, "openai_chat_memory_failing_append");
    #[tokio::test]
    memory_failing_append_streamed: ("corpus_matrix_chat/memory_failing_append_streamed", cells::MEMORY_FAILING_APPEND_STREAMED, "openai_chat_memory_failing_append_streamed");
    #[tokio::test]
    output_tool_unary: ("corpus_matrix_chat/output_tool_unary", cells::OUTPUT_TOOL_UNARY, "openai_chat_output_tool_unary");
    #[tokio::test]
    output_tool_streamed: ("corpus_matrix_chat/output_tool_streamed", cells::OUTPUT_TOOL_STREAMED, "openai_chat_output_tool_streamed");
    #[tokio::test]
    output_prompted_unary: ("corpus_matrix_chat/output_prompted_unary", cells::OUTPUT_PROMPTED_UNARY, "openai_chat_output_prompted_unary");
    #[tokio::test]
    output_prompted_streamed: ("corpus_matrix_chat/output_prompted_streamed", cells::OUTPUT_PROMPTED_STREAMED, "openai_chat_output_prompted_streamed");
    #[tokio::test]
    output_tool_with_real_tool: ("corpus_matrix_chat/output_tool_with_real_tool", cells::OUTPUT_TOOL_WITH_REAL_TOOL, "openai_chat_output_tool_with_real_tool");
    #[tokio::test]
    output_prompted_with_real_tool: ("corpus_matrix_chat/output_prompted_with_real_tool", cells::OUTPUT_PROMPTED_WITH_REAL_TOOL, "openai_chat_output_prompted_with_real_tool");
    #[tokio::test]
    output_tool_choice_specific_output: ("corpus_matrix_chat/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT, "openai_chat_output_tool_choice_specific_output");
    #[tokio::test]
    output_tool_choice_required: ("corpus_matrix_chat/output_tool_choice_required", cells::OUTPUT_TOOL_CHOICE_REQUIRED, "openai_chat_output_tool_choice_required");
    #[tokio::test]
    output_tool_under_none_degrades: ("corpus_matrix_chat/output_tool_under_none_degrades", cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES, "openai_chat_output_tool_under_none_degrades");
    #[tokio::test]
    shaping_tool_choice_required_first: ("corpus_matrix_chat/shaping_tool_choice_required_first", cells::SHAPING_TOOL_CHOICE_REQUIRED_FIRST, "openai_chat_shaping_tool_choice_required_first");
    #[tokio::test]
    shaping_tool_choice_none_on_committed_output: ("corpus_matrix_chat/shaping_tool_choice_none_on_committed_output", cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT, "openai_chat_shaping_tool_choice_none_on_committed_output");
    #[tokio::test]
    shaping_extra_context: ("corpus_matrix_chat/shaping_extra_context", cells::SHAPING_EXTRA_CONTEXT, "openai_chat_shaping_extra_context");
    #[tokio::test]
    shaping_extra_context_streamed: ("corpus_matrix_chat/shaping_extra_context_streamed", cells::SHAPING_EXTRA_CONTEXT_STREAMED, "openai_chat_shaping_extra_context_streamed");
    #[tokio::test]
    shaping_merged_three: ("corpus_matrix_chat/shaping_merged_three", cells::SHAPING_MERGED_THREE, "openai_chat_shaping_merged_three");
    #[tokio::test]
    shaping_route_on_first_turn: ("corpus_matrix_chat/shaping_route_on_first_turn", cells::SHAPING_ROUTE_ON_FIRST_TURN, "openai_chat_shaping_route_on_first_turn");
    #[tokio::test]
    shaping_late_route: ("corpus_matrix_chat/shaping_late_route", cells::SHAPING_LATE_ROUTE, "openai_chat_shaping_late_route");
    #[tokio::test]
    shaping_max_tokens_second_turn: ("corpus_matrix_chat/shaping_max_tokens_second_turn", cells::SHAPING_MAX_TOKENS_SECOND_TURN, "openai_chat_shaping_max_tokens_second_turn");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix_chat/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "openai_chat_shaping_preamble_second_turn");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix_chat/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "openai_chat_shaping_active_tools_none_second_turn");
    #[tokio::test]
    shaping_history_first_turn: ("corpus_matrix_chat/shaping_history_first_turn", cells::SHAPING_HISTORY_FIRST_TURN, "openai_chat_shaping_history_first_turn");
    #[tokio::test]
    causal_completion_serial: ("corpus_matrix_chat/causal_completion_serial", cells::CAUSAL_COMPLETION_SERIAL, "openai_chat_causal_completion_serial");
    #[tokio::test]
    causal_completion_concurrent: ("corpus_matrix_chat/causal_completion_concurrent", cells::CAUSAL_COMPLETION_CONCURRENT, "openai_chat_causal_completion_concurrent");
    #[tokio::test]
    causal_completion_streamed: ("corpus_matrix_chat/causal_completion_streamed", cells::CAUSAL_COMPLETION_STREAMED, "openai_chat_causal_completion_streamed");
    #[tokio::test]
    resume_tool_turn: ("corpus_matrix_chat/resume_tool_turn", cells::RESUME_TOOL_TURN, "openai_chat_resume_tool_turn");
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: reasoning_wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_tool_thinking: ("corpus_matrix_chat/output_tool_thinking", cells::OUTPUT_TOOL_THINKING, "openai_chat_output_tool_thinking");
    #[tokio::test]
    reasoning_text_streamed: ("reasoning_matrix_chat/text_streamed", cells::REASONING_TEXT_STREAMED, "openai_chat_reasoning_text_streamed");
    #[tokio::test]
    reasoning_off: ("reasoning_matrix_chat/off", cells::REASONING_OFF, "openai_chat_reasoning_off");
    #[tokio::test]
    reasoning_capped: ("reasoning_matrix_chat/capped", cells::REASONING_CAPPED, "openai_chat_reasoning_capped");
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
fn reasoning_wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat(rig::providers::openai::GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
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
