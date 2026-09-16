//! The ECS contract matrix's producers on the Venice wire (`mistral-small-3-2-24b-instruct`, the model the suite records tools under, temperature 0; the route is the same model under `golden/model:fast`: every other Venice model tried either thinks (its reasoning part in the history is refused by the Mistral tokenizer on the next turn) or re-calls the tool after its result, so the route is observable on the bus, not on the wire): every cell of
//! `tests/common/ecs_matrix/cells.rs` on rig-agent's builder, recorded once
//! and replayed as the golden `venice_<cell>` the world cells in
//! `ecs_matrix.rs` are compared to. The driver is
//! `tests/common/ecs_matrix/agent.rs`; this file holds the scenario
//! literals, the wire's models and the wire's `#[ignore]` reasons.
//!
//! Every scenario here is a new recording under `corpus_matrix/`; a cell of
//! the grid missing from this file reuses a recording the corpus already
//! had, whose producer stays where it is.

use rig::completion::CompletionModel;
use rig::providers::venice::MISTRAL_SMALL_3_2_24B;

use super::super::support::{BoundVenice, with_venice_cassette};
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &BoundVenice) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Venice,
        model: client.completion(MISTRAL_SMALL_3_2_24B),
        route: Some(client.completion(MISTRAL_SMALL_3_2_24B)),
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_venice_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    endings_tool_dispatch_cancelled: ("corpus_matrix/endings_tool_dispatch_cancelled", cells::ENDINGS_TOOL_DISPATCH_CANCELLED, "venice_endings_tool_dispatch_cancelled");
    #[tokio::test]
    endings_tool_outcome_cancelled: ("corpus_matrix/endings_tool_outcome_cancelled", cells::ENDINGS_TOOL_OUTCOME_CANCELLED, "venice_endings_tool_outcome_cancelled");
    #[tokio::test]
    endings_tool_call_delta_stop: ("corpus_matrix/endings_tool_call_delta_stop", cells::ENDINGS_TOOL_CALL_DELTA_STOP, "venice_endings_tool_call_delta_stop");
    #[tokio::test]
    hooks_patch_tool_args_streamed: ("corpus_matrix/hooks_patch_tool_args_streamed", cells::HOOKS_PATCH_TOOL_ARGS_STREAMED, "venice_hooks_patch_tool_args_streamed");
    #[tokio::test]
    host_custom_at_completion_call: ("corpus_matrix/host_custom_at_completion_call", cells::HOST_CUSTOM_AT_COMPLETION_CALL, "venice_host_custom_at_completion_call");
    #[tokio::test]
    host_custom_start_and_settled: ("corpus_matrix/host_custom_start_and_settled", cells::HOST_CUSTOM_START_AND_SETTLED, "venice_host_custom_start_and_settled");
    #[tokio::test]
    host_custom_twice_concurrent: ("corpus_matrix/host_custom_twice_concurrent", cells::HOST_CUSTOM_TWICE_CONCURRENT, "venice_host_custom_twice_concurrent");
    #[tokio::test]
    host_custom_at_start_streamed: ("corpus_matrix/host_custom_at_start_streamed", cells::HOST_CUSTOM_AT_START_STREAMED, "venice_host_custom_at_start_streamed");
    #[tokio::test]
    host_custom_at_outcome_streamed: ("corpus_matrix/host_custom_at_outcome_streamed", cells::HOST_CUSTOM_AT_OUTCOME_STREAMED, "venice_host_custom_at_outcome_streamed");
    #[tokio::test]
    serving_serial_concurrency_one: ("corpus_matrix/serving_serial_concurrency_one", cells::SERVING_SERIAL_CONCURRENCY_ONE, "venice_serving_serial_concurrency_one");
    #[tokio::test]
    serving_model_route_unselected: ("corpus_matrix/serving_model_route_unselected", cells::SERVING_MODEL_ROUTE_UNSELECTED, "venice_serving_model_route_unselected");
    #[tokio::test]
    layers_host_deny_over_host_bus: ("corpus_matrix/layers_host_deny_over_host_bus", cells::LAYERS_HOST_DENY_OVER_HOST_BUS, "venice_layers_host_deny_over_host_bus");
    #[tokio::test]
    layers_patch_beneath_hook_patch: ("corpus_matrix/layers_patch_beneath_hook_patch", cells::LAYERS_PATCH_BENEATH_HOOK_PATCH, "venice_layers_patch_beneath_hook_patch");
    #[tokio::test]
    memory_clear_at_start_two_runs: ("corpus_matrix/memory_clear_at_start_two_runs", cells::MEMORY_CLEAR_AT_START_TWO_RUNS, "venice_memory_clear_at_start_two_runs");
    #[tokio::test]
    memory_failing_append_streamed: ("corpus_matrix/memory_failing_append_streamed", cells::MEMORY_FAILING_APPEND_STREAMED, "venice_memory_failing_append_streamed");
    #[tokio::test]
    output_prompted_with_real_tool: ("corpus_matrix/output_prompted_with_real_tool", cells::OUTPUT_PROMPTED_WITH_REAL_TOOL, "venice_output_prompted_with_real_tool");
    #[tokio::test]
    output_tool_under_none_degrades: ("corpus_matrix/output_tool_under_none_degrades", cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES, "venice_output_tool_under_none_degrades");
    #[tokio::test]
    shaping_extra_context_streamed: ("corpus_matrix/shaping_extra_context_streamed", cells::SHAPING_EXTRA_CONTEXT_STREAMED, "venice_shaping_extra_context_streamed");
    #[tokio::test]
    shaping_max_tokens_second_turn: ("corpus_matrix/shaping_max_tokens_second_turn", cells::SHAPING_MAX_TOKENS_SECOND_TURN, "venice_shaping_max_tokens_second_turn");
    #[tokio::test]
    shaping_preamble_second_turn: ("corpus_matrix/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN, "venice_shaping_preamble_second_turn");
    #[tokio::test]
    causal_completion_concurrent: ("corpus_matrix/causal_completion_concurrent", cells::CAUSAL_COMPLETION_CONCURRENT, "venice_causal_completion_concurrent");
}

crate::matrix::golden_matrix! {
    wrapper: with_venice_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    endings_answer_outcome_cancelled: ("corpus_matrix/endings_answer_outcome_cancelled", cells::ENDINGS_ANSWER_OUTCOME_CANCELLED, "venice_endings_answer_outcome_cancelled");
    #[tokio::test]
    endings_turn_finished_stop: ("corpus_matrix/endings_turn_finished_stop", cells::ENDINGS_TURN_FINISHED_STOP, "venice_endings_turn_finished_stop");
    #[tokio::test]
    endings_answer_turn_stop: ("corpus_matrix/endings_answer_turn_stop", cells::ENDINGS_ANSWER_TURN_STOP, "venice_endings_answer_turn_stop");
    #[tokio::test]
    endings_text_delta_stop: ("corpus_matrix/endings_text_delta_stop", cells::ENDINGS_TEXT_DELTA_STOP, "venice_endings_text_delta_stop");
    #[tokio::test]
    endings_tool_dispatch_cancelled_streamed: ("corpus_matrix/endings_tool_dispatch_cancelled_streamed", cells::ENDINGS_TOOL_DISPATCH_CANCELLED_STREAMED, "venice_endings_tool_dispatch_cancelled_streamed");
    #[tokio::test]
    endings_turn_finished_stop_streamed: ("corpus_matrix/endings_turn_finished_stop_streamed", cells::ENDINGS_TURN_FINISHED_STOP_STREAMED, "venice_endings_turn_finished_stop_streamed");
    #[tokio::test]
    endings_tool_outcome_cancelled_streamed: ("corpus_matrix/endings_tool_outcome_cancelled_streamed", cells::ENDINGS_TOOL_OUTCOME_CANCELLED_STREAMED, "venice_endings_tool_outcome_cancelled_streamed");
    #[tokio::test]
    hooks_observe_everything: ("corpus_matrix/hooks_observe_everything", cells::HOOKS_OBSERVE_EVERYTHING, "venice_hooks_observe_everything");
    #[tokio::test]
    hooks_patch_tool_args: ("corpus_matrix/hooks_patch_tool_args", cells::HOOKS_PATCH_TOOL_ARGS, "venice_hooks_patch_tool_args");
    #[tokio::test]
    hooks_deny_tool: ("corpus_matrix/hooks_deny_tool", cells::HOOKS_DENY_TOOL, "venice_hooks_deny_tool");
    #[tokio::test]
    hooks_deny_tool_streamed: ("corpus_matrix/hooks_deny_tool_streamed", cells::HOOKS_DENY_TOOL_STREAMED, "venice_hooks_deny_tool_streamed");
    #[tokio::test]
    hooks_replace_tool_result: ("corpus_matrix/hooks_replace_tool_result", cells::HOOKS_REPLACE_TOOL_RESULT, "venice_hooks_replace_tool_result");
    #[tokio::test]
    hooks_replace_answer: ("corpus_matrix/hooks_replace_answer", cells::HOOKS_REPLACE_ANSWER, "venice_hooks_replace_answer");
    #[tokio::test]
    hooks_preamble_override: ("corpus_matrix/hooks_preamble_override", cells::HOOKS_PREAMBLE_OVERRIDE, "venice_hooks_preamble_override");
    #[tokio::test]
    hooks_demand_done: ("corpus_matrix/hooks_demand_done", cells::HOOKS_DEMAND_DONE, "venice_hooks_demand_done");
    #[tokio::test]
    hooks_lookup_before_run: ("corpus_matrix/hooks_lookup_before_run", cells::HOOKS_LOOKUP_BEFORE_RUN, "venice_hooks_lookup_before_run");
    #[tokio::test]
    hooks_two_hooks: ("corpus_matrix/hooks_two_hooks", cells::HOOKS_TWO_HOOKS, "venice_hooks_two_hooks");
    #[tokio::test]
    host_custom_at_start: ("corpus_matrix/host_custom_at_start", cells::HOST_CUSTOM_AT_START, "venice_host_custom_at_start");
    #[tokio::test]
    host_custom_at_outcome: ("corpus_matrix/host_custom_at_outcome", cells::HOST_CUSTOM_AT_OUTCOME, "venice_host_custom_at_outcome");
    #[tokio::test]
    host_custom_at_settled: ("corpus_matrix/host_custom_at_settled", cells::HOST_CUSTOM_AT_SETTLED, "venice_host_custom_at_settled");
    #[tokio::test]
    host_custom_twice_serial: ("corpus_matrix/host_custom_twice_serial", cells::HOST_CUSTOM_TWICE_SERIAL, "venice_host_custom_twice_serial");
    #[tokio::test]
    host_custom_unserved: ("corpus_matrix/host_custom_unserved", cells::HOST_CUSTOM_UNSERVED, "venice_host_custom_unserved");
    #[tokio::test]
    serving_concurrent_concurrency_one: ("corpus_matrix/serving_concurrent_concurrency_one", cells::SERVING_CONCURRENT_CONCURRENCY_ONE, "venice_serving_concurrent_concurrency_one");
    #[tokio::test]
    serving_concurrent_concurrency_two: ("corpus_matrix/serving_concurrent_concurrency_two", cells::SERVING_CONCURRENT_CONCURRENCY_TWO, "venice_serving_concurrent_concurrency_two");
    #[tokio::test]
    serving_concurrent_concurrency_two_events: ("corpus_matrix/serving_concurrent_concurrency_two_events", cells::SERVING_CONCURRENT_CONCURRENCY_TWO_EVENTS, "venice_serving_concurrent_concurrency_two_events");
    #[tokio::test]
    serving_capacity_one: ("corpus_matrix/serving_capacity_one", cells::SERVING_CAPACITY_ONE, "venice_serving_capacity_one");
    #[tokio::test]
    serving_serial_memory_tools: ("corpus_matrix/serving_serial_memory_tools", cells::SERVING_SERIAL_MEMORY_TOOLS, "venice_serving_serial_memory_tools");
    #[tokio::test]
    serving_model_route: ("corpus_matrix/serving_model_route", cells::SERVING_MODEL_ROUTE, "venice_serving_model_route");
    #[tokio::test]
    serving_host_bus: ("corpus_matrix/serving_host_bus", cells::SERVING_HOST_BUS, "venice_serving_host_bus");
    #[tokio::test]
    serving_host_bus_streamed: ("corpus_matrix/serving_host_bus_streamed", cells::SERVING_HOST_BUS_STREAMED, "venice_serving_host_bus_streamed");
    #[tokio::test]
    layers_deny_tool: ("corpus_matrix/layers_deny_tool", cells::LAYERS_DENY_TOOL, "venice_layers_deny_tool");
    #[tokio::test]
    layers_patch_tool_args: ("corpus_matrix/layers_patch_tool_args", cells::LAYERS_PATCH_TOOL_ARGS, "venice_layers_patch_tool_args");
    #[tokio::test]
    layers_replace_tool_result: ("corpus_matrix/layers_replace_tool_result", cells::LAYERS_REPLACE_TOOL_RESULT, "venice_layers_replace_tool_result");
    #[tokio::test]
    layers_two_layers: ("corpus_matrix/layers_two_layers", cells::LAYERS_TWO_LAYERS, "venice_layers_two_layers");
    #[tokio::test]
    layers_memory_load_replaced: ("corpus_matrix/layers_memory_load_replaced", cells::LAYERS_MEMORY_LOAD_REPLACED, "venice_layers_memory_load_replaced");
    #[tokio::test]
    memory_clear_at_start: ("corpus_matrix/memory_clear_at_start", cells::MEMORY_CLEAR_AT_START, "venice_memory_clear_at_start");
    #[tokio::test]
    memory_clear_at_settled: ("corpus_matrix/memory_clear_at_settled", cells::MEMORY_CLEAR_AT_SETTLED, "venice_memory_clear_at_settled");
    #[tokio::test]
    memory_two_runs: ("corpus_matrix/memory_two_runs", cells::MEMORY_TWO_RUNS, "venice_memory_two_runs");
    #[tokio::test]
    memory_two_runs_streamed: ("corpus_matrix/memory_two_runs_streamed", cells::MEMORY_TWO_RUNS_STREAMED, "venice_memory_two_runs_streamed");
    #[tokio::test]
    memory_clear_at_settled_two_runs: ("corpus_matrix/memory_clear_at_settled_two_runs", cells::MEMORY_CLEAR_AT_SETTLED_TWO_RUNS, "venice_memory_clear_at_settled_two_runs");
    #[tokio::test]
    memory_history_bypass: ("corpus_matrix/memory_history_bypass", cells::MEMORY_HISTORY_BYPASS, "venice_memory_history_bypass");
    #[tokio::test]
    memory_host_bus_memory: ("corpus_matrix/memory_host_bus_memory", cells::MEMORY_HOST_BUS_MEMORY, "venice_memory_host_bus_memory");
    #[tokio::test]
    memory_serial_two_tools: ("corpus_matrix/memory_serial_two_tools", cells::MEMORY_SERIAL_TWO_TOOLS, "venice_memory_serial_two_tools");
    #[tokio::test]
    memory_failing_append: ("corpus_matrix/memory_failing_append", cells::MEMORY_FAILING_APPEND, "venice_memory_failing_append");
    #[tokio::test]
    output_tool_unary: ("corpus_matrix/output_tool_unary", cells::OUTPUT_TOOL_UNARY, "venice_output_tool_unary");
    #[tokio::test]
    output_tool_streamed: ("corpus_matrix/output_tool_streamed", cells::OUTPUT_TOOL_STREAMED, "venice_output_tool_streamed");
    #[tokio::test]
    output_prompted_unary: ("corpus_matrix/output_prompted_unary", cells::OUTPUT_PROMPTED_UNARY, "venice_output_prompted_unary");
    #[tokio::test]
    output_prompted_streamed: ("corpus_matrix/output_prompted_streamed", cells::OUTPUT_PROMPTED_STREAMED, "venice_output_prompted_streamed");
    #[tokio::test]
    output_tool_choice_specific_output: ("corpus_matrix/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT, "venice_output_tool_choice_specific_output");
    #[tokio::test]
    output_tool_choice_required: ("corpus_matrix/output_tool_choice_required", cells::OUTPUT_TOOL_CHOICE_REQUIRED, "venice_output_tool_choice_required");
    #[tokio::test]
    shaping_tool_choice_required_first: ("corpus_matrix/shaping_tool_choice_required_first", cells::SHAPING_TOOL_CHOICE_REQUIRED_FIRST, "venice_shaping_tool_choice_required_first");
    #[tokio::test]
    shaping_tool_choice_none_on_committed_output: ("corpus_matrix/shaping_tool_choice_none_on_committed_output", cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT, "venice_shaping_tool_choice_none_on_committed_output");
    #[tokio::test]
    shaping_extra_context: ("corpus_matrix/shaping_extra_context", cells::SHAPING_EXTRA_CONTEXT, "venice_shaping_extra_context");
    #[tokio::test]
    shaping_merged_three: ("corpus_matrix/shaping_merged_three", cells::SHAPING_MERGED_THREE, "venice_shaping_merged_three");
    #[tokio::test]
    shaping_route_on_first_turn: ("corpus_matrix/shaping_route_on_first_turn", cells::SHAPING_ROUTE_ON_FIRST_TURN, "venice_shaping_route_on_first_turn");
    #[tokio::test]
    shaping_late_route: ("corpus_matrix/shaping_late_route", cells::SHAPING_LATE_ROUTE, "venice_shaping_late_route");
    #[tokio::test]
    shaping_active_tools_none_second_turn: ("corpus_matrix/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN, "venice_shaping_active_tools_none_second_turn");
    #[tokio::test]
    shaping_history_first_turn: ("corpus_matrix/shaping_history_first_turn", cells::SHAPING_HISTORY_FIRST_TURN, "venice_shaping_history_first_turn");
    #[tokio::test]
    causal_completion_serial: ("corpus_matrix/causal_completion_serial", cells::CAUSAL_COMPLETION_SERIAL, "venice_causal_completion_serial");
    #[tokio::test]
    causal_completion_streamed: ("corpus_matrix/causal_completion_streamed", cells::CAUSAL_COMPLETION_STREAMED, "venice_causal_completion_streamed");
    #[tokio::test]
    resume_tool_turn: ("corpus_matrix/resume_tool_turn", cells::RESUME_TOOL_TURN, "venice_resume_tool_turn");
}

#[ignore = "Venice's gateway never answers the two-turn output-tool program (a 2,200 s hang, then a 500 `cannot send request`); two recordings agreed"]
#[tokio::test]
async fn output_tool_with_real_tool() {
    with_venice_cassette(
        "corpus_matrix/output_tool_with_real_tool",
        |client| async move {
            run_agent(&wire(&client), &cells::OUTPUT_TOOL_WITH_REAL_TOOL, |_| {}).await;
        },
    )
    .await;
}

crate::matrix::golden_matrix! {
    wrapper: with_venice_cassette, wire: reasoning_wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    output_tool_thinking: ("corpus_matrix/output_tool_thinking", cells::OUTPUT_TOOL_THINKING, "venice_output_tool_thinking");
    #[tokio::test]
    reasoning_text_unary: ("reasoning_matrix/text_unary", cells::REASONING_TEXT_UNARY, "venice_reasoning_text_unary");
    #[tokio::test]
    reasoning_text_streamed: ("reasoning_matrix/text_streamed", cells::REASONING_TEXT_STREAMED, "venice_reasoning_text_streamed");
    #[tokio::test]
    reasoning_tool_unary: ("reasoning_matrix/tool_unary", cells::REASONING_TOOL_UNARY, "venice_reasoning_tool_unary");
    #[tokio::test]
    reasoning_tool_streamed: ("reasoning_matrix/tool_streamed", cells::REASONING_TOOL_STREAMED, "venice_reasoning_tool_streamed");
    #[tokio::test]
    reasoning_off: ("reasoning_matrix/off", cells::REASONING_OFF, "venice_reasoning_off");
    #[tokio::test]
    reasoning_capped: ("reasoning_matrix/capped", cells::REASONING_CAPPED, "venice_reasoning_capped");
}

crate::matrix::case_matrix! {
    wrapper: with_venice_cassette, family: wire_matrix_case;
    #[ignore = "Venice thinking disabled answered directly without the required first-turn add call in all three attempts; record-venice-shaping-thinking-second-turn-attempt-{1,2,3}.log; three attempts exhausted"]
    #[tokio::test]
    shaping_thinking_second_turn: ("corpus_matrix/shaping_thinking_second_turn", shaping_thinking_second_turn_9);
}

// Reasoning matrix: the named thinking model, with the shared knob.
fn reasoning_wire(client: &BoundVenice) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Venice,
        model: client.completion(rig::providers::venice::QWEN3_235B_A22B_THINKING),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}
