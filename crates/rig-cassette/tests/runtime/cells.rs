//! The ECS contract matrix once per cell: every cell of
//! `tests/common/ecs_matrix/cells.rs` on rig-agent's builder and as an agent
//! graph in a Bevy world, both over the bank's replies of the shapes one
//! wire recorded for it, and the two logs compared record by record. The
//! wires rotate across the cells so every wire's decoder serves some of
//! them; a reasoning cell runs on every wire it was recorded on, because
//! each wire carries reasoning in its own shape.
//!
//! The drivers are the matrix's own (`run_agent`, `run_world`), with every
//! assertion they make: the ending, the record's families, the header's
//! hooks, the world's graph, despawn and cut. The per-provider goldens are
//! replaced by the agreement of the two interpreters over the same replies.
//! Where a consumer cancels a stream mid-flight, how many of its events
//! landed before the drop is the scheduler's, so a cancelled record's events
//! need only agree up to the shorter of the two.

use rig::http_client::DynHttpClient;
use rig_cassette::effect_log::EffectLog;
use rig_core::effect::EffectRecord;
use rig_core::error::ErrorKind;
use rig_test_support::bank;

use crate::ecs_matrix::corpus;
use crate::ecs_matrix::{Wire, agent::run_agent, cells, cells::Cell, world::run_world};
use crate::goldens::capture_world_programs;

/// Run `cell` on both interpreters over `replies`, and compare their logs.
pub(crate) async fn agree<W, T>(
    wire: fn(DynHttpClient) -> Wire<rig::driver::Model<W, T>>,
    replies: Vec<bank::Entry>,
    cell: &Cell,
) where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let mut agent = run_agent(&wire(bank::client(&replies)), cell, |_| {}).await;
    let mut world =
        capture_world_programs(run_world(&wire(bank::client(&replies)), cell, |_| {})).await;
    agree_on_cuts(&mut world, &mut agent, cell.name);
    corpus::assert_same_records(&world, &agent, cell.name);
}

/// A record both interpreters cancelled mid-stream keeps events in both
/// logs; the shorter must be a prefix of the longer, and then neither is
/// compared further.
pub(crate) fn agree_on_cuts(world: &mut EffectLog, agent: &mut EffectLog, name: &str) {
    let cancelled = |record: &EffectRecord| matches!(&record.outcome, Err(report) if report.kind == ErrorKind::Cancelled);
    for (position, (ours, theirs)) in world
        .records
        .iter_mut()
        .zip(agent.records.iter_mut())
        .enumerate()
    {
        if !(cancelled(ours) && cancelled(theirs)) {
            continue;
        }
        let events = |record: &EffectRecord| -> Vec<serde_json::Value> {
            match serde_json::to_value(&record.events).expect("events serialize") {
                serde_json::Value::Array(items) => items,
                _ => Vec::new(),
            }
        };
        let (world_events, agent_events) = (events(ours), events(theirs));
        let shared = world_events.len().min(agent_events.len());
        assert_eq!(
            world_events[..shared],
            agent_events[..shared],
            "{name}: record {position}'s events before the cut"
        );
        ours.events = None;
        theirs.events = None;
    }
}

/// A row runs its cell over the bank replies of the shapes its wire
/// recorded for the scenario (`bank::script`), or, marked `recorded`, over
/// that scenario's own pinned replies (`bank::recorded`).
macro_rules! cells {
    ($($name:ident: $($pinned:ident)? ($wire:ident, $provider:literal, $scenario:literal, $cell:path);)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                let replies = cells!(@replies $($pinned)? $provider, $scenario);
                agree(crate::wires::$wire, replies, &$cell).await;
            }
        )*
    };
    (@replies recorded $provider:literal, $scenario:literal) => {
        bank::recorded($provider, $scenario)
    };
    (@replies $provider:literal, $scenario:literal) => {
        bank::script($provider, $scenario)
    };
}

cells! {
    endings_tool_dispatch_cancelled: (deepseek, "deepseek", "corpus_matrix/endings_tool_dispatch_cancelled", cells::ENDINGS_TOOL_DISPATCH_CANCELLED);
    endings_tool_outcome_cancelled: (doubleword, "doubleword", "corpus_matrix/endings_tool_outcome_cancelled", cells::ENDINGS_TOOL_OUTCOME_CANCELLED);
    endings_answer_outcome_cancelled: (gemini, "gemini", "corpus_matrix/endings_answer_outcome_cancelled", cells::ENDINGS_ANSWER_OUTCOME_CANCELLED);
    endings_turn_finished_stop: (openai_chat, "openai", "corpus_matrix_chat/endings_turn_finished_stop", cells::ENDINGS_TURN_FINISHED_STOP);
    endings_answer_turn_stop: (openai_responses, "openai", "corpus_matrix_responses/endings_answer_turn_stop", cells::ENDINGS_ANSWER_TURN_STOP);
    endings_text_delta_stop: (venice, "venice", "corpus_matrix/endings_text_delta_stop", cells::ENDINGS_TEXT_DELTA_STOP);
    endings_tool_dispatch_cancelled_streamed: (deepseek, "deepseek", "corpus_matrix/endings_tool_dispatch_cancelled_streamed", cells::ENDINGS_TOOL_DISPATCH_CANCELLED_STREAMED);
    endings_turn_finished_stop_streamed: (doubleword, "doubleword", "corpus_matrix/endings_turn_finished_stop_streamed", cells::ENDINGS_TURN_FINISHED_STOP_STREAMED);
    endings_tool_outcome_cancelled_streamed: (gemini, "gemini", "corpus_matrix/endings_tool_outcome_cancelled_streamed", cells::ENDINGS_TOOL_OUTCOME_CANCELLED_STREAMED);
    hooks_observe_everything: (openai_chat, "openai", "corpus_matrix_chat/hooks_observe_everything", cells::HOOKS_OBSERVE_EVERYTHING);
    hooks_patch_tool_args: (openai_responses, "openai", "corpus_matrix_responses/hooks_patch_tool_args", cells::HOOKS_PATCH_TOOL_ARGS);
    hooks_patch_tool_args_streamed: (venice, "venice", "corpus_matrix/hooks_patch_tool_args_streamed", cells::HOOKS_PATCH_TOOL_ARGS_STREAMED);
    hooks_deny_tool: (deepseek, "deepseek", "corpus_matrix/hooks_deny_tool", cells::HOOKS_DENY_TOOL);
    hooks_deny_tool_streamed: (doubleword, "doubleword", "corpus_matrix/hooks_deny_tool_streamed", cells::HOOKS_DENY_TOOL_STREAMED);
    hooks_replace_tool_result: (gemini, "gemini", "corpus_matrix/hooks_replace_tool_result", cells::HOOKS_REPLACE_TOOL_RESULT);
    hooks_replace_answer: (openai_chat, "openai", "corpus_matrix_chat/hooks_replace_answer", cells::HOOKS_REPLACE_ANSWER);
    hooks_preamble_override: (openai_responses, "openai", "corpus_matrix_responses/hooks_preamble_override", cells::HOOKS_PREAMBLE_OVERRIDE);
    hooks_demand_done: recorded (venice, "venice", "corpus_matrix/hooks_demand_done", cells::HOOKS_DEMAND_DONE);
    hooks_lookup_before_run: (deepseek, "deepseek", "corpus_matrix/hooks_lookup_before_run", cells::HOOKS_LOOKUP_BEFORE_RUN);
    hooks_two_hooks: (doubleword, "doubleword", "corpus_matrix/hooks_two_hooks", cells::HOOKS_TWO_HOOKS);
    host_custom_at_start: (gemini, "gemini", "corpus_matrix/host_custom_at_start", cells::HOST_CUSTOM_AT_START);
    host_custom_at_completion_call: (openai_chat, "openai", "corpus_matrix_chat/host_custom_at_completion_call", cells::HOST_CUSTOM_AT_COMPLETION_CALL);
    host_custom_at_outcome: (venice, "venice", "corpus_matrix/host_custom_at_outcome", cells::HOST_CUSTOM_AT_OUTCOME);
    host_custom_at_settled: (openai_responses, "openai", "corpus_matrix_responses/host_custom_at_settled", cells::HOST_CUSTOM_AT_SETTLED);
    host_custom_twice_serial: (deepseek, "deepseek", "corpus_matrix/host_custom_twice_serial", cells::HOST_CUSTOM_TWICE_SERIAL);
    host_custom_at_outcome_streamed: (doubleword, "doubleword", "corpus_matrix/host_custom_at_outcome_streamed", cells::HOST_CUSTOM_AT_OUTCOME_STREAMED);
    host_custom_unserved: (gemini, "gemini", "corpus_matrix/host_custom_unserved", cells::HOST_CUSTOM_UNSERVED);
    serving_serial_concurrency_one: (openai_chat, "openai", "corpus_matrix_chat/serving_serial_concurrency_one", cells::SERVING_SERIAL_CONCURRENCY_ONE);
    serving_concurrent_concurrency_one: (openai_responses, "openai", "corpus_matrix_responses/serving_concurrent_concurrency_one", cells::SERVING_CONCURRENT_CONCURRENCY_ONE);
    serving_concurrent_concurrency_two: (venice, "venice", "corpus_matrix/serving_concurrent_concurrency_two", cells::SERVING_CONCURRENT_CONCURRENCY_TWO);
    serving_concurrent_concurrency_two_events: (deepseek, "deepseek", "corpus_matrix/serving_concurrent_concurrency_two_events", cells::SERVING_CONCURRENT_CONCURRENCY_TWO_EVENTS);
    serving_capacity_one: (doubleword, "doubleword", "corpus_matrix/serving_capacity_one", cells::SERVING_CAPACITY_ONE);
    serving_serial_memory_tools: (gemini, "gemini", "corpus_matrix/serving_serial_memory_tools", cells::SERVING_SERIAL_MEMORY_TOOLS);
    serving_model_route: (openai_chat, "openai", "corpus_matrix_chat/serving_model_route", cells::SERVING_MODEL_ROUTE);
    serving_model_route_unselected: (openai_responses, "openai", "corpus_matrix_responses/serving_model_route_unselected", cells::SERVING_MODEL_ROUTE_UNSELECTED);
    serving_host_bus: (venice, "venice", "corpus_matrix/serving_host_bus", cells::SERVING_HOST_BUS);
    serving_host_bus_streamed: (deepseek, "deepseek", "corpus_matrix/serving_host_bus_streamed", cells::SERVING_HOST_BUS_STREAMED);
    layers_deny_tool: (doubleword, "doubleword", "corpus_matrix/layers_deny_tool", cells::LAYERS_DENY_TOOL);
    layers_patch_tool_args: (gemini, "gemini", "corpus_matrix/layers_patch_tool_args", cells::LAYERS_PATCH_TOOL_ARGS);
    layers_replace_tool_result: (openai_chat, "openai", "corpus_matrix_chat/layers_replace_tool_result", cells::LAYERS_REPLACE_TOOL_RESULT);
    layers_two_layers: (openai_responses, "openai", "corpus_matrix_responses/layers_two_layers", cells::LAYERS_TWO_LAYERS);
    layers_host_deny_over_host_bus: (venice, "venice", "corpus_matrix/layers_host_deny_over_host_bus", cells::LAYERS_HOST_DENY_OVER_HOST_BUS);
    layers_patch_beneath_hook_patch: (deepseek, "deepseek", "corpus_matrix/layers_patch_beneath_hook_patch", cells::LAYERS_PATCH_BENEATH_HOOK_PATCH);
    layers_memory_load_replaced: (doubleword, "doubleword", "corpus_matrix/layers_memory_load_replaced", cells::LAYERS_MEMORY_LOAD_REPLACED);
    memory_clear_at_start: (gemini, "gemini", "corpus_matrix/memory_clear_at_start", cells::MEMORY_CLEAR_AT_START);
    memory_clear_at_settled: (openai_chat, "openai", "corpus_matrix_chat/memory_clear_at_settled", cells::MEMORY_CLEAR_AT_SETTLED);
    memory_two_runs: (venice, "venice", "corpus_matrix/memory_two_runs", cells::MEMORY_TWO_RUNS);
    memory_two_runs_streamed: (openai_responses, "openai", "corpus_matrix_responses/memory_two_runs_streamed", cells::MEMORY_TWO_RUNS_STREAMED);
    memory_clear_at_settled_two_runs: (deepseek, "deepseek", "corpus_matrix/memory_clear_at_settled_two_runs", cells::MEMORY_CLEAR_AT_SETTLED_TWO_RUNS);
    memory_clear_at_start_two_runs: (doubleword, "doubleword", "corpus_matrix/memory_clear_at_start_two_runs", cells::MEMORY_CLEAR_AT_START_TWO_RUNS);
    memory_history_bypass: (gemini, "gemini", "corpus_matrix/memory_history_bypass", cells::MEMORY_HISTORY_BYPASS);
    memory_host_bus_memory: (openai_chat, "openai", "corpus_matrix_chat/memory_host_bus_memory", cells::MEMORY_HOST_BUS_MEMORY);
    memory_serial_two_tools: (openai_responses, "openai", "corpus_matrix_responses/memory_serial_two_tools", cells::MEMORY_SERIAL_TWO_TOOLS);
    memory_failing_append: (venice, "venice", "corpus_matrix/memory_failing_append", cells::MEMORY_FAILING_APPEND);
    memory_failing_append_streamed: (deepseek, "deepseek", "corpus_matrix/memory_failing_append_streamed", cells::MEMORY_FAILING_APPEND_STREAMED);
    output_tool_unary: (doubleword, "doubleword", "corpus_matrix/output_tool_unary", cells::OUTPUT_TOOL_UNARY);
    output_tool_streamed: (openai_chat, "openai", "corpus_matrix_chat/output_tool_streamed", cells::OUTPUT_TOOL_STREAMED);
    output_prompted_unary: (gemini, "gemini", "corpus_matrix/output_prompted_unary", cells::OUTPUT_PROMPTED_UNARY);
    output_prompted_streamed: (venice, "venice", "corpus_matrix/output_prompted_streamed", cells::OUTPUT_PROMPTED_STREAMED);
    output_tool_with_real_tool: (openai_responses, "openai", "corpus_matrix_responses/output_tool_with_real_tool", cells::OUTPUT_TOOL_WITH_REAL_TOOL);
    output_prompted_with_real_tool: (deepseek, "deepseek", "corpus_matrix/output_prompted_with_real_tool", cells::OUTPUT_PROMPTED_WITH_REAL_TOOL);
    output_tool_choice_specific_output: (doubleword, "doubleword", "corpus_matrix/output_tool_choice_specific_output", cells::OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT);
    output_tool_choice_required: (gemini, "gemini", "corpus_matrix/output_tool_choice_required", cells::OUTPUT_TOOL_CHOICE_REQUIRED);
    output_tool_under_none_degrades: (openai_chat, "openai", "corpus_matrix_chat/output_tool_under_none_degrades", cells::OUTPUT_TOOL_UNDER_NONE_DEGRADES);
    shaping_tool_choice_required_first: (openai_responses, "openai", "corpus_matrix_responses/shaping_tool_choice_required_first", cells::SHAPING_TOOL_CHOICE_REQUIRED_FIRST);
    shaping_tool_choice_none_on_committed_output: (venice, "venice", "corpus_matrix/shaping_tool_choice_none_on_committed_output", cells::SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT);
    shaping_extra_context: (deepseek, "deepseek", "corpus_matrix/shaping_extra_context", cells::SHAPING_EXTRA_CONTEXT);
    shaping_extra_context_streamed: (doubleword, "doubleword", "corpus_matrix/shaping_extra_context_streamed", cells::SHAPING_EXTRA_CONTEXT_STREAMED);
    shaping_merged_three: (gemini, "gemini", "corpus_matrix/shaping_merged_three", cells::SHAPING_MERGED_THREE);
    shaping_route_on_first_turn: (openai_chat, "openai", "corpus_matrix_chat/shaping_route_on_first_turn", cells::SHAPING_ROUTE_ON_FIRST_TURN);
    shaping_late_route: (openai_responses, "openai", "corpus_matrix_responses/shaping_late_route", cells::SHAPING_LATE_ROUTE);
    shaping_max_tokens_second_turn: (venice, "venice", "corpus_matrix/shaping_max_tokens_second_turn", cells::SHAPING_MAX_TOKENS_SECOND_TURN);
    shaping_active_tools_none_second_turn: (deepseek, "deepseek", "corpus_matrix/shaping_active_tools_none_second_turn", cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN);
    shaping_history_first_turn: (doubleword, "doubleword", "corpus_matrix/shaping_history_first_turn", cells::SHAPING_HISTORY_FIRST_TURN);
    causal_completion_serial: (gemini, "gemini", "corpus_matrix/causal_completion_serial", cells::CAUSAL_COMPLETION_SERIAL);
    causal_completion_streamed: (openai_chat, "openai", "corpus_matrix_chat/causal_completion_streamed", cells::CAUSAL_COMPLETION_STREAMED);
    resume_tool_turn: (openai_responses, "openai", "corpus_matrix_responses/resume_tool_turn", cells::RESUME_TOOL_TURN);
    endings_tool_call_delta_stop: (venice, "venice", "corpus_matrix/endings_tool_call_delta_stop", cells::ENDINGS_TOOL_CALL_DELTA_STOP);
    host_custom_start_and_settled: (deepseek, "deepseek", "corpus_matrix/host_custom_start_and_settled", cells::HOST_CUSTOM_START_AND_SETTLED);
    host_custom_twice_concurrent: (doubleword, "doubleword", "corpus_matrix/host_custom_twice_concurrent", cells::HOST_CUSTOM_TWICE_CONCURRENT);
    host_custom_at_start_streamed: (gemini, "gemini", "corpus_matrix/host_custom_at_start_streamed", cells::HOST_CUSTOM_AT_START_STREAMED);
    shaping_preamble_second_turn: (openai_chat, "openai", "corpus_matrix_chat/shaping_preamble_second_turn", cells::SHAPING_PREAMBLE_SECOND_TURN);
    causal_completion_concurrent: (deepseek, "deepseek", "corpus_matrix/causal_completion_concurrent", cells::CAUSAL_COMPLETION_CONCURRENT);
    output_tool_thinking_deepseek: (deepseek, "deepseek", "corpus_matrix/output_tool_thinking", cells::OUTPUT_TOOL_THINKING);
    output_tool_thinking_doubleword: (doubleword, "doubleword", "corpus_matrix/output_tool_thinking", cells::OUTPUT_TOOL_THINKING);
    output_tool_thinking_openai_chat: recorded (openai_chat, "openai", "corpus_matrix_chat/output_tool_thinking", cells::OUTPUT_TOOL_THINKING);
    output_tool_thinking_openai_responses: (openai_responses, "openai", "corpus_matrix_responses/output_tool_thinking", cells::OUTPUT_TOOL_THINKING);
    reasoning_text_unary_deepseek: recorded (deepseek, "deepseek", "reasoning_matrix/text_unary", cells::REASONING_TEXT_UNARY);
    reasoning_text_unary_doubleword: (doubleword, "doubleword", "reasoning_matrix/text_unary", cells::REASONING_TEXT_UNARY);
    reasoning_text_unary_openai_responses: (openai_responses, "openai", "reasoning_matrix_responses/text_unary", cells::REASONING_TEXT_UNARY);
    reasoning_text_unary_venice: (venice, "venice", "reasoning_matrix/text_unary", cells::REASONING_TEXT_UNARY);
    reasoning_text_streamed_deepseek: (deepseek, "deepseek", "reasoning_matrix/text_streamed", cells::REASONING_TEXT_STREAMED);
    reasoning_text_streamed_doubleword: (doubleword, "doubleword", "reasoning_matrix/text_streamed", cells::REASONING_TEXT_STREAMED);
    reasoning_text_streamed_gemini: recorded (gemini, "gemini", "reasoning_matrix/text_streamed", cells::REASONING_TEXT_STREAMED);
    reasoning_text_streamed_openai_chat: recorded (openai_chat, "openai", "reasoning_matrix_chat/text_streamed", cells::REASONING_TEXT_STREAMED);
    reasoning_text_streamed_openai_responses: (openai_responses, "openai", "reasoning_matrix_responses/text_streamed", cells::REASONING_TEXT_STREAMED);
    reasoning_text_streamed_venice: (venice, "venice", "reasoning_matrix/text_streamed", cells::REASONING_TEXT_STREAMED);
    reasoning_tool_unary_deepseek: (deepseek, "deepseek", "reasoning_matrix/tool_unary", cells::REASONING_TOOL_UNARY);
    reasoning_tool_unary_doubleword: (doubleword, "doubleword", "reasoning_matrix/tool_unary", cells::REASONING_TOOL_UNARY);
    reasoning_tool_unary_venice: (venice, "venice", "reasoning_matrix/tool_unary", cells::REASONING_TOOL_UNARY);
    reasoning_tool_streamed_deepseek: (deepseek, "deepseek", "reasoning_matrix/tool_streamed", cells::REASONING_TOOL_STREAMED);
    reasoning_tool_streamed_doubleword: (doubleword, "doubleword", "reasoning_matrix/tool_streamed", cells::REASONING_TOOL_STREAMED);
    reasoning_tool_streamed_venice: (venice, "venice", "reasoning_matrix/tool_streamed", cells::REASONING_TOOL_STREAMED);
    reasoning_off_deepseek: (deepseek, "deepseek", "reasoning_matrix/off", cells::REASONING_OFF);
    reasoning_off_doubleword: (doubleword, "doubleword", "reasoning_matrix/off", cells::REASONING_OFF);
    reasoning_off_openai_chat: (openai_chat, "openai", "reasoning_matrix_chat/off", cells::REASONING_OFF);
    reasoning_off_venice: (venice, "venice", "reasoning_matrix/off", cells::REASONING_OFF);
    reasoning_capped_deepseek: (deepseek, "deepseek", "reasoning_matrix/capped", cells::REASONING_CAPPED);
    reasoning_capped_doubleword: (doubleword, "doubleword", "reasoning_matrix/capped", cells::REASONING_CAPPED);
    reasoning_capped_gemini: (gemini, "gemini", "reasoning_matrix/capped", cells::REASONING_CAPPED);
    reasoning_capped_openai_chat: (openai_chat, "openai", "reasoning_matrix_chat/capped", cells::REASONING_CAPPED);
    reasoning_capped_openai_responses: (openai_responses, "openai", "reasoning_matrix_responses/capped", cells::REASONING_CAPPED);
    reasoning_capped_venice: (venice, "venice", "reasoning_matrix/capped", cells::REASONING_CAPPED);
    shaping_thinking_second_turn_doubleword: (doubleword, "doubleword", "corpus_matrix/shaping_thinking_second_turn", cells::SHAPING_THINKING_SECOND_TURN);
    reasoning_capped_streamed_gemini: recorded (gemini, "gemini", "reasoning_matrix/capped_streamed", cells::REASONING_CAPPED_STREAMED);
    reasoning_capped_streamed_openai_responses: (openai_responses, "openai", "reasoning_matrix_responses/capped_streamed", cells::REASONING_CAPPED_STREAMED);
}

// The image cells (`tests/common/ecs_matrix/image.rs`): the image in every
// request and in history, then the cell's answer.
cells! {
    image_inline_text_unary: recorded (anthropic_sonnet, "anthropic", "image_matrix/inline_text_unary", cells::IMAGE_INLINE_TEXT_UNARY);
    image_inline_text_streamed: recorded (gemini_preview, "gemini", "image_matrix/inline_text_streamed", cells::IMAGE_INLINE_TEXT_STREAMED);
    image_inline_mixed_order: recorded (openai_chat_image, "openai", "image_matrix_chat/inline_mixed_order", cells::IMAGE_INLINE_MIXED_ORDER);
    image_inline_tool_unary: recorded (openai_responses_image, "openai", "image_matrix_responses/inline_tool_unary", cells::IMAGE_INLINE_TOOL_UNARY);
    image_inline_tool_streamed: recorded (anthropic_sonnet, "anthropic", "image_matrix/inline_tool_streamed", cells::IMAGE_INLINE_TOOL_STREAMED);
    image_inline_followup: recorded (gemini_preview, "gemini", "image_matrix/inline_followup", cells::IMAGE_INLINE_FOLLOWUP);
    image_url_text_unary: recorded (openai_chat_image, "openai", "image_matrix_chat/url_text_unary", cells::IMAGE_URL_TEXT_UNARY);
    image_url_tool_unary: recorded (openai_responses_image, "openai", "image_matrix_responses/url_tool_unary", cells::IMAGE_URL_TOOL_UNARY);
}
