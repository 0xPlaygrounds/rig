//! The long tool loop's native column on OpenAiChat: gpt-4.1-mini (Chat Completions).
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::Scripted`), no
//! cassette; the negative probe mutates the streamed recording's last tool
//! result and proves the strict matcher refuses it.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, long_loop, long_loop_world};
use rig::providers::openai::wire::OpenAIConfig;
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::Chat>> {
    mini(&client.openai)
}

fn mini(models: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::Chat>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: models.chat("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

fn task_wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::Chat>> {
    Wire {
        additional_params: Some(
            || serde_json::json!({"prompt_cache_key": "rig-native-long-tasks"}),
        ),
        ..wire(client)
    }
}

fn assert_task_requests(scenario: &str) {
    crate::ecs_matrix::long_tasks::assert_requests("openai", scenario);
}

crate::matrix::resume_matrix! {
    wrapper: with_openai_cassette, wire: task_wire, run: crate::ecs_matrix::long_tasks::run_world, after: assert_task_requests;
    #[tokio::test]
    task_repair: ("long_task_matrix/chat_repair", crate::ecs_matrix::long_tasks::REPAIR, None, "openai_chat_long_task_repair");
    #[tokio::test]
    task_repair_streamed: ("long_task_matrix/chat_repair_streamed", crate::ecs_matrix::long_tasks::REPAIR_STREAMED, None, "openai_chat_long_task_repair_streamed");
    #[tokio::test]
    task_reconcile: ("long_task_matrix/chat_reconcile", crate::ecs_matrix::long_tasks::RECONCILE, None, "openai_chat_long_task_reconcile");
    #[tokio::test]
    task_inventory: ("long_task_matrix/chat_inventory", crate::ecs_matrix::long_tasks::INVENTORY, None, "openai_chat_long_task_inventory");
    #[tokio::test]
    task_inventory_restore: ("long_task_matrix/chat_inventory", crate::ecs_matrix::long_tasks::INVENTORY, Some(5), "openai_chat_long_task_inventory_restore");
}

/// The scripted rows' wire; the provider fault rewrites the recorded setup
/// failure to a retryable status.
const SCRIPTED: long_loop::Scripted<rig::Model<rig::providers::openai::wire::Chat>> =
    long_loop::Scripted {
        wire: |http| {
            mini(&OpenAiModels::new(
                OpenAIConfig::new("sk-scripted-fault-key-7f3a9c"),
                http,
            ))
        },
        fault_reply: || {
            crate::stream_faults::status_reply(
                "openai",
                "corpus_faults_chat/setup_unary",
                503,
                false,
            )
        },
    };

crate::matrix::resume_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop_world::run_world;
    #[tokio::test]
    long_unary: ("long_loop_matrix_chat/long_unary", long_loop::LONG_UNARY, None, "openai_chat_long_unary");
    #[tokio::test]
    long_unary_cut_1: ("long_loop_matrix_chat/long_unary", long_loop::LONG_UNARY, Some(1), "openai_chat_long_unary_cut_1");
    #[tokio::test]
    long_unary_cut_2: ("long_loop_matrix_chat/long_unary", long_loop::LONG_UNARY, Some(2), "openai_chat_long_unary_cut_2");
    #[tokio::test]
    long_unary_cut_3: ("long_loop_matrix_chat/long_unary", long_loop::LONG_UNARY, Some(3), "openai_chat_long_unary_cut_3");
    #[tokio::test]
    long_unary_cut_final: ("long_loop_matrix_chat/long_unary", long_loop::LONG_UNARY, Some(usize::MAX), "openai_chat_long_unary_cut_final");
}

crate::matrix::resume_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop_world::run_world;
    #[tokio::test]
    long_streamed: ("long_loop_matrix_chat/long_streamed", long_loop::LONG_STREAMED, long_loop::LONG_STREAMED.resume_after, "openai_chat_long_streamed");
    #[tokio::test]
    parallel_calls: ("long_loop_matrix_chat/parallel_calls", long_loop::PARALLEL_CALLS, long_loop::PARALLEL_CALLS.resume_after, "openai_chat_parallel_calls");
    #[tokio::test]
    big_result: ("long_loop_matrix_chat/big_result", long_loop::BIG_RESULT, long_loop::BIG_RESULT.resume_after, "openai_chat_big_result");
    #[tokio::test]
    tool_error_midway: ("long_loop_matrix_chat/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY, long_loop::TOOL_ERROR_MIDWAY.resume_after, "openai_chat_tool_error_midway");
    #[tokio::test]
    max_turns_midway: ("long_loop_matrix_chat/max_turns_midway", long_loop::MAX_TURNS_MIDWAY, long_loop::MAX_TURNS_MIDWAY.resume_after, "openai_chat_max_turns_midway");
    /// Failed(Response): the chat decoder drops the cut call and
    /// rig-agent refuses the turn ("produced no answer ...
    /// finish_reason=Length"; round 3).
    #[tokio::test]
    output_cap_midway: ("long_loop_matrix_chat/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY, long_loop::OUTPUT_CAP_MIDWAY.resume_after, "openai_matrix_long_loop_chat_output_cap_midway");
}

crate::matrix::case_matrix! {
    family: ecs_faults_case;
    #[tokio::test]
    invalid_args_midway: SCRIPTED => "openai_matrix_long_loop_chat_invalid_args_midway";
    #[tokio::test]
    provider_fault_midway: SCRIPTED => "openai_matrix_long_loop_chat_provider_fault_midway";
}

/// Negative matcher probe against the streamed loop the native consumers
/// replay: the last tool result's final byte, altered, is refused.
#[tokio::test]
async fn streamed_tool_result_mismatch() {
    long_loop::assert_stream_request_rejected("openai", "long_loop_matrix_chat/long_streamed")
        .await;
}
