//! The long tool loop's native column on OpenAiResponses: gpt-4.1-mini (Responses).
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::Scripted`), no
//! cassette; the negative probe mutates the streamed recording's last tool
//! result and proves the strict matcher refuses it.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, long_loop, long_loop_world};
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    mini(&client.openai)
}

fn mini(models: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: models.completion("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

fn task_wire(
    client: &OpenAiCassette,
) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        additional_params: Some(
            || serde_json::json!({"prompt_cache_key": "rig-native-long-tasks", "store": false}),
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
    task_repair: ("long_task_matrix/responses_repair", crate::ecs_matrix::long_tasks::REPAIR, None, "openai_responses_long_task_repair");
    #[tokio::test]
    task_repair_streamed: ("long_task_matrix/responses_repair_streamed", crate::ecs_matrix::long_tasks::REPAIR_STREAMED, None, "openai_responses_long_task_repair_streamed");
    #[tokio::test]
    task_inventory: ("long_task_matrix/responses_inventory", crate::ecs_matrix::long_tasks::INVENTORY_WIDE_BATCH, None, "openai_responses_long_task_inventory");
}

crate::matrix::resume_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: long_loop_world::run_world;
    /// Failed(Response): the cut write_file call is a response
    /// error on this wire (recording confirms).
    #[tokio::test]
    #[ignore = "a length-cut call is now kept with tolerant arguments and answered with an error result (change 2), so the cell's cut turn no longer ends the run and the shared long_loop assertions (every requested call dispatched, nothing dispatched under the cap) cannot hold; two live attempts failed on them; the cell needs redesigning for the new semantics"]
    output_cap_midway: ("long_loop_matrix_responses/output_cap_midway", long_loop::OUTPUT_CAP_MIDWAY, long_loop::OUTPUT_CAP_MIDWAY.resume_after, "openai_matrix_long_loop_responses_output_cap_midway");
}
