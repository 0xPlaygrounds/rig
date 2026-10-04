//! Native long loops on Gemini 2.5 Flash and long tasks on Gemini 3.8 Flash.
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::Scripted`), no
//! cassette; the negative probe mutates the streamed recording's last tool
//! result and proves the strict matcher refuses it.

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells};
use rig_test_support::cassette_models::GeminiModels;

fn task_wire(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        model: client.completion("gemini-3.8-flash"),
        thinking: cells::ThinkingWire::Gemini,
        route: None,
        temperature: Some(0.0),
        additional_params: Some(
            || serde_json::json!({"generationConfig":{"thinkingConfig":{"thinkingLevel":"low"}}}),
        ),
    }
}

fn assert_task_requests(scenario: &str) {
    crate::ecs_matrix::long_tasks::assert_requests("gemini", scenario);
}

crate::matrix::resume_matrix! {
    wrapper: with_gemini_cassette, wire: task_wire, run: crate::ecs_matrix::long_tasks::run_world, after: assert_task_requests;
    #[tokio::test]
    task_repair_streamed: ("long_task_matrix/repair_streamed", crate::ecs_matrix::long_tasks::REPAIR_STREAMED, None, "gemini_long_task_repair_streamed");
    #[tokio::test]
    task_reconcile: ("long_task_matrix/reconcile", crate::ecs_matrix::long_tasks::RECONCILE, None, "gemini_long_task_reconcile");
}

crate::matrix::resume_matrix! {
    wrapper: with_gemini_cassette, wire: task_wire, run: crate::ecs_matrix::long_tasks::run_world, after: assert_task_requests;
    #[tokio::test]
    task_inventory: ("long_task_matrix/inventory", crate::ecs_matrix::long_tasks::INVENTORY, None, "gemini_long_task_inventory");
}
