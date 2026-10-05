//! The long tool loop's native column on DeepSeek: deepseek-flash (thinking disabled).
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::Scripted`), no
//! cassette; the negative probe mutates the streamed recording's last tool
//! result and proves the strict matcher refuses it.

use crate::deepseek::support::with_deepseek_cassette;
use crate::ecs_matrix::{Wire, cells};
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(|| serde_json::json!({"thinking":{"type":"disabled"}})),
    }
}

fn assert_task_requests(scenario: &str) {
    crate::ecs_matrix::long_tasks::assert_requests("deepseek", scenario);
}

crate::matrix::resume_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: crate::ecs_matrix::long_tasks::run_world, after: assert_task_requests;
    #[tokio::test]
    task_inventory: ("long_task_matrix/inventory", crate::ecs_matrix::long_tasks::INVENTORY, None, "deepseek_long_task_inventory");
}
