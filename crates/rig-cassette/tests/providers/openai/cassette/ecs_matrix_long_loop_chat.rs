//! The long tool loop's native column on OpenAiChat: gpt-4.1-mini (Chat Completions).
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::Scripted`), no
//! cassette; the negative probe mutates the streamed recording's last tool
//! result and proves the strict matcher refuses it.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells};
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
        options: None,
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
    task_inventory_restore: ("long_task_matrix/chat_inventory", crate::ecs_matrix::long_tasks::INVENTORY, Some(5), "openai_chat_long_task_inventory_restore");
}
