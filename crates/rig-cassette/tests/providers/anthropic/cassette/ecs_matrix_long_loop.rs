//! The long tool loop's native column on Anthropic: claude-haiku-4-5-20251001.
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::Scripted`), no
//! cassette; the negative probe mutates the streamed recording's last tool
//! result and proves the strict matcher refuses it.

use rig_test_support::cassette_models::AnthropicModels;
use rig_test_support::cassette_models::MapWire;

use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, cells};

const THINKING: cells::ThinkingWire = cells::ThinkingWire::Anthropic;

fn task_wire(
    client: &AnthropicModels,
) -> Wire<rig::Model<rig::providers::anthropic::wire::Messages>> {
    Wire {
        thinking: THINKING,
        model: client
            .completion("claude-haiku-4-5-20251001")
            .map_wire(|wire| wire.with_prompt_caching()),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The harness carries per-cell request parameters only as raw JSON, so the
/// one-hour automatic marker (`CacheRetention::Long`'s body) is sent raw.
fn automatic_task_wire(
    client: &AnthropicModels,
) -> Wire<rig::Model<rig::providers::anthropic::wire::Messages>> {
    Wire {
        thinking: THINKING,
        model: client.completion("claude-haiku-4-5-20251001"),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(
            || serde_json::json!({"cache_control": {"type": "ephemeral", "ttl": "1h"}}),
        ),
    }
}

fn mixed_task_wire(
    client: &AnthropicModels,
) -> Wire<rig::Model<rig::providers::anthropic::wire::Messages>> {
    Wire {
        thinking: THINKING,
        model: client
            .completion("claude-haiku-4-5-20251001")
            .map_wire(|wire| {
                wire.with_static_prefix_cache_ttl(
                    rig::providers::anthropic::completion::CacheTtl::OneHour,
                )
            }),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(|| serde_json::json!({"cache_control": {"type": "ephemeral"}})),
    }
}

fn assert_task_requests(scenario: &str) {
    crate::ecs_matrix::long_tasks::assert_requests("anthropic", scenario);
}

crate::matrix::resume_matrix! {
    wrapper: with_anthropic_cassette, wire: task_wire, run: crate::ecs_matrix::long_tasks::run_world, after: assert_task_requests;
    #[tokio::test]
    #[ignore = "Anthropic workspace API quota exhausted; native recording returned HTTP 400, reset 2026-10-01; unrecorded"]
    task_repair: ("long_task_matrix/repair", crate::ecs_matrix::long_tasks::REPAIR, None, "anthropic_long_task_repair");
    #[tokio::test]
    #[ignore = "Anthropic workspace API quota exhausted; unrecorded"]
    task_reconcile: ("long_task_matrix/reconcile", crate::ecs_matrix::long_tasks::RECONCILE, None, "anthropic_long_task_reconcile");
}

crate::matrix::resume_matrix! {
    wrapper: with_anthropic_cassette, wire: automatic_task_wire, run: crate::ecs_matrix::long_tasks::run_world, after: assert_task_requests;
    #[tokio::test]
    #[ignore = "Anthropic workspace API quota exhausted; unrecorded"]
    task_repair_streamed: ("long_task_matrix/repair_streamed", crate::ecs_matrix::long_tasks::REPAIR_STREAMED, None, "anthropic_long_task_repair_streamed");
}

crate::matrix::resume_matrix! {
    wrapper: with_anthropic_cassette, wire: mixed_task_wire, run: crate::ecs_matrix::long_tasks::run_world, after: assert_task_requests;
    #[tokio::test]
    #[ignore = "Anthropic workspace API quota exhausted; unrecorded"]
    task_inventory: ("long_task_matrix/inventory", crate::ecs_matrix::long_tasks::INVENTORY, None, "anthropic_long_task_inventory");
    #[tokio::test]
    #[ignore = "Anthropic workspace API quota exhausted; baseline unrecorded"]
    task_inventory_restore: ("long_task_matrix/inventory", crate::ecs_matrix::long_tasks::INVENTORY, Some(5), "anthropic_long_task_inventory_restore");
}
