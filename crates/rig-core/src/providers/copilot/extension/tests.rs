//! Copilot's extras read from recorded replies on both routes.

use serde_json::json;

use super::*;
use crate::providers::copilot::CopilotConfig;
use crate::providers::openai::extension::OpenAi;
use crate::test_utils::provider_extensions::{recorded_reply, reply_of};

fn copilot() -> CopilotConfig {
    CopilotConfig::new("tid=copilot-session-token")
}

#[tokio::test]
async fn chat_extras_from_a_unary_recording() {
    let reply = reply_of(
        copilot().completion(crate::providers::copilot::GPT_4O),
        recorded_reply("copilot", "agent/completion_smoke", 0),
    )
    .await;
    let extras = reply
        .extras::<Copilot>()
        .unwrap_or_else(|| panic!("a Copilot reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    let usage = extras.copilot_usage.unwrap_or_default();
    assert_eq!(usage.total_nano_aiu, Some(0));
    let first = usage
        .token_details
        .unwrap_or_default()
        .first()
        .cloned()
        .unwrap_or_default();
    assert_eq!(first.batch_size, Some(1_000_000));
    assert_eq!(first.cost_per_batch, Some(0));
    assert_eq!(first.token_count, Some(38));
    assert_eq!(first.token_type.as_deref(), Some("input"));
    let filters = extras.prompt_filter_results.unwrap_or_default();
    assert_eq!(
        filters.first().map(|result| &result["prompt_index"]),
        Some(&json!(0))
    );
    assert!(reply.extras::<OpenAi>().is_none());
}

#[tokio::test]
async fn responses_extras_from_a_unary_recording() {
    let reply = reply_of(
        copilot().completion(crate::providers::copilot::GPT_5_3_CODEX),
        recorded_reply("copilot", "reasoning_roundtrip/nonstreaming", 0),
    )
    .await;
    let extras = reply
        .extras::<Copilot>()
        .unwrap_or_else(|| panic!("a Copilot reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    let usage = extras.copilot_usage.unwrap_or_default();
    let first = usage
        .token_details
        .unwrap_or_default()
        .first()
        .cloned()
        .unwrap_or_default();
    assert_eq!(first.token_count, Some(73));
    assert_eq!(first.cost_per_batch, Some(175_000_000_000));
    assert_eq!(usage.total_nano_aiu, Some(726_775_000));
    assert_eq!(extras.prompt_filter_results, None);
}
