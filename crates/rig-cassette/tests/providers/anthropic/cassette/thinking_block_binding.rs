//! Claude Opus 5.5 binds each thinking block to the conversation it was
//! produced in, tools included. After the tool list changes, a request that
//! replays an earlier thinking block is refused unless it asks the API to drop
//! the stale block, which rig does by default.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `ANTHROPIC_API_KEY`; see
//! `tests/README.md`.

use rig::AgentBuilder;
use rig::agent::Agent;
use rig::completion::Message;
use rig::message::AssistantContent;
use rig::providers::anthropic::completion::CLAUDE_OPUS_5_5;
use rig::providers::anthropic::{THINKING_BINDING_BETA, ThinkingPrefixMismatch};
use rig_test_support::cache_longrun::workloads::{OrderHistory, ToolSchedule};
use rig_test_support::cache_longrun::{LookupOrder, SUPPORT_PREAMBLE, question};
use rig_test_support::cassette_models::AnthropicModels;
use serde_json::Value;

use super::super::support::with_anthropic_cassette;

const SCENARIO: &str = "thinking_block_binding/changed_tools";

fn agent(models: &AnthropicModels, schedule: &ToolSchedule) -> Agent {
    AgentBuilder::new(models.completion(CLAUDE_OPUS_5_5))
        .preamble(SUPPORT_PREAMBLE)
        .tool(LookupOrder)
        .tool(OrderHistory)
        .max_tokens(800)
        .default_max_turns(4)
        .add_hook(schedule.clone())
        .build()
}

fn has_reasoning(history: &[Message]) -> bool {
    history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|part| matches!(part, AssistantContent::Reasoning(_)))
        )
    })
}

fn replays_thinking(body: &Value) -> bool {
    body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|message| {
            message["content"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|block| {
                    matches!(
                        block["type"].as_str(),
                        Some("thinking" | "redacted_thinking")
                    )
                })
        })
}

/// The support chat runs on one tool until the model has produced thinking to
/// replay, then advertises a second tool. That turn is refused when sent
/// without a binding; sent as rig sends it by default, the API drops the stale
/// thinking block and answers.
#[tokio::test]
async fn a_changed_tool_list_drops_stale_thinking_blocks() {
    with_anthropic_cassette(
        "thinking_block_binding/changed_tools",
        |models| async move {
            let schedule = ToolSchedule::new(
                1,
                vec![vec!["lookup_order"], vec!["lookup_order", "order_history"]],
            );
            let dropping = agent(&models, &schedule);
            let rejecting = agent(
                &models.clone().map_config(|config| {
                    config.with_thinking_prefix_mismatch(ThinkingPrefixMismatch::Reject)
                }),
                &schedule,
            );

            let mut history = Vec::new();
            let mut turn = 0;
            schedule.start_turn(1);
            while !has_reasoning(&history) {
                turn += 1;
                assert!(turn <= 3, "three turns produced no thinking to replay");
                dropping
                    .chat(question(turn, "T"), &mut history)
                    .await
                    .expect("a turn on the first tool set completes");
            }

            schedule.start_turn(2);
            let changed = question(turn + 1, "T");
            let error = rejecting
                .chat(changed.as_str(), &mut history.clone())
                .await
                .expect_err("a replayed block under a changed tool list is refused");
            let error = error.to_string();
            assert!(
                error.contains("400")
                    && error.contains(
                        "The `tools` list differs from the one this block was created with"
                    ),
                "the refusal is the thinking-block binding: {error}"
            );

            dropping
                .chat(changed.as_str(), &mut history)
                .await
                .expect("the stale block is dropped and the turn completes");
        },
    )
    .await;

    let turns = crate::cassettes::recorded_json_turns("anthropic", SCENARIO);
    let statuses = crate::cassettes::recorded_statuses_and_bodies("anthropic", SCENARIO);
    let headers = crate::cassettes::recorded_request_header_pairs("anthropic", SCENARIO);
    let refused = statuses
        .iter()
        .position(|(status, _)| *status == 400)
        .expect("one request was refused");
    let (rejected, _) = &turns[refused];
    let (dropped, _) = &turns[refused + 1];
    let beta = |index: usize| {
        headers[index]
            .iter()
            .find(|(name, _)| name == "anthropic-beta")
            .map(|(_, value)| value.clone())
    };

    assert!(
        replays_thinking(rejected),
        "the refused request replays thinking"
    );
    assert_eq!(rejected.get("thinking"), None);
    assert_eq!(beta(refused), None);

    assert_eq!(statuses[refused + 1].0, 200);
    assert_eq!(dropped["messages"], rejected["messages"]);
    assert_eq!(dropped["tools"], rejected["tools"]);
    assert_eq!(
        dropped["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
    assert_eq!(beta(refused + 1).as_deref(), Some(THINKING_BINDING_BETA));
}
