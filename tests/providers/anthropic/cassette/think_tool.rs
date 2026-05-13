//! Migrated from `examples/anthropic_think_tool.rs`.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::anthropic;
use rig::tools::ThinkTool;

use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

#[tokio::test]
async fn think_tool_menu_planning() {
    let (cassette, client) =
        super::super::support::anthropic_cassette("think_tool/think_tool_menu_planning").await;
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .name("Anthropic Thinker")
        .preamble(
            "You are a helpful assistant that can solve complex problems. \
             Use the 'think' tool to reason through complex problems step by step.",
        )
        .tool(ThinkTool)
        .build();

    let response = agent
        .prompt(
            "I need to plan a dinner party for 8 people, including 2 vegetarians and \
             1 person with a gluten allergy. Create appetizers, mains, and desserts.",
        )
        .max_turns(10)
        .await
        .expect("think tool prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["appetizer", "main", "dessert"]);

    cassette.finish().await;
}
