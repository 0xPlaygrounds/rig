//! Migrated from `examples/anthropic_think_tool.rs`.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::anthropic;
use rig_core::tools::ThinkTool;

use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn think_tool_menu_planning() {
    let agent = anthropic::Client::from_env()
        .expect("client should build")
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
}
