//! Migrated from `examples/agent_with_tools.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;
use rig::tool::ToolDyn;

use crate::support::{Adder, Subtract, assert_mentions_expected_number};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn boxed_tools_prompt() {
    let client = openai::Client::from_env();
    let tools: Vec<Box<dyn ToolDyn>> = vec![Box::new(Adder), Box::new(Subtract)];
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             You must use the provided tools before answering.",
        )
        .tools(tools)
        .max_tokens(1024)
        .build();

    let response = agent
        .prompt("Calculate 2 - 5.")
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
