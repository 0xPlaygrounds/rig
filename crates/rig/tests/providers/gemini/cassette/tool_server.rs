//! Runtime mutation of a shared `ToolServerHandle`: tools added or removed
//! between turns change the definitions advertised on the next request, and a
//! single handle backs multiple agents. This is the surface `McpClientHandler`
//! drives on tool-list-changed notifications, so its semantics must survive
//! the rmcp migration unchanged.

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::providers::gemini;
use rig::tool::server::ToolServer;

use super::super::agent_run_support::{history_has_assistant_tool_call, tool_result_texts};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract, FORCE_TOOLS_PREAMBLE};
use crate::support::assert_mentions_expected_number;

#[tokio::test]
async fn add_tool_between_turns_appears_in_next_request() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let subtract_counter = subtract.counter.clone();

    with_gemini_cassette(
        "tool_server/add_tool_between_turns_appears_in_next_request",
        |client| async move {
            let handle = ToolServer::new().tool(add).run();
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_server_handle(handle.clone())
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let first = agent
                .chat("What is 19 + 23?", &mut history)
                .await
                .expect("first prompt should succeed with only the add tool");
            assert_mentions_expected_number(&first, 42);

            handle
                .add_tool(subtract)
                .await
                .expect("adding a tool to a live server should succeed");

            let mut history = Vec::<Message>::new();
            let second = agent
                .chat("What is 50 - 8?", &mut history)
                .await
                .expect("second prompt should see the newly added subtract tool");

            assert_mentions_expected_number(&second, 42);
            assert!(
                history_has_assistant_tool_call(&history, "subtract"),
                "the added tool should be called on the next request: {history:?}"
            );
            assert_eq!(
                subtract_counter.count(),
                1,
                "the added tool should execute exactly once"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn remove_tool_between_turns_drops_definition() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_counter = add.counter.clone();

    with_gemini_cassette(
        "tool_server/remove_tool_between_turns_drops_definition",
        |client| async move {
            let handle = ToolServer::new().tool(add).tool(subtract).run();
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_server_handle(handle.clone())
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let first = agent
                .chat("What is 19 + 23?", &mut history)
                .await
                .expect("first prompt should succeed with both tools advertised");
            assert_mentions_expected_number(&first, 42);
            assert_eq!(add_counter.count(), 1, "add should execute on the first prompt");

            handle
                .remove_tool("subtract")
                .await
                .expect("removing a tool from a live server should succeed");

            let mut history = Vec::<Message>::new();
            let second = agent
                .chat(
                    "List the names of the tools you currently have access to, as a plain comma-separated list.",
                    &mut history,
                )
                .await
                .expect("second prompt should succeed with the reduced tool set");

            assert!(
                second.to_ascii_lowercase().contains("add"),
                "the remaining tool should still be advertised: {second:?}"
            );
            assert!(
                !second.to_ascii_lowercase().contains("subtract"),
                "the removed tool should no longer be advertised: {second:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn shared_tool_server_handle_updates_all_agents() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let subtract_counter = subtract.counter.clone();

    with_gemini_cassette(
        "tool_server/shared_tool_server_handle_updates_all_agents",
        |client| async move {
            let handle = ToolServer::new().tool(add).run();
            let first_agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_server_handle(handle.clone())
                .default_max_turns(3)
                .build();
            let second_agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_server_handle(handle.clone())
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let first = first_agent
                .chat("What is 19 + 23?", &mut history)
                .await
                .expect("the first agent should use the shared add tool");
            assert_mentions_expected_number(&first, 42);

            handle
                .add_tool(subtract)
                .await
                .expect("adding a tool to the shared server should succeed");

            let mut history = Vec::<Message>::new();
            let second = second_agent
                .chat("What is 50 - 8?", &mut history)
                .await
                .expect("the second agent should see the tool added through the shared handle");

            assert_mentions_expected_number(&second, 42);
            let result_texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
            assert_eq!(
                result_texts,
                vec!["42".to_string()],
                "the shared tool should execute for the second agent"
            );
            assert_eq!(subtract_counter.count(), 1);
        },
    )
    .await;
}
