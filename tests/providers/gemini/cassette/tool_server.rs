//! Runtime mutation of the advertised tool set between turns. These cassettes
//! were recorded against the removed `ToolServer`/`ToolServerHandle` registry:
//! tools added or removed between turns changed the definitions advertised on
//! the next request. With the data-oriented runtime the executable set is
//! fixed per agent, so each mutation is ported as building a fresh agent with
//! the post-mutation tool set — the per-request tool declarations (and thus
//! the recorded wire bytes) are identical, and the model-visible consequences
//! (which tools are advertised, called, and executed) stay pinned.

use rig::completion::{Chat, Message};
use rig::prelude::*;
use rig::providers::gemini;

use super::super::agent_run_support::{history_has_assistant_tool_call, tool_result_texts};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract, FORCE_TOOLS_PREAMBLE};
use crate::support::assert_mentions_expected_number;

fn calculator_agent(
    client: &gemini::Client,
    add: CountingAdd,
    subtract: Option<CountingSubtract>,
) -> rig::agent::Agent {
    let builder = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(FORCE_TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool(add);
    let builder = match subtract {
        Some(subtract) => builder.tool(subtract),
        None => builder,
    };
    builder.default_max_turns(3).build()
}

#[tokio::test]
async fn add_tool_between_turns_appears_in_next_request() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let subtract_counter = subtract.counter.clone();

    with_gemini_cassette(
        "tool_server/add_tool_between_turns_appears_in_next_request",
        |client| async move {
            let agent = calculator_agent(&client, add.clone(), None);

            let mut history = Vec::<Message>::new();
            let first = agent
                .chat("What is 19 + 23?", &mut history)
                .await
                .expect("first prompt should succeed with only the add tool");
            assert_mentions_expected_number(&first, 42);

            // Classic: `handle.add_tool(subtract)` on the shared server.
            // Ported: the next prompt runs with the widened tool set.
            let agent = calculator_agent(&client, add, Some(subtract));

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
            let agent = calculator_agent(&client, add.clone(), Some(subtract));

            let mut history = Vec::<Message>::new();
            let first = agent
                .chat("What is 19 + 23?", &mut history)
                .await
                .expect("first prompt should succeed with both tools advertised");
            assert_mentions_expected_number(&first, 42);
            assert_eq!(add_counter.count(), 1, "add should execute on the first prompt");

            // Classic: `handle.remove_tool("subtract")` on the shared server.
            // Ported: the next prompt runs with the narrowed tool set.
            let agent = calculator_agent(&client, add, None);

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
            let first_agent = calculator_agent(&client, add.clone(), None);
            // Classic: a second agent shared the same handle and saw the
            // `add_tool(subtract)` mutation. Ported: the second agent is
            // rebuilt with the widened tool set before its prompt.
            let second_agent = calculator_agent(&client, add, Some(subtract));

            let mut history = Vec::<Message>::new();
            let first = first_agent
                .chat("What is 19 + 23?", &mut history)
                .await
                .expect("the first agent should use the shared add tool");
            assert_mentions_expected_number(&first, 42);

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
