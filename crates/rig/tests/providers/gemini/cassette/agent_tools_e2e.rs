//! End-to-end tool execution through the built-in agent drivers: real
//! `ToolSet` execution behind `agent.prompt()` / `agent.chat()` /
//! `agent.stream_prompt()`, pinning the wire contract of the handrolled tool
//! pipeline ahead of the rmcp migration.

use rig::client::CompletionClient;
use rig::completion::{Chat, Message, Prompt};
use rig::providers::gemini;
use rig::streaming::StreamingPrompt;

use super::super::agent_run_support::{
    assistant_tool_call_names, is_tool_result_user_message, tool_result_texts,
};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    ConfigTool, CountingAdd, CountingPing, CountingSubtract, FORCE_TOOLS_PREAMBLE, MOTTO_OUTPUT,
    MottoTool, PING_OUTPUT,
};
use crate::support::{assert_mentions_expected_number, assert_nonempty_response};

const CHAINED_PROMPT: &str =
    "Calculate 12 - 5 using the subtract tool, then add 30 to that result using the add tool.";
const CHAINED_RESULT: i32 = 37;

const PARALLEL_PROMPT: &str = "Compute 3 + 4 and 10 - 2. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.";

/// All tool-result texts across every user message in the history.
fn all_tool_result_texts(history: &[Message]) -> Vec<String> {
    history.iter().flat_map(tool_result_texts).collect()
}

/// The first assistant message carrying exactly two tool calls, paired with
/// the message that follows it. The downstream assertions already pin the
/// pair to `["add", "subtract"]`, so requiring exactly two here is equivalent
/// and clearer.
fn parallel_call_turn(history: &[Message]) -> (&Message, &Message) {
    let index = history
        .iter()
        .position(|message| assistant_tool_call_names(message).len() == 2)
        .unwrap_or_else(|| {
            panic!("expected an assistant message with parallel tool calls: {history:?}")
        });
    (
        &history[index],
        history
            .get(index + 1)
            .expect("a tool-result message should follow the parallel calls"),
    )
}

#[tokio::test]
async fn nonstreaming_multi_turn_executes_tools_and_reports_usage() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let (add_counter, subtract_counter) = (add.counter.clone(), subtract.counter.clone());

    with_gemini_cassette(
        "agent_tools/nonstreaming_multi_turn_executes_tools_and_reports_usage",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .prompt(CHAINED_PROMPT)
                .max_turns(5)
                .extended_details()
                .await
                .expect("multi-turn tool prompt should succeed");

            assert_mentions_expected_number(&response.output, CHAINED_RESULT);
            assert_eq!(add_counter.count(), 1, "add should execute exactly once");
            assert_eq!(
                subtract_counter.count(),
                1,
                "subtract should execute exactly once"
            );
            assert!(
                response.requests() >= 2,
                "a chained tool run should take multiple completion calls, got {}",
                response.requests()
            );
            assert!(
                response.usage.total_tokens > 0,
                "aggregated usage should be recorded: {:?}",
                response.usage
            );

            let messages = response
                .messages
                .expect("extended details should carry the run's messages");
            assert!(
                messages.iter().any(is_tool_result_user_message),
                "history should carry tool results: {messages:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_multi_turn_executes_tools_via_builtin_driver() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let (add_counter, subtract_counter) = (add.counter.clone(), subtract.counter.clone());

    with_gemini_cassette(
        "agent_tools/streaming_multi_turn_executes_tools_via_builtin_driver",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let mut stream = agent.stream_prompt(CHAINED_PROMPT).multi_turn(5).await;
            let observation = crate::support::collect_stream_observation(&mut stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            assert!(
                observation.tool_calls.iter().any(|name| name == "subtract"),
                "expected a subtract call, saw {:?}",
                observation.tool_calls
            );
            assert!(
                observation.tool_calls.iter().any(|name| name == "add"),
                "expected an add call, saw {:?}",
                observation.tool_calls
            );
            assert!(
                observation.tool_results >= 2,
                "expected at least two tool results, got {}",
                observation.tool_results
            );
            assert_eq!(add_counter.count(), 1, "add should execute exactly once");
            assert_eq!(
                subtract_counter.count(),
                1,
                "subtract should execute exactly once"
            );
            assert_mentions_expected_number(
                observation
                    .final_response_text
                    .as_deref()
                    .expect("stream should yield a final response"),
                CHAINED_RESULT,
            );
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_tool_calls_land_in_one_tool_result_message() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let (add_counter, subtract_counter) = (add.counter.clone(), subtract.counter.clone());

    with_gemini_cassette(
        "agent_tools/parallel_tool_calls_land_in_one_tool_result_message",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat(PARALLEL_PROMPT, &mut history)
                .await
                .expect("parallel tool prompt should succeed");

            let (calls_message, results_message) = parallel_call_turn(&history);
            let mut call_names = assistant_tool_call_names(calls_message);
            call_names.sort();
            assert_eq!(
                call_names,
                vec!["add".to_string(), "subtract".to_string()],
                "both tools should be called in one assistant turn"
            );

            assert!(
                is_tool_result_user_message(results_message),
                "parallel calls should be answered by a tool-result message: {results_message:?}"
            );
            let mut result_texts = tool_result_texts(results_message);
            result_texts.sort();
            assert_eq!(
                result_texts,
                vec!["7".to_string(), "8".to_string()],
                "both results should land in the single following user message"
            );

            assert_eq!(add_counter.count(), 1, "add should execute exactly once");
            assert_eq!(
                subtract_counter.count(),
                1,
                "subtract should execute exactly once"
            );
            assert_mentions_expected_number(&response, 7);
            assert_mentions_expected_number(&response, 8);
        },
    )
    .await;
}

#[tokio::test]
async fn tool_concurrency_one_preserves_parallel_call_contract() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let (add_counter, subtract_counter) = (add.counter.clone(), subtract.counter.clone());

    with_gemini_cassette(
        "agent_tools/tool_concurrency_one_preserves_parallel_call_contract",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .prompt(PARALLEL_PROMPT)
                .max_turns(3)
                .with_tool_concurrency(1)
                .extended_details()
                .await
                .expect("serial tool execution should succeed");

            let messages = response
                .messages
                .expect("extended details should carry the run's messages");
            let (_, results_message) = parallel_call_turn(&messages);
            let mut result_texts = tool_result_texts(results_message);
            result_texts.sort();
            assert_eq!(
                result_texts,
                vec!["7".to_string(), "8".to_string()],
                "serial execution should still produce one combined tool-result message"
            );

            assert_eq!(add_counter.count(), 1, "add should execute exactly once");
            assert_eq!(
                subtract_counter.count(),
                1,
                "subtract should execute exactly once"
            );
            assert_mentions_expected_number(&response.output, 7);
            assert_mentions_expected_number(&response.output, 8);
        },
    )
    .await;
}

#[tokio::test]
async fn zero_arg_tool_call_round_trips() {
    let ping = CountingPing::default();
    let counter = ping.counter.clone();

    with_gemini_cassette(
        "agent_tools/zero_arg_tool_call_round_trips",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(
                    "You must use the provided tools. Report tool outputs exactly as returned.",
                )
                .temperature(0.0)
                .tool(ping)
                .default_max_turns(2)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat(
                    "Call the ping tool, then report the exact marker it returns.",
                    &mut history,
                )
                .await
                .expect("zero-arg tool prompt should succeed");

            assert_eq!(counter.count(), 1, "ping should execute exactly once");
            assert_eq!(
                all_tool_result_texts(&history),
                vec![PING_OUTPUT.to_string()],
                "the verbatim string output should be the tool result"
            );
            assert!(
                response.contains(PING_OUTPUT),
                "final answer should repeat the marker, got {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn string_output_sent_verbatim_and_struct_output_serialized_as_json() {
    with_gemini_cassette(
        "agent_tools/string_output_verbatim_struct_output_json",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble("You must use the provided tools before answering.")
                .temperature(0.0)
                .tool(MottoTool)
                .tool(ConfigTool)
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat(
                    "Call fetch_motto and fetch_config, then summarize both outputs in one sentence.",
                    &mut history,
                )
                .await
                .expect("string/struct output prompt should succeed");

            let texts = all_tool_result_texts(&history);
            assert!(
                texts.iter().any(|text| text == MOTTO_OUTPUT),
                "string outputs should be sent verbatim with newlines preserved: {texts:?}"
            );
            assert!(
                texts
                    .iter()
                    .any(|text| text == &ConfigTool::expected_output_json()),
                "struct outputs should be JSON-serialized: {texts:?}"
            );
            assert_nonempty_response(&response);
        },
    )
    .await;
}
