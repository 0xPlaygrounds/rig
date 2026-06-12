//! Tool-failure round trips through the built-in agent driver: tool `Err`
//! values, argument deserialization failures, and execution-time
//! `ToolNotFoundError` are all stringified into tool results the model can
//! react to — the run keeps going instead of aborting.

use rig::client::CompletionClient;
use rig::completion::{Chat, Message, Prompt};
use rig::providers::gemini;
use rig::tool::{Tool, server::ToolServer};

use super::super::agent_run_support::tool_result_texts;
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    BLUE_CODEWORD, CODEWORD_GUIDANCE, CodewordLookup, CountingAdd, RemoveToolBeforeExecutionHook,
    StrictRegister,
};
use crate::support::assert_nonempty_response;

fn all_tool_result_texts(history: &[Message]) -> Vec<String> {
    history.iter().flat_map(tool_result_texts).collect()
}

#[tokio::test]
async fn tool_error_string_reaches_model_and_model_recovers() {
    let lookup = CodewordLookup::default();
    let counter = lookup.counter.clone();

    with_gemini_cassette(
        "tool_errors/tool_error_string_reaches_model_and_model_recovers",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(
                    "You retrieve codewords with the lookup_codeword tool. If a lookup fails, read the error text and follow its guidance, then report the codeword you obtained.",
                )
                .temperature(0.0)
                .tool(lookup)
                .default_max_turns(4)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat(
                    "Look up the codeword for the red team.",
                    &mut history,
                )
                .await
                .expect("the run should survive a failed tool call");

            assert_eq!(
                counter.count(),
                2,
                "the tool should run twice: the failing red lookup, then the blue retry"
            );

            let texts = all_tool_result_texts(&history);
            let error_text = texts
                .iter()
                .find(|text| text.contains(CODEWORD_GUIDANCE))
                .unwrap_or_else(|| {
                    panic!("the tool error text should reach the model verbatim: {texts:?}")
                });
            assert!(
                error_text.contains("ToolCallError"),
                "tool errors should keep the ToolSet error wrapper the model currently sees: {error_text:?}"
            );
            assert!(
                texts.iter().any(|text| text == BLUE_CODEWORD),
                "the recovered lookup should return the blue codeword: {texts:?}"
            );
            assert!(
                response.contains(BLUE_CODEWORD),
                "final answer should report the recovered codeword, got {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn invalid_args_deserialization_error_reaches_model() {
    let register = StrictRegister::default();
    let counter = register.counter.clone();

    with_gemini_cassette(
        "tool_errors/invalid_args_deserialization_error_reaches_model",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(
                    "You register guests with the register_guests tool, passing the seat count spelled out as a lowercase English word. If the tool reports an error, do not retry; apologize and explain that registration failed.",
                )
                .temperature(0.0)
                .tool(register)
                .default_max_turns(2)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat("Register four guests for the event.", &mut history)
                .await
                .expect("the run should survive an argument deserialization failure");

            assert_eq!(
                counter.count(),
                0,
                "the tool body should never run when its arguments fail to deserialize"
            );

            let texts = all_tool_result_texts(&history);
            assert!(
                texts.iter().any(|text| text.contains("JsonError")
                    || text.to_ascii_lowercase().contains("invalid type")),
                "the serde failure should be stringified into the tool result: {texts:?}"
            );
            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn missing_tool_at_execution_time_reports_tool_not_found() {
    let add = CountingAdd::default();
    let counter = add.counter.clone();

    with_gemini_cassette(
        "tool_errors/missing_tool_at_execution_time_reports_tool_not_found",
        |client| async move {
            let handle = ToolServer::new().tool(add).run();
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(
                    "You must use the add tool for arithmetic. If a tool reports an error, do not retry; explain that the tool is unavailable.",
                )
                .temperature(0.0)
                .tool_server_handle(handle.clone())
                .build();

            let response = agent
                .prompt("What is 19 + 23?")
                .max_turns(2)
                .with_hook(RemoveToolBeforeExecutionHook {
                    handle: handle.clone(),
                    tool_name: CountingAdd::NAME,
                })
                .extended_details()
                .await
                .expect("the run should survive an execution-time missing tool");

            assert_eq!(
                counter.count(),
                0,
                "the removed tool should never execute"
            );

            let messages = response
                .messages
                .expect("extended details should carry the run's messages");
            let texts = all_tool_result_texts(&messages);
            assert!(
                texts
                    .iter()
                    .any(|text| text.contains("ToolNotFoundError") || text.contains("not found")),
                "the ToolNotFound failure should be stringified into the tool result: {texts:?}"
            );
            assert_nonempty_response(&response.output);
        },
    )
    .await;
}
