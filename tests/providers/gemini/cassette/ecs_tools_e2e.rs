//! Native counterparts of the original agent-tool conformance cases.
//! Real tools and provider adapters execute through native ECS systems.
use rig::prelude::*;
use rig::providers::gemini;
#[path = "ecs_tools_e2e/conformance.rs"]
mod conformance;
#[path = "ecs_tools_e2e/runtime.rs"]
mod runtime;
use super::super::agent_run_support::is_tool_result_user_message;
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract, FORCE_TOOLS_PREAMBLE};
use crate::support::assert_mentions_expected_number;
use conformance::{parallel_tools, tool_output_serialization, zero_argument_tool};
use runtime::{configured, execute};
const CHAINED_PROMPT: &str =
    "Calculate 12 - 5 using the subtract tool, then add 30 to that result using the add tool.";
const CHAINED_RESULT: i32 = 37;
#[tokio::test]
async fn nonstreaming_multi_turn_executes_tools_and_reports_usage() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let (add_counter, subtract_counter) = (add.counter.clone(), subtract.counter.clone());
    with_gemini_cassette(
        "agent_tools/nonstreaming_multi_turn_executes_tools_and_reports_usage",
        |client| async move {
            let mut ecs = configured(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                None,
            );
            ecs.tool(add);
            ecs.tool(subtract);
            let response = execute(&mut ecs, CHAINED_PROMPT, false, Some(5), None).await;
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
            let mut ecs = configured(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                None,
            );
            ecs.tool(add);
            ecs.tool(subtract);
            crate::ecs_observation::install_observers(&mut ecs);
            let _response = execute(&mut ecs, CHAINED_PROMPT, true, Some(5), None).await;
            let observation = crate::ecs_observation::observation(&ecs);
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
    with_gemini_cassette(
        "agent_tools/parallel_tool_calls_land_in_one_tool_result_message",
        |client| async move {
            let report = parallel_tools(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                None,
            )
            .await
            .expect("parallel-tool conformance scenario should succeed");
            eprintln!("[gemini] {report:?}");
        },
    )
    .await;
}
#[tokio::test]
async fn tool_concurrency_one_preserves_parallel_call_contract() {
    with_gemini_cassette(
        "agent_tools/tool_concurrency_one_preserves_parallel_call_contract",
        |client| async move {
            let report = parallel_tools(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                Some(1),
            )
            .await
            .expect("serial parallel-tool conformance scenario should succeed");
            eprintln!("[gemini] {report:?}");
        },
    )
    .await;
}
#[tokio::test]
async fn zero_arg_tool_call_round_trips() {
    with_gemini_cassette(
        "agent_tools/zero_arg_tool_call_round_trips",
        |client| async move {
            let report =
                zero_argument_tool(client.completion_model(gemini::completion::GEMINI_2_5_FLASH))
                    .await
                    .expect("zero-argument conformance scenario should succeed");
            eprintln!("[gemini] {report:?}");
        },
    )
    .await;
}
#[tokio::test]
async fn string_output_sent_verbatim_and_struct_output_serialized_as_json() {
    with_gemini_cassette(
        "agent_tools/string_output_verbatim_struct_output_json",
        |client| async move {
            let report = tool_output_serialization(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
            )
            .await
            .expect("tool-output serialization conformance scenario should succeed");
            eprintln!("[gemini] {report:?}");
        },
    )
    .await;
}
