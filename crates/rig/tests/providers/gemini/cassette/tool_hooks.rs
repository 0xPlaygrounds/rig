//! Prompt-hook dispatch on the tool execution path: skip-with-reason,
//! terminate-early, and observation of every call/result pair.

use rig::agent::PromptHook;
use rig::client::CompletionClient;
use rig::completion::{Prompt, PromptError};
use rig::providers::gemini;
use rig::tool::Tool;

use super::super::agent_run_support::tool_result_texts;
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    CountingAdd, FORCE_TOOLS_PREAMBLE, SkipToolHook, TerminateOnToolHook, ToolEventRecorder,
};
use crate::support::assert_nonempty_response;

const SKIP_REASON: &str = "the add tool is down for maintenance; report exactly that to the user";
const TERMINATE_REASON: &str = "tool execution vetoed by policy hook";

#[tokio::test]
async fn on_tool_call_skip_returns_reason_without_executing() {
    let add = CountingAdd::default();
    let counter = add.counter.clone();

    with_gemini_cassette(
        "tool_hooks/on_tool_call_skip_returns_reason_without_executing",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .prompt("What is 19 + 23?")
                .max_turns(3)
                .with_hook(SkipToolHook {
                    tool_name: CountingAdd::NAME,
                    reason: SKIP_REASON,
                })
                .extended_details()
                .await
                .expect("a skipped tool call should not fail the run");

            assert_eq!(counter.count(), 0, "the skipped tool should never execute");

            let messages = response
                .messages
                .expect("extended details should carry the run's messages");
            let texts: Vec<String> = messages.iter().flat_map(tool_result_texts).collect();
            assert_eq!(
                texts,
                vec![SKIP_REASON.to_string()],
                "the skip reason should be the synthetic tool result"
            );
            assert_nonempty_response(&response.output);
        },
    )
    .await;
}

#[tokio::test]
async fn on_tool_call_terminate_cancels_run() {
    let add = CountingAdd::default();
    let counter = add.counter.clone();

    with_gemini_cassette(
        "tool_hooks/on_tool_call_terminate_cancels_run",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            let error = agent
                .prompt("What is 19 + 23?")
                .max_turns(3)
                .with_hook(TerminateOnToolHook {
                    tool_name: CountingAdd::NAME,
                    reason: TERMINATE_REASON,
                })
                .extended_details()
                .await
                .expect_err("a terminating hook should cancel the run");

            assert_eq!(counter.count(), 0, "the vetoed tool should never execute");
            match &error {
                PromptError::PromptCancelled { reason, .. } => {
                    // The hook's reason passes through verbatim (no model
                    // content), so this is exact.
                    assert_eq!(
                        reason, TERMINATE_REASON,
                        "cancellation should carry the hook's reason verbatim"
                    );
                }
                other => panic!("expected PromptCancelled, got {other:?}"),
            }
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_observe_every_tool_call_and_result() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_for_test = recorder.clone();

    with_gemini_cassette(
        "tool_hooks/hooks_observe_every_tool_call_and_result",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .prompt("Use the add tool to calculate 19 + 23, then report the result.")
                .max_turns(3)
                .with_hook(recorder)
                .await
                .expect("recorded tool prompt should succeed");

            let calls = recorder_for_test.recorded_calls();
            assert_eq!(
                calls.len(),
                1,
                "exactly one tool call should be observed: {calls:?}"
            );
            let (call_name, call_args) = &calls[0];
            assert_eq!(call_name, CountingAdd::NAME);
            let args: serde_json::Value =
                serde_json::from_str(call_args).expect("observed args should be JSON");
            assert_eq!(args, serde_json::json!({ "x": 19, "y": 23 }));

            let results = recorder_for_test.recorded_results();
            assert_eq!(
                results.len(),
                1,
                "exactly one tool result should be observed"
            );
            let (result_name, result_args, result_output) = &results[0];
            assert_eq!(result_name, CountingAdd::NAME);
            assert_eq!(
                result_args, call_args,
                "result hook should see the same args"
            );
            assert_eq!(
                result_output, "42",
                "result hook should see the raw tool output"
            );

            assert!(
                response.contains("42"),
                "final answer should report 42: {response:?}"
            );
        },
    )
    .await;
}

// Compile-time check that the fixtures implement the hook trait for the
// Gemini model, so failures surface here instead of inside macro-expanded
// builder code.
#[allow(unused)]
fn assert_hook_impls() {
    fn requires_hook<H: PromptHook<gemini::completion::CompletionModel>>(_hook: H) {}
    requires_hook(ToolEventRecorder::default());
    requires_hook(SkipToolHook {
        tool_name: "add",
        reason: "",
    });
    requires_hook(TerminateOnToolHook {
        tool_name: "add",
        reason: "",
    });
}
