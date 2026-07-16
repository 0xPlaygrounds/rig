//! Hook-system stress suite: the tool-execution lifecycle — chained
//! `ToolCallAction::Rewrite`, chained `ToolResultAction::Rewrite` (redact / wrap / truncate),
//! `Terminate` from a `ToolResult` (post-execution), and model-driven recovery
//! from a tool error. Recorded against real Gemini.

use rig::client::CompletionClient;
use rig::completion::{Prompt, PromptError};
use rig::providers::gemini;
use rig::test_utils::{
    validate_cancelled_failure, validate_result_redaction, validate_rewritten_arguments,
};
use serde_json::json;

use super::super::hook_stress_support::{
    ResultRewrite, RewriteToolResult, SetArg, TerminateOnResult,
};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CodewordLookup, CountingAdd, MottoTool, ToolEventRecorder};
use crate::support::assert_nonempty_response;

// ---------------------------------------------------------------------------
// Argument rewriting.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn arg_rewrite_sets_one_key_preserving_rest_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress_tools/arg_rewrite_sets_one_key_preserving_rest_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant. Use the add tool for the addition.")
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .prompt("Use the add tool to add 3 and 4, then report the tool's result.")
                .max_turns(4)
                // Force x = 100 but leave y untouched, then observe.
                .add_hook(SetArg {
                    tool: "add",
                    key: "x",
                    value: json!(100),
                })
                .add_hook(recorder)
                .await
                .expect("single-key arg rewrite run should succeed");

            assert_nonempty_response(&response);
            assert!(add_calls.count() >= 1, "the tool should execute");
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1, "one add call, saw {calls:?}");
            let observed: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            validate_rewritten_arguments(
                "gemini_arg_rewrite_preserves_fields",
                std::slice::from_ref(&observed),
                &json!({ "x": 100 }),
            )
            .expect("portable argument-rewrite contract should hold");
            assert_eq!(observed["x"], json!(100), "x must be the rewritten value");
            assert!(
                observed.get("y").is_some(),
                "the model's y argument must be preserved: {observed}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn two_arg_rewrites_chain_blocking() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress_tools/two_arg_rewrites_chain_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant. Use the add tool for the addition.")
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .prompt("Use the add tool to add 1 and 1, then report the tool's result.")
                .max_turns(4)
                // Two rewriters chain: the second sees the first's output and adds
                // its own key, so the tool executes against {x:7, y:8}.
                .add_hook(SetArg {
                    tool: "add",
                    key: "x",
                    value: json!(7),
                })
                .add_hook(SetArg {
                    tool: "add",
                    key: "y",
                    value: json!(8),
                })
                .add_hook(recorder)
                .await
                .expect("chained arg rewrite run should succeed");

            assert_nonempty_response(&response);
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1);
            let observed: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            validate_rewritten_arguments(
                "gemini_chained_arg_rewrites",
                std::slice::from_ref(&observed),
                &json!({ "x": 7, "y": 8 }),
            )
            .expect("portable chained argument-rewrite contract should hold");
            assert_eq!(
                observed,
                json!({ "x": 7, "y": 8 }),
                "both chained rewrites must compose"
            );
            let results = recorder_probe.recorded_results();
            assert_eq!(
                results[0].2, "15",
                "the tool executed against the composed args"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Result rewriting.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_result_rewrites_chain_redact_then_wrap_blocking() {
    let add = CountingAdd::default();

    with_gemini_cassette(
        "hook_stress_tools/two_result_rewrites_chain_redact_then_wrap_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. Use the add tool, then report the exact tool \
                     result text verbatim.",
                )
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .prompt("Use the add tool to add 2 and 2, then report the exact tool result.")
                .max_turns(4)
                // Redact -> wrap: the model sees "[SECRET]".
                .add_hook(RewriteToolResult {
                    tool: "add",
                    rewrite: ResultRewrite::Replace("SECRET"),
                })
                .add_hook(RewriteToolResult {
                    tool: "add",
                    rewrite: ResultRewrite::Wrap {
                        prefix: "[",
                        suffix: "]",
                    },
                })
                .await
                .expect("chained result rewrite run should succeed");

            assert!(
                response.contains("[SECRET]"),
                "both chained result rewrites must compose (redact then wrap): {response:?}"
            );
            assert!(
                !response.contains('4'),
                "the raw tool result must not reach the model: {response:?}"
            );
            validate_result_redaction("gemini_chained_result_rewrites", true, &response, "4")
                .expect("portable result-rewrite contract should hold");
        },
    )
    .await;
}

#[tokio::test]
async fn result_truncation_reaches_model_blocking() {
    with_gemini_cassette(
        "hook_stress_tools/result_truncation_reaches_model_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "Call the fetch_motto tool, then report the exact tool result text verbatim.",
                )
                .temperature(0.0)
                .tool(MottoTool)
                .build();

            // The motto is "steady hands\ncalm waters"; truncate to its first 6
            // chars ("steady") before the model sees it.
            let response = agent
                .prompt("Call fetch_motto and report exactly what it returns.")
                .max_turns(4)
                .add_hook(RewriteToolResult {
                    tool: "fetch_motto",
                    rewrite: ResultRewrite::Truncate(6),
                })
                .await
                .expect("result truncation run should succeed");

            assert!(
                response.contains("steady"),
                "the truncated result prefix must reach the model: {response:?}"
            );
            assert!(
                !response.contains("waters"),
                "the truncated-off suffix must not reach the model: {response:?}"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Terminate from a ToolResult (post-execution).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn terminate_from_tool_result_cancels_after_execution_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();

    with_gemini_cassette(
        "hook_stress_tools/terminate_from_tool_result_cancels_after_execution_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant. Use the add tool for the addition.")
                .temperature(0.0)
                .tool(add)
                .build();

            let error = agent
                .prompt("Use the add tool to add 21 and 21, then report the result.")
                .max_turns(4)
                // Unlike a ToolCall terminate, the tool body DOES run first; the
                // hook then vetoes the run when it sees the result.
                .add_hook(TerminateOnResult {
                    tool: "add",
                    reason: "result vetoed by policy hook",
                })
                .await
                .expect_err("a ToolResult terminate should cancel the run");

            validate_cancelled_failure(&error, "result vetoed by policy hook", "add")
                .expect("portable cancellation diagnostics should hold");

            // The tool executed before the terminate fired.
            assert!(
                add_calls.count() >= 1,
                "the tool body must have run before the ToolResult terminate"
            );
            match &error {
                PromptError::PromptCancelled { reason, .. } => assert_eq!(
                    reason, "result vetoed by policy hook",
                    "the cancellation must carry the hook reason verbatim"
                ),
                other => panic!("expected PromptCancelled, got {other:?}"),
            }
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Model-driven recovery from a tool error.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_error_guidance_drives_model_retry_blocking() {
    let lookup = CodewordLookup::default();
    let lookup_calls = lookup.counter.clone();

    with_gemini_cassette(
        "hook_stress_tools/tool_error_guidance_drives_model_retry_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You look up team codewords with the lookup_codeword tool. If the tool returns \
                     an error with guidance, follow that guidance and try again, then report the \
                     codeword you obtain.",
                )
                .temperature(0.0)
                .tool(lookup)
                .build();

            // The first (red) lookup errors with corrective guidance pointing at
            // the blue team; the model should retry and obtain the blue codeword.
            let response = agent
                .prompt("Look up the codeword for the red team.")
                .max_turns(5)
                .await
                .expect("tool-error recovery run should succeed");

            assert!(
                lookup_calls.count() >= 2,
                "the model should retry the lookup after the error guidance, saw {} call(s)",
                lookup_calls.count()
            );
            assert!(
                response.to_ascii_lowercase().contains("azure-falcon"),
                "the model should report the recovered codeword: {response:?}"
            );
        },
    )
    .await;
}
