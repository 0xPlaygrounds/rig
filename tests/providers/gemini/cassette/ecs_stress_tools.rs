//! Native tool stress: real provider, real tools, ordered application policies.
use super::super::{
    hook_stress_support::ResultRewrite,
    support::with_gemini_cassette,
    tools_support::{CodewordLookup, CountingAdd, MottoTool, ToolEventRecorder},
};
use super::ecs_stress_tools_runtime as runtime;
use crate::support::assert_nonempty_response;
use rig::{completion::PromptError, prelude::*, providers::gemini};
use rig_agent::test_utils::{
    validate_cancelled_failure, validate_result_redaction, validate_rewritten_arguments,
};
use serde_json::json;
#[tokio::test]
async fn arg_rewrite_sets_one_key_preserving_rest_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();
    with_gemini_cassette(
        "hook_stress_tools/arg_rewrite_sets_one_key_preserving_rest_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. Use the add tool for the addition.",
                "stress-agent",
                0.0,
            );
            ecs.tool(add);
            runtime::set_arg(&mut ecs, "add", "x", json!(100));
            runtime::record(&mut ecs, recorder);
            let response = runtime::prompt(
                &mut ecs,
                "Use the add tool to add 3 and 4, then report the tool's result.",
                4,
            )
            .await;
            assert_nonempty_response(&response);
            assert!(add_calls.count() >= 1, "the tool should execute");
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1, "one add call, saw {calls:?}");
            let observed: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            validate_rewritten_arguments(
                "gemini_arg_rewrite_preserves_fields",
                std::slice::from_ref(&observed),
                &json!({ "x" : 100 }),
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
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. Use the add tool for the addition.",
                "stress-agent",
                0.0,
            );
            ecs.tool(add);
            runtime::set_arg(&mut ecs, "add", "x", json!(7));
            runtime::set_arg(&mut ecs, "add", "y", json!(8));
            runtime::record(&mut ecs, recorder);
            let response = runtime::prompt(
                &mut ecs,
                "Use the add tool to add 1 and 1, then report the tool's result.",
                4,
            )
            .await;
            assert_nonempty_response(&response);
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1);
            let observed: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            validate_rewritten_arguments(
                "gemini_chained_arg_rewrites",
                std::slice::from_ref(&observed),
                &json!({ "x" : 7, "y" : 8 }),
            )
            .expect("portable chained argument-rewrite contract should hold");
            assert_eq!(
                observed,
                json!({ "x" : 7, "y" : 8 }),
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
#[tokio::test]
async fn two_result_rewrites_chain_redact_then_wrap_blocking() {
    let add = CountingAdd::default();
    with_gemini_cassette(
        "hook_stress_tools/two_result_rewrites_chain_redact_then_wrap_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. Use the add tool, then report the exact tool \
                     result text verbatim.",
                "stress-agent",
                0.0,
            );
            ecs.tool(add);
            runtime::rewrite_result(&mut ecs, "add", ResultRewrite::Replace("SECRET"));
            runtime::rewrite_result(
                &mut ecs,
                "add",
                ResultRewrite::Wrap {
                    prefix: "[",
                    suffix: "]",
                },
            );
            let response = runtime::prompt(
                &mut ecs,
                "Use the add tool to add 2 and 2, then report the exact tool result.",
                4,
            )
            .await;
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
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "Call the fetch_motto tool, then report the exact tool result text verbatim.",
                "stress-agent",
                0.0,
            );
            ecs.tool(MottoTool);
            runtime::rewrite_result(&mut ecs, "fetch_motto", ResultRewrite::Truncate(6));
            let response = runtime::prompt(
                &mut ecs,
                "Call fetch_motto and report exactly what it returns.",
                4,
            )
            .await;
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
#[tokio::test]
async fn terminate_from_tool_result_cancels_after_execution_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();
    with_gemini_cassette(
        "hook_stress_tools/terminate_from_tool_result_cancels_after_execution_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. Use the add tool for the addition.",
                "stress-agent",
                0.0,
            );
            ecs.tool(add);
            runtime::terminate_result(&mut ecs, "add", "result vetoed by policy hook");
            let error = runtime::cancelled(
                &mut ecs,
                "Use the add tool to add 21 and 21, then report the result.",
                4,
            )
            .await;
            validate_cancelled_failure(&error, "result vetoed by policy hook", "add")
                .expect("portable cancellation diagnostics should hold");
            assert!(
                add_calls.count() >= 1,
                "the tool body must have run before the ToolResult terminate"
            );
            match &error {
                PromptError::PromptCancelled { reason, .. } => {
                    assert_eq!(
                        reason, "result vetoed by policy hook",
                        "the cancellation must carry the hook reason verbatim"
                    )
                }
                other => panic!("expected PromptCancelled, got {other:?}"),
            }
        },
    )
    .await;
}
#[tokio::test]
async fn tool_error_guidance_drives_model_retry_blocking() {
    let lookup = CodewordLookup::default();
    let lookup_calls = lookup.counter.clone();
    with_gemini_cassette(
        "hook_stress_tools/tool_error_guidance_drives_model_retry_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You look up team codewords with the lookup_codeword tool. If the tool returns \
                     an error with guidance, follow that guidance and try again, then report the \
                     codeword you obtain.",
                "stress-agent",
                0.0,
            );
            ecs.tool(lookup);
            let response =
                runtime::prompt(&mut ecs, "Look up the codeword for the red team.", 5).await;
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
