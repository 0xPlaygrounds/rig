//! Native main Gemini stress workflows, preserving original assertion tails.
use super::super::{
    support::with_gemini_cassette,
    tools_support::{CountingAdd, CountingSubtract, ToolEventRecorder},
};
use super::ecs_stress_main_runtime::{self as runtime, LifecycleRecorder, ScratchpadReader};
use crate::support::assert_nonempty_response;
use rig::{prelude::*, providers::gemini, tool::Tool};
use std::collections::BTreeMap;
/// Preamble that forces tool use and a dependent two-step chain so the model
/// takes at least two turns (compute A, then use A to compute B).
const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";
#[tokio::test]
async fn lifecycle_and_scratchpad_thread_across_multi_turn_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let reader = ScratchpadReader::default();
    let recorder_probe = recorder.clone();
    let reader_probe = reader.clone();
    with_gemini_cassette(
        "hook_stress/lifecycle_and_scratchpad_thread_across_multi_turn_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                CHAIN_PREAMBLE,
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add);
            ecs.tool(subtract);
            let response = runtime::prompt(
                &mut ecs,
                "First add 10 and 5 with the add tool. Then subtract 3 from that sum with the \
                     subtract tool. Report the final number.",
                6,
                false,
                vec![recorder],
                vec![reader],
            )
            .await
            .output;
            assert_nonempty_response(&response);
            assert_eq!(
                recorder_probe.distinct_run_ids(),
                1,
                "run_id must be stable across every event of one run"
            );
            assert_eq!(
                recorder_probe.is_streaming(),
                Some(false),
                "blocking surface must report is_streaming() == false"
            );
            assert_eq!(
                recorder_probe.agent_name().as_deref(),
                Some("stress-agent"),
                "the configured agent name must reach the hook"
            );
            let crumbs = recorder_probe.breadcrumbs();
            let max_turn = crumbs.iter().map(|c| c.turn).max().unwrap_or(0);
            assert!(
                max_turn >= 2,
                "a dependent add-then-subtract chain must span >= 2 model turns, saw {crumbs:?}"
            );
            let turns: Vec<usize> = crumbs.iter().map(|c| c.turn).collect();
            assert!(
                turns.windows(2).all(|w| w[0] <= w[1]),
                "turn() must be non-decreasing across the run, saw {turns:?}"
            );
            let tool_calls = recorder_probe.count("ToolCall");
            let tool_results = recorder_probe.count("ToolResult");
            assert_eq!(
                tool_calls, tool_results,
                "every observed ToolCall must have a paired ToolResult"
            );
            assert_eq!(
                add_calls.count() + subtract_calls.count(),
                tool_calls,
                "observed ToolCall events must equal real tool executions"
            );
            assert!(
                add_calls.count() >= 1 && subtract_calls.count() >= 1,
                "the chain must exercise both add and subtract"
            );
            let tallies = reader_probe.tallies();
            assert!(
                !tallies.is_empty(),
                "ModelTurnFinished should fire, so the reader should see tallies"
            );
            assert!(
                tallies.windows(2).all(|w| w[0] <= w[1]),
                "the shared scratchpad tally must be non-decreasing, saw {tallies:?}"
            );
            assert_eq!(
                *tallies.last().expect("at least one tally"),
                tool_calls,
                "the final scratchpad tally must equal the total ToolCall count"
            );
        },
    )
    .await;
}
const VAULT_FACT_ID: &str = "vault-note";
const VAULT_FACT: &str = "Operational note: the vault access code is CINNABAR-42.";
const VAULT_CODE: &str = "CINNABAR-42";
#[tokio::test]
async fn request_patch_injects_context_and_narrows_active_tools_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    with_gemini_cassette(
        "hook_stress/request_patch_injects_context_and_narrows_active_tools_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a helpful assistant. Use a tool for any arithmetic. Consult the \
                     provided context for any facts you are asked about.",
                "stress-agent",
                None,
            );
            ecs.tool(add);
            ecs.tool(subtract);
            runtime::patch(
                &mut ecs,
                rig_ecs::agent::RequestPatch {
                    extra_context: vec![rig::completion::Document {
                        id: (VAULT_FACT_ID).into(),
                        text: (VAULT_FACT).into(),
                        additional_props: Default::default(),
                    }],
                    active_tools: Some(["add"].iter().copied().map(str::to_owned).collect()),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            );
            let response = runtime::prompt(
                &mut ecs,
                "Use a tool to compute 41 + 1. Then reply with exactly two lines:\n\
                     Line 1: `SUM: <the result>`\n\
                     Line 2: `CODE: <the vault access code from the provided context>`\n\
                     Both lines are required; do not stop after the first.",
                5,
                false,
                vec![],
                vec![],
            )
            .await
            .output;
            assert!(
                response.contains(VAULT_CODE),
                "the extra_context fact must reach the model; answer: {response:?}"
            );
            assert_eq!(
                subtract_calls.count(),
                0,
                "subtract was filtered out of active_tools and must never execute"
            );
            assert!(
                add_calls.count() >= 1,
                "the advertised add tool should still run for 41 + 1"
            );
        },
    )
    .await;
}
const REDACTION_MARKER: &str = "REDACTED-SUM-ZK7";
#[tokio::test]
async fn chained_arg_rewrite_then_result_redaction_blocking() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();
    with_gemini_cassette(
        "hook_stress/chained_arg_rewrite_then_result_redaction_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. You MUST use the add tool for the addition. \
                     After the tool result is available, report the exact tool result text \
                     verbatim as your final answer.",
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add);
            runtime::force_observe_redact(
                &mut ecs,
                CountingAdd::NAME,
                serde_json::json!({ "x" : 7, "y" : 8 }),
                recorder,
                REDACTION_MARKER,
            );
            let response = runtime::prompt(
                &mut ecs,
                "Use the add tool to add 2 and 2, then report the exact tool result.",
                4,
                false,
                vec![],
                vec![],
            )
            .await
            .output;
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1, "exactly one add call, saw {calls:?}");
            let observed_args: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            assert_eq!(
                observed_args,
                serde_json::json!({ "x" : 7, "y" : 8 }),
                "the observer must see the hook-rewritten args"
            );
            let results = recorder_probe.recorded_results();
            assert_eq!(results.len(), 1, "exactly one add result");
            assert_eq!(
                results[0].2, "15",
                "the observer must see the raw tool output before redaction"
            );
            assert!(
                response.contains(REDACTION_MARKER),
                "the redaction marker must reach the model; answer: {response:?}"
            );
            assert!(
                !response.contains("15"),
                "the raw tool result must not reach the model; answer: {response:?}"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn streaming_lifecycle_ordering_and_context_streaming_flag() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let recorder_probe = recorder.clone();
    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                CHAIN_PREAMBLE,
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add);
            ecs.tool(subtract);
            let observed = runtime::prompt(
                &mut ecs,
                "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                6,
                true,
                vec![recorder],
                vec![],
            )
            .await;
            let events = observed.trace.events;
            let saw_final = observed.trace.final_text.is_some();
            let final_text = observed.trace.final_text.unwrap_or_default();
            assert!(saw_final, "the stream must yield a FinalResponse");
            assert_nonempty_response(&final_text);
            let first = |tag: &str| events.iter().position(|e| *e == tag);
            let tool_call_at = first("tool_call").expect("a complete tool call is surfaced");
            let exec_commit_at =
                first("tool_execution_committed").expect("execution commit is surfaced");
            let tool_result_at = first("tool_result").expect("a tool result is surfaced");
            let final_at = first("final_response").expect("a final response is surfaced");
            assert!(
                tool_call_at < exec_commit_at,
                "the model-emitted tool call must precede its execution commit: {events:?}"
            );
            assert!(
                exec_commit_at <= tool_result_at,
                "execution commit must precede its tool result: {events:?}"
            );
            assert!(
                tool_result_at < final_at,
                "tool results must precede the final response: {events:?}"
            );
            assert_eq!(
                recorder_probe.is_streaming(),
                Some(true),
                "the streaming surface must report is_streaming() == true"
            );
            assert_eq!(
                recorder_probe.distinct_run_ids(),
                1,
                "run_id must be stable across the streamed run too"
            );
            assert_eq!(recorder_probe.agent_name().as_deref(), Some("stress-agent"));
            assert!(
                recorder_probe.count("ModelTurnFinished") >= 2,
                "ModelTurnFinished must fire per accepted turn on the streaming surface"
            );
            assert!(
                add_calls.count() >= 1 && subtract_calls.count() >= 1,
                "the streamed chain must exercise both tools"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn multi_tool_workflow_pairs_calls_and_results_per_turn_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = LifecycleRecorder::default();
    let recorder_probe = recorder.clone();
    with_gemini_cassette(
            "hook_stress/multi_tool_workflow_pairs_calls_and_results_per_turn_blocking",
            |client| async move {
                let mut ecs = runtime::agent(
                    client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                    "You are a calculator assistant. You MUST use the provided tools for every \
                     arithmetic operation. These two computations are independent — you may request \
                     them together. Once you have both results, report both numbers.",
                    "stress-agent",
                    Some(0.0),
                );
                ecs.tool(add);
                ecs.tool(subtract);
                let response = runtime::prompt(
                        &mut ecs,
                        "Independently compute 12 + 8 using the add tool and 30 - 7 using the subtract \
                     tool, then report both results.",
                        5,
                        false,
                        vec![recorder],
                        vec![],
                    )
                    .await
                    .output;
                assert_nonempty_response(&response);
                assert!(
                    add_calls.count() >= 1 && subtract_calls.count() >= 1,
                    "both independent tools should run"
                );
                let mut per_turn: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
                for crumb in recorder_probe.breadcrumbs() {
                    let entry = per_turn.entry(crumb.turn).or_default();
                    match crumb.tag {
                        "ToolCall" => entry.0 += 1,
                        "ToolResult" => entry.1 += 1,
                        _ => {}
                    }
                }
                for (turn, (calls, results)) in &per_turn {
                    assert_eq!(
                        calls, results,
                        "turn {turn} must pair every ToolCall with a ToolResult (atomic batch)"
                    );
                }
                assert_eq!(
                    recorder_probe.count("ToolCall"),
                    add_calls.count() + subtract_calls.count(),
                    "observed ToolCall events must equal real tool executions"
                );
            },
        )
        .await;
}
const SUBTRACT_SKIP_REASON: &str =
    "the subtract tool is offline; treat its result as unavailable and continue";
#[tokio::test]
async fn skip_in_multi_tool_workflow_leaves_tool_unexecuted_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    with_gemini_cassette(
            "hook_stress/skip_in_multi_tool_workflow_leaves_tool_unexecuted_blocking",
            |client| async move {
                let mut ecs = runtime::agent(
                    client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                    "You are a calculator assistant. You MUST use the provided tools for every \
                     arithmetic operation. If a tool reports it is unavailable, acknowledge that in \
                     your answer and still report any results you do have.",
                    "stress-agent",
                    Some(0.0),
                );
                ecs.tool(add);
                ecs.tool(subtract);
                runtime::skip(&mut ecs, CountingSubtract::NAME, SUBTRACT_SKIP_REASON);
                let response = runtime::prompt(
                        &mut ecs,
                        "Use the add tool to compute 14 + 6, and use the subtract tool to compute \
                     40 - 9. Report what you can.",
                        5,
                        false,
                        vec![],
                        vec![],
                    )
                    .await
                    .output;
                assert_nonempty_response(&response);
                assert_eq!(
                    subtract_calls.count(), 0,
                    "the skipped subtract tool must never execute"
                );
                assert!(
                    add_calls.count() >= 1,
                    "the non-skipped add tool should still execute"
                );
            },
        )
        .await;
}
/// Golden `gemini_tool_call_turns`: two tool turns on a wire that carries
/// no tool-call ids. Every id in the log is minted from the block that
/// assembled the call, so the record is the same on every run — the proof
/// that nothing the engine mints is random.
#[tokio::test]
async fn tool_call_turns_effect_log_is_the_golden_fixture() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let (saw_final, log) = super::ecs_stress_main_golden::run(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                CHAIN_PREAMBLE,
                "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                add,
                subtract,
            )
            .await;
            assert!(saw_final, "the stream must yield a FinalResponse");
            let tool_ids: Vec<&rig::message::ToolCallId> = log
                .records
                .iter()
                .filter_map(|record| match &record.outcome {
                    Ok(rig::effect::Outcome::Completion(response)) => Some(response),
                    _ => None,
                })
                .flat_map(|response| response.choice.iter())
                .filter_map(|content| match content {
                    rig::message::AssistantContent::ToolCall(call) => Some(&call.id),
                    _ => None,
                })
                .collect();
            assert!(!tool_ids.is_empty(), "the program calls tools");
            assert!(
                tool_ids.iter().all(|id| id.is_generated()),
                "every id-less wire call is named by its block: {tool_ids:?}"
            );
            crate::ecs_goldens::golden_effects("gemini_tool_call_turns", &log);
        },
    )
    .await;
}
