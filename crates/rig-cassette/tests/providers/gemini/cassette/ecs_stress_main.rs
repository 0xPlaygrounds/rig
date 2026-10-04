//! Native main Gemini stress workflows, preserving original assertion tails.
use super::super::{
    support::with_gemini_cassette,
    tools_support::{CountingAdd, CountingSubtract},
};
use super::ecs_stress_main_runtime::{self as runtime, LifecycleRecorder};
use crate::support::assert_nonempty_response;
use rig::providers::gemini;
use std::collections::BTreeMap;
#[tokio::test]
async fn multi_tool_workflow_pairs_calls_and_results_per_turn_blocking() {
    rig_test_support::goldens::world_golden_test(async {

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
                    client.completion(gemini::completion::GEMINI_2_5_FLASH),
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

}, |log| rig_test_support::goldens::world_golden_effects("gemini_stress_main_multi_tool_workflow_pairs_calls_and_results_per_turn_blocking", log)).await
}
