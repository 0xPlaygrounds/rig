//! Native Gemini context/composition stress cases, preserving original assertions.
use super::super::hook_stress_support::CountingMultiply;
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract};
use super::ecs_stress_runtime::{self as runtime};
use crate::support::assert_nonempty_response;
use rig::providers::gemini;
#[tokio::test]
async fn two_hooks_narrow_active_tools_to_intersection_blocking() {
    rig_test_support::goldens::world_golden_test(
        async {
            let add = CountingAdd::default();
            let subtract = CountingSubtract::default();
            let multiply = CountingMultiply::default();
            let add_calls = add.counter.clone();
            let subtract_calls = subtract.counter.clone();
            let multiply_calls = multiply.counter.clone();
            with_gemini_cassette(
        "hook_stress_context/two_hooks_narrow_active_tools_to_intersection_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. Use a provided tool for any arithmetic you \
                     can. If a needed tool is unavailable, say so and move on.",
                Some("stress-agent"),
                None,
            );
            ecs.tool(add);
            ecs.tool(subtract);
            ecs.tool(multiply);
            runtime::install_patch(
                &mut ecs,
                rig_ecs::agent::RequestPatch {
                    active_tools: Some(
                        (["add", "subtract"])
                            .into_iter()
                            .map(str::to_owned)
                            .collect(),
                    ),
                    temperature: Some(0.0),
                    ..Default::default()
                },
                false,
            );
            runtime::install_patch(
                &mut ecs,
                rig_ecs::agent::RequestPatch {
                    active_tools: Some(
                        (["add", "multiply"])
                            .into_iter()
                            .map(str::to_owned)
                            .collect(),
                    ),
                    ..Default::default()
                },
                false,
            );
            let response = runtime::prompt(
                &mut ecs,
                "Compute 6 + 2, then 10 - 3, then 4 * 5. Report whichever results you can \
                     obtain.",
                5,
                vec![],
                vec![],
            )
            .await;
            assert_nonempty_response(&response);
            assert!(
                add_calls.count() >= 1,
                "add is in the intersection and should run"
            );
            assert_eq!(
                subtract_calls.count(),
                0,
                "subtract is outside the intersection and must never execute"
            );
            assert_eq!(
                multiply_calls.count(),
                0,
                "multiply is outside the intersection and must never execute"
            );
        },
    )
    .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "gemini_stress_context_two_hooks_narrow_active_tools_to_intersection_blocking",
                log,
            )
        },
    )
    .await
}
