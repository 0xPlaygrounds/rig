//! Hook-system stress suite: turn advancement, host state shared across hooks
//! and turns, and `Hooks` composition (multiple entries, observe-only both-fire,
//! `add_hook` append, `CompletionCall` patch accumulation, `active_tools`
//! intersection). Recorded against real Gemini.
//!
//! Assertions are loose for model-shaped values and exact only for
//! rig-synthesized values (see `tools_support`'s note).

use rig::providers::gemini;

use super::super::hook_stress_support::{
    CHAIN_PREAMBLE, CountingMultiply, EventTap, TallyReader, apply_patch, fact_doc, tally_reader,
};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract};
use crate::support::assert_nonempty_response;

use rig::agent::RequestPatch;

// ---------------------------------------------------------------------------
// Turn advancement across a dependent multi-turn chain.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hook_context_identity_stable_and_turn_advances_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let tap = EventTap::default();
    let probe = tap.clone();
    let tap_entry = tap.entry();

    with_gemini_cassette(
        "hook_stress_context/hook_context_identity_stable_and_turn_advances_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .runner(
                    "First add 9 and 6 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .add_hook(tap_entry)
                .run()
                .await
                .map(|response| response.output)
                .expect("dependent chain should succeed");

            assert_nonempty_response(&response);
            // The blocking surface's medium-specific observation fires, and the
            // streamed response-finish never does.
            assert!(probe.count("CompletionResponse") >= 1);
            assert_eq!(probe.count("StreamResponseFinish"), 0);

            let turns = probe.distinct_turns();
            assert_eq!(
                turns.first().copied(),
                Some(1),
                "the first observed turn must be turn 1, saw {turns:?}"
            );
            assert!(
                turns.len() >= 2 && *turns.last().expect("turn") >= 2,
                "a dependent chain must reach >= 2 turns, saw {turns:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn unnamed_agent_still_dispatches_every_hook_event_blocking() {
    // The classic hook context exposed the configured agent name; hooks are
    // closures now and capture whatever identity they need, so what is left to
    // pin here is that an agent with no `.name(..)` still dispatches the full
    // event set. Same recorded request bytes as before.
    let add = CountingAdd::default();
    let tap = EventTap::default();
    let probe = tap.clone();
    let tap_entry = tap.entry();

    with_gemini_cassette(
        "hook_stress_context/agent_name_absent_when_unconfigured_blocking",
        |client| async move {
            // No `.name(..)` on the builder.
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .runner("Use the add tool to add 3 and 4, then report the result.")
                .max_turns(4)
                .add_hook(tap_entry)
                .run()
                .await
                .map(|response| response.output)
                .expect("run should succeed");

            assert_nonempty_response(&response);
            for tag in [
                "CompletionCall",
                "CompletionResponse",
                "ToolCall",
                "ToolResult",
                "ModelTurnFinished",
            ] {
                assert!(
                    probe.count(tag) >= 1,
                    "an unnamed agent must still dispatch {tag}"
                );
            }
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Shared host state: read by two hooks and growing across turns.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scratchpad_tally_grows_across_turns_and_is_read_by_second_hook_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let tap = EventTap::default();
    let reader = TallyReader::default();
    let tap_probe = tap.clone();
    let reader_probe = reader.clone();
    let tap_entry = tap.entry();
    let reader_entry = tally_reader(&reader, tap.tally());

    with_gemini_cassette(
        "hook_stress_context/scratchpad_tally_grows_across_turns_and_is_read_by_second_hook_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .runner(
                    "First add 30 and 12 with the add tool. Then subtract 5 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                // Writer (tap) bumps the shared tally on each ToolCall; the
                // reader (a *different* hook) reads it on each ModelTurnFinished.
                .add_hook(tap_entry)
                .add_hook(reader_entry).run()
                .await.map(|response| response.output)
                .expect("dependent chain should succeed");

            assert_nonempty_response(&response);
            let total_calls = add_calls.count() + subtract_calls.count();
            let tallies = reader_probe.tallies();
            assert!(
                tallies.len() >= 2,
                "a multi-turn run should fire ModelTurnFinished at least twice, saw {tallies:?}"
            );
            assert!(
                tallies.windows(2).all(|w| w[0] <= w[1]),
                "the shared tally must be non-decreasing, saw {tallies:?}"
            );
            assert!(
                tallies.first() < tallies.last(),
                "the tally must grow across turns (cross-turn shared state), saw {tallies:?}"
            );
            assert_eq!(
                *tallies.last().expect("a tally"),
                total_calls,
                "the final tally must equal the total ToolCall count"
            );
            assert_eq!(
                tap_probe.count("ToolCall"),
                total_calls,
                "observed ToolCall events must equal real executions"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Tool-call correlation: internal_call_id pairs ToolCall with ToolResult.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn internal_call_id_correlates_tool_call_and_result_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let tap = EventTap::default();
    let probe = tap.clone();
    let tap_entry = tap.entry();

    with_gemini_cassette(
        "hook_stress_context/internal_call_id_correlates_tool_call_and_result_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let response = agent
                .runner(
                    "First add 7 and 7 with the add tool. Then subtract 2 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .add_hook(tap_entry)
                .run()
                .await
                .map(|response| response.output)
                .expect("dependent chain should succeed");

            assert_nonempty_response(&response);
            let call_ids = probe.call_ids();
            let result_ids = probe.result_ids();
            assert!(!call_ids.is_empty(), "the run should make tool calls");
            assert_eq!(
                call_ids, result_ids,
                "each ToolResult must carry the same internal_call_id as its ToolCall, in order"
            );
            assert!(
                call_ids.iter().all(|id| !id.is_empty()),
                "internal_call_ids must be non-empty"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Hooks composition: multiple observe-only entries, add_hook append.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_observe_only_hooks_both_observe_the_run_blocking() {
    let add = CountingAdd::default();
    let first = EventTap::default();
    let second = EventTap::default();
    let first_probe = first.clone();
    let second_probe = second.clone();
    let first_entry = first.entry();
    let second_entry = second.entry();

    with_gemini_cassette(
        "hook_stress_context/two_observe_only_hooks_both_observe_the_run_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            let response = agent
                .runner("Use the add tool to add 8 and 8, then report the result.")
                .max_turns(4)
                .add_hook(first_entry)
                .add_hook(second_entry)
                .run()
                .await
                .map(|response| response.output)
                .expect("run should succeed");

            assert_nonempty_response(&response);
            // Both observe-only hooks see the same run — neither short-circuits.
            for tag in [
                "CompletionCall",
                "ToolCall",
                "ToolResult",
                "ModelTurnFinished",
            ] {
                assert_eq!(
                    first_probe.count(tag),
                    second_probe.count(tag),
                    "both hooks must observe the same number of {tag} events"
                );
            }
            assert!(first_probe.count("ToolCall") >= 1);
        },
    )
    .await;
}

#[tokio::test]
async fn add_hook_appends_across_builder_and_request_blocking() {
    let add = CountingAdd::default();
    let builder_hook = EventTap::default();
    let request_hook = EventTap::default();
    let builder_probe = builder_hook.clone();
    let request_probe = request_hook.clone();
    let builder_entry = builder_hook.entry();
    let request_entry = request_hook.entry();

    with_gemini_cassette(
        "hook_stress_context/add_hook_appends_across_builder_and_request_blocking",
        |client| async move {
            // One hook on the agent builder, one on the request: both must fire
            // (the request-level add_hook appends to the agent-default stack).
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .add_hook(builder_entry)
                .build();

            let response = agent
                .runner("Use the add tool to add 5 and 6, then report the result.")
                .max_turns(4)
                .add_hook(request_entry)
                .run()
                .await
                .map(|response| response.output)
                .expect("run should succeed");

            assert_nonempty_response(&response);
            assert!(
                builder_probe.count("CompletionCall") >= 1,
                "the agent-default (builder) hook must still fire"
            );
            assert!(
                request_probe.count("CompletionCall") >= 1,
                "the request-level hook must fire too"
            );
            assert_eq!(
                builder_probe.count("ToolCall"),
                request_probe.count("ToolCall"),
                "both stacked hooks observe the same tool calls"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn completion_call_patches_accumulate_from_two_hooks_blocking() {
    let add = CountingAdd::default();

    with_gemini_cassette(
        "hook_stress_context/completion_call_patches_accumulate_from_two_hooks_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a helpful assistant. Consult the provided context for any facts you \
                     are asked about; use a tool for arithmetic.",
                )
                .tool(add)
                .build();

            // Two independent hooks each inject a different fact via extra_context.
            // Patches must accumulate (append), so BOTH facts reach the model.
            let response = agent
                .runner("Tell me both the harbor code and the orchard code.")
                .max_turns(4)
                .add_hook(apply_patch(
                    RequestPatch::new()
                        .context(fact_doc("harbor", "The harbor code is ALPHA-11."))
                        .temperature(0.0),
                ))
                .add_hook(apply_patch(
                    RequestPatch::new()
                        .context(fact_doc("orchard", "The orchard code is BETA-22.")),
                ))
                .run()
                .await
                .map(|response| response.output)
                .expect("accumulated context run should succeed");

            assert!(
                response.contains("ALPHA-11"),
                "the first hook's injected fact must reach the model: {response:?}"
            );
            assert!(
                response.contains("BETA-22"),
                "the second hook's injected fact must reach the model too (patches accumulate): \
                 {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn two_hooks_narrow_active_tools_to_intersection_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let multiply = CountingMultiply::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let multiply_calls = multiply.counter.clone();

    with_gemini_cassette(
        "hook_stress_context/two_hooks_narrow_active_tools_to_intersection_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. Use a provided tool for any arithmetic you \
                     can. If a needed tool is unavailable, say so and move on.",
                )
                .tool(add)
                .tool(subtract)
                .tool(multiply)
                .build();

            // Two narrowing hooks: {add, subtract} ∩ {add, multiply} == {add}.
            let response = agent
                .runner(
                    "Compute 6 + 2, then 10 - 3, then 4 * 5. Report whichever results you can \
                     obtain.",
                )
                .max_turns(5)
                .add_hook(apply_patch(
                    RequestPatch::new()
                        .active_tools(["add", "subtract"])
                        .temperature(0.0),
                ))
                .add_hook(apply_patch(
                    RequestPatch::new().active_tools(["add", "multiply"]),
                ))
                .run()
                .await
                .map(|response| response.output)
                .expect("intersected-tools run should succeed");

            assert_nonempty_response(&response);
            // Only `add` is in the intersection, so only it can execute.
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
}
use rig::prelude::*;
