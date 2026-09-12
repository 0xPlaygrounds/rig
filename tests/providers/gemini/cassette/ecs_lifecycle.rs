//! Native gemini lifecycle counterparts with shared original transport/assertion helpers.
//! Entry storage, startup rewrites and settlement observations run in native systems.

use rig::prelude::*;
use rig::providers::gemini;

use super::super::support::with_gemini_lifecycle_cassette;
use crate::support::{
    Adder, BASIC_PREAMBLE, BASIC_PROMPT, STREAMING_PREAMBLE, STREAMING_PROMPT, WireProbe,
    assert_nonempty_response,
};

use crate::ecs_lifecycle::{self, LifecycleProbe};

const MODEL: &str = gemini::completion::GEMINI_2_5_FLASH;

#[tokio::test]
async fn middleware_phases_observe_a_unary_completion() {
    let probe = WireProbe::default();
    with_gemini_lifecycle_cassette(
        "lifecycle_matrix/middleware_unary",
        probe.clone(),
        |client| async move {
            let mut ecs = ecs_lifecycle::agent(client.completion_model(MODEL), BASIC_PREAMBLE);
            let response = ecs.prompt(BASIC_PROMPT, false).await;
            assert_nonempty_response(&response);
        },
    )
    .await;
    probe.assert_single_exchange();
}

#[tokio::test]
async fn middleware_response_phase_precedes_stream_consumption() {
    let probe = WireProbe::default();
    let hook = LifecycleProbe::default();
    let settle_hook = hook.clone();
    with_gemini_lifecycle_cassette(
        "lifecycle_matrix/middleware_streaming",
        probe.clone(),
        |client| async move {
            let mut ecs = ecs_lifecycle::agent(client.completion_model(MODEL), STREAMING_PREAMBLE);
            ecs_lifecycle::install(&mut ecs, settle_hook);
            let response = ecs.prompt(STREAMING_PROMPT, true).await;
            let provider_final = ecs_lifecycle::provider_final(&mut ecs);
            assert_nonempty_response(&response);
            assert!(provider_final.usage.total_tokens > 0);
        },
    )
    .await;
    probe.assert_single_exchange();
    // The streamed run settled exactly once, with a response.
    assert_eq!(hook.settle_outcomes(), ["response"]);
    assert_eq!(hook.starts.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn run_start_rewrite_reaches_the_provider() {
    let hook = LifecycleProbe::rewriting_to(
        "Reply with exactly the single word PINEAPPLE and nothing else.",
    );
    let agent_hook = hook.clone();
    with_gemini_lifecycle_cassette(
        "lifecycle_matrix/run_start_rewrite",
        WireProbe::default(),
        |client| async move {
            let mut ecs = ecs_lifecycle::agent(client.completion_model(MODEL), BASIC_PREAMBLE);
            ecs_lifecycle::install(&mut ecs, agent_hook);
            // The original prompt says nothing about pineapples; only the
            // pre-run rewrite can put the marker into the model's reply.
            let response = ecs
                .prompt("Tell me about the Rust borrow checker.", false)
                .await;
            assert!(
                response.to_uppercase().contains("PINEAPPLE"),
                "the provider answered the rewritten prompt, not the original: {response:?}"
            );
        },
    )
    .await;
    assert_eq!(hook.starts.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(hook.settle_outcomes(), ["response"]);
}

#[tokio::test]
async fn entry_log_orders_and_turn_stamps_across_a_streamed_tool_run() {
    let probe = LifecycleProbe::entry_log();
    let agent_hook = probe.clone();
    with_gemini_lifecycle_cassette(
        "lifecycle_matrix/entry_log_order",
        WireProbe::default(),
        |client| async move {
            let mut ecs = ecs_lifecycle::agent(
                client.completion_model(MODEL),
                "You are a calculator. Use the add tool for arithmetic.",
            );
            ecs.tool(Adder);
            ecs_lifecycle::install(&mut ecs, agent_hook);
            let response = ecs
                .prompt_with_max_turns(
                    "What is 9 + 16? Use the add tool, then reply with just the number.",
                    true,
                    Some(3),
                )
                .await;
            let _final = ecs_lifecycle::provider_final(&mut ecs);
            assert!(
                response.contains("25"),
                "the tool result reached the final answer: {response:?}"
            );
        },
    )
    .await;
    // Turn-0 run_start append, then one turn-stamped snapshot per model call
    // (a tool run makes at least two), replayed in append order at settle.
    probe.assert_phases(2);
}

#[tokio::test]
async fn run_settles_once_across_a_multi_turn_tool_run_with_durable_state() {
    let hook = LifecycleProbe::default();
    let agent_hook = hook.clone();
    with_gemini_lifecycle_cassette(
        "lifecycle_matrix/run_settled_tool_run",
        WireProbe::default(),
        |client| async move {
            let mut ecs = ecs_lifecycle::agent(
                client.completion_model(MODEL),
                "You are a calculator. Use the add tool for arithmetic.",
            );
            ecs.tool(Adder);
            ecs_lifecycle::install(&mut ecs, agent_hook);
            let response = ecs
                .prompt_with_max_turns(
                    "What is 7 + 15? Use the add tool, then reply with just the number.",
                    false,
                    Some(3),
                )
                .await;
            assert!(
                response.contains("22"),
                "the tool result reached the final answer: {response:?}"
            );
        },
    )
    .await;
    assert_eq!(hook.starts.load(std::sync::atomic::Ordering::SeqCst), 1);
    // Terminal, not per-turn: two model calls, one settle.
    assert_eq!(hook.settle_outcomes(), ["response"]);
    let calls = hook
        .exported_completion_calls()
        .expect("the settle export carries the durable counter");
    assert!(
        calls >= 2,
        "a tool run makes at least two model calls; durable counter was {calls}"
    );
}
