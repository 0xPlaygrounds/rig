//! Run-lifecycle matrix (PR #2407): transport middleware, `on_run_start`,
//! `on_run_settled`, and durable scratchpad state, recorded against the live
//! Anthropic API.
//!
//! Every cell sends through a `BoxedHttpClient` carrying a `WireProbe`
//! middleware, so one recorded exchange proves both the transport seam (the
//! phases fired, the serialized body and response status were visible) and
//! the hook-level lifecycle claims. All assertions hold in both cassette
//! modes: on replay the same code paths run against the replay server.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;

use super::super::support::with_anthropic_lifecycle_cassette;
use crate::support::{
    Adder, BASIC_PREAMBLE, BASIC_PROMPT, LifecycleHookProbe, STREAMING_PREAMBLE, STREAMING_PROMPT,
    WireProbe, assert_nonempty_response, collect_stream_final_response_and_provider_final,
};

const MODEL: &str = anthropic::completion::CLAUDE_SONNET_4_6;

#[tokio::test]
async fn middleware_phases_observe_a_unary_completion() {
    let probe = WireProbe::default();
    with_anthropic_lifecycle_cassette(
        "lifecycle_matrix/middleware_unary",
        probe.clone(),
        |client| async move {
            let agent = client.agent(MODEL).preamble(BASIC_PREAMBLE).build();
            let response = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect("completion should succeed");
            assert_nonempty_response(&response);
        },
    )
    .await;
    probe.assert_single_exchange();
}

#[tokio::test]
async fn middleware_response_phase_precedes_stream_consumption() {
    let probe = WireProbe::default();
    let hook = LifecycleHookProbe::default();
    let settle_hook = hook.clone();
    with_anthropic_lifecycle_cassette(
        "lifecycle_matrix/middleware_streaming",
        probe.clone(),
        |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(STREAMING_PREAMBLE)
                .add_hook(settle_hook)
                .build();
            let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
            let (response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");
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
    let hook = LifecycleHookProbe::rewriting_to(
        "Reply with exactly the single word PINEAPPLE and nothing else.",
    );
    let agent_hook = hook.clone();
    with_anthropic_lifecycle_cassette(
        "lifecycle_matrix/run_start_rewrite",
        WireProbe::default(),
        |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(BASIC_PREAMBLE)
                .add_hook(agent_hook)
                .build();
            // The original prompt says nothing about pineapples; only the
            // pre-run rewrite can put the marker into the model's reply.
            let response = agent
                .prompt("Tell me about the Rust borrow checker.")
                .await
                .expect("completion should succeed");
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
async fn run_settles_once_across_a_multi_turn_tool_run_with_durable_state() {
    let hook = LifecycleHookProbe::default();
    let agent_hook = hook.clone();
    with_anthropic_lifecycle_cassette(
        "lifecycle_matrix/run_settled_tool_run",
        WireProbe::default(),
        |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble("You are a calculator. Use the add tool for arithmetic.")
                .tool(Adder)
                .add_hook(agent_hook)
                .build();
            let response = agent
                .prompt("What is 7 + 15? Use the add tool, then reply with just the number.")
                .max_turns(3)
                .await
                .expect("tool run should succeed");
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
