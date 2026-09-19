//! The long tool loop's native column on Gemini: gemini-2.5-flash.
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::scripted_replies`),
//! no cassette and no golden; the negative probe mutates the streamed
//! recording's last tool result and proves the strict matcher refuses it.

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells, long_loop, long_loop_world};
use rig::completion::CompletionModel;
use rig::driver::{Bound, Socket};
use rig::prelude::*;
use rig::providers::gemini::Gemini;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};

const THINKING: cells::ThinkingWire = cells::ThinkingWire::Gemini;

// gemini-2.5-flash, not flash-lite: at temperature 0 flash-lite answered the
// `list_files` functionResponse with an empty candidate (no parts,
// finishReason STOP) on both endpoints, 3 attempts, first recording round.
fn wire<H: Socket>(client: &Bound<Gemini, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion("gemini-2.5-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// A key the scripted cells send: it must never reach a recording or a
/// trace.
const SCRIPTED_KEY: &str = "scripted-fault-key-7f3a9c";

/// The wire over a transport that answers each unary request with the
/// next of `replies`.
fn scripted_unary(replies: Vec<MockHttpResponse>) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = Gemini::new(SCRIPTED_KEY).bind(SequencedHttpClient::new(replies));
    Wire {
        thinking: THINKING,
        model: client.completion("gemini-2.5-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The recorded setup failure the scripted provider fault rewrites to a
/// retryable status (the failure rows' `SETUP_REPLY`).
const SETUP_REPLY: &str = "corpus_faults/setup_unary";

fn fault_reply() -> MockHttpResponse {
    crate::stream_faults::status_reply("gemini", SETUP_REPLY, 503, false)
}

crate::matrix::resume_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: long_loop_world::run_world;
    #[tokio::test]
    long_unary: ("long_loop_matrix/long_unary", long_loop::LONG_UNARY, None, golden_gemini_long_loop_long_unary);
    #[tokio::test]
    long_unary_cut_1: ("long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(1), golden_gemini_long_loop_long_unary);
    #[tokio::test]
    long_unary_cut_2: ("long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(2), golden_gemini_long_loop_long_unary);
    #[tokio::test]
    long_unary_cut_3: ("long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(3), golden_gemini_long_loop_long_unary);
    #[tokio::test]
    long_unary_cut_final: ("long_loop_matrix/long_unary", long_loop::LONG_UNARY, Some(usize::MAX), golden_gemini_long_loop_long_unary);
}

crate::matrix::resume_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: long_loop_world::run_world;
    #[tokio::test]
    long_streamed: ("long_loop_matrix/long_streamed", long_loop::LONG_STREAMED, long_loop::LONG_STREAMED.resume_after, golden_gemini_long_loop_long_streamed);
    #[tokio::test]
    parallel_calls: ("long_loop_matrix/parallel_calls", long_loop::PARALLEL_CALLS, long_loop::PARALLEL_CALLS.resume_after, golden_gemini_long_loop_parallel_calls);
    #[tokio::test]
    big_result: ("long_loop_matrix/big_result", long_loop::BIG_RESULT, long_loop::BIG_RESULT.resume_after, golden_gemini_long_loop_big_result);
    #[tokio::test]
    tool_error_midway: ("long_loop_matrix/tool_error_midway", long_loop::TOOL_ERROR_MIDWAY, long_loop::TOOL_ERROR_MIDWAY.resume_after, golden_gemini_long_loop_tool_error_midway);
    #[tokio::test]
    max_turns_midway: ("long_loop_matrix/max_turns_midway", long_loop::MAX_TURNS_MIDWAY, long_loop::MAX_TURNS_MIDWAY.resume_after, golden_gemini_long_loop_max_turns_midway);
}

#[tokio::test]
async fn output_cap_midway() {
    with_gemini_cassette("long_loop_matrix/output_cap_midway", |client| async move {
        long_loop_world::run_world(
            &wire(&client),
            // Failed(Response): under maxOutputTokens 32 the first request
            // comes back with finishReason MALFORMED_FUNCTION_CALL and no
            // content, a response error at turn 1 (round 3 recording).
            &long_loop::OUTPUT_CAP_MIDWAY,
            golden_gemini_long_loop_output_cap_midway,
        )
        .await;
    })
    .await;
}

crate::matrix::case_matrix! {
    family: wire_matrix_case;
    /// Row 4, scripted (`long_loop`'s module doc): the row-1 unary recording
    /// with turn `FAULT_TURN`'s arguments rewritten to the wrong type; both
    /// interpreters answer `invalid_args`, run nothing, and go on.
    #[tokio::test]
    invalid_args_midway: invalid_args_midway_13;
    /// Row 4, scripted and world-only (`long_loop`'s module doc): a retryable
    /// 503 before turn `FAULT_TURN`'s completion, re-issued under the default
    /// budget (CONTRACT §5) with the same history; rig-agent has no budget, so
    /// there is no producer and no golden — `long_loop::assert_log` is the
    /// oracle.
    #[tokio::test]
    provider_fault_midway: provider_fault_midway_14;
}

fn golden_gemini_long_loop_long_unary(log: &rig::cassette::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_long_loop_long_unary", log);
}

fn golden_gemini_long_loop_long_streamed(log: &rig::cassette::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_long_loop_long_streamed", log);
}

fn golden_gemini_long_loop_parallel_calls(log: &rig::cassette::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_long_loop_parallel_calls", log);
}

fn golden_gemini_long_loop_big_result(log: &rig::cassette::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_long_loop_big_result", log);
}

fn golden_gemini_long_loop_tool_error_midway(log: &rig::cassette::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_long_loop_tool_error_midway", log);
}

fn golden_gemini_long_loop_max_turns_midway(log: &rig::cassette::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_long_loop_max_turns_midway", log);
}

fn golden_gemini_long_loop_output_cap_midway(log: &rig::cassette::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_long_loop_output_cap_midway", log);
}

/// Negative matcher probe against the streamed loop the native consumers
/// replay: the last tool result's final byte, altered, is refused.
#[tokio::test]
async fn streamed_tool_result_mismatch() {
    long_loop::assert_stream_request_rejected("gemini", "long_loop_matrix/long_streamed").await;
}
