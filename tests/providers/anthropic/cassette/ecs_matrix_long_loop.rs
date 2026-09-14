//! The long tool loop's native column on Anthropic: claude-haiku-4-5-20251001.
//! Every live cell reuses the producer's recording with strict matching; the
//! row-1 unary recording is also cut at tool turns 1, 2, 3 and last and
//! resumed live in a fresh world. The scripted row-4 cells serve that same
//! recording through the sequenced transport (`long_loop::scripted_replies`),
//! no cassette and no golden; the negative probe mutates the streamed
//! recording's last tool result and proves the strict matcher refuses it.

use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, cells, long_loop, long_loop_world};
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};

const THINKING: cells::ThinkingWire = cells::ThinkingWire::Anthropic;

fn wire(
    client: &rig::providers::anthropic::Client,
) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Anthropic,
        model: client.completion_model("claude-haiku-4-5-20251001"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// A key the scripted cells send: it must never reach a recording or a
/// trace.
const SCRIPTED_KEY: &str = "sk-ant-scripted-fault-key-7f3a9c";

/// The wire over a transport that answers each unary request with the
/// next of `replies`.
fn scripted_unary(replies: Vec<MockHttpResponse>) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = rig::providers::anthropic::Client::builder()
        .api_key(SCRIPTED_KEY)
        .http_client(SequencedHttpClient::new(replies))
        .build()
        .expect("client should build");
    Wire {
        thinking: THINKING,
        model: client.completion_model("claude-haiku-4-5-20251001"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// This wire has no recorded setup failure to rewrite: the scripted provider
/// fault answers the Anthropic Messages error envelope the adapter's own
/// tests pin (`crates/rig-core/src/providers/anthropic/completion/tests.rs`,
/// `overloaded_error`), under a retryable 503.
const OVERLOADED_BODY: &str =
    r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;

fn fault_reply() -> MockHttpResponse {
    MockHttpResponse::error(
        rig::http_client::StatusCode::SERVICE_UNAVAILABLE,
        OVERLOADED_BODY,
    )
}

#[tokio::test]
async fn long_unary() {
    with_anthropic_cassette("long_loop_matrix/long_unary", |client| async move {
        let cell = cells::Cell {
            resume_after: None,
            ..long_loop::LONG_UNARY
        };
        long_loop_world::run_world(&wire(&client), &cell, golden_anthropic_long_loop_long_unary)
            .await;
    })
    .await;
}

#[tokio::test]
async fn long_unary_cut_1() {
    with_anthropic_cassette("long_loop_matrix/long_unary", |client| async move {
        let cell = cells::Cell {
            resume_after: Some(1),
            ..long_loop::LONG_UNARY
        };
        long_loop_world::run_world(&wire(&client), &cell, golden_anthropic_long_loop_long_unary)
            .await;
    })
    .await;
}

#[tokio::test]
async fn long_unary_cut_2() {
    with_anthropic_cassette("long_loop_matrix/long_unary", |client| async move {
        let cell = cells::Cell {
            resume_after: Some(2),
            ..long_loop::LONG_UNARY
        };
        long_loop_world::run_world(&wire(&client), &cell, golden_anthropic_long_loop_long_unary)
            .await;
    })
    .await;
}

#[tokio::test]
async fn long_unary_cut_3() {
    with_anthropic_cassette("long_loop_matrix/long_unary", |client| async move {
        let cell = cells::Cell {
            resume_after: Some(3),
            ..long_loop::LONG_UNARY
        };
        long_loop_world::run_world(&wire(&client), &cell, golden_anthropic_long_loop_long_unary)
            .await;
    })
    .await;
}

#[tokio::test]
async fn long_unary_cut_final() {
    with_anthropic_cassette("long_loop_matrix/long_unary", |client| async move {
        let cell = cells::Cell {
            resume_after: Some(usize::MAX),
            ..long_loop::LONG_UNARY
        };
        long_loop_world::run_world(&wire(&client), &cell, golden_anthropic_long_loop_long_unary)
            .await;
    })
    .await;
}

#[tokio::test]
async fn long_streamed() {
    with_anthropic_cassette("long_loop_matrix/long_streamed", |client| async move {
        long_loop_world::run_world(
            &wire(&client),
            &long_loop::LONG_STREAMED,
            golden_anthropic_long_loop_long_streamed,
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn parallel_calls() {
    with_anthropic_cassette("long_loop_matrix/parallel_calls", |client| async move {
        long_loop_world::run_world(
            &wire(&client),
            &long_loop::PARALLEL_CALLS,
            golden_anthropic_long_loop_parallel_calls,
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn big_result() {
    with_anthropic_cassette("long_loop_matrix/big_result", |client| async move {
        long_loop_world::run_world(
            &wire(&client),
            &long_loop::BIG_RESULT,
            golden_anthropic_long_loop_big_result,
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn tool_error_midway() {
    with_anthropic_cassette("long_loop_matrix/tool_error_midway", |client| async move {
        long_loop_world::run_world(
            &wire(&client),
            &long_loop::TOOL_ERROR_MIDWAY,
            golden_anthropic_long_loop_tool_error_midway,
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn max_turns_midway() {
    with_anthropic_cassette("long_loop_matrix/max_turns_midway", |client| async move {
        long_loop_world::run_world(
            &wire(&client),
            &long_loop::MAX_TURNS_MIDWAY,
            golden_anthropic_long_loop_max_turns_midway,
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn output_cap_midway() {
    with_anthropic_cassette("long_loop_matrix/output_cap_midway", |client| async move {
        long_loop_world::run_world(
            &wire(&client),
            // Failed(Response): the cut write_file call is a response error
            // on this wire (recording confirms).
            &long_loop::OUTPUT_CAP_MIDWAY_LENGTH_ANSWER,
            golden_anthropic_long_loop_output_cap_midway,
        )
        .await;
    })
    .await;
}

/// Row 4, scripted (`long_loop`'s module doc): the row-1 unary recording
/// with turn `FAULT_TURN`'s arguments rewritten to the wrong type; both
/// interpreters answer `invalid_args`, run nothing, and go on.
#[tokio::test]
async fn invalid_args_midway() {
    let replies = long_loop::scripted_replies(THINKING, &long_loop::INVALID_ARGS_MIDWAY, None);
    long_loop::run_scripted(&long_loop::INVALID_ARGS_MIDWAY, || {
        scripted_unary(replies.clone())
    })
    .await;
}

/// Row 4, scripted and world-only (`long_loop`'s module doc): a retryable
/// 503 before turn `FAULT_TURN`'s completion, re-issued under the default
/// budget (CONTRACT §5) with the same history; rig-agent has no budget, so
/// there is no producer and no golden — `long_loop::assert_log` is the
/// oracle.
#[tokio::test]
async fn provider_fault_midway() {
    let replies = long_loop::scripted_replies(
        THINKING,
        &long_loop::PROVIDER_FAULT_MIDWAY,
        Some(fault_reply()),
    );
    long_loop_world::run_world(
        &scripted_unary(replies),
        &long_loop::PROVIDER_FAULT_MIDWAY,
        |_| {},
    )
    .await;
}

fn golden_anthropic_long_loop_long_unary(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("anthropic_long_loop_long_unary", log);
}

fn golden_anthropic_long_loop_long_streamed(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("anthropic_long_loop_long_streamed", log);
}

fn golden_anthropic_long_loop_parallel_calls(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("anthropic_long_loop_parallel_calls", log);
}

fn golden_anthropic_long_loop_big_result(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("anthropic_long_loop_big_result", log);
}

fn golden_anthropic_long_loop_tool_error_midway(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("anthropic_long_loop_tool_error_midway", log);
}

fn golden_anthropic_long_loop_max_turns_midway(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("anthropic_long_loop_max_turns_midway", log);
}

fn golden_anthropic_long_loop_output_cap_midway(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("anthropic_long_loop_output_cap_midway", log);
}

/// Negative matcher probe against the streamed loop the native consumers
/// replay: the last tool result's final byte, altered, is refused.
#[tokio::test]
async fn streamed_tool_result_mismatch() {
    long_loop::assert_stream_request_rejected("anthropic", "long_loop_matrix/long_streamed").await;
}
