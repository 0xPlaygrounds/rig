//! The failure rows on the Doubleword wire (`Qwen/Qwen3.5-397B-A17B-FP8`, `reasoning_effort: none`): every cell of
//! `tests/common/ecs_matrix/faults.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the wire's own recording (the recorded
//! rows, the producer's rows in `corpus_faults.rs`), over a
//! recording another cell owns (rows 9 and 13),
//! or over the sequenced transport serving labelled frames cut or rewritten
//! from this wire's #2501 recordings (the scripted rows, against the
//! rig-agent runner over the same frames). This file holds the scenario
//! literals, the frames' provenance, the wire's models and its `#[ignore]`
//! reasons; the drivers are `tests/common/ecs_matrix/{world,agent,extra}.rs`.

use rig::completion::CompletionModel;
use rig::driver::{Bound, Socket};
use rig::prelude::*;
use rig::providers::doubleword::QWEN3_5_397B_A17B;
use rig::providers::openai::wire::{DOUBLEWORD, OpenAI};
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};

use super::super::support::with_doubleword_cassette;
use crate::ecs_matrix::{
    Wire, cells,
    cells::Cell,
    corpus::Program,
    extra::{Cut, cancel_at},
    faults::{self, Fault},
    world::{run_scripted, run_world},
};
use crate::stream_faults::{SseShape, recorded_sse_frames, scripted, sse_bytes, status_reply};

fn wire<H: Socket>(client: &Bound<OpenAI, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion(QWEN3_5_397B_A17B),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(cells::reasoning_off),
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing<H: Socket>(client: &Bound<OpenAI, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion("rig/definitely-not-a-doubleword-model"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
pub(super) const SETUP_UNARY: Cell = Cell {
    program: Program {
        prompt: "Reply with error-probe.",
        max_tokens: Some(8),
        ..faults::SETUP_UNARY.program
    },
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("model_not_found"),
    }),
    ..faults::SETUP_UNARY
};
pub(super) const SETUP_STREAMED: Cell = Cell {
    program: Program {
        streamed: true,
        ..SETUP_UNARY.program
    },
    name: faults::SETUP_STREAMED.name,
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("model_not_found"),
    }),
    ..SETUP_UNARY
};

/// A key the scripted cells send: it must never reach a recording or a
/// trace.
const SCRIPTED_KEY: &str = "scripted-fault-key-7f3a9c";
const SHAPE: SseShape = SseShape::Chat;
/// The #2501 recordings the scripted rows cut: a streamed text answer and
/// a streamed tool call, on this wire's own model.
const TEXT_STREAM: &str = "corpus_matrix/shaping_extra_context_streamed";
const TOOL_STREAM: &str = "corpus_matrix/hooks_patch_tool_args_streamed";
/// The recorded setup failure the status rows rewrite.
const SETUP_REPLY: &str = "corpus_faults/setup_unary";

fn recorded(scenario: &str) -> Vec<String> {
    recorded_sse_frames("doubleword", scenario, 0)
}

fn reply(status: u16, retry_after: bool) -> MockHttpResponse {
    status_reply("doubleword", SETUP_REPLY, status, retry_after)
}

/// The wire over a transport that answers one streaming request with
/// `frames`, then EOF.
fn scripted_stream(frames: &[String]) -> Wire<impl CompletionModel + Clone + 'static> {
    let client =
        OpenAI::with_key(&DOUBLEWORD, SCRIPTED_KEY).bind(scripted(vec![sse_bytes(frames)]));
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion(QWEN3_5_397B_A17B),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(cells::reasoning_off),
    }
}

/// The wire over a transport that answers each unary request with the
/// next of `replies`.
fn scripted_unary(replies: Vec<MockHttpResponse>) -> Wire<impl CompletionModel + Clone + 'static> {
    let client =
        OpenAI::with_key(&DOUBLEWORD, SCRIPTED_KEY).bind(SequencedHttpClient::new(replies));
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion(QWEN3_5_397B_A17B),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(cells::reasoning_off),
    }
}

crate::matrix::native_matrix! {
    wrapper: with_doubleword_cassette, wire: missing, run: run_world;
    #[tokio::test]
    setup_unary: ("corpus_faults/setup_unary", SETUP_UNARY, "doubleword_setup_unary");
    #[tokio::test]
    setup_streamed: ("error_matrix/unknown_model_streaming", SETUP_STREAMED, "doubleword_setup_streamed");
}

crate::matrix::native_matrix! {
    wrapper: with_doubleword_cassette, wire: wire, run: run_world;
    #[tokio::test]
    tool_error: ("corpus_faults/tool_error", faults::TOOL_ERROR, "doubleword_tool_error");
    #[tokio::test]
    tool_error_streamed: ("corpus_faults/tool_error_streamed", faults::TOOL_ERROR_STREAMED, "doubleword_tool_error_streamed");
    #[tokio::test]
    batch_second_fails: ("corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS, "doubleword_batch_second_fails");
    #[tokio::test]
    batch_second_fails_concurrent: ("corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS_CONCURRENT, "doubleword_batch_second_fails_concurrent");
}

/// Row 9 over `endings_tool_outcome_cancelled`'s recording.
#[tokio::test]
async fn stop_while_tool_runs() {
    crate::goldens::capture_world_programs(async {
        with_doubleword_cassette(
            "corpus_matrix/endings_tool_outcome_cancelled",
            |client| async move {
                run_world(&wire(&client), &faults::STOP_WHILE_TOOL_RUNS, |log| {
                    crate::goldens::world_golden_effects(
                        "doubleword_faults_stop_while_tool_runs",
                        log,
                    )
                })
                .await;
            },
        )
        .await;
    })
    .await
}

/// Row 13 over `resume_tool_turn`'s recording.
#[tokio::test]
async fn scene_tool_in_flight() {
    crate::goldens::capture_world_programs(async {
        with_doubleword_cassette("corpus_matrix/resume_tool_turn", |client| async move {
            run_world(&wire(&client), &faults::SCENE_TOOL_IN_FLIGHT, |log| {
                crate::goldens::world_golden_effects("doubleword_faults_scene_tool_in_flight", log)
            })
            .await;
        })
        .await;
    })
    .await
}

crate::matrix::case_matrix! {
    family: wire_matrix_case;
    #[tokio::test]
    truncated_after_text: truncated_after_text_0 => "doubleword_faults_truncated_after_text";
    #[tokio::test]
    truncated_after_tool_call: truncated_after_tool_call_1 => "doubleword_faults_truncated_after_tool_call";
    /// The in-band error frame's facts, as the funnel reports them on this
    /// shape (`SseShape::error_{code,message,status}`).
    #[tokio::test]
    error_after_text: error_after_text_2 => "doubleword_faults_error_after_text";
    #[tokio::test]
    filtered_with_text: filtered_with_text_3 => "doubleword_faults_filtered_with_text";
    #[tokio::test]
    filtered_empty: filtered_empty_4 => "doubleword_faults_filtered_empty";
    /// Row 10: no request reaches the wire; the transport answers nothing.
    #[tokio::test]
    failing_load: failing_load_5 => "doubleword_faults_failing_load";
    #[tokio::test]
    failing_load_streamed: failing_load_streamed_6 => "doubleword_faults_failing_load_streamed";
    /// Row 11: a bare `Cancelled` at the first tool-call delta; the stream is
    /// left to its handler, the tool never dispatched. The recorded tool turn
    /// is served whole by the sequenced transport: a cancelled run makes one
    /// request, and the recording holds two.
    #[tokio::test]
    cancel_at_first_tool_call_delta: cancel_at_first_tool_call_delta_7 => "doubleword_faults_cancel_at_first_tool_call_delta";
    #[tokio::test]
    status_429: status_429_23 => "doubleword_faults_status_429";
    #[tokio::test]
    status_503: status_503_24 => "doubleword_faults_status_503";
    /// World-only: the default budget re-issues the completion three times.
    #[tokio::test]
    status_503_retried: status_503_retried_25 => "doubleword_faults_status_503_retried";
}

crate::matrix::case_matrix! {
    wrapper: with_doubleword_cassette, family: ecs_faults_case;
    # [doc = " Row 11: a bare `Cancelled` once the terminal record has landed and"]
    # [doc = " before `Fold`: a whole completion, the run cancelled, despawned at once."]
    #[tokio::test]
    cancel_after_terminal: ("corpus_matrix/endings_text_delta_stop", cancel_after_terminal_6, "doubleword_faults_cancel_after_terminal");
}
