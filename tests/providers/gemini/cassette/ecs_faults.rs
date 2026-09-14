//! The failure rows on the Gemini REST wire (`gemini-3-flash-preview`): every cell of
//! `tests/common/ecs_matrix/faults.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the wire's own recording (the recorded
//! rows, against the producer's golden in `corpus_faults.rs`), over a
//! recording another cell owns (rows 9 and 13, against that cell's golden),
//! or over the sequenced transport serving labelled frames cut or rewritten
//! from this wire's #2501 recordings (the scripted rows, against the
//! rig-agent runner over the same frames). This file holds the scenario
//! literals, the frames' provenance, the wire's models and its `#[ignore]`
//! reasons; the drivers are `tests/common/ecs_matrix/{world,agent,extra}.rs`.

use rig::completion::CompletionModel;
use rig::error::ErrorKind;
use rig::prelude::*;
use rig::providers::gemini::completion::{GEMINI_2_5_FLASH, GEMINI_3_FLASH_PREVIEW};
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{
    Wire, cells,
    cells::Cell,
    corpus::{Ending, Program},
    extra::{Cut, cancel_at},
    faults::{self, Fault},
    world::{run_scripted, run_world},
};
use crate::stream_faults::{SseShape, recorded_sse_frames, scripted, sse_bytes, status_reply};

fn wire(client: &rig::providers::gemini::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion_model(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(
    client: &rig::providers::gemini::Client,
) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion_model("gemini-nonexistent-rig-test"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The recording's own model, for a cell that reuses a recording the
/// corpus already had.
fn legacy(client: &rig::providers::gemini::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion_model(GEMINI_2_5_FLASH),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
pub(super) const SETUP_UNARY: Cell = Cell {
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("NOT_FOUND"),
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
        code: Some("NOT_FOUND"),
    }),
    ..SETUP_UNARY
};

/// A key the scripted cells send: it must never reach a recording or a
/// trace.
const SCRIPTED_KEY: &str = "scripted-fault-key-7f3a9c";
const SHAPE: SseShape = SseShape::Gemini;
/// The #2501 recordings the scripted rows cut: a streamed text answer and
/// a streamed tool call, on this wire's own model.
const TEXT_STREAM: &str = "corpus_matrix/shaping_extra_context_streamed";
const TOOL_STREAM: &str = "corpus_matrix/hooks_patch_tool_args_streamed";
/// The recorded setup failure the status rows rewrite.
const SETUP_REPLY: &str = "corpus_faults/setup_unary";

fn recorded(scenario: &str) -> Vec<String> {
    recorded_sse_frames("gemini", scenario, 0)
}

fn reply(status: u16, retry_after: bool) -> MockHttpResponse {
    status_reply("gemini", SETUP_REPLY, status, retry_after)
}

/// The wire over a transport that answers one streaming request with
/// `frames`, then EOF.
fn scripted_stream(frames: &[String]) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = rig::providers::gemini::Client::builder()
        .api_key(SCRIPTED_KEY)
        .http_client(scripted(vec![sse_bytes(frames)]))
        .build()
        .expect("client should build");
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion_model(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The wire over a transport that answers each unary request with the
/// next of `replies`.
fn scripted_unary(replies: Vec<MockHttpResponse>) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = rig::providers::gemini::Client::builder()
        .api_key(SCRIPTED_KEY)
        .http_client(SequencedHttpClient::new(replies))
        .build()
        .expect("client should build");
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion_model(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: missing, run: run_world, oracle: crate::ecs_goldens::golden_effects;
    #[tokio::test]
    setup_unary: ("corpus_faults/setup_unary", SETUP_UNARY, "gemini_fault_setup_unary");
    #[tokio::test]
    setup_streamed: ("error_envelope/nonexistent_model_streaming_error_preserves_status_and_body", SETUP_STREAMED, "gemini_fault_setup_streamed");
}

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: run_world, oracle: crate::ecs_goldens::golden_effects;
    #[tokio::test]
    tool_error: ("corpus_faults/tool_error", faults::TOOL_ERROR, "gemini_fault_tool_error");
    #[tokio::test]
    tool_error_streamed: ("corpus_faults/tool_error_streamed", faults::TOOL_ERROR_STREAMED, "gemini_fault_tool_error_streamed");
    #[tokio::test]
    batch_second_fails: ("corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS, "gemini_fault_batch_second_fails");
    #[tokio::test]
    batch_second_fails_concurrent: ("corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS_CONCURRENT, "gemini_fault_batch_second_fails_concurrent");
}

/// Row 9 over `endings_tool_outcome_cancelled`'s recording and golden.
#[tokio::test]
async fn stop_while_tool_runs() {
    with_gemini_cassette(
        "corpus_matrix/endings_tool_outcome_cancelled",
        |client| async move {
            run_world(&wire(&client), &faults::STOP_WHILE_TOOL_RUNS, |log| {
                crate::ecs_goldens::compare_to_original(
                    "gemini_endings_tool_outcome_cancelled",
                    log,
                )
            })
            .await;
        },
    )
    .await;
}

/// Row 13 over `resume_tool_turn`'s recording and golden.
#[tokio::test]
async fn scene_tool_in_flight() {
    with_gemini_cassette("corpus_matrix/resume_tool_turn", |client| async move {
        run_world(&wire(&client), &faults::SCENE_TOOL_IN_FLIGHT, |log| {
            crate::ecs_goldens::compare_to_original("gemini_resume_tool_turn", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn status_429() {
    let cell = Cell {
        fault: Some(Fault::Status {
            status: 429,
            code: Some("NOT_FOUND"),
            retry_after: true,
        }),
        ..faults::STATUS_429
    };
    run_scripted(&cell, || scripted_unary(vec![reply(429, true)])).await;
}

#[tokio::test]
async fn status_503() {
    let cell = Cell {
        fault: Some(Fault::Status {
            status: 503,
            code: Some("NOT_FOUND"),
            retry_after: false,
        }),
        ..faults::STATUS_503
    };
    run_scripted(&cell, || scripted_unary(vec![reply(503, false)])).await;
}

/// World-only: the default budget re-issues the completion three times.
#[tokio::test]
async fn status_503_retried() {
    let cell = Cell {
        fault: Some(Fault::Status {
            status: 503,
            code: Some("NOT_FOUND"),
            retry_after: false,
        }),
        ..faults::STATUS_503_RETRIED
    };
    let replies = || (0..4).map(|_| reply(503, false)).collect();
    run_world(&scripted_unary(replies()), &cell, |_| {}).await;
}

crate::matrix::case_matrix! {
    family: wire_matrix_case;
    #[tokio::test]
    truncated_after_text: truncated_after_text_0;
    #[tokio::test]
    truncated_after_tool_call: truncated_after_tool_call_1;
    /// The in-band error frame's facts, as the funnel reports them on this
    /// shape (`SseShape::error_{code,message,status}`).
    #[tokio::test]
    error_after_text: error_after_text_2;
    #[tokio::test]
    filtered_with_text: filtered_with_text_3;
    #[tokio::test]
    filtered_empty: filtered_empty_4;
    /// Row 10: no request reaches the wire; the transport answers nothing.
    #[tokio::test]
    failing_load: failing_load_5;
    #[tokio::test]
    failing_load_streamed: failing_load_streamed_6;
    /// Row 11: a bare `Cancelled` at the first tool-call delta; the stream is
    /// left to its handler, the tool never dispatched. The recorded tool turn
    /// is served whole by the sequenced transport: a cancelled run makes one
    /// request, and the recording holds two.
    #[tokio::test]
    cancel_at_first_tool_call_delta: cancel_at_first_tool_call_delta_7;
}

/// Gemini refuses the prompt outright (`promptFeedback.blockReason`, no
/// candidates): a non-retryable provider failure naming the block.
#[tokio::test]
async fn refusal() {
    let cell = Cell {
        program: Program {
            ending: Ending::Failed(ErrorKind::ProviderResponse),
            ..faults::REFUSAL.program
        },
        ..faults::REFUSAL
    };
    let frames = SHAPE.refusal(&[]);
    run_scripted(&cell, || scripted_stream(&frames)).await;
}

crate::matrix::case_matrix! {
    wrapper: with_gemini_cassette, family: wire_matrix_case;
    /// Row 11: a bare `Cancelled` once the terminal record has landed and
    /// before `Fold`: a whole completion, the run cancelled, despawned at once.
    #[tokio::test]
    cancel_after_terminal: ("corpus_breadth/text_delta_stop", cancel_after_terminal_8);
}
