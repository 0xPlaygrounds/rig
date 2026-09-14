//! The failure rows on the DeepSeek wire (`deepseek-chat`): every cell of
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
use rig::prelude::*;

use rig::test_utils::{MockHttpResponse, SequencedHttpClient};

use crate::deepseek::support::with_deepseek_cassette;
use crate::ecs_matrix::{
    Wire, cells,
    cells::Cell,
    corpus::Program,
    extra::{Cut, cancel_at},
    faults::{self, Fault},
    world::{run_scripted, run_world},
};
use crate::stream_faults::{SseShape, recorded_sse_frames, scripted, sse_bytes, status_reply};

/// The recorded DeepSeek setup failure asked the wire not to think.
fn thinking_disabled() -> serde_json::Value {
    serde_json::json!({ "thinking": { "type": "disabled" } })
}

fn wire(client: &rig::providers::deepseek::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion_model("deepseek-chat"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(
    client: &rig::providers::deepseek::Client,
) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion_model("deepseek-v9-nonexistent"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
pub(super) const SETUP_UNARY: Cell = Cell {
    program: Program {
        prompt: "hi",
        max_tokens: Some(8),
        additional_params: Some(thinking_disabled),
        ..faults::SETUP_UNARY.program
    },
    fault: Some(Fault::Setup {
        status: 400,
        code: Some("invalid_request_error"),
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
        status: 400,
        code: Some("invalid_request_error"),
    }),
    ..SETUP_UNARY
};

/// A key the scripted cells send: it must never reach a recording or a
/// trace.
const SCRIPTED_KEY: &str = "sk-scripted-fault-key-7f3a9c";
const SHAPE: SseShape = SseShape::Chat;
/// The #2501 recordings the scripted rows cut: a streamed text answer and
/// a streamed tool call, on this wire's own model.
const TEXT_STREAM: &str = "corpus_matrix/shaping_extra_context_streamed";
const TOOL_STREAM: &str = "corpus_matrix/hooks_patch_tool_args_streamed";
/// The recorded setup failure the status rows rewrite.
const SETUP_REPLY: &str = "corpus_faults/setup_unary";

fn recorded(scenario: &str) -> Vec<String> {
    recorded_sse_frames("deepseek", scenario, 0)
}

fn reply(status: u16, retry_after: bool) -> MockHttpResponse {
    status_reply("deepseek", SETUP_REPLY, status, retry_after)
}

/// The wire over a transport that answers one streaming request with
/// `frames`, then EOF.
fn scripted_stream(frames: &[String]) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = rig::providers::deepseek::Client::builder()
        .api_key(SCRIPTED_KEY)
        .http_client(scripted(vec![sse_bytes(frames)]))
        .build()
        .expect("client should build");
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion_model("deepseek-chat"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The wire over a transport that answers each unary request with the
/// next of `replies`.
fn scripted_unary(replies: Vec<MockHttpResponse>) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = rig::providers::deepseek::Client::builder()
        .api_key(SCRIPTED_KEY)
        .http_client(SequencedHttpClient::new(replies))
        .build()
        .expect("client should build");
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion_model("deepseek-chat"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn setup_unary() {
    with_deepseek_cassette("corpus_faults/setup_unary", |client| async move {
        run_world(&missing(&client), &SETUP_UNARY, |log| {
            crate::ecs_goldens::golden_effects("deepseek_fault_setup_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn setup_streamed() {
    with_deepseek_cassette("corpus_matrix/error_facts_streamed", |client| async move {
        run_world(&missing(&client), &SETUP_STREAMED, |log| {
            crate::ecs_goldens::golden_effects("deepseek_fault_setup_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn tool_error() {
    with_deepseek_cassette("corpus_faults/tool_error", |client| async move {
        run_world(&wire(&client), &faults::TOOL_ERROR, |log| {
            crate::ecs_goldens::golden_effects("deepseek_fault_tool_error", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn tool_error_streamed() {
    with_deepseek_cassette("corpus_faults/tool_error_streamed", |client| async move {
        run_world(&wire(&client), &faults::TOOL_ERROR_STREAMED, |log| {
            crate::ecs_goldens::golden_effects("deepseek_fault_tool_error_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn batch_second_fails() {
    with_deepseek_cassette("corpus_faults/batch_second_fails", |client| async move {
        run_world(&wire(&client), &faults::BATCH_SECOND_FAILS, |log| {
            crate::ecs_goldens::golden_effects("deepseek_fault_batch_second_fails", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn batch_second_fails_concurrent() {
    with_deepseek_cassette("corpus_faults/batch_second_fails", |client| async move {
        run_world(
            &wire(&client),
            &faults::BATCH_SECOND_FAILS_CONCURRENT,
            |log| {
                crate::ecs_goldens::golden_effects(
                    "deepseek_fault_batch_second_fails_concurrent",
                    log,
                )
            },
        )
        .await;
    })
    .await;
}

/// Row 9 over `endings_tool_outcome_cancelled`'s recording and golden.
#[tokio::test]
async fn stop_while_tool_runs() {
    with_deepseek_cassette(
        "corpus_matrix/endings_tool_outcome_cancelled",
        |client| async move {
            run_world(&wire(&client), &faults::STOP_WHILE_TOOL_RUNS, |log| {
                crate::ecs_goldens::compare_to_original(
                    "deepseek_endings_tool_outcome_cancelled",
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
    with_deepseek_cassette("corpus_matrix/resume_tool_turn", |client| async move {
        run_world(&wire(&client), &faults::SCENE_TOOL_IN_FLIGHT, |log| {
            crate::ecs_goldens::compare_to_original("deepseek_resume_tool_turn", log)
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
            code: Some("invalid_request_error"),
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
            code: Some("invalid_request_error"),
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
            code: Some("invalid_request_error"),
            retry_after: false,
        }),
        ..faults::STATUS_503_RETRIED
    };
    let replies = || (0..4).map(|_| reply(503, false)).collect();
    run_world(&scripted_unary(replies()), &cell, |_| {}).await;
}

#[tokio::test]
async fn truncated_after_text() {
    let frames = SHAPE.text_prefix(&recorded(TEXT_STREAM));
    run_scripted(&faults::TRUNCATED_AFTER_TEXT, || scripted_stream(&frames)).await;
}

#[tokio::test]
async fn truncated_after_tool_call() {
    let frames = SHAPE.tool_prefix(&recorded(TOOL_STREAM));
    run_scripted(&faults::TRUNCATED_AFTER_TOOL_CALL, || {
        scripted_stream(&frames)
    })
    .await;
}

/// The in-band error frame's facts, as the funnel reports them on this
/// shape (`SseShape::error_{code,message,status}`).
#[tokio::test]
async fn error_after_text() {
    let cell = Cell {
        fault: Some(Fault::ErrorAfterText {
            code: SHAPE.error_code(),
            message: SHAPE.error_message(),
            status: SHAPE.error_status(),
        }),
        ..faults::ERROR_AFTER_TEXT
    };
    let frames = SHAPE.error_frames(&recorded(TEXT_STREAM));
    run_scripted(&cell, || scripted_stream(&frames)).await;
}

#[ignore = "the deepseek wire reports no refusal: it is OpenAI-compatible and sends neither a `refusal` part nor a blocked prompt"]
#[tokio::test]
async fn refusal() {}

#[tokio::test]
async fn filtered_with_text() {
    let frames = SHAPE.filtered(&recorded(TEXT_STREAM), true);
    run_scripted(&faults::FILTERED_WITH_TEXT, || scripted_stream(&frames)).await;
}

#[tokio::test]
async fn filtered_empty() {
    let frames = SHAPE.filtered(&recorded(TEXT_STREAM), false);
    run_scripted(&faults::FILTERED_EMPTY, || scripted_stream(&frames)).await;
}

/// Row 10: no request reaches the wire; the transport answers nothing.
#[tokio::test]
async fn failing_load() {
    run_scripted(&faults::FAILING_LOAD, || scripted_stream(&[])).await;
}

#[tokio::test]
async fn failing_load_streamed() {
    run_scripted(&faults::FAILING_LOAD_STREAMED, || scripted_stream(&[])).await;
}

/// Row 11: a bare `Cancelled` at the first tool-call delta; the stream is
/// left to its handler, the tool never dispatched. The recorded tool turn
/// is served whole by the sequenced transport: a cancelled run makes one
/// request, and the recording holds two.
#[tokio::test]
async fn cancel_at_first_tool_call_delta() {
    let frames = recorded(TOOL_STREAM);
    cancel_at(
        &scripted_stream(&frames),
        &cells::HOOKS_PATCH_TOOL_ARGS_STREAMED,
        Cut::FirstToolCallDelta,
    )
    .await;
}

/// Row 11: a bare `Cancelled` once the terminal record has landed and
/// before `Fold`: a whole completion, the run cancelled, despawned at once.
#[tokio::test]
async fn cancel_after_terminal() {
    with_deepseek_cassette(
        "corpus_matrix/endings_text_delta_stop",
        |client| async move {
            cancel_at(
                &wire(&client),
                &cells::ENDINGS_TEXT_DELTA_STOP,
                Cut::AfterTerminal,
            )
            .await;
        },
    )
    .await;
}
