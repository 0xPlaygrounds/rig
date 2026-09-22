//! The failure rows on the OpenAI Chat Completions wire (`gpt-5-mini`): every cell of
//! `tests/common/ecs_matrix/faults.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the wire's own recording (the recorded
//! rows, the producer's rows in `corpus_faults_chat.rs`), over a
//! recording another cell owns (rows 9 and 13),
//! or over the sequenced transport serving labelled frames cut or rewritten
//! from this wire's #2501 recordings (the scripted rows, against the
//! rig-agent runner over the same frames). This file holds the scenario
//! literals, the frames' provenance, the wire's models and its `#[ignore]`
//! reasons; the drivers are `tests/common/ecs_matrix/{world,agent,extra}.rs`.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::GPT_5_MINI;
use rig::providers::openai::wire::OpenAI;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{
    Wire, cells,
    cells::Cell,
    corpus::Program,
    extra::{Cut, cancel_at},
    faults::{self, Fault},
    world::{run_scripted, run_world},
};
use crate::stream_faults::{
    CHAT_REFUSAL_TEXT, SseShape, recorded_sse_frames, scripted, sse_bytes, status_reply,
};

fn wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat(GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat("gpt-5-mini-nonexistent-rig-test"),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
pub(super) const SETUP_UNARY: Cell = Cell {
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
const SCRIPTED_KEY: &str = "sk-scripted-fault-key-7f3a9c";
const SHAPE: SseShape = SseShape::Chat;
/// The #2501 recordings the scripted rows cut: a streamed text answer and
/// a streamed tool call, on this wire's own model.
const TEXT_STREAM: &str = "corpus_matrix_chat/shaping_extra_context_streamed";
const TOOL_STREAM: &str = "corpus_matrix_chat/hooks_patch_tool_args_streamed";
/// The recorded setup failure the status rows rewrite.
const SETUP_REPLY: &str = "corpus_faults_chat/setup_unary";

fn recorded(scenario: &str) -> Vec<String> {
    recorded_sse_frames("openai", scenario, 0)
}

fn reply(status: u16, retry_after: bool) -> MockHttpResponse {
    status_reply("openai", SETUP_REPLY, status, retry_after)
}

/// The wire over a transport that answers one streaming request with
/// `frames`, then EOF.
fn scripted_stream(frames: &[String]) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = OpenAI::new(SCRIPTED_KEY).bind(scripted(vec![sse_bytes(frames)]));
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client.chat(GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

/// The wire over a transport that answers each unary request with the
/// next of `replies`.
fn scripted_unary(replies: Vec<MockHttpResponse>) -> Wire<impl CompletionModel + Clone + 'static> {
    let client = OpenAI::new(SCRIPTED_KEY).bind(SequencedHttpClient::new(replies));
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client.chat(GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

crate::matrix::native_matrix! {
    wrapper: with_openai_cassette, wire: missing, run: run_world;
    #[tokio::test]
    setup_unary: ("corpus_faults_chat/setup_unary", SETUP_UNARY, "openai_chat_setup_unary");
    #[tokio::test]
    setup_streamed: ("corpus_matrix_chat/error_facts_streamed", SETUP_STREAMED, "openai_chat_setup_streamed");
}

crate::matrix::native_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: run_world;
    #[tokio::test]
    tool_error: ("corpus_faults_chat/tool_error", faults::TOOL_ERROR, "openai_chat_tool_error");
    #[tokio::test]
    tool_error_streamed: ("corpus_faults_chat/tool_error_streamed", faults::TOOL_ERROR_STREAMED, "openai_chat_tool_error_streamed");
    #[tokio::test]
    batch_second_fails: ("corpus_faults_chat/batch_second_fails", faults::BATCH_SECOND_FAILS, "openai_chat_batch_second_fails");
    #[tokio::test]
    batch_second_fails_concurrent: ("corpus_faults_chat/batch_second_fails", faults::BATCH_SECOND_FAILS_CONCURRENT, "openai_chat_batch_second_fails_concurrent");
}

/// Row 9 over `endings_tool_outcome_cancelled`'s recording.
#[tokio::test]
async fn stop_while_tool_runs() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "corpus_matrix_chat/endings_tool_outcome_cancelled",
            |client| async move {
                run_world(&wire(&client), &faults::STOP_WHILE_TOOL_RUNS, |log| {
                    crate::goldens::world_golden_effects(
                        "openai_faults_chat_stop_while_tool_runs",
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
        with_openai_cassette("corpus_matrix_chat/resume_tool_turn", |client| async move {
            run_world(&wire(&client), &faults::SCENE_TOOL_IN_FLIGHT, |log| {
                crate::goldens::world_golden_effects("openai_faults_chat_scene_tool_in_flight", log)
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
    truncated_after_text: truncated_after_text_0 => "openai_faults_chat_truncated_after_text";
    #[tokio::test]
    truncated_after_tool_call: truncated_after_tool_call_1 => "openai_faults_chat_truncated_after_tool_call";
    /// The in-band error frame's facts, as the funnel reports them on this
    /// shape (`SseShape::error_{code,message,status}`).
    #[tokio::test]
    error_after_text: error_after_text_2 => "openai_faults_chat_error_after_text";
    #[tokio::test]
    filtered_with_text: filtered_with_text_3 => "openai_faults_chat_filtered_with_text";
    #[tokio::test]
    filtered_empty: filtered_empty_4 => "openai_faults_chat_filtered_empty";
    /// Row 10: no request reaches the wire; the transport answers nothing.
    #[tokio::test]
    failing_load: failing_load_5 => "openai_faults_chat_failing_load";
    #[tokio::test]
    failing_load_streamed: failing_load_streamed_6 => "openai_faults_chat_failing_load_streamed";
    /// Row 11: a bare `Cancelled` at the first tool-call delta; the stream is
    /// left to its handler, the tool never dispatched. The recorded tool turn
    /// is served whole by the sequenced transport: a cancelled run makes one
    /// request, and the recording holds two.
    #[tokio::test]
    cancel_at_first_tool_call_delta: cancel_at_first_tool_call_delta_7 => "openai_faults_chat_cancel_at_first_tool_call_delta";
    #[tokio::test]
    status_429: status_429_23 => "openai_faults_chat_status_429";
    #[tokio::test]
    status_503: status_503_24 => "openai_faults_chat_status_503";
    /// World-only: the default budget re-issues the completion three times.
    #[tokio::test]
    status_503_retried: status_503_retried_25 => "openai_faults_chat_status_503_retried";
}

/// The refusal streams as the answer: the text the adapter's unit tests
/// pin (`CHAT_REFUSAL_TEXT`), whole.
#[tokio::test]
async fn refusal() {
    crate::goldens::capture_world_programs(async {
        let frames = SHAPE.refusal(&[]);
        let log = run_scripted(
            &faults::REFUSAL,
            || scripted_stream(&frames),
            |log| crate::goldens::world_golden_effects("openai_faults_chat_refusal", log),
        )
        .await;
        assert_eq!(
            crate::ecs_matrix::corpus::golden_answer(&log),
            CHAT_REFUSAL_TEXT
        );
    })
    .await
}

/// Row 11: a bare `Cancelled` once the terminal record has landed and
/// before `Fold`: a whole completion, the run cancelled, despawned at once.
#[tokio::test]
async fn cancel_after_terminal() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "corpus_matrix_chat/endings_text_delta_stop",
            |client| async move {
                cancel_at(
                    &wire(&client),
                    &cells::ENDINGS_TEXT_DELTA_STOP,
                    Cut::AfterTerminal,
                    |log| {
                        crate::goldens::world_golden_effects(
                            "openai_faults_chat_cancel_after_terminal",
                            log,
                        )
                    },
                )
                .await;
            },
        )
        .await;
    })
    .await
}
