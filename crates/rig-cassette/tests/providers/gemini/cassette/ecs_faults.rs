//! The failure rows on the Gemini REST wire (`gemini-3-flash-preview`): every cell of
//! `tests/common/ecs_matrix/faults.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the wire's own recording (the recorded
//! rows, the producer's rows in `corpus_faults.rs`), over a
//! recording another cell owns (rows 9 and 13),
//! or over the sequenced transport serving labelled frames cut or rewritten
//! from this wire's #2501 recordings (the scripted rows, against the
//! rig-agent runner over the same frames). This file holds the scenario
//! literals, the frames' provenance, the wire's models and its `#[ignore]`
//! reasons; the drivers are `tests/common/ecs_matrix/{world,agent,extra,faults}.rs`.

use rig::error::ErrorKind;
use rig::providers::gemini::GeminiConfig;
use rig::providers::gemini::completion::{GEMINI_2_5_FLASH, GEMINI_3_FLASH_PREVIEW};
use rig_test_support::cassette_models::GeminiModels;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{
    Wire, cells,
    cells::Cell,
    corpus::{Ending, Program},
    faults::{self, Fault},
    world::{run_scripted, run_world},
};
use crate::stream_faults::SseShape;

fn wire(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion("gemini-nonexistent-rig-test"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The recording's own model, for a cell that reuses a recording the
/// corpus already had.
fn legacy(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion(GEMINI_2_5_FLASH),
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

/// The scripted rows' facts: the #2501 recordings they cut (a streamed
/// text answer and a streamed tool call, on this wire's own model) and the
/// recorded setup failure the status rows rewrite.
const SCRIPTED: faults::Scripted<rig::Model<rig::providers::gemini::completion::GenerateContent>> =
    faults::Scripted {
        provider: "gemini",
        shape: SseShape::Gemini,
        text_stream: "corpus_matrix/shaping_extra_context_streamed",
        tool_stream: "corpus_matrix/hooks_patch_tool_args_streamed",
        setup_reply: "corpus_faults/setup_unary",
        code: Some("NOT_FOUND"),
        wire: |http| {
            wire(&GeminiModels::new(
                GeminiConfig::new("scripted-fault-key-7f3a9c"),
                http,
            ))
        },
    };

crate::matrix::native_matrix! {
    wrapper: with_gemini_cassette, wire: missing, run: run_world;
    #[tokio::test]
    setup_unary: ("corpus_faults/setup_unary", SETUP_UNARY, "gemini_setup_unary");
    #[tokio::test]
    setup_streamed: ("error_envelope/nonexistent_model_streaming_error_preserves_status_and_body", SETUP_STREAMED, "gemini_setup_streamed");
}

crate::matrix::native_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: run_world;
    #[tokio::test]
    tool_error: ("corpus_faults/tool_error", faults::TOOL_ERROR, "gemini_tool_error");
    #[tokio::test]
    tool_error_streamed: ("corpus_faults/tool_error_streamed", faults::TOOL_ERROR_STREAMED, "gemini_tool_error_streamed");
    #[tokio::test]
    batch_second_fails: ("corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS, "gemini_batch_second_fails");
    #[tokio::test]
    batch_second_fails_concurrent: ("corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS_CONCURRENT, "gemini_batch_second_fails_concurrent");
    /// Row 9 over `endings_tool_outcome_cancelled`'s recording.
    #[tokio::test]
    stop_while_tool_runs: ("corpus_matrix/endings_tool_outcome_cancelled", faults::STOP_WHILE_TOOL_RUNS, "gemini_faults_stop_while_tool_runs");
    /// Row 13 over `resume_tool_turn`'s recording.
    #[tokio::test]
    scene_tool_in_flight: ("corpus_matrix/resume_tool_turn", faults::SCENE_TOOL_IN_FLIGHT, "gemini_faults_scene_tool_in_flight");
}

crate::matrix::native_matrix! {
    wrapper: with_gemini_cassette, wire: legacy, run: faults::cancel_after_terminal;
    /// Row 11: a bare `Cancelled` once the terminal record has landed and
    /// before `Fold`: a whole completion, the run cancelled, despawned at once.
    #[tokio::test]
    cancel_after_terminal: ("corpus_breadth/text_delta_stop", cells::BREADTH_TEXT_DELTA_STOP, "gemini_faults_cancel_after_terminal");
}

crate::matrix::case_matrix! {
    family: ecs_faults_case;
    #[tokio::test]
    truncated_after_text: SCRIPTED => "gemini_faults_truncated_after_text";
    #[tokio::test]
    truncated_after_tool_call: SCRIPTED => "gemini_faults_truncated_after_tool_call";
    #[tokio::test]
    error_after_text: SCRIPTED => "gemini_faults_error_after_text";
    #[tokio::test]
    filtered_with_text: SCRIPTED => "gemini_faults_filtered_with_text";
    #[tokio::test]
    filtered_empty: SCRIPTED => "gemini_faults_filtered_empty";
    #[tokio::test]
    failing_load: SCRIPTED => "gemini_faults_failing_load";
    #[tokio::test]
    failing_load_streamed: SCRIPTED => "gemini_faults_failing_load_streamed";
    #[tokio::test]
    cancel_at_first_tool_call_delta: SCRIPTED => "gemini_faults_cancel_at_first_tool_call_delta";
    #[tokio::test]
    status_429: SCRIPTED => "gemini_faults_status_429";
    #[tokio::test]
    status_503: SCRIPTED => "gemini_faults_status_503";
    #[tokio::test]
    status_503_retried: SCRIPTED => "gemini_faults_status_503_retried";
}

/// Gemini refuses the prompt outright (`promptFeedback.blockReason`, no
/// candidates): a non-retryable provider failure naming the block.
#[tokio::test]
async fn refusal() {
    crate::goldens::capture_world_programs(async {
        let cell = Cell {
            program: Program {
                ending: Ending::Failed(ErrorKind::ProviderResponse),
                ..faults::REFUSAL.program
            },
            ..faults::REFUSAL
        };
        let frames = SCRIPTED.shape.refusal(&[]);
        run_scripted(
            &cell,
            || SCRIPTED.stream(&frames),
            |log| crate::goldens::world_golden_effects("gemini_faults_refusal", log),
        )
        .await;
    })
    .await
}
