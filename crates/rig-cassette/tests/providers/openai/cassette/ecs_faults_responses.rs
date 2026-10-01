//! The failure rows on the OpenAI Responses wire (`gpt-5-mini`): every cell of
//! `tests/common/ecs_matrix/faults.rs` as an agent graph in a Bevy `World`,
//! served by the real adapter over the wire's own recording (the recorded
//! rows, the producer's rows in `corpus_faults_responses.rs`), over a
//! recording another cell owns (rows 9 and 13),
//! or over the sequenced transport serving labelled frames cut or rewritten
//! from this wire's #2501 recordings (the scripted rows, against the
//! rig-agent runner over the same frames). This file holds the scenario
//! literals, the frames' provenance, the wire's models and its `#[ignore]`
//! reasons; the drivers are `tests/common/ecs_matrix/{world,agent,extra,faults}.rs`.

use rig::providers::openai::OpenAIConfig;
use rig::providers::openai::{GPT_4O, GPT_5_MINI};
use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{
    Wire, cells,
    cells::Cell,
    corpus::Program,
    faults::{self, Fault},
    world::{run_scripted, run_world},
};
use crate::stream_faults::SseShape;

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    gpt_5_mini(&client.openai)
}

fn gpt_5_mini(models: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: models.completion(GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion("gpt-4o-mini-nonexistent-rig-test"),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

/// The recording's own model, for a cell that reuses a recording the
/// corpus already had.
fn legacy(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion(GPT_4O),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
/// The unary reply the wire gives today (404) differs from the streamed
/// recording's (400, recorded earlier): the wire's own two replies.
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
        status: 400,
        code: Some("model_not_found"),
    }),
    ..SETUP_UNARY
};

/// The scripted rows' facts: the #2501 recordings they cut (a streamed
/// text answer and a streamed tool call, on this wire's own model) and the
/// recorded setup failure the status rows rewrite.
const SCRIPTED: faults::Scripted<rig::Model<rig::providers::openai::wire::OpenAiWire>> =
    faults::Scripted {
        provider: "openai",
        shape: SseShape::Responses,
        text_stream: "corpus_matrix_responses/shaping_extra_context_streamed",
        tool_stream: "corpus_matrix_responses/hooks_patch_tool_args_streamed",
        setup_reply: "corpus_faults_responses/setup_unary",
        code: Some("model_not_found"),
        wire: |http| {
            gpt_5_mini(&OpenAiModels::new(
                OpenAIConfig::new("sk-scripted-fault-key-7f3a9c"),
                http,
            ))
        },
    };

crate::matrix::native_matrix! {
    wrapper: with_openai_cassette, wire: missing, run: run_world;
    #[tokio::test]
    setup_unary: ("corpus_faults_responses/setup_unary", SETUP_UNARY, "openai_responses_setup_unary");
    #[tokio::test]
    setup_streamed: ("error_envelope/nonexistent_model_streaming_error_preserves_status_and_body", SETUP_STREAMED, "openai_responses_setup_streamed");
}

crate::matrix::native_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: run_world;
    #[tokio::test]
    tool_error: ("corpus_faults_responses/tool_error", faults::TOOL_ERROR, "openai_responses_tool_error");
    #[tokio::test]
    tool_error_streamed: ("corpus_faults_responses/tool_error_streamed", faults::TOOL_ERROR_STREAMED, "openai_responses_tool_error_streamed");
    #[tokio::test]
    batch_second_fails: ("corpus_faults_responses/batch_second_fails", faults::BATCH_SECOND_FAILS, "openai_responses_batch_second_fails");
    #[tokio::test]
    batch_second_fails_concurrent: ("corpus_faults_responses/batch_second_fails", faults::BATCH_SECOND_FAILS_CONCURRENT, "openai_responses_batch_second_fails_concurrent");
    /// Row 9 over `endings_tool_outcome_cancelled`'s recording.
    #[tokio::test]
    stop_while_tool_runs: ("corpus_matrix_responses/endings_tool_outcome_cancelled", faults::STOP_WHILE_TOOL_RUNS, "openai_faults_responses_stop_while_tool_runs");
    /// Row 13 over `resume_tool_turn`'s recording.
    #[tokio::test]
    scene_tool_in_flight: ("corpus_matrix_responses/resume_tool_turn", faults::SCENE_TOOL_IN_FLIGHT, "openai_faults_responses_scene_tool_in_flight");
}

crate::matrix::native_matrix! {
    wrapper: with_openai_cassette, wire: legacy, run: faults::cancel_after_terminal;
    /// Row 11: a bare `Cancelled` once the terminal record has landed and
    /// before `Fold`: a whole completion, the run cancelled, despawned at once.
    #[tokio::test]
    cancel_after_terminal: ("corpus_breadth/text_delta_stop", cells::BREADTH_TEXT_DELTA_STOP, "openai_faults_responses_cancel_after_terminal");
}

crate::matrix::case_matrix! {
    family: ecs_faults_case;
    #[tokio::test]
    truncated_after_text: SCRIPTED => "openai_faults_responses_truncated_after_text";
    #[tokio::test]
    truncated_after_tool_call: SCRIPTED => "openai_faults_responses_truncated_after_tool_call";
    #[tokio::test]
    error_after_text: SCRIPTED => "openai_faults_responses_error_after_text";
    #[tokio::test]
    filtered_with_text: SCRIPTED => "openai_faults_responses_filtered_with_text";
    #[tokio::test]
    filtered_empty: SCRIPTED => "openai_faults_responses_filtered_empty";
    #[tokio::test]
    failing_load: SCRIPTED => "openai_faults_responses_failing_load";
    #[tokio::test]
    failing_load_streamed: SCRIPTED => "openai_faults_responses_failing_load_streamed";
    #[tokio::test]
    cancel_at_first_tool_call_delta: SCRIPTED => "openai_faults_responses_cancel_at_first_tool_call_delta";
    #[tokio::test]
    status_429: SCRIPTED => "openai_faults_responses_status_429";
    #[tokio::test]
    status_503: SCRIPTED => "openai_faults_responses_status_503";
    #[tokio::test]
    status_503_retried: SCRIPTED => "openai_faults_responses_status_503_retried";
}

/// The refusal streams as the answer: the recorded text turn rewritten to
/// `refusal` parts and deltas, whole.
#[tokio::test]
async fn refusal() {
    crate::goldens::capture_world_programs(async {
        let recorded = SCRIPTED.recorded(SCRIPTED.text_stream);
        let frames = SCRIPTED.shape.refusal(&recorded);
        let log = run_scripted(
            &faults::REFUSAL,
            || SCRIPTED.stream(&frames),
            |log| crate::goldens::world_golden_effects("openai_faults_responses_refusal", log),
        )
        .await;
        assert_eq!(
            crate::ecs_matrix::corpus::golden_answer(&log),
            SCRIPTED.shape.delta_text(&recorded)
        );
    })
    .await
}
