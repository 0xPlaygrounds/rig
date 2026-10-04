//! The failure rows (`tests/common/ecs_matrix/faults.rs`) once. A recorded
//! row runs on one wire, the producer and the world over the bank, their
//! logs compared. A scripted row cuts or rewrites a streamed reply in its
//! wire's stream shape, so it runs once per shape (Chat Completions,
//! Responses, Gemini) over the bank's replies of the shapes the wire's
//! scripted rows cut (`faults::Frames::Bank`), with every assertion the
//! drivers make.

use rig::http_client::DynHttpClient;
use rig_test_support::bank;

use super::cells::agree;
use crate::ecs_matrix::{
    Wire,
    cells::{self, Cell},
    faults::{self, Fault, Frames},
    world::run_world,
};
use crate::goldens::capture_world_programs;
use crate::stream_faults::SseShape;

/// Gemini's refusal of a model it does not serve, as its recordings carry it.
const SETUP_UNARY: Cell = Cell {
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("NOT_FOUND"),
    }),
    ..faults::SETUP_UNARY
};
const SETUP_STREAMED: Cell = Cell {
    program: crate::ecs_matrix::corpus::Program {
        streamed: true,
        ..SETUP_UNARY.program
    },
    name: faults::SETUP_STREAMED.name,
    ..SETUP_UNARY
};

/// A world-only row: the world over the bank, every assertion its driver
/// makes.
async fn world<W, T>(
    wire: fn(DynHttpClient) -> Wire<rig::driver::Model<W, T>>,
    replies: Vec<bank::Entry>,
    cell: &Cell,
) where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let wire = wire(bank::client(&replies));
    capture_world_programs(run_world(&wire, cell, |_| {})).await;
}

macro_rules! rows {
    ($($name:ident: $run:ident $($pinned:ident)? ($wire:ident, $provider:literal, $scenario:literal, $cell:expr);)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                let replies = rows!(@replies $($pinned)? $provider, $scenario);
                $run(crate::wires::$wire, replies, &$cell).await;
            }
        )*
    };
    (@replies recorded $provider:literal, $scenario:literal) => {
        bank::recorded($provider, $scenario)
    };
    (@replies $provider:literal, $scenario:literal) => {
        bank::script($provider, $scenario)
    };
}

rows! {
    setup_unary: agree (gemini_preview, "gemini", "corpus_faults/setup_unary", SETUP_UNARY);
    setup_streamed: agree (gemini_preview, "gemini", "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body", SETUP_STREAMED);
    tool_error: agree (doubleword, "doubleword", "corpus_faults/tool_error", faults::TOOL_ERROR);
    tool_error_streamed: agree (venice, "venice", "corpus_faults/tool_error_streamed", faults::TOOL_ERROR_STREAMED);
    batch_second_fails: agree (gemini, "gemini", "corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS);
    batch_second_fails_concurrent: agree (openai_chat, "openai", "corpus_faults_chat/batch_second_fails", faults::BATCH_SECOND_FAILS_CONCURRENT);
    stop_while_tool_runs: world recorded (deepseek, "deepseek", "corpus_matrix/endings_tool_outcome_cancelled", faults::STOP_WHILE_TOOL_RUNS);
    scene_tool_in_flight: world (openai_responses, "openai", "corpus_matrix_responses/resume_tool_turn", faults::SCENE_TOOL_IN_FLIGHT);
}

#[tokio::test]
async fn cancel_after_terminal() {
    let replies = bank::script("venice", "corpus_matrix/endings_text_delta_stop");
    let wire = crate::wires::venice(bank::client(&replies));
    capture_world_programs(faults::cancel_after_terminal(
        &wire,
        &cells::ENDINGS_TEXT_DELTA_STOP,
        |_| {},
    ))
    .await;
}

const CHAT: faults::Scripted<rig::Model<rig::providers::openai::wire::OpenAiWire>> =
    faults::Scripted {
        provider: "deepseek",
        shape: SseShape::Chat,
        text_stream: "corpus_matrix/shaping_extra_context_streamed",
        tool_stream: "corpus_matrix/hooks_patch_tool_args_streamed",
        setup_reply: "corpus_faults/setup_unary",
        code: Some("invalid_request_error"),
        wire: crate::wires::deepseek_scripted,
        frames: Frames::Bank,
    };

const RESPONSES: faults::Scripted<rig::Model<rig::providers::openai::wire::OpenAiWire>> =
    faults::Scripted {
        provider: "openai",
        shape: SseShape::Responses,
        text_stream: "corpus_matrix_responses/shaping_extra_context_streamed",
        tool_stream: "corpus_matrix_responses/hooks_patch_tool_args_streamed",
        setup_reply: "corpus_faults_responses/setup_unary",
        code: Some("model_not_found"),
        wire: crate::wires::openai_responses_image,
        frames: Frames::Bank,
    };

const GEMINI: faults::Scripted<
    rig::Model<rig::providers::gemini::completion::GenerateContent, DynHttpClient>,
> = faults::Scripted {
    provider: "gemini",
    shape: SseShape::Gemini,
    text_stream: "corpus_matrix/shaping_extra_context_streamed",
    tool_stream: "corpus_matrix/hooks_patch_tool_args_streamed",
    setup_reply: "corpus_faults/setup_unary",
    code: Some("NOT_FOUND"),
    wire: crate::wires::gemini_preview,
    frames: Frames::Bank,
};

macro_rules! scripted {
    ($suite:ident: $module:ident; $($row:ident),* $(,)?) => {
        mod $module {
            $(
                #[tokio::test]
                async fn $row() {
                    super::$suite.$row(|_| {}).await;
                }
            )*
        }
    };
}

scripted!(CHAT: chat; truncated_after_text, truncated_after_tool_call, error_after_text, filtered_with_text, filtered_empty, failing_load, failing_load_streamed, cancel_at_first_tool_call_delta, status_429, status_503, status_503_retried);
scripted!(RESPONSES: responses; truncated_after_text, truncated_after_tool_call, error_after_text, filtered_with_text, filtered_empty, failing_load, failing_load_streamed, cancel_at_first_tool_call_delta, status_429, status_503, status_503_retried);
scripted!(GEMINI: gemini; truncated_after_text, truncated_after_tool_call, error_after_text, filtered_with_text, filtered_empty, failing_load, failing_load_streamed, cancel_at_first_tool_call_delta, status_429, status_503, status_503_retried);
