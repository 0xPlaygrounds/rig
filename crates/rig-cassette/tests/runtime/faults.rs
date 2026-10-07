//! The failure rows (`tests/common/corpus_matrix/faults.rs`) once. A recorded
//! row runs the producer on one wire over the bank. A scripted row cuts or
//! rewrites a streamed reply in its wire's stream shape, so it runs once per
//! shape (Chat Completions, Responses, Gemini) over the bank's replies of
//! the shapes the wire's scripted rows cut, with every assertion the driver
//! makes.

use rig::http_client::DynHttpClient;
use rig_test_support::bank;

use super::cells::produce;
use crate::corpus_matrix::faults;
use crate::stream_faults::SseShape;

macro_rules! rows {
    ($($name:ident: $($pinned:ident)? ($wire:ident, $provider:literal, $scenario:literal, $cell:expr);)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                let replies = rows!(@replies $($pinned)? $provider, $scenario);
                produce(crate::wires::$wire, replies, &$cell).await;
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
    setup_unary: (gemini_preview, "gemini", "corpus_faults/setup_unary", faults::SETUP_UNARY);
    setup_streamed: (gemini_preview, "gemini", "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body", faults::SETUP_STREAMED);
    tool_error: (doubleword, "doubleword", "corpus_faults/tool_error", faults::TOOL_ERROR);
    tool_error_streamed: (venice, "venice", "corpus_faults/tool_error_streamed", faults::TOOL_ERROR_STREAMED);
    batch_second_fails: (gemini, "gemini", "corpus_faults/batch_second_fails", faults::BATCH_SECOND_FAILS);
    batch_second_fails_concurrent: (openai_chat, "openai", "corpus_faults_chat/batch_second_fails", faults::BATCH_SECOND_FAILS_CONCURRENT);
}

const CHAT: faults::Scripted<rig::Model<rig::providers::openai::wire::OpenAiWire>> =
    faults::Scripted {
        provider: "deepseek",
        shape: SseShape::Chat,
        text_stream: "corpus_matrix/shaping_extra_context_streamed",
        tool_stream: "corpus_matrix/hooks_patch_tool_args_streamed",
        setup_reply: "corpus_faults/setup_unary",
        wire: crate::wires::deepseek_scripted,
    };

const RESPONSES: faults::Scripted<rig::Model<rig::providers::openai::wire::OpenAiWire>> =
    faults::Scripted {
        provider: "openai",
        shape: SseShape::Responses,
        text_stream: "corpus_matrix_responses/shaping_extra_context_streamed",
        tool_stream: "corpus_matrix_responses/hooks_patch_tool_args_streamed",
        setup_reply: "corpus_faults_responses/setup_unary",
        wire: crate::wires::openai_responses_image,
    };

const GEMINI: faults::Scripted<
    rig::Model<rig::providers::gemini::completion::GenerateContent, DynHttpClient>,
> = faults::Scripted {
    provider: "gemini",
    shape: SseShape::Gemini,
    text_stream: "corpus_matrix/shaping_extra_context_streamed",
    tool_stream: "corpus_matrix/hooks_patch_tool_args_streamed",
    setup_reply: "corpus_faults/setup_unary",
    wire: crate::wires::gemini_preview,
};

macro_rules! scripted {
    ($suite:ident: $module:ident; $($row:ident),* $(,)?) => {
        mod $module {
            $(
                #[tokio::test]
                async fn $row() {
                    super::$suite.$row().await;
                }
            )*
        }
    };
}

scripted!(CHAT: chat; truncated_after_text, truncated_after_tool_call, error_after_text, filtered_with_text, filtered_empty, failing_load, failing_load_streamed, status_429, status_503);
scripted!(RESPONSES: responses; truncated_after_text, truncated_after_tool_call, error_after_text, filtered_with_text, filtered_empty, failing_load, failing_load_streamed, status_429, status_503);
scripted!(GEMINI: gemini; truncated_after_text, truncated_after_tool_call, error_after_text, filtered_with_text, filtered_empty, failing_load, failing_load_streamed, status_429, status_503);
