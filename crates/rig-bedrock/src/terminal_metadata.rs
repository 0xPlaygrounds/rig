use rig_core::completion::{CompletionFinishReason, CompletionTerminalMetadata};

pub(crate) fn from_stop_reason(raw_reason: &str) -> CompletionTerminalMetadata {
    let reason = match raw_reason {
        "content_filtered" | "guardrail_intervened" => CompletionFinishReason::ContentFilter,
        "end_turn" | "stop_sequence" => CompletionFinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => CompletionFinishReason::Length,
        "tool_use" => CompletionFinishReason::ToolCalls,
        _ => CompletionFinishReason::Unknown,
    };

    CompletionTerminalMetadata::new(reason).with_raw_reason(raw_reason)
}
