//! Classic-runtime streaming interfaces.

pub use rig_core::streaming::{
    PauseControl, RawStreamingChoice, RawStreamingToolCall, StreamedAssistantContent,
    StreamedUserContent, StreamingCompletionResponse, StreamingResult, ToolCallDeltaContent,
};

use crate::agent::StreamingPromptRequest;
use rig_core::{
    completion::{CompletionModel, GetTokenUsage, Message},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// High-level classic streaming prompt interface.
pub trait StreamingPrompt<M, R>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Start a streaming prompt request.
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M>;
}

/// High-level classic streaming chat interface.
pub trait StreamingChat<M, R>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel + 'static,
    M::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Start a streaming chat request with existing history.
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest<M>
    where
        I: IntoIterator<Item = T> + WasmCompatSend,
        T: Into<Message>;
}
