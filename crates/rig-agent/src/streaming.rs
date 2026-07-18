//! Classic high-level streaming traits and portable stream values.

pub use rig_core::streaming::*;

use rig_core::{
    completion::{CompletionModel, GetTokenUsage, Message},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use crate::agent::prompt_request::streaming::StreamingPromptRequest;

/// High-level streaming one-shot prompt interface.
pub trait StreamingPrompt<M, R>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Create a streaming request for `prompt`.
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M>;
}

/// High-level streaming chat interface with caller-managed history.
pub trait StreamingChat<M, R>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel + 'static,
    M::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Create a streaming request with `chat_history`.
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest<M>
    where
        I: IntoIterator<Item = T> + WasmCompatSend,
        T: Into<Message>;
}
