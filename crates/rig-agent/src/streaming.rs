//! High-level streaming prompting traits for the classic agent runtime.

use crate::{
    agent::StreamingPromptRequest,
    completion::{CompletionModel, GetTokenUsage, Message},
};
use rig_core::wasm_compat::{MaybeSend, MaybeSync};

pub use rig_core::streaming::*;

/// High-level one-shot streaming prompt interface.
pub trait StreamingPrompt<M, R>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: MaybeSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Create a classic streaming request for `prompt`.
    fn stream_prompt(&self, prompt: impl Into<Message> + MaybeSend) -> StreamingPromptRequest<M>;
}

/// High-level streaming chat interface with caller-provided history.
pub trait StreamingChat<M, R>: MaybeSend + MaybeSync
where
    M: CompletionModel + 'static,
    M::StreamingResponse: MaybeSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// Create a classic streaming request with canonical chat history.
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + MaybeSend,
        chat_history: I,
    ) -> StreamingPromptRequest<M>
    where
        I: IntoIterator<Item = T> + MaybeSend,
        T: Into<Message>;
}
