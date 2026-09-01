//! High-level streaming prompting traits for the classic agent runtime.

use crate::{agent::StreamingPromptRequest, completion::Message};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

pub use rig_core::streaming::*;

/// High-level one-shot streaming prompt interface.
pub trait StreamingPrompt {
    /// Create a classic streaming request for `prompt`.
    fn stream_prompt(&self, prompt: impl Into<Message> + WasmCompatSend) -> StreamingPromptRequest;
}

/// High-level streaming chat interface with caller-provided history.
pub trait StreamingChat: WasmCompatSend + WasmCompatSync {
    /// Create a classic streaming request with canonical chat history.
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest
    where
        I: IntoIterator<Item = T> + WasmCompatSend,
        T: Into<Message>;
}
