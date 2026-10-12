//! Estimating how many tokens a message costs.

use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Counts the tokens contributed by a single [`Message`].
///
/// Implementors should pick a counting strategy appropriate for their target
/// provider (for example, `tiktoken-rs` for OpenAI). Counting must be cheap;
/// it runs once per message a window or a compaction weighs.
pub trait TokenCounter: WasmCompatSend + WasmCompatSync {
    /// Approximate the number of tokens contributed by `message`.
    fn count(&self, message: &Message) -> usize;
}

impl<F> TokenCounter for F
where
    F: Fn(&Message) -> usize + WasmCompatSend + WasmCompatSync,
{
    fn count(&self, message: &Message) -> usize {
        (self)(message)
    }
}

impl<C> TokenCounter for Arc<C>
where
    C: TokenCounter + ?Sized,
{
    fn count(&self, message: &Message) -> usize {
        (**self).count(message)
    }
}

impl TokenCounter for Box<dyn TokenCounter> {
    fn count(&self, message: &Message) -> usize {
        (**self).count(message)
    }
}

/// Estimates token cost from UTF-8 lengths, rounded up per text-bearing part.
/// Tool calls count name bytes plus serialized argument bytes. Attachments and
/// message overhead have fixed costs. These estimates do not bound provider usage.
///
/// ```
/// use rig_core::completion::Message;
/// use rig_memory::{HeuristicTokenCounter, TokenCounter};
///
/// let counter = HeuristicTokenCounter::default();
/// assert_eq!(counter.count(&Message::user("12345678")), 2 + 4);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HeuristicTokenCounter {
    bytes_per_token: f32,
    per_message_overhead: usize,
    per_attachment_tokens: usize,
}

impl HeuristicTokenCounter {
    /// Create a counter with explicit parameters.
    ///
    /// Non-finite ratios and ratios below `1.0` become `1.0`.
    /// Overhead and attachment costs may be zero.
    pub fn new(
        bytes_per_token: f32,
        per_message_overhead: usize,
        per_attachment_tokens: usize,
    ) -> Self {
        let bytes_per_token = if bytes_per_token.is_finite() && bytes_per_token >= 1.0 {
            bytes_per_token
        } else {
            1.0
        };
        Self {
            bytes_per_token,
            per_message_overhead,
            per_attachment_tokens,
        }
    }

    /// Uses 3.5 bytes per token, four tokens per message, and 256 per attachment.
    pub fn anthropic() -> Self {
        Self::new(3.5, 4, 256)
    }

    fn bytes_to_tokens(&self, bytes: usize) -> usize {
        // `bytes_per_token` is clamped to >= 1.0 in the constructor, so the
        // division is well-defined. We round up so a single non-empty
        // input still costs at least one token.
        let tokens = (bytes as f32) / self.bytes_per_token;
        tokens.ceil() as usize
    }

    /// Approximate the tokens of one item of a user message, without the
    /// per-message overhead.
    pub(crate) fn count_user(&self, content: &rig_core::message::UserContent) -> usize {
        use rig_core::message::UserContent;
        match content {
            UserContent::Text(text) => self.bytes_to_tokens(text.text.len()),
            UserContent::ToolResult(result) => result
                .content
                .iter()
                .map(|c| match c {
                    rig_core::message::ToolResultContent::Text(t) => {
                        self.bytes_to_tokens(t.text.len())
                    }
                    rig_core::message::ToolResultContent::Json { value } => {
                        self.bytes_to_tokens(value.to_string().len())
                    }
                    rig_core::message::ToolResultContent::Image(_) => self.per_attachment_tokens,
                })
                .sum(),
            UserContent::Image(_)
            | UserContent::Audio(_)
            | UserContent::Video(_)
            | UserContent::Document(_) => self.per_attachment_tokens,
        }
    }

    /// Approximate the tokens of one item of an assistant message, without
    /// the per-message overhead.
    fn count_assistant(&self, content: &rig_core::message::AssistantContent) -> usize {
        use rig_core::message::AssistantContent;
        match content {
            AssistantContent::Text(text) => self.bytes_to_tokens(text.text.len()),
            AssistantContent::Reasoning(reasoning) => self.bytes_to_tokens(reasoning.text.len()),
            AssistantContent::ToolCall(call) => {
                let name_bytes = call.function.name.len();
                let args_bytes = call.function.arguments_value().to_string().len();
                self.bytes_to_tokens(name_bytes + args_bytes)
            }
            AssistantContent::Image(_) => self.per_attachment_tokens,
            // A provider-only item is sent back to the model that made it.
            AssistantContent::Opaque(opaque) => self.bytes_to_tokens(opaque.item.to_string().len()),
            other => self.bytes_to_tokens(
                rig_core::serde_json::to_string(other).map_or(0, |json| json.len()),
            ),
        }
    }
}

impl Default for HeuristicTokenCounter {
    /// Uses four bytes per token, four tokens per message, and 256 per attachment.
    fn default() -> Self {
        Self::new(4.0, 4, 256)
    }
}

impl TokenCounter for HeuristicTokenCounter {
    fn count(&self, message: &Message) -> usize {
        let content_tokens: usize = match message {
            Message::User { content } => content.iter().map(|c| self.count_user(c)).sum(),
            Message::Assistant(turn) => turn.content.iter().map(|c| self.count_assistant(c)).sum(),
            Message::System { content } => self.bytes_to_tokens(content.len()),
        };
        content_tokens.saturating_add(self.per_message_overhead)
    }
}
