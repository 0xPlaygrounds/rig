//! Estimating how many tokens a message, and a request built to send
//! ([`request_estimate`]), cost.

use rig_core::completion::{CompletionRequest, Message, ToolDefinition, Usage};
use rig_core::conversation::MessageId;
use rig_core::json_utils::json_len;
use serde::{Deserialize, Serialize};

/// Estimates token cost from UTF-8 lengths, rounded up per text-bearing part.
/// Tool calls count name bytes plus serialized argument bytes. Attachments and
/// message overhead have fixed costs.
///
/// ```
/// use rig_core::completion::Message;
/// use rig_memory::HeuristicTokenCounter;
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
    fn bytes_to_tokens(&self, bytes: usize) -> usize {
        // Rounded up, so a non-empty input costs at least one token.
        let tokens = (bytes as f32) / self.bytes_per_token;
        tokens.ceil() as usize
    }

    /// About how many tokens the definition of a tool costs as a request
    /// sends it: its name, description and parameters' JSON.
    pub fn count_tool(&self, tool: &ToolDefinition) -> usize {
        let bytes = tool.name.len().saturating_add(tool.description.len());
        self.bytes_to_tokens(bytes.saturating_add(json_len(&tool.parameters)))
            .saturating_add(self.per_message_overhead)
    }

    /// About how many tokens one item of a user message costs, without the
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
                        self.bytes_to_tokens(json_len(value))
                    }
                    rig_core::message::ToolResultContent::Image(_) => self.per_attachment_tokens,
                })
                .fold(0, usize::saturating_add),
            UserContent::Image(_)
            | UserContent::Audio(_)
            | UserContent::Video(_)
            | UserContent::Document(_) => self.per_attachment_tokens,
        }
    }

    /// About how many tokens one item of an assistant message costs, without
    /// the per-message overhead.
    fn count_assistant(&self, content: &rig_core::message::AssistantContent) -> usize {
        use rig_core::message::AssistantContent;
        match content {
            AssistantContent::Text(text) => self.bytes_to_tokens(text.text.len()),
            AssistantContent::Reasoning(reasoning) => self.bytes_to_tokens(reasoning.text.len()),
            AssistantContent::ToolCall(call) => {
                let name_bytes = call.function.name.len();
                let args_bytes = json_len(&call.function.arguments);
                self.bytes_to_tokens(name_bytes + args_bytes)
            }
            AssistantContent::Image(_) => self.per_attachment_tokens,
            // A provider-only item is sent back to the model that made it.
            AssistantContent::Opaque(opaque) => self.bytes_to_tokens(json_len(&opaque.item)),
            other => self.bytes_to_tokens(json_len(other)),
        }
    }
}

/// About how many tokens an image, or another attachment, costs.
pub const ATTACHMENT_TOKENS: usize = 1_200;

impl Default for HeuristicTokenCounter {
    /// Uses four bytes per token, four tokens per message, and
    /// [`ATTACHMENT_TOKENS`] per attachment.
    fn default() -> Self {
        Self {
            bytes_per_token: 4.0,
            per_message_overhead: 4,
            per_attachment_tokens: ATTACHMENT_TOKENS,
        }
    }
}

impl HeuristicTokenCounter {
    /// About how many tokens `message` costs.
    pub fn count(&self, message: &Message) -> usize {
        let content_tokens: usize = match message {
            Message::User { content } => content
                .iter()
                .map(|c| self.count_user(c))
                .fold(0, usize::saturating_add),
            Message::Assistant(turn) => turn
                .content
                .iter()
                .map(|c| self.count_assistant(c))
                .fold(0, usize::saturating_add),
            Message::System { content } => self.bytes_to_tokens(content.len()),
        };
        content_tokens.saturating_add(self.per_message_overhead)
    }
}

/// The usage a provider reported for an agent's last reply, and the last
/// entry of the conversation it counts: the reply, or the request's last
/// message when the reply added none. It anchors the estimate of the next
/// request: the reported tokens plus an estimate of the messages after it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// What the provider reported. Its input counts every token the
    /// request sent, the tool definitions included.
    pub usage: Usage,
    /// The last entry it counts.
    pub through: MessageId,
}

/// The estimate of a request sent as it is: its messages and tools, by the
/// default [`HeuristicTokenCounter`]. Its documents are not counted.
///
/// ```
/// use rig_core::completion::{CompletionRequest, Message};
/// use rig_memory::request_estimate;
///
/// let request = CompletionRequest::new(Message::user("12345678"));
/// assert_eq!(request_estimate(&request), 2 + 4);
/// ```
pub fn request_estimate(request: &CompletionRequest) -> u64 {
    let counter = HeuristicTokenCounter::default();
    let tools = request.tools.iter().map(|tool| counter.count_tool(tool));
    let messages = request
        .chat_history
        .iter()
        .map(|message| counter.count(message));
    tools.chain(messages).fold(0, usize::saturating_add) as u64
}

#[cfg(test)]
mod tests;
