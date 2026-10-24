//! Estimating how many tokens a message, and the next request, cost.
//!
//! The next request is estimated from the last usage a provider reported
//! while it still describes the request ([`next_request_tokens`]), and
//! from the request as a whole otherwise.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::io;
use std::sync::Arc;

use rig_core::completion::{Message, ToolDefinition, Usage};
use rig_core::conversation::{Conversation, MessageId};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};

/// Counts the tokens of one [`Message`]: the provider's tokenizer (such as
/// `tiktoken-rs` for OpenAI) or an estimate. It runs once per message a
/// window or a compaction weighs, so it must be cheap.
///
/// A counter chosen at runtime is shared as `Arc<dyn TokenCounter>`, owned as
/// `Box<dyn TokenCounter>` or borrowed as `&dyn TokenCounter`; each counts as
/// the counter it holds. (A generic `Box<T>` or `&T` would overlap the
/// closure impl, since a boxed or borrowed closure is a closure too.)
///
/// ```
/// use std::sync::Arc;
/// use rig_core::completion::Message;
/// use rig_memory::{HeuristicTokenCounter, TokenCounter, recent::TokenBudget};
///
/// let counter: Arc<dyn TokenCounter> = Arc::new(HeuristicTokenCounter::default());
/// let budget = TokenBudget { max_tokens: 1_000, counter: Arc::clone(&counter) };
/// assert_eq!(budget.counter.count(&Message::user("12345678")), 2 + 4);
/// ```
pub trait TokenCounter: WasmCompatSend + WasmCompatSync {
    /// About how many tokens `message` costs.
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

impl<C: TokenCounter + ?Sized> TokenCounter for Arc<C> {
    fn count(&self, message: &Message) -> usize {
        (**self).count(message)
    }
}

impl TokenCounter for Box<dyn TokenCounter> {
    fn count(&self, message: &Message) -> usize {
        (**self).count(message)
    }
}

impl TokenCounter for &dyn TokenCounter {
    fn count(&self, message: &Message) -> usize {
        (**self).count(message)
    }
}

/// Estimates token cost from UTF-8 lengths, rounded up per text-bearing part.
/// Tool calls count name bytes plus serialized argument bytes. Attachments and
/// message overhead have fixed costs.
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
    /// A counter with these parameters.
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

    /// Uses 3.5 bytes per token, four tokens per message, and
    /// [`ATTACHMENT_TOKENS`] per attachment.
    pub fn anthropic() -> Self {
        Self::new(3.5, 4, ATTACHMENT_TOKENS)
    }

    fn bytes_to_tokens(&self, bytes: usize) -> usize {
        // `bytes_per_token` is clamped to >= 1.0 in the constructor, so the
        // division is well-defined. We round up so a single non-empty
        // input still costs at least one token.
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

/// The bytes of `value`'s JSON text, counted as it is written; 0 when it
/// does not serialize.
fn json_len(value: &impl serde::Serialize) -> usize {
    struct Len(usize);
    impl io::Write for Len {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut len = Len(0);
    serde_json::to_writer(&mut len, value).map_or(0, |()| len.0)
}

/// About how many tokens an image, or another attachment, costs.
pub const ATTACHMENT_TOKENS: usize = 1_200;

impl Default for HeuristicTokenCounter {
    /// Uses four bytes per token, four tokens per message, and
    /// [`ATTACHMENT_TOKENS`] per attachment.
    fn default() -> Self {
        Self::new(4.0, 4, ATTACHMENT_TOKENS)
    }
}

impl TokenCounter for HeuristicTokenCounter {
    fn count(&self, message: &Message) -> usize {
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
/// request ([`next_request_tokens`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// What the provider reported. Its input counts every token the
    /// request sent, the tool definitions included.
    pub usage: Usage,
    /// The last entry it counts.
    pub through: MessageId,
}

/// A request as prepared to be sent, as [`next_request_tokens`] reads it.
#[derive(Clone, Copy, Debug)]
pub struct Prepared<'a> {
    /// The model it goes to.
    pub model: &'a str,
    /// The system prompt sent before the messages, if any.
    pub prompt: Option<&'a str>,
    /// The tools it offers.
    pub tools: &'a [Arc<ToolDefinition>],
    /// Its messages, with their ids.
    pub history: &'a Conversation,
}

/// What a sent request started with, to tell whether the usage reported
/// for its reply still describes the next request: a fingerprint of its
/// model, system prompt, tools and messages (each by id, the time it was
/// made and the shared message it holds, so an edit, a splice or a restore
/// changes it), and how many messages it had. It holds in the process that
/// made it only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestStart {
    fingerprint: u64,
    messages: usize,
}

impl Prepared<'_> {
    /// What this request starts with, to keep once it is sent.
    pub fn start(&self) -> RequestStart {
        let messages = self.history.messages().len();
        RequestStart {
            fingerprint: self.fingerprint(messages),
            messages,
        }
    }

    /// The fingerprint of the model, prompt, tools and first `messages`
    /// messages.
    fn fingerprint(&self, messages: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        (self.model, self.prompt).hash(&mut hasher);
        for tool in self.tools {
            let mut bytes = HashWriter(&mut hasher);
            let _ = serde_json::to_writer(&mut bytes, &**tool);
        }
        for entry in self.history.entries().take(messages) {
            (entry.key(), std::ptr::from_ref(entry.message).addr()).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The estimate of the whole request: its system prompt, tools and
    /// messages, by the default [`HeuristicTokenCounter`].
    pub fn whole_tokens(&self) -> u64 {
        let counter = HeuristicTokenCounter::default();
        let prompt = self
            .prompt
            .map_or(0, |prompt| counter.count(&Message::system(prompt)));
        let tools = self.tools.iter().map(|tool| counter.count_tool(tool));
        let tools = tools.fold(0, usize::saturating_add);
        let messages = estimate_messages(self.history.messages());
        (prompt.saturating_add(tools) as u64).saturating_add(messages)
    }
}

/// Feeds bytes written to it to a hasher.
struct HashWriter<'h>(&'h mut DefaultHasher);

impl io::Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.write(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The estimated tokens of `messages`, by the default
/// [`HeuristicTokenCounter`].
fn estimate_messages(messages: &[Arc<Message>]) -> u64 {
    let counter = HeuristicTokenCounter::default();
    messages
        .iter()
        .map(|message| counter.count(message) as u64)
        .fold(0, u64::saturating_add)
}

/// About how many tokens `request` sends. Anchored, it is the usage the
/// provider reported for the reply to the request that started as `sent`
/// says, plus the estimate of the messages after the anchor's entry: the
/// anchor holds while that entry is the reply to that request (or its last
/// message) and `request` starts as that request did, with the same
/// model, system prompt, tools and messages. Otherwise, such as after a
/// compaction, a model change, a prompt or tool change, or a cleared copy,
/// it is [`Prepared::whole_tokens`]. The tool definitions are counted
/// once either way: reported input counts them already.
pub fn next_request_tokens(
    anchor: Option<&Anchor>,
    sent: Option<RequestStart>,
    request: &Prepared<'_>,
) -> u64 {
    let anchored = anchor.zip(sent).and_then(|(anchor, sent)| {
        let at = request.history.index_of(anchor.through)?;
        let follows = at == sent.messages || at.saturating_add(1) == sent.messages;
        let same = follows && request.fingerprint(sent.messages) == sent.fingerprint;
        let reported = anchor.usage.context_tokens().filter(|_| same)?;
        let after = request.history.messages().get(at.saturating_add(1)..);
        Some(reported.saturating_add(estimate_messages(after.unwrap_or_default())))
    });
    anchored.unwrap_or_else(|| request.whole_tokens())
}

/// Default answer budget for a turn, before adding a thinking budget.
pub const TURN_OUTPUT: u64 = 64_000;

/// Tokens a request leaves free below the window, besides the output it
/// asks for: a margin for the estimate's error.
pub const OUTPUT_MARGIN: u64 = 4_096;

#[cfg(test)]
mod tests;
