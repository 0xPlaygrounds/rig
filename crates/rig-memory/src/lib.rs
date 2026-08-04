#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Conversation memory policies for the Rig agent framework.
//!
//! `rig-core` provides the concrete in-process
//! [`InMemoryConversationMemory`] store. This crate adds reusable,
//! *data-shaped* history policies plus one concrete adapter,
//! [`PolicyMemory`], that reports what a policy did as **owned data** the
//! host acts on. There are no behavior traits here: policies, token
//! counters, and compactors are exhaustive enums.
//!
//! - [`MemoryPolicy`] — `Noop`, `SlidingWindow`, `TokenWindow`. Its
//!   [`apply`](MemoryPolicy::apply) returns a [`PolicyOutcome`]
//!   (`kept` + `demoted`).
//! - [`TokenCounter`] — `Heuristic` ([`HeuristicTokenCounter`], a
//!   provider-agnostic byte-length approximation) or `Fixed` (a constant
//!   per-message cost, handy for tests and coarse budgets).
//! - [`Compactor`] — `Template` ([`TemplateCompactor`], a zero-dependency
//!   textual rollup producing a [`TextSummary`]).
//! - [`PolicyMemory`] — an [`InMemoryConversationMemory`] plus a
//!   [`MemoryPolicy`] and an optional [`Compactor`].
//!
//! All windowing policies drop a leading orphan tool-result message when the
//! preceding assistant tool call has been truncated (most providers reject
//! unpaired tool results); the orphan is reported as *demoted* so a host that
//! archives demotions never loses it.
//!
//! # Host recipe
//!
//! Memory is host-owned: nothing in `rig-agent` loads or saves history for
//! you. Wrap a run like this (the classic driver's semantics: a load failure
//! is fatal, an append failure warns and proceeds, explicit caller history
//! bypasses memory entirely):
//!
//! ```
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_memory::{MemoryPolicy, PolicyMemory, InMemoryConversationMemory};
//! use rig_core::completion::Message;
//!
//! let memory = PolicyMemory::new(
//!     InMemoryConversationMemory::new(),
//!     MemoryPolicy::sliding_window(20),
//! );
//!
//! // Before the run: load the policy-shaped history and pass it as history.
//! let history = memory.load("user-42")?;
//! # let _ = history;
//!
//! // After a successful run: append the run's new messages. The outcome tells
//! // you what the policy pushed out of the active window.
//! let outcome = memory.append(
//!     "user-42",
//!     vec![Message::user("hi"), Message::assistant("hello")],
//! )?;
//! for demoted in &outcome.demoted {
//!     // Archive into a long-tail store (vector RAG, episodic recall, ...).
//!     # let _ = demoted;
//! }
//! if let Some(request) = &outcome.compaction {
//!     // A compactor is configured: roll the evicted prefix into a summary
//!     // that `load` will splice in front of the active window.
//!     memory.compact(request);
//! }
//! # Ok(()) }
//! # run().unwrap();
//! ```
//!
//! Because demotion and compaction are values rather than callbacks, there is
//! no in-process delivery watermark and no idempotency contract to honour:
//! the host observes each demotion exactly once, when it appends.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex as StdMutex},
};

/// Re-exports of the core memory store and error type so callers only need a
/// single dependency on `rig-memory` for both the store and the policies.
///
/// The store is *defined* in `rig_core::memory` (this crate depends on
/// `rig-core`, so defining it here would create a dependency cycle);
/// `rig-memory` is the canonical companion path for everything
/// memory-related.
pub use rig_core::memory::{InMemoryConversationMemory, MemoryBackendError, MemoryError};

#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;

use rig_core::completion::Message;
use rig_core::message::UserContent;

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

/// What a [`MemoryPolicy`] did to a history.
///
/// `demoted` is the prefix of the input that `kept` does not retain, in
/// original conversation order. A host that archives demotions can persist
/// `demoted` verbatim.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyOutcome {
    /// The history that should be sent to the model, in original order.
    pub kept: Vec<Message>,
    /// Messages the policy pushed out of the active window, in original order.
    pub demoted: Vec<Message>,
}

/// A history-shaping policy, as data.
///
/// Policies are pure and infallible: [`MemoryPolicy::apply`] is a total
/// function from a history to a [`PolicyOutcome`].
#[derive(Debug, Clone, Copy, Default)]
pub enum MemoryPolicy {
    /// Identity: keep everything, demote nothing.
    #[default]
    Noop,
    /// Keep the most recent `N` messages.
    SlidingWindow(SlidingWindowMemory),
    /// Keep the most recent messages that fit within a token budget.
    TokenWindow(TokenWindowMemory),
}

impl MemoryPolicy {
    /// Keep at most `max_messages` of the most recent messages.
    pub fn sliding_window(max_messages: usize) -> Self {
        Self::SlidingWindow(SlidingWindowMemory::last_messages(max_messages))
    }

    /// Keep the most recent messages that fit within `max_tokens`, as counted
    /// by `counter`.
    pub fn token_window(max_tokens: usize, counter: TokenCounter) -> Self {
        Self::TokenWindow(TokenWindowMemory::new(max_tokens, counter))
    }

    /// Shape `messages`, reporting both the kept window and what fell out.
    pub fn apply(&self, messages: Vec<Message>) -> PolicyOutcome {
        match self {
            Self::Noop => PolicyOutcome {
                kept: messages,
                demoted: Vec::new(),
            },
            Self::SlidingWindow(policy) => policy.apply(messages),
            Self::TokenWindow(policy) => policy.apply(messages),
        }
    }

    /// Shape `messages` and return only the kept window.
    pub fn shape(&self, messages: Vec<Message>) -> Vec<Message> {
        self.apply(messages).kept
    }
}

/// Retains only the most recent `max_messages` entries.
///
/// When the window starts mid-conversation, a leading orphan tool-result
/// message (a [`Message::User`] whose first content is a tool result without
/// its preceding [`Message::Assistant`] tool call) is demoted to preserve the
/// tool-call/result pairing required by most providers.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindowMemory {
    max_messages: usize,
}

impl SlidingWindowMemory {
    /// Keep at most `n` messages.
    pub fn last_messages(n: usize) -> Self {
        Self { max_messages: n }
    }

    /// The configured message budget.
    pub fn max_messages(&self) -> usize {
        self.max_messages
    }

    /// Shape `messages`, reporting the kept window and the demoted prefix.
    pub fn apply(&self, messages: Vec<Message>) -> PolicyOutcome {
        if messages.len() <= self.max_messages {
            return PolicyOutcome {
                kept: messages,
                demoted: Vec::new(),
            };
        }

        let start = messages.len() - self.max_messages;
        let mut iter = messages.into_iter();
        let demoted: Vec<Message> = (&mut iter).take(start).collect();
        let window: Vec<Message> = iter.collect();

        split_orphan_tool_result(window, demoted)
    }
}

/// Retains the most recent messages up to a token budget.
///
/// Messages are walked from newest to oldest, accumulating token counts
/// produced by a [`TokenCounter`]. Once including a message would exceed
/// `max_tokens`, the walk stops and the included messages are returned in
/// original (oldest-first) order. As with [`SlidingWindowMemory`], a leading
/// orphan tool-result is demoted when its paired assistant tool call has been
/// truncated.
#[derive(Debug, Clone, Copy)]
pub struct TokenWindowMemory {
    max_tokens: usize,
    counter: TokenCounter,
}

impl TokenWindowMemory {
    /// Create a new policy with a token budget and a counter.
    pub fn new(max_tokens: usize, counter: TokenCounter) -> Self {
        Self {
            max_tokens,
            counter,
        }
    }

    /// The configured token budget.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The configured counter.
    pub fn counter(&self) -> &TokenCounter {
        &self.counter
    }

    /// Shape `messages`, reporting the kept window and the demoted prefix.
    pub fn apply(&self, messages: Vec<Message>) -> PolicyOutcome {
        let mut budget = self.max_tokens;
        let mut keep_from = messages.len();

        for (idx, msg) in messages.iter().enumerate().rev() {
            let cost = self.counter.count(msg);
            if cost > budget {
                break;
            }
            budget -= cost;
            keep_from = idx;
        }

        let mut iter = messages.into_iter();
        let demoted: Vec<Message> = (&mut iter).take(keep_from).collect();
        let window: Vec<Message> = iter.collect();

        split_orphan_tool_result(window, demoted)
    }
}

/// Move a leading orphan tool-result out of `window` and onto `demoted`, so
/// the model never sees an unpaired tool result but a host archiving
/// demotions still receives it.
fn split_orphan_tool_result(mut window: Vec<Message>, mut demoted: Vec<Message>) -> PolicyOutcome {
    if demoted.is_empty() {
        return PolicyOutcome {
            kept: window,
            demoted,
        };
    }
    if let Some(Message::User { content }) = window.first()
        && matches!(content.first_ref(), UserContent::ToolResult(_))
    {
        demoted.push(window.remove(0));
    }
    PolicyOutcome {
        kept: window,
        demoted,
    }
}

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

/// Counts the tokens contributed by a single [`Message`], as data.
#[derive(Debug, Clone, Copy)]
pub enum TokenCounter {
    /// Approximate token counts from UTF-8 byte lengths.
    Heuristic(HeuristicTokenCounter),
    /// Charge every message the same constant cost. Useful for coarse
    /// message-count budgets and for deterministic tests.
    Fixed(usize),
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self::Heuristic(HeuristicTokenCounter::default())
    }
}

impl TokenCounter {
    /// The default [`HeuristicTokenCounter`] (OpenAI preset).
    pub fn heuristic() -> Self {
        Self::Heuristic(HeuristicTokenCounter::default())
    }

    /// Charge every message `cost` tokens.
    pub fn fixed(cost: usize) -> Self {
        Self::Fixed(cost)
    }

    /// Approximate the number of tokens contributed by `message`.
    pub fn count(&self, message: &Message) -> usize {
        match self {
            Self::Heuristic(counter) => counter.count(message),
            Self::Fixed(cost) => *cost,
        }
    }
}

impl From<HeuristicTokenCounter> for TokenCounter {
    fn from(value: HeuristicTokenCounter) -> Self {
        Self::Heuristic(value)
    }
}

/// A provider-agnostic token counter that approximates token counts from
/// UTF-8 byte lengths.
///
/// This is intended as a zero-dependency default. It is **not** a substitute
/// for a tokenizer and will under- or over-count by up to ~30 % on real
/// content, but it is monotonic in message size and stable across runs, which
/// is enough for [`TokenWindowMemory`] to enforce a budget that *trends*
/// with provider billing.
///
/// # Strategy
///
/// For every text-bearing block (`Text`, reasoning text, tool-result text)
/// the counter sums UTF-8 byte lengths (`str::len`, an O(1) call) and divides
/// by `bytes_per_token`, rounded up. Bytes are used instead of Unicode
/// scalars because the cost is O(1), modern BPE tokenizers operate on byte
/// sequences, and per-message budgeting only needs the rough order of
/// magnitude. For ASCII text bytes and characters coincide; for non-ASCII
/// text the counter slightly over-estimates, which is the safe direction
/// for a hard budget.
///
/// Tool calls are charged the JSON-serialised length of their `ToolFunction`
/// payload. Each message is charged a flat `per_message_overhead` to model
/// the per-turn role/separator tokens that providers add internally. Non-text
/// blocks (images, audio, video, documents) are charged
/// `per_attachment_tokens` each because their real cost is provider-specific
/// and rarely text-derived.
///
/// # Presets
///
/// The defaults match OpenAI's published rule of thumb (~4 bytes per token,
/// ~4 tokens of per-message overhead). [`HeuristicTokenCounter::anthropic`]
/// uses a slightly denser ratio that better fits Claude's tokenizer.
///
/// # Example
///
/// ```
/// use rig_memory::{HeuristicTokenCounter, MemoryPolicy, TokenCounter};
///
/// let policy = MemoryPolicy::token_window(
///     2_000,
///     TokenCounter::Heuristic(HeuristicTokenCounter::default()),
/// );
/// # let _ = policy;
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
    /// `bytes_per_token` is clamped to a minimum of `1.0` so the counter
    /// never panics or produces zero-cost messages on degenerate input.
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

    /// Preset matching OpenAI's chat-completion token rule of thumb.
    ///
    /// Equivalent to [`HeuristicTokenCounter::default`].
    pub fn openai() -> Self {
        Self::new(4.0, 4, 256)
    }

    /// Preset tuned for Anthropic Claude's tokenizer.
    pub fn anthropic() -> Self {
        Self::new(3.5, 4, 256)
    }

    /// Preset tuned for Google Gemini.
    pub fn gemini() -> Self {
        Self::new(4.0, 4, 256)
    }

    /// Approximate the number of tokens contributed by `message`.
    pub fn count(&self, message: &Message) -> usize {
        let content_tokens: usize = match message {
            Message::User { content } => content.iter().map(|c| self.count_user(c)).sum(),
            Message::Assistant { content, .. } => {
                content.iter().map(|c| self.count_assistant(c)).sum()
            }
            Message::System { content } => self.bytes_to_tokens(content.len()),
        };
        content_tokens.saturating_add(self.per_message_overhead)
    }

    fn bytes_to_tokens(&self, bytes: usize) -> usize {
        // `bytes_per_token` is clamped to >= 1.0 in the constructor, so the
        // division is well-defined. We round up so a single non-empty
        // input still costs at least one token.
        let tokens = (bytes as f32) / self.bytes_per_token;
        tokens.ceil() as usize
    }

    fn count_user(&self, content: &rig_core::message::UserContent) -> usize {
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

    fn count_assistant(&self, content: &rig_core::message::AssistantContent) -> usize {
        use rig_core::message::AssistantContent;
        match content {
            AssistantContent::Text(text) => self.bytes_to_tokens(text.text.len()),
            AssistantContent::Reasoning(reasoning) => {
                self.bytes_to_tokens(reasoning.display_text().len())
            }
            AssistantContent::ToolCall(call) => {
                let name_bytes = call.function.name.len();
                // `serde_json::Value::to_string` is the canonical compact JSON
                // encoding and never fails, so we charge tool calls by the
                // length of their serialised arguments without pulling in a
                // direct `serde_json` dependency.
                let args_bytes = call.function.arguments.to_string().len();
                self.bytes_to_tokens(name_bytes + args_bytes)
            }
            AssistantContent::Image(_) => self.per_attachment_tokens,
        }
    }
}

impl Default for HeuristicTokenCounter {
    fn default() -> Self {
        Self::openai()
    }
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

/// A request to fold an evicted prefix into a rolling summary.
///
/// Produced by [`PolicyMemory::append`] when a [`Compactor`] is configured
/// and the policy demoted messages. The host decides when to act on it —
/// [`PolicyMemory::compact`] is the built-in way to do so.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionRequest {
    /// The conversation whose prefix was evicted.
    pub conversation_id: String,
    /// The evicted messages, in original conversation order.
    pub evicted: Vec<Message>,
}

/// Derives a [`Message`]-shaped artifact from messages a policy evicted, as
/// data.
///
/// Where demotion is a one-way drain — observe what fell out — compaction is
/// the inverse: it takes the evicted prefix (and the previous summary) and
/// produces a derived artifact that [`PolicyMemory::load`] splices back in
/// front of the active window, so the loaded prompt is
/// `[summary, ...recent_window]`.
#[derive(Debug, Clone)]
pub enum Compactor {
    /// A zero-dependency textual rollup.
    Template(TemplateCompactor),
}

impl Default for Compactor {
    fn default() -> Self {
        Self::Template(TemplateCompactor::new())
    }
}

impl Compactor {
    /// A [`TemplateCompactor`] with the default header and no size cap.
    pub fn template() -> Self {
        Self::Template(TemplateCompactor::new())
    }

    /// Fold `request.evicted` into a summary, rolling `carry_over` (the
    /// previous summary for this conversation, if any) forward so context
    /// lost in earlier compactions is preserved transitively.
    pub fn compact(
        &self,
        request: &CompactionRequest,
        carry_over: Option<&TextSummary>,
    ) -> TextSummary {
        match self {
            Self::Template(compactor) => compactor.compact(&request.evicted, carry_over),
        }
    }
}

impl From<TemplateCompactor> for Compactor {
    fn from(value: TemplateCompactor) -> Self {
        Self::Template(value)
    }
}

/// A zero-dependency compactor that produces a textual rollup of evicted
/// messages without calling an LLM.
///
/// The artifact is a single [`TextSummary`] (convertible into a
/// [`Message::System`]) whose body concatenates a header, the previous
/// summary (if any), and the textual content of each newly-evicted message.
/// It is intentionally simple: useful as a default for tests and examples,
/// and as a placeholder before wiring a real summarising model.
///
/// # Bounding the summary
///
/// By default the summary grows monotonically: every compaction pass embeds
/// the previous summary verbatim and appends newly-evicted lines.
/// Long-running conversations should call [`Self::with_max_bytes`] to cap the
/// rolled-up text. When the cap is exceeded, the oldest portion of the body
/// (after the header) is dropped at a UTF-8 boundary and replaced with a
/// `"[…truncated…]"` marker, preserving the most recent context.
///
/// Compaction is spliced **outside** the policy's budget: the loaded prompt
/// is `[summary, ...kept_window]`. Callers combining compaction with a
/// token-budgeted policy should cap the summary with
/// [`Self::with_max_bytes`], or accept a prompt that exceeds the policy
/// budget by the size of the summary.
///
/// # Example
///
/// ```
/// use rig_memory::TemplateCompactor;
///
/// // Default header is "[Conversation summary so far]", unbounded.
/// let _compactor = TemplateCompactor::new();
///
/// // Custom header plus a 4 KiB cap for use with token-budgeted policies.
/// let _bounded = TemplateCompactor::with_header("Earlier context")
///     .with_max_bytes(4 * 1024);
/// ```
#[derive(Debug, Clone)]
pub struct TemplateCompactor {
    header: String,
    max_bytes: Option<usize>,
}

impl TemplateCompactor {
    /// Create a [`TemplateCompactor`] with the default header
    /// `"[Conversation summary so far]"` and no size cap.
    pub fn new() -> Self {
        Self::with_header("[Conversation summary so far]")
    }

    /// Create a [`TemplateCompactor`] with a custom header line and no
    /// size cap.
    pub fn with_header(header: impl Into<String>) -> Self {
        Self {
            header: header.into(),
            max_bytes: None,
        }
    }

    /// Cap the rolled-up summary at `max_bytes` bytes (UTF-8). When the
    /// assembled body exceeds the cap, the oldest portion after the
    /// header is dropped at a char boundary and replaced with a
    /// `"[…truncated…]"` marker.
    ///
    /// `max_bytes` of `0` disables truncation (equivalent to the default
    /// unbounded behaviour). The header line plus the marker are always
    /// preserved even if they exceed the cap.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = if max_bytes == 0 {
            None
        } else {
            Some(max_bytes)
        };
        self
    }

    /// Produce a summary for `evicted`, embedding `carry_over` when present.
    pub fn compact(&self, evicted: &[Message], carry_over: Option<&TextSummary>) -> TextSummary {
        let mut buf = String::new();
        buf.push_str(&self.header);
        buf.push('\n');
        if let Some(prev) = carry_over {
            buf.push_str(prev.as_str());
            buf.push('\n');
        }
        for msg in evicted {
            let line = render_message_line(msg);
            if !line.is_empty() {
                buf.push_str(&line);
                buf.push('\n');
            }
        }
        if let Some(cap) = self.max_bytes
            && buf.len() > cap
        {
            buf = truncate_summary(&buf, cap);
        }
        TextSummary(buf)
    }
}

impl Default for TemplateCompactor {
    fn default() -> Self {
        Self::new()
    }
}

/// Plain-text artifact produced by [`TemplateCompactor`].
///
/// Convertible into a [`Message::System`] whose body is the rolled-up
/// text. The system role is used because the rollup represents
/// out-of-band context about the prior conversation, not a turn from
/// any participant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSummary(String);

impl TextSummary {
    /// Wrap `text` as a summary artifact.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// Borrow the underlying summary text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the wrapper and return the underlying `String`.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<TextSummary> for Message {
    fn from(value: TextSummary) -> Self {
        Message::System { content: value.0 }
    }
}

/// Truncate `buf` to fit within `cap` bytes by dropping the oldest
/// content after the header line. Always preserves the header plus a
/// `"[\u{2026}truncated\u{2026}]"` marker, even if they alone exceed `cap`.
///
/// The header boundary is located by scanning `buf` for the first `\n`
/// rather than by trusting any caller-supplied header length, so a
/// header containing embedded newlines does not mis-locate the body.
fn truncate_summary(buf: &str, cap: usize) -> String {
    const MARKER: &str = "[\u{2026}truncated\u{2026}]\n";
    // Body starts right after the first newline in `buf`. If `buf` has
    // no newline at all there is no body to drop, so return as-is.
    let header_prefix_len = match buf.find('\n') {
        Some(i) => i + 1,
        None => return buf.to_string(),
    };
    if buf.len() <= header_prefix_len {
        return buf.to_string();
    }
    let preserved = header_prefix_len + MARKER.len();
    // Number of bytes of the body we can keep after the marker.
    let keep_bytes = cap.saturating_sub(preserved);
    let body_start = header_prefix_len;
    let body = match buf.get(body_start..) {
        Some(b) => b,
        None => return buf.to_string(),
    };
    // Take the suffix of `body` whose length is at most `keep_bytes`,
    // walking forward to a UTF-8 char boundary.
    let mut cut = body.len().saturating_sub(keep_bytes);
    while cut < body.len() && !body.is_char_boundary(cut) {
        cut += 1;
    }
    let suffix: &str = body.get(cut..).unwrap_or_default();
    let header_with_nl = match buf.get(..header_prefix_len) {
        Some(h) => h,
        None => return buf.to_string(),
    };
    let mut out = String::with_capacity(header_prefix_len + MARKER.len() + suffix.len());
    out.push_str(header_with_nl);
    out.push_str(MARKER);
    out.push_str(suffix);
    out
}

/// Render a single message as a `"role: text"` line for [`TemplateCompactor`].
///
/// Non-textual content (tool calls, tool results, attachments) is rendered
/// as a short marker so the rollup does not silently drop them but also
/// does not balloon with serialized JSON.
fn render_message_line(msg: &Message) -> String {
    use rig_core::message::AssistantContent;

    match msg {
        Message::System { content } => {
            if content.is_empty() {
                String::new()
            } else {
                format!("system: {content}")
            }
        }
        Message::User { content } => {
            let mut text = String::new();
            for c in content.iter() {
                match c {
                    UserContent::Text(t) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&t.text);
                    }
                    UserContent::ToolResult(_) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str("[tool result]");
                    }
                    _ => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str("[attachment]");
                    }
                }
            }
            if text.is_empty() {
                String::new()
            } else {
                format!("user: {text}")
            }
        }
        Message::Assistant { content, .. } => {
            let mut text = String::new();
            for c in content.iter() {
                match c {
                    AssistantContent::Text(t) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&t.text);
                    }
                    AssistantContent::ToolCall(call) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&format!("[tool call: {}]", call.function.name));
                    }
                    _ => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str("[reasoning]");
                    }
                }
            }
            if text.is_empty() {
                String::new()
            } else {
                format!("assistant: {text}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PolicyMemory
// ---------------------------------------------------------------------------

/// What an [`PolicyMemory::append`] did, as owned data.
///
/// The host acts on `demoted` (archive it) and `compaction` (fold it into a
/// rolling summary) instead of a callback firing behind its back.
#[derive(Debug, Clone, Default)]
pub struct AppendOutcome {
    /// The messages that were appended to the store, in the order given.
    pub stored: Vec<Message>,
    /// Messages the policy pushed out of the active window as a result, in
    /// original conversation order. Empty when the policy retained
    /// everything.
    pub demoted: Vec<Message>,
    /// Present when a [`Compactor`] is configured and the policy demoted
    /// messages. Pass it to [`PolicyMemory::compact`] (or fold it yourself)
    /// to update the conversation's rolling summary.
    pub compaction: Option<CompactionRequest>,
}

/// A concrete [`InMemoryConversationMemory`] plus a [`MemoryPolicy`] and an
/// optional [`Compactor`].
///
/// - [`load`](Self::load) returns `[summary?, ...kept_window]`: the stored
///   history shaped by the policy, with the conversation's rolling summary
///   spliced in front when one exists.
/// - [`append`](Self::append) stores the messages and hands back an
///   [`AppendOutcome`] describing what the policy demoted and whether
///   compaction is due.
///
/// Cloning shares both the store and the summaries.
///
/// # Example
///
/// ```
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// use rig_core::completion::Message;
/// use rig_memory::{Compactor, InMemoryConversationMemory, MemoryPolicy, PolicyMemory};
///
/// let memory = PolicyMemory::new(
///     InMemoryConversationMemory::new(),
///     MemoryPolicy::sliding_window(2),
/// )
/// .with_compactor(Compactor::template());
///
/// let outcome = memory.append(
///     "c",
///     vec![
///         Message::user("first"),
///         Message::assistant("second"),
///         Message::user("third"),
///         Message::assistant("fourth"),
///     ],
/// )?;
/// assert_eq!(outcome.demoted.len(), 2);
/// let request = outcome.compaction.expect("compactor configured");
/// memory.compact(&request);
///
/// // [summary, third, fourth]
/// assert_eq!(memory.load("c")?.len(), 3);
/// # Ok(()) }
/// # run().unwrap();
/// ```
#[derive(Debug, Clone, Default)]
pub struct PolicyMemory {
    inner: InMemoryConversationMemory,
    policy: MemoryPolicy,
    compactor: Option<Compactor>,
    summaries: Arc<StdMutex<HashMap<String, TextSummary>>>,
}

impl PolicyMemory {
    /// Wrap `inner` so every loaded history is shaped by `policy`.
    pub fn new(inner: InMemoryConversationMemory, policy: MemoryPolicy) -> Self {
        Self {
            inner,
            policy,
            compactor: None,
            summaries: Arc::new(StdMutex::new(HashMap::new())),
        }
    }

    /// Configure a compactor, so appends that demote messages also produce a
    /// [`CompactionRequest`].
    pub fn with_compactor(mut self, compactor: Compactor) -> Self {
        self.compactor = Some(compactor);
        self
    }

    /// Return a reference to the wrapped store.
    pub fn inner(&self) -> &InMemoryConversationMemory {
        &self.inner
    }

    /// Return the configured policy.
    pub fn policy(&self) -> &MemoryPolicy {
        &self.policy
    }

    /// Return the configured compactor, if any.
    pub fn compactor(&self) -> Option<&Compactor> {
        self.compactor.as_ref()
    }

    /// Consume the wrapper and return its components.
    pub fn into_inner(self) -> (InMemoryConversationMemory, MemoryPolicy, Option<Compactor>) {
        (self.inner, self.policy, self.compactor)
    }

    /// Load the policy-shaped history for `conversation_id`, with the
    /// conversation's rolling summary spliced in front when one exists.
    pub fn load(&self, conversation_id: &str) -> Result<Vec<Message>, MemoryError> {
        Ok(self.load_with_outcome(conversation_id)?.kept)
    }

    /// Like [`load`](Self::load), but also reports what the policy demoted
    /// from the stored history. The summary (when present) is the first
    /// element of `kept`.
    pub fn load_with_outcome(&self, conversation_id: &str) -> Result<PolicyOutcome, MemoryError> {
        let stored = self.inner.load(conversation_id)?;
        let mut outcome = self.policy.apply(stored);
        if let Some(summary) = self.summary(conversation_id)? {
            let mut kept = Vec::with_capacity(outcome.kept.len() + 1);
            kept.push(summary.into());
            kept.append(&mut outcome.kept);
            outcome.kept = kept;
        }
        Ok(outcome)
    }

    /// Append `messages` and report what the policy demoted as a result.
    ///
    /// `demoted` carries only the messages this append pushed out of the
    /// active window — messages an earlier append already reported are not
    /// replayed — so a host archiving demotions sees each message once
    /// without keeping a delivery watermark.
    pub fn append(
        &self,
        conversation_id: &str,
        messages: Vec<Message>,
    ) -> Result<AppendOutcome, MemoryError> {
        // Demotion is a growing prefix of the stored history, so the messages
        // *newly* pushed out by this append are the difference between the
        // demoted prefix before and after it.
        let already_demoted = self
            .policy
            .apply(self.inner.load(conversation_id)?)
            .demoted
            .len();

        self.inner.append(conversation_id, messages.clone())?;

        let stored_history = self.inner.load(conversation_id)?;
        let mut demoted = self.policy.apply(stored_history).demoted;
        let split = already_demoted.min(demoted.len());
        let demoted = demoted.split_off(split);
        let compaction = match (&self.compactor, demoted.is_empty()) {
            (Some(_), false) => Some(CompactionRequest {
                conversation_id: conversation_id.to_string(),
                evicted: demoted.clone(),
            }),
            _ => None,
        };

        Ok(AppendOutcome {
            stored: messages,
            demoted,
            compaction,
        })
    }

    /// Fold `request` into the conversation's rolling summary using the
    /// configured compactor, store the result, and return it.
    ///
    /// Returns `None` when no compactor is configured. The previous summary
    /// is passed to the compactor as carry-over, so summaries roll forward.
    pub fn compact(&self, request: &CompactionRequest) -> Option<TextSummary> {
        let compactor = self.compactor.as_ref()?;
        let mut guard = self.summaries.lock().ok()?;
        let carry_over = guard.get(&request.conversation_id);
        let summary = compactor.compact(request, carry_over);
        guard.insert(request.conversation_id.clone(), summary.clone());
        Some(summary)
    }

    /// The conversation's current rolling summary, if any.
    pub fn summary(&self, conversation_id: &str) -> Result<Option<TextSummary>, MemoryError> {
        Ok(self.lock_summaries()?.get(conversation_id).cloned())
    }

    /// Replace the conversation's rolling summary. Use this when the host
    /// summarises with its own model instead of the configured compactor.
    pub fn set_summary(
        &self,
        conversation_id: &str,
        summary: TextSummary,
    ) -> Result<(), MemoryError> {
        self.lock_summaries()?
            .insert(conversation_id.to_string(), summary);
        Ok(())
    }

    /// Drop the stored history **and** the rolling summary for
    /// `conversation_id`.
    pub fn clear(&self, conversation_id: &str) -> Result<(), MemoryError> {
        self.inner.clear(conversation_id)?;
        self.forget(conversation_id);
        Ok(())
    }

    /// Drop only the rolling summary for `conversation_id`, leaving the
    /// stored history intact. Call this when a conversation has ended to
    /// bound memory usage; the summary map is otherwise unbounded.
    ///
    /// A poisoned internal lock makes this a no-op (the summary is dropped
    /// naturally when the last handle is dropped).
    pub fn forget(&self, conversation_id: &str) {
        if let Ok(mut guard) = self.summaries.lock() {
            guard.remove(conversation_id);
        }
    }

    /// Number of conversations with a stored rolling summary. Useful for
    /// telemetry and leak detection. Returns `0` if the internal lock is
    /// poisoned.
    pub fn tracked_summaries(&self) -> usize {
        self.summaries.lock().map(|g| g.len()).unwrap_or(0)
    }

    fn lock_summaries(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<String, TextSummary>>, MemoryError> {
        self.summaries
            .lock()
            .map_err(|err| MemoryError::Internal(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::OneOrMany;
    use rig_core::message::{
        AssistantContent, ToolCall, ToolFunction, ToolResult, ToolResultContent, UserContent,
    };

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    fn tool_call_msg() -> Message {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                "call_1".into(),
                ToolFunction::new("t".into(), serde_json::json!({})),
            ))),
        }
    }

    fn tool_result_msg() -> Message {
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: "call_1".into(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text("ok")),
            })),
        }
    }

    // ----- policies -----

    #[test]
    fn noop_policy_is_identity() {
        let out = MemoryPolicy::Noop.apply(vec![user("a"), assistant("b")]);
        assert_eq!(out.kept.len(), 2);
        assert!(out.demoted.is_empty());
    }

    #[test]
    fn sliding_window_passthrough_when_under_limit() {
        let policy = MemoryPolicy::sliding_window(5);
        let out = policy.apply(vec![user("1"), assistant("2")]);
        assert_eq!(out.kept.len(), 2);
        assert!(out.demoted.is_empty());
    }

    #[test]
    fn sliding_window_truncates_history() {
        let policy = MemoryPolicy::sliding_window(2);
        let kept = policy.shape(vec![user("1"), assistant("2"), user("3"), assistant("4")]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn sliding_window_drops_leading_orphan_tool_result() {
        let policy = MemoryPolicy::sliding_window(3);
        let kept = policy.shape(vec![
            tool_call_msg(),
            tool_result_msg(),
            user("after"),
            assistant("done"),
        ]);

        assert_eq!(kept.len(), 2);
        assert!(matches!(kept.first(), Some(Message::User { content })
            if matches!(content.first(), UserContent::Text(_))));
    }

    #[test]
    fn sliding_window_reports_demoted_prefix() {
        let out = MemoryPolicy::sliding_window(2).apply(vec![
            user("oldest"),
            assistant("old"),
            user("recent"),
            assistant("latest"),
        ]);
        assert_eq!(out.kept.len(), 2);
        assert_eq!(out.demoted.len(), 2);
    }

    #[test]
    fn sliding_window_demotes_orphan_tool_result_with_prefix() {
        // Window keeps the last 2 messages, but the leading message of that
        // window is an orphan tool result; it must be moved into `demoted`
        // so the host can preserve it.
        let out = MemoryPolicy::sliding_window(2).apply(vec![
            tool_call_msg(),
            tool_result_msg(),
            user("after"),
            assistant("done"),
        ]);
        assert_eq!(out.kept.len(), 2);
        assert!(matches!(out.kept.first(), Some(Message::User { content })
            if matches!(content.first(), UserContent::Text(_))));
        assert_eq!(out.demoted.len(), 2);
    }

    #[test]
    fn token_window_keeps_within_budget() {
        let policy = MemoryPolicy::token_window(2, TokenCounter::fixed(1));
        let kept = policy.shape(vec![
            user("aaaa"),
            assistant("bbbb"),
            user("cccc"),
            assistant("dddd"),
        ]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn token_window_passes_through_when_under_budget() {
        let policy = MemoryPolicy::token_window(usize::MAX, TokenCounter::fixed(1));
        let kept = policy.shape(vec![user("a"), assistant("b")]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn token_window_drops_leading_orphan_tool_result() {
        let policy = MemoryPolicy::token_window(25, TokenCounter::fixed(10));
        let kept = policy.shape(vec![tool_call_msg(), tool_result_msg(), user("after")]);
        assert_eq!(kept.len(), 1);
        assert!(matches!(kept.first(), Some(Message::User { content })
            if matches!(content.first(), UserContent::Text(_))));
    }

    #[test]
    fn token_window_skips_message_larger_than_budget() {
        let policy = MemoryPolicy::token_window(5, TokenCounter::fixed(10));
        assert!(policy.shape(vec![user("anything")]).is_empty());
    }

    #[test]
    fn token_window_reports_demoted_prefix() {
        let out = MemoryPolicy::token_window(2, TokenCounter::fixed(1)).apply(vec![
            user("a"),
            assistant("b"),
            user("c"),
            assistant("d"),
        ]);
        assert_eq!(out.kept.len(), 2);
        assert_eq!(out.demoted.len(), 2);
    }

    // ----- token counters -----

    #[test]
    fn heuristic_counter_charges_overhead_per_message() {
        let counter = TokenCounter::heuristic();
        assert!(
            counter.count(&user("")) >= 4,
            "default per-message overhead is at least 4 tokens"
        );
    }

    #[test]
    fn heuristic_counter_is_monotonic_in_text_length() {
        let counter = TokenCounter::heuristic();
        let small = counter.count(&user("hi"));
        let big = counter.count(&user(&"x".repeat(400)));
        assert!(big > small);
    }

    #[test]
    fn heuristic_counter_handles_tool_calls() {
        assert!(TokenCounter::heuristic().count(&tool_call_msg()) > 0);
    }

    #[test]
    fn heuristic_counter_handles_system_messages() {
        let cost = TokenCounter::heuristic().count(&Message::System {
            content: "you are helpful".into(),
        });
        assert!(cost > 0);
    }

    #[test]
    fn heuristic_counter_clamps_invalid_bytes_per_token() {
        // Zero/NaN/negative ratios fall back to 1.0 instead of panicking.
        let counter = HeuristicTokenCounter::new(0.0, 0, 0);
        assert!(counter.count(&user("abcd")) >= 4);
        let nan = HeuristicTokenCounter::new(f32::NAN, 0, 0);
        assert!(nan.count(&user("abcd")) >= 4);
    }

    #[test]
    fn heuristic_counter_drives_token_window() {
        let policy = MemoryPolicy::token_window(100, TokenCounter::heuristic());
        let kept = policy.shape(vec![user(&"a".repeat(2_000)), user("short")]);
        // The huge message must be evicted; the short one retained.
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn fixed_counter_drives_token_window() {
        let policy = MemoryPolicy::token_window(2, TokenCounter::Fixed(1));
        let kept = policy.shape(vec![user("a"), assistant("b"), user("c")]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn heuristic_counter_converts_into_token_counter() {
        let counter: TokenCounter = HeuristicTokenCounter::anthropic().into();
        assert!(counter.count(&user("hello")) > 0);
    }

    // ----- PolicyMemory -----

    #[test]
    fn policy_memory_truncates_loaded_history() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(2),
        );

        mem.append(
            "c",
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .unwrap();

        assert_eq!(mem.load("c").unwrap().len(), 2);
    }

    #[test]
    fn policy_memory_append_and_clear_delegate_to_store() {
        let mem = PolicyMemory::new(InMemoryConversationMemory::new(), MemoryPolicy::Noop);
        let outcome = mem.append("c", vec![user("hi"), assistant("ok")]).unwrap();
        assert_eq!(outcome.stored.len(), 2);
        assert_eq!(mem.load("c").unwrap().len(), 2);

        mem.clear("c").unwrap();
        assert!(mem.load("c").unwrap().is_empty());
    }

    #[test]
    fn policy_memory_append_reports_demotions_once_each() {
        // The classic demotion hook fired once per newly-evicted batch; the
        // owned outcome reports exactly the messages newly pushed out.
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(2),
        );

        let first = mem
            .append(
                "c",
                vec![user("1"), assistant("2"), user("3"), assistant("4")],
            )
            .unwrap();
        assert_eq!(first.demoted.len(), 2);
        assert!(
            first.compaction.is_none(),
            "no compactor configured, no compaction request"
        );

        // Loading does not re-report or replay the demotion.
        assert_eq!(mem.load("c").unwrap().len(), 2);
    }

    #[test]
    fn policy_memory_append_reports_nothing_when_nothing_evicted() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(10),
        );
        let outcome = mem.append("c", vec![user("1"), assistant("2")]).unwrap();
        assert!(outcome.demoted.is_empty());
        assert!(outcome.compaction.is_none());
    }

    #[test]
    fn policy_memory_load_with_outcome_reports_demoted() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        );
        mem.inner()
            .append("c", vec![user("old"), assistant("new")])
            .unwrap();

        let outcome = mem.load_with_outcome("c").unwrap();
        assert_eq!(outcome.kept.len(), 1);
        assert_eq!(outcome.demoted.len(), 1);
    }

    #[test]
    fn policy_memory_isolates_conversations() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        )
        .with_compactor(Compactor::template());

        for id in ["a", "b"] {
            let outcome = mem
                .append(id, vec![user("1"), assistant("2"), user("3")])
                .unwrap();
            let request = outcome.compaction.expect("compaction due");
            assert_eq!(request.conversation_id, id);
            mem.compact(&request);
        }

        assert_eq!(mem.load("a").unwrap().len(), 2);
        assert_eq!(mem.load("b").unwrap().len(), 2);
        assert_eq!(mem.tracked_summaries(), 2);
    }

    #[test]
    fn policy_memory_into_inner_returns_components() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        )
        .with_compactor(Compactor::template());
        let (_store, _policy, compactor) = mem.into_inner();
        assert!(compactor.is_some());
    }

    // ----- compaction -----

    #[test]
    fn compaction_not_requested_without_demotion() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(10),
        )
        .with_compactor(Compactor::template());

        let outcome = mem
            .append("c", vec![user("hi"), assistant("hello")])
            .unwrap();
        assert!(outcome.compaction.is_none());
        let loaded = mem.load("c").unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(matches!(&loaded[0], Message::User { .. }));
        assert_eq!(mem.tracked_summaries(), 0);
    }

    #[test]
    fn compaction_splices_summary_in_front_of_window() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(2),
        )
        .with_compactor(Compactor::template());

        let outcome = mem
            .append(
                "c",
                vec![
                    user("first"),
                    assistant("second"),
                    user("third"),
                    assistant("fourth"),
                ],
            )
            .unwrap();
        mem.compact(&outcome.compaction.expect("compaction due"));

        let loaded = mem.load("c").unwrap();
        // Expected shape: [summary, third, fourth]
        assert_eq!(loaded.len(), 3);
        let Message::System { content } = &loaded[0] else {
            panic!("expected summary as system message");
        };
        assert!(content.contains("[Conversation summary so far]"));
        assert!(content.contains("user: first"));
        assert!(content.contains("assistant: second"));
        let Message::User { content } = &loaded[1] else {
            panic!("expected kept user message");
        };
        let UserContent::Text(t) = content.first_ref() else {
            panic!("expected text");
        };
        assert_eq!(t.text, "third");
    }

    #[test]
    fn compaction_rolls_summary_forward() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(2),
        )
        .with_compactor(Compactor::template());

        let outcome = mem
            .append(
                "c",
                vec![user("a"), assistant("b"), user("c"), assistant("d")],
            )
            .unwrap();
        mem.compact(&outcome.compaction.expect("compaction due"));
        let first_summary = mem.summary("c").unwrap().expect("summary stored");
        assert!(first_summary.as_str().contains("user: a"));
        assert!(first_summary.as_str().contains("assistant: b"));

        // Append more turns; the next compaction folds the previous summary
        // into a new one that also covers the newly-evicted prefix.
        let outcome = mem.append("c", vec![user("e"), assistant("f")]).unwrap();
        let request = outcome.compaction.expect("compaction due");
        // Only the newly-evicted prefix is handed to the compactor.
        assert_eq!(request.evicted.len(), 2);
        let second = mem.compact(&request).expect("compactor configured");
        assert!(second.as_str().contains(first_summary.as_str()));
        assert!(second.as_str().contains("user: c"));
        assert!(second.as_str().contains("assistant: d"));
    }

    #[test]
    fn compaction_is_not_requested_without_a_compactor() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        );
        let outcome = mem
            .append("c", vec![user("a"), assistant("b"), user("c")])
            .unwrap();
        assert_eq!(outcome.demoted.len(), 2);
        assert!(outcome.compaction.is_none());
        assert!(
            mem.compact(&CompactionRequest {
                conversation_id: "c".into(),
                evicted: outcome.demoted,
            })
            .is_none(),
            "compact is a no-op without a compactor"
        );
    }

    #[test]
    fn loading_twice_returns_the_same_summary() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        )
        .with_compactor(Compactor::template());
        let outcome = mem
            .append("c", vec![user("a"), assistant("b"), user("c")])
            .unwrap();
        mem.compact(&outcome.compaction.expect("compaction due"));

        let first = mem.load("c").unwrap();
        let second = mem.load("c").unwrap();
        assert_eq!(first.len(), second.len());
        let (Message::System { content: c1 }, Message::System { content: c2 }) =
            (&first[0], &second[0])
        else {
            panic!("expected summaries");
        };
        assert_eq!(c1, c2);
    }

    #[test]
    fn clear_drops_history_and_summary() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        )
        .with_compactor(Compactor::template());
        let outcome = mem
            .append("c", vec![user("a"), assistant("b"), user("c")])
            .unwrap();
        mem.compact(&outcome.compaction.expect("compaction due"));
        assert_eq!(mem.tracked_summaries(), 1);

        mem.clear("c").unwrap();
        assert_eq!(mem.tracked_summaries(), 0);
        assert!(mem.load("c").unwrap().is_empty());

        // A fresh conversation under the same id starts from no carry-over.
        let outcome = mem
            .append("c", vec![user("x"), assistant("y"), user("z")])
            .unwrap();
        let summary = mem
            .compact(&outcome.compaction.expect("compaction due"))
            .expect("compactor configured");
        assert!(!summary.as_str().contains("user: a"), "no stale carry-over");
    }

    #[test]
    fn forget_drops_summary_but_keeps_history() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        )
        .with_compactor(Compactor::template());
        let outcome = mem
            .append("c", vec![user("a"), assistant("b"), user("c")])
            .unwrap();
        mem.compact(&outcome.compaction.expect("compaction due"));
        assert_eq!(mem.tracked_summaries(), 1);

        mem.forget("c");
        assert_eq!(mem.tracked_summaries(), 0);
        assert_eq!(mem.load("c").unwrap().len(), 1, "history is untouched");
    }

    #[test]
    fn set_summary_overrides_the_stored_summary() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(1),
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .unwrap();
        mem.set_summary("c", TextSummary::new("host-authored summary"))
            .unwrap();

        let loaded = mem.load("c").unwrap();
        assert_eq!(loaded.len(), 2);
        let Message::System { content } = &loaded[0] else {
            panic!("expected summary")
        };
        assert_eq!(content, "host-authored summary");
    }

    #[test]
    fn compaction_composes_with_token_window() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::token_window(
                30,
                TokenCounter::Heuristic(HeuristicTokenCounter::openai()),
            ),
        )
        .with_compactor(Compactor::template());
        let outcome = mem
            .append(
                "c",
                vec![
                    user("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                    assistant("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
                    user("cccccccccccccccccccc"),
                    assistant("d"),
                ],
            )
            .unwrap();
        mem.compact(&outcome.compaction.expect("prefix should be evicted"));

        let loaded = mem.load("c").unwrap();
        assert!(loaded.len() >= 2);
        assert!(matches!(&loaded[0], Message::System { .. }));
    }

    #[test]
    fn summary_stays_bounded_across_rolls() {
        let cap = 512;
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            MemoryPolicy::sliding_window(2),
        )
        .with_compactor(Compactor::Template(
            TemplateCompactor::new().with_max_bytes(cap),
        ));
        mem.append("c", vec![user("seed-a"), assistant("seed-b")])
            .unwrap();
        for i in 0..30 {
            let outcome = mem
                .append(
                    "c",
                    vec![
                        user(&format!("user line {i} ----- padding padding padding")),
                        assistant(&format!("assistant line {i} ----- padding padding")),
                    ],
                )
                .unwrap();
            if let Some(request) = &outcome.compaction {
                mem.compact(request);
            }
        }
        let summary = mem.summary("c").unwrap().expect("summary stored");
        // Allow some slack for header + marker overhead.
        let slack = "[Conversation summary so far]\n[\u{2026}truncated\u{2026}]\n".len();
        assert!(
            summary.as_str().len() <= cap + slack,
            "summary grew to {} bytes (cap {}, slack {})",
            summary.as_str().len(),
            cap,
            slack,
        );
    }

    // ----- TemplateCompactor -----

    #[test]
    fn template_compactor_renders_system_messages() {
        let summary = TemplateCompactor::new().compact(
            &[
                Message::System {
                    content: "you are helpful".into(),
                },
                user("hi"),
                assistant("hello"),
            ],
            None,
        );
        let s = summary.as_str();
        assert!(s.contains("system: you are helpful"), "got: {s}");
        assert!(s.contains("user: hi"));
        assert!(s.contains("assistant: hello"));
    }

    #[test]
    fn template_compactor_renders_tool_call_marker() {
        let summary = TemplateCompactor::new().compact(&[tool_call_msg(), tool_result_msg()], None);
        let s = summary.as_str();
        assert!(s.contains("[tool call: t]"), "got: {s}");
        assert!(s.contains("[tool result]"), "got: {s}");
    }

    #[test]
    fn template_compactor_carry_over_threaded() {
        let compactor = TemplateCompactor::new();
        let first = compactor.compact(&[user("hello")], None);
        assert!(!first.as_str().is_empty());

        let second = compactor.compact(&[assistant("world")], Some(&first));
        assert!(second.as_str().contains(first.as_str()));
        assert!(second.as_str().contains("assistant: world"));
    }

    #[test]
    fn template_compactor_artifact_into_message() {
        let msg: Message = TextSummary::new("rolled-up text").into();
        let Message::System { content } = msg else {
            panic!("expected system message");
        };
        assert_eq!(content, "rolled-up text");
    }

    #[test]
    fn template_compactor_caps_summary_at_max_bytes() {
        let cap = 256;
        let compactor = TemplateCompactor::new().with_max_bytes(cap);
        let evicted: Vec<Message> = (0..50)
            .map(|i| user(&format!("message number {i} with some filler")))
            .collect();
        let summary = compactor.compact(&evicted, None);
        assert!(
            summary.as_str().len()
                <= cap + "[Conversation summary so far]\n[\u{2026}truncated\u{2026}]\n".len(),
            "summary len {} exceeds cap {} (plus header+marker)",
            summary.as_str().len(),
            cap,
        );
        assert!(
            summary
                .as_str()
                .starts_with("[Conversation summary so far]\n")
        );
        assert!(summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
        assert!(summary.as_str().contains("message number 49"));
    }

    #[test]
    fn template_compactor_unbounded_by_default() {
        let evicted: Vec<Message> = (0..200).map(|i| user(&format!("msg {i}"))).collect();
        let summary = TemplateCompactor::new().compact(&evicted, None);
        assert!(!summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
        assert!(summary.as_str().contains("msg 0"));
        assert!(summary.as_str().contains("msg 199"));
    }

    #[test]
    fn template_compactor_with_max_bytes_zero_is_unbounded() {
        let evicted: Vec<Message> = (0..200).map(|i| user(&format!("msg {i}"))).collect();
        let summary = TemplateCompactor::new()
            .with_max_bytes(0)
            .compact(&evicted, None);
        assert!(!summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
    }

    #[test]
    fn template_compactor_caps_summary_with_multiline_header() {
        // A header containing embedded newlines must not break the
        // truncation boundary calculation. The first newline in the
        // assembled buffer marks the header/body split, regardless of
        // how the caller chose to format the header.
        let cap = 256;
        let compactor = TemplateCompactor::with_header("line one\nline two").with_max_bytes(cap);
        let evicted: Vec<Message> = (0..50)
            .map(|i| user(&format!("message number {i} with some filler")))
            .collect();
        let text = compactor.compact(&evicted, None).into_string();

        assert!(text.starts_with("line one\n"));
        assert!(text.contains("[\u{2026}truncated\u{2026}]"));
        assert!(text.contains("message number 49"));
        let overhead = "line one\n".len() + "[\u{2026}truncated\u{2026}]\n".len();
        assert!(
            text.len() <= cap + overhead,
            "summary len {} exceeds cap {} plus overhead {}",
            text.len(),
            cap,
            overhead,
        );
    }
}
